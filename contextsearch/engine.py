"""The engine: one object that owns the store, models, indexes and workers.

Library use::

    from contextsearch import Engine

    engine = Engine.open("./data")            # creates data/contextsearch.db
    engine.start()
    source, job = engine.add_folder("/srv/docs")
    job.wait()                                # or let it run in the background
    for hit in engine.search("refund policy for late deliveries").hits:
        print(hit.confidence_label, hit.document.relpath, hit.chunk.page, hit.snippet)
    engine.close()

The API server and the CLI are thin layers over this class.
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

from . import __version__
from .chunking import get_chunker
from .config import Settings
from .embedding import EmbedderRegistry, QueryEmbeddingCache, dense_available
from .errors import ConfigurationError, ConflictError, ForbiddenError, ModelError, NotFoundError, ValidationError
from .ingest import FolderWatcher, Ingestor, IngestWorker, Job, JobRegistry, normalize_path
from .observability import configure_logging, get_logger, metrics
from .readers import ocr, register_builtin_readers, supported_extensions
from .rerank import Reranker, RerankerRegistry
from .retrieval import DenseRetriever, Searcher, SearchRequest, SearchResponse
from .store import Document, Source, Store
from .vectors import VectorIndex

log = get_logger("contextsearch.engine")

UPLOADS_SOURCE_NAME = "Uploads"


class Engine:
    def __init__(self, settings: Settings | None = None, configure_logs: bool = True):
        self.settings = settings or Settings.from_env()
        if configure_logs:
            configure_logging(self.settings.log_level, self.settings.log_format, self.settings.log_file_path)
        self.settings.ensure_dirs()
        try:
            get_chunker(self.settings.chunker)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        register_builtin_readers(self.settings)
        self.store = Store(self.settings.db_path)
        self.embedders = EmbedderRegistry(self.settings)
        self.rerankers = RerankerRegistry(self.settings)
        self.indexes: dict[str, VectorIndex] = {}
        self.query_cache = QueryEmbeddingCache(self.settings.query_cache_size)
        self.jobs = JobRegistry()
        self.ingestor = Ingestor(self.store, self.settings, self.embedders, self.indexes)
        self.worker = IngestWorker(self.store, self.settings, self.ingestor, self.jobs)
        self.watcher = FolderWatcher(self.store, self.settings, self.worker)
        self.searcher = Searcher(
            self.store, self.settings, self._dense_retriever, self.available_models, self._reranker
        )
        self._retrievers: dict[str, DenseRetriever] = {}
        self._lock = threading.RLock()
        self.started_at = time.time()
        self.running = False
        self._warm_thread: threading.Thread | None = None
        self._uploads_source: Source | None = None

    @classmethod
    def open(cls, data_dir: str | Path | None = None, **overrides: Any) -> Engine:
        if data_dir is not None:
            overrides["data_dir"] = str(data_dir)
        return cls(Settings.from_env(**overrides))

    # ---------------------------------------------------------------- lifecycle
    def start(self, warm: bool = True, workers: bool = True) -> None:
        """Open the index and, unless ``workers`` is False, start the background threads.

        One-shot commands (search from the CLI, evaluation) pass ``workers=False``
        so nothing runs in the background and the process exits promptly.
        """
        with self._lock:
            if self.running:
                return
            self._ensure_uploads_source()
            self._load_indexes()
            if workers:
                self.worker.start()
                if self.settings.watch_folders:
                    self.watcher.start()
                for model in self.settings.embedding_model_list:
                    if model not in self.indexes:
                        self.worker.enqueue_build_model(model)
            self.running = True
            if warm and workers:
                self._warm_thread = threading.Thread(target=self._warm, name="contextsearch-warm", daemon=True)
                self._warm_thread.start()
            log.info(
                "engine started",
                extra_fields={
                    "version": __version__,
                    "data_dir": str(self.settings.data_path),
                    "profile": self.settings.profile,
                    "models": ",".join(self.indexes) or "none",
                    "documents": self.store.stats()["documents"],
                    "workers": workers,
                },
            )

    def close(self) -> None:
        with self._lock:
            if self.running:
                self.watcher.stop()
                self.worker.stop()
                self.running = False
            self.store.close()

    def __enter__(self) -> Engine:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_uploads_source(self) -> Source:
        path = normalize_path(self.settings.uploads_path)
        source = self.store.get_source_by_path(path)
        if source is None:
            source = self.store.add_source("uploads", UPLOADS_SOURCE_NAME, path, recursive=True)
        self._uploads_source = source
        return source

    def _load_indexes(self) -> None:
        for model in self.store.list_models():
            name = model["name"]
            if name in self.indexes:
                continue
            index = VectorIndex(name, int(model["dimension"]), self.settings.vector_compact_ratio)
            index.load(self.store)
            self.indexes[name] = index
        metrics.gauge("contextsearch_vector_indexes", len(self.indexes))

    def _warm(self) -> None:
        """Load models in the background so the first search does not pay for it."""
        for name in list(self.indexes):
            try:
                self.embedders.get(name).warm_up()
            except ModelError as exc:
                log.warning("could not warm embedding model", extra_fields={"model": name, "error": exc.message})
        if self.settings.rerank:
            try:
                self.rerankers.get(self.settings.rerank_model).score("warm up", ["warm up"])
            except ModelError as exc:
                log.warning(
                    "could not warm reranker", extra_fields={"model": self.settings.rerank_model, "error": exc.message}
                )

    # ---------------------------------------------------------------- search
    def search(self, query: str | SearchRequest, **options: Any) -> SearchResponse:
        request = query if isinstance(query, SearchRequest) else SearchRequest(query=query, **options)
        return self.searcher.search(request)

    def compare(self, query: str, variants: list[dict[str, Any]]) -> list[SearchResponse]:
        """Run the same query under several configurations, for side-by-side views."""
        if not variants:
            raise ValidationError("at least one variant is required")
        if len(variants) > 4:
            raise ValidationError("at most four variants can be compared at once")
        return [self.searcher.search(SearchRequest(query=query, **v)) for v in variants]

    def available_models(self) -> list[str]:
        return [name for name, index in self.indexes.items() if index.loaded]

    def _dense_retriever(self, model: str) -> DenseRetriever | None:
        index = self.indexes.get(model)
        if index is None:
            return None
        retriever = self._retrievers.get(model)
        if retriever is None:
            retriever = DenseRetriever(self.embedders.get(model), index, self.query_cache)
            self._retrievers[model] = retriever
        return retriever

    def _reranker(self, model: str) -> Reranker:
        return self.rerankers.get(model)

    # ---------------------------------------------------------------- sources
    def add_folder(
        self, path: str, name: str = "", recursive: bool = True, include: str = "", exclude: str = ""
    ) -> tuple[Source, Job]:
        if not (path or "").strip():
            raise ValidationError("a folder path is required")
        normalized = normalize_path(Path(path.strip()).expanduser())
        if not os.path.isdir(normalized):
            raise ValidationError(f"folder does not exist: {normalized}")
        self._check_allowed(normalized)
        for existing in self.store.list_sources():
            if existing.kind == "folder" and _is_within(normalized, existing.path):
                raise ConflictError(
                    f"this folder is already covered by source '{existing.name}' ({existing.path})",
                    {"source_id": existing.id},
                )
            if existing.kind == "folder" and _is_within(existing.path, normalized):
                raise ConflictError(
                    f"source '{existing.name}' sits inside this folder; remove it first", {"source_id": existing.id}
                )
        source = self.store.add_source(
            "folder", name or Path(normalized).name or normalized, normalized, recursive, include, exclude
        )
        if self.running and self.settings.watch_folders:
            self.watcher.watch(source)
        job = self.worker.enqueue_sync(source)
        return source, job

    def remove_source(self, source_id: str) -> Source:
        source = self.store.get_source(source_id)
        if source.kind == "uploads":
            raise ConflictError("the uploads source cannot be removed; delete its documents instead")
        self.watcher.unwatch(source_id)
        chunk_ids = self.store.allowed_chunk_ids(source_id=source_id)
        removed = self.store.remove_source(source_id)
        if chunk_ids is not None:
            for index in self.indexes.values():
                index.remove(chunk_ids)
        return removed

    def sync_source(self, source_id: str) -> Job:
        return self.worker.enqueue_sync(self.store.get_source(source_id))

    def sync_all(self) -> list[Job]:
        return [self.worker.enqueue_sync(s) for s in self.store.list_sources() if s.enabled]

    def set_source_enabled(self, source_id: str, enabled: bool) -> Source:
        source = self.store.set_source_enabled(source_id, enabled)
        if enabled:
            if self.running and self.settings.watch_folders and source.kind == "folder":
                self.watcher.watch(source)
        else:
            self.watcher.unwatch(source_id)
        return source

    def _check_allowed(self, path: str) -> None:
        roots = self.settings.allowed_root_list
        if not roots:
            if self.settings.is_production:
                raise ForbiddenError(
                    "CSE_ALLOWED_ROOTS is not set; in production every source folder must sit inside an allowed root"
                )
            return
        if not any(_is_within(path, str(root)) for root in roots):
            raise ForbiddenError(f"{path} is outside the allowed roots", {"allowed_roots": [str(r) for r in roots]})

    # ---------------------------------------------------------------- uploads
    @property
    def uploads_source(self) -> Source:
        if self._uploads_source is None:
            self._ensure_uploads_source()
        assert self._uploads_source is not None
        return self._uploads_source

    def save_upload(self, filename: str, stream: BinaryIO, wait: float | None = 120.0) -> tuple[Document | None, Job]:
        """Store an uploaded file under the uploads folder and index it.

        With ``wait`` the call blocks until indexing finishes (or the timeout
        passes) and returns the document. Without it the job is returned
        straight away and the document may still be pending.
        """
        safe = _safe_filename(filename)
        ext = Path(safe).suffix.lower().lstrip(".")
        if ext not in supported_extensions() or ext not in self.settings.include_extension_set:
            raise ValidationError(
                f"file type .{ext} is not indexed; allowed: {', '.join(sorted(self.settings.include_extension_set & supported_extensions()))}"
            )
        folder = self.settings.uploads_path / datetime.now().strftime("%Y%m%d")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / safe
        counter = 2
        while target.exists():
            target = folder / f"{Path(safe).stem} ({counter}){Path(safe).suffix}"
            counter += 1
        limit = self.settings.max_upload_mb * 1024 * 1024
        written = 0
        with open(target, "wb") as out:
            while True:
                block = stream.read(1 << 20)
                if not block:
                    break
                written += len(block)
                if written > limit:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise ValidationError(f"file is larger than the {self.settings.max_upload_mb} MB upload limit")
                out.write(block)
        job = self.worker.enqueue_file(self.uploads_source, normalize_path(target))
        document = None
        if wait:
            job.wait(wait)
            doc_id = job.result.get("document_id")
            if doc_id:
                document = self.store.get_document(doc_id)
        return document, job

    def delete_document(self, document_id: str, delete_file: bool = True) -> Document:
        doc = self.store.get_document(document_id)
        source = self.store.get_source(doc.source_id)
        if source.kind != "uploads":
            raise ConflictError(
                "documents in a watched folder follow the folder: delete or move the file on disk, or add an exclude pattern to the source",
                {"path": doc.relpath, "source_id": source.id},
            )
        chunk_ids = [c.id for c in self.store.chunks_for_document(document_id)]
        self.store.delete_document(document_id)
        for index in self.indexes.values():
            index.remove(chunk_ids)
        if delete_file:
            try:
                Path(doc.path).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not delete uploaded file", extra_fields={"path": doc.path, "error": str(exc)})
        return doc

    # ---------------------------------------------------------------- models
    def build_dense_index(self, model: str) -> Job:
        ok, reason = dense_available()
        if not ok and not model.startswith("test/") and not self.embedders.has(model):
            raise ModelError(f"dense search needs the dense extra: {reason}")
        return self.worker.enqueue_build_model(model)

    def drop_dense_index(self, model: str) -> None:
        self.store.remove_model(model)
        self.indexes.pop(model, None)
        self._retrievers.pop(model, None)
        metrics.gauge("contextsearch_vector_indexes", len(self.indexes))

    def models_info(self) -> list[dict[str, Any]]:
        out = []
        stored = {m["name"]: m for m in self.store.list_models()}
        for name, row in stored.items():
            index = self.indexes.get(name)
            info: dict[str, Any] = {
                "name": name,
                "dimension": row["dimension"],
                "vectors": row["vectors"],
                "missing_vectors": self.store.count_chunks_missing_vectors(name),
                "index": index.info() if index else None,
                "embedder": self._embedder_info(name),
                "building": self.jobs.active_for("build_model", name) is not None,
            }
            out.append(info)
        return out

    def _embedder_info(self, name: str) -> dict[str, Any]:
        try:
            return self.embedders.get(name).info()
        except Exception as exc:
            return {"name": name, "error": str(exc)}

    def reindex(self, force: bool = False) -> Job:
        return self.worker.enqueue_reindex(force)

    # ---------------------------------------------------------------- documents
    def document_page(self, document_id: str, page: int) -> dict[str, Any]:
        doc = self.store.get_document(document_id)
        if doc.status != "indexed":
            raise NotFoundError("document has no indexed content")
        data = self.store.get_page(document_id, page)
        chunks = self.store.chunks_for_document(document_id, page)
        data["chunks"] = [
            {"chunk_id": c.id, "chunk_index": c.chunk_index, "char_start": c.char_start, "char_end": c.char_end}
            for c in chunks
        ]
        data["pages"] = doc.pages
        return data

    def document_detail(self, document_id: str) -> dict[str, Any]:
        doc = self.store.get_document(document_id)
        source = self.store.get_source(doc.source_id)
        return {
            **doc.to_dict(include_path=source.kind == "folder"),
            "source": {"id": source.id, "name": source.name, "kind": source.kind},
            "page_list": self.store.list_page_numbers(document_id),
            "revisions": self.store.revisions(document_id),
            "file_available": os.path.isfile(doc.path),
        }

    def document_file(self, document_id: str) -> tuple[Document, Path]:
        doc = self.store.get_document(document_id)
        path = Path(doc.path)
        if not path.is_file():
            raise NotFoundError("the file is no longer on disk")
        return doc, path

    # ---------------------------------------------------------------- status
    def health(self) -> dict[str, Any]:
        ok, reason = dense_available()
        current = self.worker.current
        return {
            "status": "ok",
            "version": __version__,
            "profile": self.settings.profile,
            "uptime_s": round(time.time() - self.started_at, 1),
            "database": {"path": str(self.settings.db_path), "ok": True},
            "dense": {"available": ok, "reason": reason or None, "device": self.settings.embedding_device or None},
            "ocr": {"enabled": self.settings.ocr, "available": ocr.available()},
            "models": [{"name": n, "vectors": i.size, "loaded": i.loaded} for n, i in self.indexes.items()],
            "watcher": {
                "enabled": self.settings.watch_folders,
                "running": self.watcher._observer is not None,
                "watched": self.watcher.watched(),
            },
            "worker": {
                "running": self.worker._thread is not None,
                "queued": self.worker.queued,
                "current_job": current.to_dict() if current else None,
            },
            "readers": sorted(supported_extensions()),
        }

    def stats(self) -> dict[str, Any]:
        data = self.store.stats()
        data["indexes"] = [i.info() for i in self.indexes.values()]
        data["memory_mb"] = round(sum(i.memory_bytes for i in self.indexes.values()) / (1024 * 1024), 1)
        data["query_cache"] = {
            "hits": self.query_cache.hits,
            "misses": self.query_cache.misses,
            "size": self.query_cache.size,
        }
        data["uptime_s"] = round(time.time() - self.started_at, 1)
        data["jobs"] = {
            "queued": self.worker.queued,
            "current": self.worker.current.to_dict() if self.worker.current else None,
        }
        return data


# ---------------------------------------------------------------------- helpers

_UNSAFE = re.compile(r"[^\w.\- ()]+", re.UNICODE)


def _safe_filename(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/")).strip()
    base = _UNSAFE.sub("_", base).strip(". ")
    if not base or base.startswith("."):
        base = "upload" + (base or "")
    return base[:180]


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(path), os.path.normcase(root)]) == os.path.normcase(root)
    except ValueError:
        return False
