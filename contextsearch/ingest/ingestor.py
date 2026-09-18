"""Index one file: hash, read, chunk, store, embed.

The text is committed before any embedding runs, so keyword search works
even if a model is slow or fails, and a dense index for a new model can be
built later from the stored chunks (``Engine.build_dense_index``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..chunking import chunk_document
from ..errors import ModelError, ReadError, UnsupportedFileError
from ..observability import get_logger, metrics
from ..readers import read_document
from .scanner import FileStat, hash_file

if TYPE_CHECKING:
    from ..config import Settings
    from ..embedding import EmbedderRegistry
    from ..store import ScanSnapshot, Source, Store
    from ..vectors import VectorIndex

log = get_logger("contextsearch.ingest")

EMBED_BATCH = 256  # chunks handed to the model per call; the model batches internally


@dataclass
class IngestOutcome:
    outcome: str  # added | updated | unchanged | failed
    document_id: str = ""
    chunks: int = 0
    pages: int = 0
    seconds: float = 0.0
    error: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


class Ingestor:
    def __init__(self, store: Store, settings: Settings, embedders: EmbedderRegistry, indexes: dict[str, VectorIndex]):
        self.store = store
        self.settings = settings
        self.embedders = embedders
        self.indexes = indexes  # model name -> in-memory index, shared with the engine

    @property
    def models(self) -> list[str]:
        return list(self.indexes)

    def ingest(self, source: Source, stat: FileStat, stored: ScanSnapshot | None = None) -> IngestOutcome:
        started = time.perf_counter()
        doc = self.store.upsert_pending(source, stat.path, stat.relpath, stat.size, stat.mtime)
        try:
            content_hash = hash_file(stat.path)
        except PermissionError as exc:
            return self._fail(doc.id, f"file is locked or not readable: {exc}", stat, started, locked=True)
        except OSError as exc:
            return self._fail(doc.id, f"cannot read file: {exc}", stat, started)

        signature = self.settings.chunking_signature()
        if (
            stored
            and stored.status == "indexed"
            and stored.content_hash == content_hash
            and stored.chunking_signature == signature
        ):
            self.store.touch_document(doc.id, stat.size, stat.mtime)
            return IngestOutcome("unchanged", doc.id, seconds=time.perf_counter() - started)

        try:
            parsed = read_document(stat.path, self.settings)
        except (ReadError, UnsupportedFileError) as exc:
            return self._fail(doc.id, exc.message, stat, started, locked=bool(exc.details.get("locked")))

        chunks = chunk_document(parsed, self.settings)
        old_chunk_ids = [c.id for c in self.store.chunks_for_document(doc.id)]
        metadata = dict(parsed.metadata)
        if not chunks:
            metadata["empty"] = True
        document, chunk_ids = self.store.store_indexed(
            doc.id,
            title=parsed.title,
            size=stat.size,
            mtime=stat.mtime,
            content_hash=content_hash,
            reader=parsed.reader,
            pages=[(p.number, p.text, p.ocr) for p in parsed.pages],
            chunks=[(c.page, c.index, c.char_start, c.char_end, c.word_count, c.text) for c in chunks],
            chunking_signature=signature,
            metadata=metadata,
        )
        for index in self.indexes.values():
            index.remove(old_chunk_ids)

        embed_errors: dict[str, str] = {}
        if chunks:
            texts = [c.text for c in chunks]
            for model in self.models:
                try:
                    self.embed_chunks(model, chunk_ids, texts)
                except ModelError as exc:
                    embed_errors[model] = exc.message
                    log.error(
                        "embedding failed; keyword index is still complete",
                        extra_fields={"model": model, "path": stat.path, "error": exc.message},
                    )

        outcome = "added" if document.revision == 1 or (stored and stored.status != "indexed") else "updated"
        seconds = time.perf_counter() - started
        metrics.inc("contextsearch_documents_indexed", outcome=outcome)
        metrics.observe("contextsearch_ingest_seconds", seconds * 1000)
        log.info(
            "indexed",
            extra_fields={
                "path": stat.relpath,
                "outcome": outcome,
                "revision": document.revision,
                "pages": len(parsed.pages),
                "chunks": len(chunks),
                "ms": round(seconds * 1000),
            },
        )
        return IngestOutcome(
            outcome,
            document.id,
            len(chunks),
            len(parsed.pages),
            seconds,
            detail={"embed_errors": embed_errors} if embed_errors else {},
        )

    def embed_chunks(self, model: str, chunk_ids: list[int], texts: list[str]) -> None:
        """Embed texts with one model, store the vectors and add them to the live index."""
        embedder = self.embedders.get(model)
        index = self.indexes.get(model)
        for start in range(0, len(texts), EMBED_BATCH):
            ids = chunk_ids[start : start + EMBED_BATCH]
            vectors = embedder.embed_documents(texts[start : start + EMBED_BATCH])
            self.store.store_vectors(model, ids, vectors)
            if index is not None:
                index.add(ids, vectors)

    def remove(self, path: str) -> IngestOutcome:
        doc = self.store.get_document_by_path(path)
        if doc is None or doc.status == "removed":
            return IngestOutcome("unchanged")
        old_chunk_ids = [c.id for c in self.store.chunks_for_document(doc.id)]
        self.store.mark_removed(doc.id)
        for index in self.indexes.values():
            index.remove(old_chunk_ids)
        metrics.inc("contextsearch_documents_indexed", outcome="removed")
        log.info("removed", extra_fields={"path": doc.relpath, "revision": doc.revision})
        return IngestOutcome("removed", doc.id)

    def _fail(
        self, document_id: str, error: str, stat: FileStat, started: float, locked: bool = False
    ) -> IngestOutcome:
        self.store.mark_failed(document_id, error, stat.size, stat.mtime)
        metrics.inc("contextsearch_documents_indexed", outcome="failed")
        level = log.warning if locked else log.error
        level("indexing failed", extra_fields={"path": stat.relpath, "error": error, "locked": locked})
        return IngestOutcome(
            "failed", document_id, seconds=time.perf_counter() - started, error=error, detail={"locked": locked}
        )


def stat_for(path: str, root: str) -> FileStat:
    from .scanner import normalize_path, relative_posix

    path = normalize_path(path)
    st = os.stat(path)
    return FileStat(path, relative_posix(path, root), st.st_size, st.st_mtime)
