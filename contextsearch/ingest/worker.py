"""Background indexing: one worker thread, a queue, and jobs you can watch.

All writes to the index go through this one thread. That keeps SQLite happy
(one writer), keeps memory bounded (one embedding batch in flight) and makes
progress easy to report. File events from the watcher are coalesced per path
so a file saved ten times in a row is read once.

Jobs are kept in memory; the last hundred are visible through the API.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..errors import ConflictError, ModelError, NotFoundError
from ..observability import get_logger, metrics, set_request_id
from ..vectors import VectorIndex
from .ingestor import Ingestor, stat_for
from .scanner import plan_sync, scan_folder

if TYPE_CHECKING:
    from ..config import Settings
    from ..store import Source, Store

log = get_logger("contextsearch.worker")

JOB_KINDS = ("sync", "file", "build_model", "reindex")


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round((self.finished_at or time.time()) - self.started_at, 1) if self.started_at else None,
            "progress": self.progress,
            "errors": self.errors[-50:],
            "error_count": len(self.errors),
            "result": self.result,
            "meta": self.meta,
            "cancel_requested": self.cancel_requested,
        }

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed", "cancelled")

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)


class JobRegistry:
    def __init__(self, keep: int = 100):
        self.keep = keep
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def create(self, kind: str, label: str, meta: dict[str, Any] | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label, meta=meta or {})
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self.keep:
                old = self._order.pop(0)
                if self._jobs[old].finished:
                    self._jobs.pop(old, None)
                else:
                    self._order.insert(0, old)
                    break
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"job {job_id} not found")
        return job

    def list(self, limit: int = 50, status: str | None = None) -> list[Job]:
        with self._lock:
            jobs = [self._jobs[i] for i in reversed(self._order) if i in self._jobs]
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs[:limit]

    def active_for(self, kind: str, key: str) -> Job | None:
        with self._lock:
            for job_id in reversed(self._order):
                job = self._jobs.get(job_id)
                if job and job.kind == kind and not job.finished and job.meta.get("key") == key:
                    return job
        return None

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.finished:
            raise ConflictError("job has already finished")
        job.cancel_requested = True
        if job.status == "queued":
            _finish(job, "cancelled")
        return job


def _finish(job: Job, status: str, result: dict[str, Any] | None = None) -> None:
    job.status = status
    job.finished_at = time.time()
    if result:
        job.result.update(result)
    job._done.set()


@dataclass
class _Task:
    job: Job
    kind: str
    source_id: str = ""
    path: str = ""
    model: str = ""
    force: bool = False


class IngestWorker:
    def __init__(self, store: Store, settings: Settings, ingestor: Ingestor, jobs: JobRegistry):
        self.store = store
        self.settings = settings
        self.ingestor = ingestor
        self.jobs = jobs
        self._queue: queue.Queue[_Task | None] = queue.Queue()
        self._pending_files: dict[str, _Task] = {}
        self._pending_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._scheduler: threading.Thread | None = None
        self._stop = threading.Event()
        self.current: Job | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="contextsearch-ingest", daemon=True)
        self._thread.start()
        if self.settings.scan_interval_seconds > 0:
            self._scheduler = threading.Thread(target=self._schedule, name="contextsearch-scheduler", daemon=True)
            self._scheduler.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if self._scheduler is not None:
            self._scheduler.join(timeout)
            self._scheduler = None

    @property
    def queued(self) -> int:
        with self._pending_lock:
            pending = len(self._pending_files)
        return self._queue.qsize() + pending

    # ---------------------------------------------------------------- enqueue
    def enqueue_sync(self, source: Source) -> Job:
        active = self.jobs.active_for("sync", source.id)
        if active is not None:
            return active
        job = self.jobs.create(
            "sync", f"Scan {source.name}", {"key": source.id, "source_id": source.id, "path": source.path}
        )
        self._queue.put(_Task(job, "sync", source_id=source.id))
        return job

    def enqueue_file(self, source: Source, path: str) -> Job:
        """Index or re-index one file. Repeated calls for the same path share one job until it starts."""
        with self._pending_lock:
            task = self._pending_files.get(path)
            if task is not None and task.job.status == "queued":
                return task.job
            job = self.jobs.create("file", f"Index {path}", {"key": path, "source_id": source.id, "path": path})
            task = _Task(job, "file", source_id=source.id, path=path)
            self._pending_files[path] = task
        self._queue.put(task)
        return job

    def enqueue_remove(self, path: str) -> Job:
        job = self.jobs.create("file", f"Remove {path}", {"key": path, "path": path, "remove": True})
        self._queue.put(_Task(job, "remove", path=path))
        return job

    def enqueue_build_model(self, model: str) -> Job:
        active = self.jobs.active_for("build_model", model)
        if active is not None:
            return active
        job = self.jobs.create("build_model", f"Build dense index for {model}", {"key": model, "model": model})
        self._queue.put(_Task(job, "build_model", model=model))
        return job

    def enqueue_reindex(self, force: bool = False) -> Job:
        active = self.jobs.active_for("reindex", "all")
        if active is not None:
            return active
        job = self.jobs.create(
            "reindex", "Reindex documents with the current chunking settings", {"key": "all", "force": force}
        )
        self._queue.put(_Task(job, "reindex", force=force))
        return job

    # ---------------------------------------------------------------- loop
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            if task.kind == "file":
                with self._pending_lock:
                    if self._pending_files.get(task.path) is task:
                        del self._pending_files[task.path]
            if task.job.cancel_requested or task.job.finished:
                continue
            self._execute(task)

    def _execute(self, task: _Task) -> None:
        job = task.job
        job.status = "running"
        job.started_at = time.time()
        self.current = job
        set_request_id(f"job-{job.id}")
        metrics.inc("contextsearch_jobs", kind=task.kind)
        try:
            if task.kind == "sync":
                self._run_sync(task)
            elif task.kind == "file":
                self._run_file(task)
            elif task.kind == "remove":
                self._run_remove(task)
            elif task.kind == "build_model":
                self._run_build_model(task)
            elif task.kind == "reindex":
                self._run_reindex(task)
            if job.status == "running":
                _finish(job, "cancelled" if job.cancel_requested else "done")
        except Exception as exc:
            log.error(
                "job failed", extra_fields={"job": job.id, "kind": task.kind, "error": f"{type(exc).__name__}: {exc}"}
            )
            job.errors.append({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]})
            _finish(job, "failed")
        finally:
            self.current = None
            set_request_id("")

    def _schedule(self) -> None:
        interval = self.settings.scan_interval_seconds
        while not self._stop.wait(interval):
            try:
                for source in self.store.list_sources():
                    if source.enabled and source.kind == "folder":
                        self.enqueue_sync(source)
            except Exception as exc:  # the scheduler must survive a transient DB error
                log.warning("periodic scan could not be queued", extra_fields={"error": str(exc)})

    # ---------------------------------------------------------------- job bodies
    def _run_sync(self, task: _Task) -> None:
        job = task.job
        source = self.store.get_source(task.source_id)
        started = time.perf_counter()
        extensions = set(self.settings.include_extension_set)
        if source.include:
            extensions = {e.strip().lower().lstrip(".") for e in source.include.split(",") if e.strip()}
        from ..readers import supported_extensions

        extensions &= supported_extensions()
        exclude = self.settings.exclude_glob_list + [g.strip() for g in source.exclude.split(",") if g.strip()]
        scan = scan_folder(source.path, source.recursive, extensions, exclude, self.settings.max_file_mb * 1024 * 1024)
        snapshot = self.store.snapshot_for_source(source.id)
        plan = plan_sync(source, snapshot, scan, self.settings.chunking_signature())
        counts = {"added": 0, "updated": 0, "unchanged": plan.unchanged, "removed": 0, "failed": 0}
        total = len(plan.new) + len(plan.changed) + len(plan.retry) + len(plan.removed)
        job.progress.update({"total": total, "done": 0, "current": "", **counts, "scan": plan.summary()})
        if scan.root_missing:
            job.errors.append({"error": f"folder is missing or not mounted: {source.path}"})
            self.store.add_event("scan", source_id=source.id, path=source.path, detail={"root_missing": True})
            return

        for stat in plan.work:
            if job.cancel_requested:
                break
            job.progress["current"] = stat.relpath
            outcome = self.ingestor.ingest(source, stat, snapshot.get(stat.path))
            counts[outcome.outcome if outcome.outcome in counts else "unchanged"] += 1
            if outcome.outcome == "failed":
                job.errors.append({"path": stat.relpath, "error": outcome.error})
            job.progress["done"] += 1
            job.progress.update(counts)
        for path, _stored in plan.removed:
            if job.cancel_requested:
                break
            job.progress["current"] = path
            self.ingestor.remove(path)
            counts["removed"] += 1
            job.progress["done"] += 1
            job.progress.update(counts)
        job.progress["current"] = ""
        summary = {
            **plan.summary(),
            **counts,
            "seconds": round(time.perf_counter() - started, 1),
            "cancelled": job.cancel_requested,
        }
        self.store.record_scan(source.id, summary)
        self.store.add_event("scan", source_id=source.id, path=source.path, detail=summary)
        self.store.trim_events()
        job.result.update(summary)
        log.info("scan finished", extra_fields={"source": source.name, **summary})

    def _run_file(self, task: _Task) -> None:
        job = task.job
        try:
            source = self.store.get_source(task.source_id)
        except NotFoundError:
            job.errors.append({"error": "source no longer exists"})
            return
        try:
            stat = stat_for(task.path, source.path)
        except FileNotFoundError:
            outcome = self.ingestor.remove(task.path)
            job.result.update({"outcome": outcome.outcome})
            return
        snapshot = self.store.snapshot_for_source(source.id).get(stat.path)
        outcome = self.ingestor.ingest(source, stat, snapshot)
        if outcome.outcome == "failed":
            job.errors.append({"path": stat.relpath, "error": outcome.error})
        job.result.update(
            {
                "outcome": outcome.outcome,
                "document_id": outcome.document_id,
                "chunks": outcome.chunks,
                "pages": outcome.pages,
                **outcome.detail,
            }
        )

    def _run_remove(self, task: _Task) -> None:
        outcome = self.ingestor.remove(task.path)
        task.job.result.update({"outcome": outcome.outcome, "document_id": outcome.document_id})

    def _run_build_model(self, task: _Task) -> None:
        job = task.job
        model = task.model
        embedder = self.ingestor.embedders.get(model)
        try:
            dimension = embedder.dimension
        except ModelError as exc:
            job.errors.append({"error": exc.message})
            _finish(job, "failed")
            return
        self.store.register_model(model, dimension, kind=type(embedder).__name__)
        index = self.ingestor.indexes.get(model)
        if index is None:
            index = VectorIndex(model, dimension, self.settings.vector_compact_ratio)
            index.load(self.store)
            self.ingestor.indexes[model] = index
        total = self.store.count_chunks_missing_vectors(model)
        job.progress.update({"total": total, "done": 0, "model": model})
        batch = 512
        while not job.cancel_requested:
            chunks = self.store.chunks_missing_vectors(model, batch)
            if not chunks:
                break
            self.ingestor.embed_chunks(model, [c.id for c in chunks], [c.text for c in chunks])
            job.progress["done"] += len(chunks)
        self.store.add_event("model_built", detail={"model": model, "vectors": self.store.count_vectors(model)})
        job.result.update({"vectors": self.store.count_vectors(model), "dimension": dimension})

    def _run_reindex(self, task: _Task) -> None:
        job = task.job
        signature = self.settings.chunking_signature()
        if task.force:
            docs, _ = self.store.list_documents(status="indexed", limit=10_000_000)
        else:
            docs = self.store.documents_with_signature_other_than(signature)
        sources = {s.id: s for s in self.store.list_sources()}
        counts = {"updated": 0, "failed": 0, "missing": 0}
        job.progress.update({"total": len(docs), "done": 0, **counts})
        for doc in docs:
            if job.cancel_requested:
                break
            job.progress["current"] = doc.relpath
            source = sources.get(doc.source_id)
            try:
                stat = stat_for(doc.path, source.path if source else "")
            except (FileNotFoundError, OSError):
                counts["missing"] += 1
                self.ingestor.remove(doc.path)
            else:
                outcome = self.ingestor.ingest(source, stat, None) if source else None
                if outcome is None or outcome.outcome == "failed":
                    counts["failed"] += 1
                    job.errors.append({"path": doc.relpath, "error": outcome.error if outcome else "source missing"})
                else:
                    counts["updated"] += 1
            job.progress["done"] += 1
            job.progress.update(counts)
        job.progress["current"] = ""
        self.store.set_meta("chunking_signature", signature)
        job.result.update(counts)
