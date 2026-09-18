"""React to file system events on watched folders.

watchdog delivers create, modify, move and delete events. Editors and copy
tools produce bursts of them for one file, so every path is debounced: it is
handed to the worker only after ``watch_debounce_seconds`` of quiet. Events
on folders (a subtree moved or deleted) trigger a full scan of the source,
which is the simplest way to get that right on every platform.

Events are an accelerator, not the source of truth. Periodic scans catch
anything they miss, which happens on network shares and some Windows setups.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any

from ..observability import get_logger
from .scanner import is_excluded, normalize_path, relative_posix

if TYPE_CHECKING:
    from ..config import Settings
    from ..store import Source, Store
    from .worker import IngestWorker

log = get_logger("contextsearch.watcher")


class FolderWatcher:
    def __init__(self, store: Store, settings: Settings, worker: IngestWorker):
        self.store = store
        self.settings = settings
        self.worker = worker
        self._observer: Any = None
        self._watches: dict[str, Any] = {}
        self._sources: dict[str, Source] = {}
        self._pending: dict[str, tuple[float, str, str]] = {}  # path -> (last_event, kind, source_id)
        self._lock = threading.Lock()
        self._flusher: threading.Thread | None = None
        self._stop = threading.Event()
        self.available = True
        self.error = ""

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        try:
            from watchdog.observers import Observer
        except ImportError as exc:  # pragma: no cover
            self.available = False
            self.error = f"watchdog is not installed: {exc}"
            log.warning(self.error)
            return
        self._observer = Observer()
        self._observer.daemon = True
        self._observer.start()
        self._stop.clear()
        self._flusher = threading.Thread(target=self._flush_loop, name="contextsearch-watch-flush", daemon=True)
        self._flusher.start()
        for source in self.store.list_sources():
            if source.kind == "folder" and source.enabled:
                self.watch(source)

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(5)
            except Exception:
                pass
            self._observer = None
        if self._flusher is not None:
            self._flusher.join(5)
            self._flusher = None
        self._watches.clear()

    def watch(self, source: Source) -> None:
        if self._observer is None or source.id in self._watches:
            return
        if not os.path.isdir(source.path):
            log.warning("cannot watch a folder that does not exist", extra_fields={"path": source.path})
            return
        from watchdog.events import FileSystemEventHandler

        watcher = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event: Any) -> None:
                watcher._on_event(source.id, event)

        try:
            self._watches[source.id] = self._observer.schedule(Handler(), source.path, recursive=source.recursive)
            self._sources[source.id] = source
            log.info("watching folder", extra_fields={"path": source.path, "recursive": source.recursive})
        except Exception as exc:
            log.warning("cannot watch folder", extra_fields={"path": source.path, "error": str(exc)})

    def unwatch(self, source_id: str) -> None:
        watch = self._watches.pop(source_id, None)
        self._sources.pop(source_id, None)
        if watch is not None and self._observer is not None:
            try:
                self._observer.unschedule(watch)
            except Exception:
                pass

    def watched(self) -> list[str]:
        return list(self._watches)

    # ---------------------------------------------------------------- events
    def _on_event(self, source_id: str, event: Any) -> None:
        source = self._sources.get(source_id)
        if source is None:
            return
        kind = event.event_type  # created, modified, moved, deleted, closed
        if kind == "opened" or getattr(event, "is_synthetic", False):
            return
        if event.is_directory:
            if kind in ("moved", "deleted", "created"):
                self._queue(source_id, source.path, "sync", 0.0)
            return
        src = normalize_path(event.src_path)
        if kind == "moved":
            self._queue(source_id, src, "remove")
            dest = normalize_path(getattr(event, "dest_path", "") or "")
            if dest:
                self._queue(source_id, dest, "file")
            return
        if kind == "deleted":
            self._queue(source_id, src, "remove")
            return
        self._queue(source_id, src, "file")

    def _queue(self, source_id: str, path: str, kind: str, delay: float | None = None) -> None:
        with self._lock:
            self._pending[path] = (
                time.time() + (self.settings.watch_debounce_seconds if delay is None else delay),
                kind,
                source_id,
            )

    def _flush_loop(self) -> None:
        while not self._stop.wait(0.5):
            now = time.time()
            ready: list[tuple[str, str, str]] = []
            with self._lock:
                for path, (due, kind, source_id) in list(self._pending.items()):
                    if due <= now:
                        ready.append((path, kind, source_id))
                        del self._pending[path]
            for path, kind, source_id in ready:
                self._dispatch(path, kind, source_id)

    def _dispatch(self, path: str, kind: str, source_id: str) -> None:
        source = self._sources.get(source_id)
        if source is None:
            return
        try:
            if kind == "sync":
                self.worker.enqueue_sync(source)
                return
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            if ext not in self._extensions(source):
                return
            rel = relative_posix(path, source.path)
            if is_excluded(
                rel, self.settings.exclude_glob_list + [g.strip() for g in source.exclude.split(",") if g.strip()]
            ):
                return
            if kind == "remove":
                self.worker.enqueue_remove(path)
            else:
                if not os.path.isfile(path):
                    self.worker.enqueue_remove(path)
                else:
                    self.worker.enqueue_file(source, path)
        except Exception as exc:
            log.warning("could not queue file event", extra_fields={"path": path, "error": str(exc)})

    def _extensions(self, source: Source) -> set[str]:
        from ..readers import supported_extensions

        exts = set(self.settings.include_extension_set)
        if source.include:
            exts = {e.strip().lower().lstrip(".") for e in source.include.split(",") if e.strip()}
        return exts & supported_extensions()
