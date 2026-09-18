"""Scanning folders, indexing files, background jobs and file watching."""

from .ingestor import Ingestor, IngestOutcome, stat_for
from .scanner import (
    FileStat,
    ScanResult,
    SyncPlan,
    hash_file,
    is_excluded,
    normalize_path,
    plan_sync,
    relative_posix,
    scan_folder,
)
from .watcher import FolderWatcher
from .worker import IngestWorker, Job, JobRegistry

__all__ = [
    "FileStat",
    "FolderWatcher",
    "IngestOutcome",
    "IngestWorker",
    "Ingestor",
    "Job",
    "JobRegistry",
    "ScanResult",
    "SyncPlan",
    "hash_file",
    "is_excluded",
    "normalize_path",
    "plan_sync",
    "relative_posix",
    "scan_folder",
    "stat_for",
]
