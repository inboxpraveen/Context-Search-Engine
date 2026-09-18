"""Walk a folder tree and decide which files need work.

A scan is cheap on purpose: it reads directory entries and file stats, never
file contents. Hashing happens later, only for files whose size or mtime
changed, so a rescan of a large tree with nothing new finishes in seconds.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..observability import get_logger

if TYPE_CHECKING:
    from ..store import ScanSnapshot, Source

log = get_logger("contextsearch.scanner")


@dataclass
class FileStat:
    path: str
    relpath: str
    size: int
    mtime: float

    @property
    def ext(self) -> str:
        return os.path.splitext(self.path)[1].lower().lstrip(".")


@dataclass
class ScanResult:
    files: list[FileStat] = field(default_factory=list)
    skipped_large: list[str] = field(default_factory=list)
    unreadable_dirs: list[str] = field(default_factory=list)
    ignored: int = 0  # unsupported extension or excluded by a glob
    root_missing: bool = False


def normalize_path(path: str | Path) -> str:
    """Absolute path with the OS separator, without resolving symlinks (slow on network shares)."""
    return os.path.normpath(os.path.abspath(str(path)))


def relative_posix(path: str, root: str) -> str:
    try:
        rel = os.path.relpath(path, root)
    except ValueError:  # different drive on Windows
        rel = os.path.basename(path)
    return rel.replace(os.sep, "/")


def is_excluded(relpath: str, patterns: list[str]) -> bool:
    name = relpath.rsplit("/", 1)[-1]
    for pattern in patterns:
        if fnmatch.fnmatch(relpath, pattern):
            return True
        if pattern.startswith("**/"):
            tail = pattern[3:]
            if fnmatch.fnmatch(relpath, tail) or fnmatch.fnmatch(name, tail):
                return True
    return False


def scan_folder(root: str, recursive: bool, extensions: set[str], exclude: list[str], max_bytes: int) -> ScanResult:
    result = ScanResult()
    root = normalize_path(root)
    if not os.path.isdir(root):
        result.root_missing = True
        return result
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            result.unreadable_dirs.append(current)
            log.warning("cannot list folder", extra_fields={"path": current, "error": str(exc)})
            continue
        for entry in entries:
            rel = relative_posix(entry.path, root)
            try:
                if entry.is_dir(follow_symlinks=False):
                    if recursive and not is_excluded(rel + "/probe", exclude):
                        stack.append(entry.path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                ext = os.path.splitext(entry.name)[1].lower().lstrip(".")
                if ext not in extensions or is_excluded(rel, exclude):
                    result.ignored += 1
                    continue
                stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                log.warning("cannot stat file", extra_fields={"path": entry.path, "error": str(exc)})
                continue
            if stat.st_size > max_bytes:
                result.skipped_large.append(entry.path)
                continue
            result.files.append(FileStat(normalize_path(entry.path), rel, stat.st_size, stat.st_mtime))
    result.files.sort(key=lambda f: f.relpath)
    return result


def hash_file(path: str, block_size: int = 1 << 20) -> str:
    digest = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SyncPlan:
    new: list[FileStat] = field(default_factory=list)
    changed: list[FileStat] = field(default_factory=list)
    retry: list[FileStat] = field(default_factory=list)
    removed: list[tuple[str, ScanSnapshot]] = field(default_factory=list)
    unchanged: int = 0
    scan: ScanResult = field(default_factory=ScanResult)

    @property
    def work(self) -> Iterator[FileStat]:
        yield from self.new
        yield from self.changed
        yield from self.retry

    def summary(self) -> dict[str, int | bool]:
        return {
            "seen": len(self.scan.files),
            "new": len(self.new),
            "changed": len(self.changed),
            "retry": len(self.retry),
            "removed": len(self.removed),
            "unchanged": self.unchanged,
            "ignored": self.scan.ignored,
            "too_large": len(self.scan.skipped_large),
            "unreadable_dirs": len(self.scan.unreadable_dirs),
            "root_missing": self.scan.root_missing,
        }


def plan_sync(source: Source, snapshot: dict[str, ScanSnapshot], scan: ScanResult, chunking_signature: str) -> SyncPlan:
    """Compare what is on disk with what is stored and sort files into buckets."""
    plan = SyncPlan(scan=scan)
    if scan.root_missing:
        return plan  # never treat an unmounted share as "everything was deleted"
    seen: set[str] = set()
    for stat in scan.files:
        seen.add(stat.path)
        stored = snapshot.get(stat.path)
        if stored is None:
            plan.new.append(stat)
        elif stored.status == "failed":
            plan.retry.append(stat)
        elif (
            stored.status != "indexed"
            or stored.chunking_signature != chunking_signature
            or stored.size != stat.size
            or abs(stored.mtime - stat.mtime) > 1e-6
        ):
            plan.changed.append(stat)
        else:
            plan.unchanged += 1
    unreadable = tuple(d.rstrip(os.sep) + os.sep for d in scan.unreadable_dirs)
    for path, stored in snapshot.items():
        if path in seen:
            continue
        if unreadable and path.startswith(unreadable):
            continue  # the folder could not be listed; do not guess
        plan.removed.append((path, stored))
    return plan
