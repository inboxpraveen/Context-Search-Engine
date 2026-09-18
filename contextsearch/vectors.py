"""In-memory exact vector search, one matrix per embedding model.

The database keeps every vector; this class keeps a float32 copy in memory so
a query is a single matrix-vector product followed by a top-k selection. It
is exact, deterministic and fast enough for corpora up to a few million
chunks (see docs/PERFORMANCE.md for the sizing table).

Rows are appended in place; deleted chunks are tombstoned and the matrix is
compacted once the dead fraction passes ``compact_ratio``. Everything is
guarded by one lock, since a search during an append would read half-written
rows.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from .observability import get_logger

if TYPE_CHECKING:
    from .store import Store

log = get_logger("contextsearch.vectors")


class VectorIndex:
    def __init__(self, model: str, dimension: int, compact_ratio: float = 0.2):
        self.model = model
        self.dimension = dimension
        self.compact_ratio = compact_ratio
        self._lock = threading.RLock()
        self._capacity = 1024
        self._matrix = np.empty((self._capacity, dimension), dtype=np.float32)
        self._ids = np.empty(self._capacity, dtype=np.int64)
        self._alive = np.zeros(self._capacity, dtype=bool)
        self._n = 0
        self._dead = 0
        self.loaded = False
        self.load_seconds = 0.0

    # ---------------------------------------------------------------- loading
    def load(self, store: Store) -> None:
        started = time.perf_counter()
        with self._lock:
            total = store.count_vectors(self.model)
            self._reset(max(1024, total))
            for ids, matrix in store.iter_vectors(self.model):
                self._append(ids, matrix)
            self.loaded = True
            self.load_seconds = time.perf_counter() - started
        log.info(
            "vector index loaded",
            extra_fields={"model": self.model, "vectors": self._n, "seconds": round(self.load_seconds, 2)},
        )

    def _reset(self, capacity: int) -> None:
        self._capacity = capacity
        self._matrix = np.empty((capacity, self.dimension), dtype=np.float32)
        self._ids = np.empty(capacity, dtype=np.int64)
        self._alive = np.zeros(capacity, dtype=bool)
        self._n = 0
        self._dead = 0

    def _grow(self, needed: int) -> None:
        capacity = self._capacity
        while capacity < needed:
            capacity *= 2
        matrix = np.empty((capacity, self.dimension), dtype=np.float32)
        matrix[: self._n] = self._matrix[: self._n]
        ids = np.empty(capacity, dtype=np.int64)
        ids[: self._n] = self._ids[: self._n]
        alive = np.zeros(capacity, dtype=bool)
        alive[: self._n] = self._alive[: self._n]
        self._matrix, self._ids, self._alive, self._capacity = matrix, ids, alive, capacity

    def _append(self, ids: np.ndarray, matrix: np.ndarray) -> None:
        count = len(ids)
        if not count:
            return
        if matrix.shape[1] != self.dimension:
            raise ValueError(f"vector dimension {matrix.shape[1]} does not match index dimension {self.dimension}")
        if self._n + count > self._capacity:
            self._grow(self._n + count)
        self._matrix[self._n : self._n + count] = matrix
        self._ids[self._n : self._n + count] = ids
        self._alive[self._n : self._n + count] = True
        self._n += count

    # ---------------------------------------------------------------- updates
    def add(self, ids: list[int] | np.ndarray, matrix: np.ndarray) -> None:
        ids_arr = np.asarray(ids, dtype=np.int64)
        with self._lock:
            # A chunk id that is already present (re-embedded) is replaced, not duplicated.
            # Ids only grow, so a batch above the current maximum cannot contain duplicates
            # and the scan is skipped; that is every batch during a first index.
            if self._n and len(ids_arr) and ids_arr.min() <= self._ids[: self._n].max():
                dup = np.isin(self._ids[: self._n], ids_arr) & self._alive[: self._n]
                if dup.any():
                    self._alive[: self._n][dup] = False
                    self._dead += int(dup.sum())
            self._append(ids_arr, np.ascontiguousarray(matrix, dtype=np.float32))
            self._maybe_compact()

    def remove(self, ids: list[int] | np.ndarray) -> int:
        ids_arr = np.asarray(ids, dtype=np.int64)
        if not len(ids_arr):
            return 0
        with self._lock:
            if not self._n:
                return 0
            hit = np.isin(self._ids[: self._n], ids_arr) & self._alive[: self._n]
            removed = int(hit.sum())
            if removed:
                self._alive[: self._n][hit] = False
                self._dead += removed
                self._maybe_compact()
            return removed

    def _maybe_compact(self) -> None:
        if self._n and self._dead / self._n >= self.compact_ratio and self._dead > 256:
            self.compact()

    def compact(self) -> None:
        with self._lock:
            keep = self._alive[: self._n]
            matrix = self._matrix[: self._n][keep]
            ids = self._ids[: self._n][keep]
            self._reset(max(1024, len(ids)))
            self._append(ids, matrix)
            log.debug("vector index compacted", extra_fields={"model": self.model, "vectors": self._n})

    # ---------------------------------------------------------------- search
    def search(self, query: np.ndarray, k: int, allowed_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Top-k chunk ids and cosine scores for a unit-length query vector."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        with self._lock:
            n = self._n
            if not n or k <= 0:
                return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
            scores = self._matrix[:n] @ q
            mask = self._alive[:n]
            if allowed_ids is not None:
                mask = mask & np.isin(self._ids[:n], allowed_ids)
            scores = np.where(mask, scores, -np.inf)
            k = min(k, int(mask.sum()))
            if k <= 0:
                return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]
            return self._ids[:n][top].copy(), scores[top].copy()

    # ---------------------------------------------------------------- info
    @property
    def size(self) -> int:
        with self._lock:
            return self._n - self._dead

    @property
    def memory_bytes(self) -> int:
        return int(self._matrix.nbytes + self._ids.nbytes + self._alive.nbytes)

    def info(self) -> dict[str, object]:
        with self._lock:
            return {
                "model": self.model,
                "dimension": self.dimension,
                "vectors": self._n - self._dead,
                "tombstones": self._dead,
                "memory_mb": round(self.memory_bytes / (1024 * 1024), 1),
                "loaded": self.loaded,
            }
