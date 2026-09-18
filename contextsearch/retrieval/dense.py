"""Semantic retrieval: embed the query, compare with every chunk vector.

One :class:`DenseRetriever` serves one embedding model. It owns the in-memory
:class:`VectorIndex` for that model and a small cache of query vectors.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..embedding import Embedder, QueryEmbeddingCache
from ..vectors import VectorIndex
from .types import RetrieverHit, RetrieverResult

if TYPE_CHECKING:
    from ..store import Store


class DenseRetriever:
    kind = "dense"

    def __init__(self, embedder: Embedder, index: VectorIndex, cache: QueryEmbeddingCache | None = None):
        self.embedder = embedder
        self.index = index
        self.cache = cache or QueryEmbeddingCache(0)

    @property
    def model(self) -> str:
        return self.embedder.name

    @property
    def name(self) -> str:
        return f"dense:{self.model}"

    def embed_query(self, query: str) -> tuple[np.ndarray, bool]:
        cached = self.cache.get(self.model, query)
        if cached is not None:
            return cached, True
        vec = self.embedder.embed_query(query)
        self.cache.put(self.model, query, vec)
        return vec, False

    def search(
        self,
        query: str,
        limit: int,
        store: Store,
        source_id: str | None = None,
        ext: str | None = None,
        document_id: str | None = None,
    ) -> RetrieverResult:
        started = time.perf_counter()
        vec, cached = self.embed_query(query)
        embed_ms = (time.perf_counter() - started) * 1000
        allowed = store.allowed_chunk_ids(source_id=source_id, ext=ext, document_id=document_id)
        ids, scores = self.index.search(vec, limit, allowed_ids=allowed)
        hits = [
            RetrieverHit(chunk_id=int(cid), score=float(score), rank=i + 1)
            for i, (cid, score) in enumerate(zip(ids, scores, strict=True))
        ]
        details: dict[str, Any] = {
            "model": self.model,
            "vectors_searched": self.index.size,
            "query_embedding_cached": cached,
            "embed_ms": round(embed_ms, 2),
        }
        return RetrieverResult(self.name, self.kind, hits, (time.perf_counter() - started) * 1000, details)
