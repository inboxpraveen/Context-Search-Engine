"""Shared result types for the retrievers and the fusion step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrieverHit:
    chunk_id: int
    score: float
    rank: int
    highlight: str | None = None


@dataclass
class RetrieverResult:
    """What one retriever (lexical, or dense for one model) found."""

    name: str
    kind: str
    hits: list[RetrieverHit]
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def by_chunk(self) -> dict[int, RetrieverHit]:
        return {h.chunk_id: h for h in self.hits}
