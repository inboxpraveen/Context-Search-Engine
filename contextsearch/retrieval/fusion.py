"""Combining ranked lists from several retrievers into one.

Two methods:

* Reciprocal rank fusion (RRF). Each retriever contributes ``1 / (k + rank)``
  for every chunk it returned; the sums are the fused scores. It ignores the
  raw scores entirely: BM25 values and cosine similarities live on different
  scales, and this way nothing needs calibrating. A chunk found by both
  retrievers reliably outranks one found by only one. Cormack, Clarke and
  Buettcher, SIGIR 2009.

* Weighted. Each retriever's scores are min-max normalised to 0..1 over its
  own candidate list, dense retrievers are averaged, and the result is
  ``alpha * dense + (1 - alpha) * lexical``. Chunks missing from a list get 0
  for it. This lets you lean towards keywords or meaning, at the cost of
  being sensitive to how the scores are distributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import RetrieverResult


@dataclass
class FusedCandidate:
    chunk_id: int
    score: float
    contributions: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)

    @property
    def found_by(self) -> list[str]:
        return list(self.ranks)


def rrf(results: list[RetrieverResult], k: int = 60) -> list[FusedCandidate]:
    fused: dict[int, FusedCandidate] = {}
    for result in results:
        for hit in result.hits:
            cand = fused.get(hit.chunk_id)
            if cand is None:
                cand = fused[hit.chunk_id] = FusedCandidate(hit.chunk_id, 0.0)
            contribution = 1.0 / (k + hit.rank)
            cand.score += contribution
            cand.contributions[result.name] = contribution
            cand.ranks[result.name] = hit.rank
            cand.raw_scores[result.name] = hit.score
    return sorted(fused.values(), key=lambda c: (-c.score, c.chunk_id))


def weighted(results: list[RetrieverResult], alpha: float = 0.5) -> list[FusedCandidate]:
    lexical = [r for r in results if r.kind == "lexical"]
    dense = [r for r in results if r.kind == "dense"]
    normalised: dict[str, dict[int, float]] = {r.name: _min_max(r) for r in results}

    fused: dict[int, FusedCandidate] = {}
    for result in results:
        for hit in result.hits:
            cand = fused.get(hit.chunk_id)
            if cand is None:
                cand = fused[hit.chunk_id] = FusedCandidate(hit.chunk_id, 0.0)
            cand.ranks[result.name] = hit.rank
            cand.raw_scores[result.name] = hit.score

    for cand in fused.values():
        lex = max((normalised[r.name].get(cand.chunk_id, 0.0) for r in lexical), default=0.0)
        dense_scores = [normalised[r.name].get(cand.chunk_id, 0.0) for r in dense]
        den = sum(dense_scores) / len(dense_scores) if dense_scores else 0.0
        if lexical and dense:
            cand.score = alpha * den + (1 - alpha) * lex
        elif dense:
            cand.score = den
        else:
            cand.score = lex
        for r in lexical:
            cand.contributions[r.name] = (1 - alpha if dense else 1.0) * normalised[r.name].get(cand.chunk_id, 0.0)
        for r in dense:
            share = (alpha if lexical else 1.0) / len(dense)
            cand.contributions[r.name] = share * normalised[r.name].get(cand.chunk_id, 0.0)
    return sorted(fused.values(), key=lambda c: (-c.score, c.chunk_id))


def _min_max(result: RetrieverResult) -> dict[int, float]:
    if not result.hits:
        return {}
    scores = [h.score for h in result.hits]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return {h.chunk_id: 1.0 for h in result.hits}
    return {h.chunk_id: (h.score - lo) / (hi - lo) for h in result.hits}


def fuse(results: list[RetrieverResult], method: str = "rrf", k: int = 60, alpha: float = 0.5) -> list[FusedCandidate]:
    if method == "weighted":
        return weighted(results, alpha)
    return rrf(results, k)
