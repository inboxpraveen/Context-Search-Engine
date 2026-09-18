"""Measure retrieval quality on your own labelled queries.

A labels file is JSON::

    {
      "queries": [
        {"query": "refund for late delivery", "relevant": ["policies/refunds.md"]},
        {"query": "laptop on the first day", "relevant": ["onboarding.docx#3"]}
      ]
    }

Each ``relevant`` entry names a document by a case-insensitive substring of
its relative path, optionally followed by ``#page``. A result counts as a hit
when its document (and page, if given) matches.

For every strategy the report gives recall@k (was a relevant document in the
top k), MRR@k (how high the first relevant one ranked) and nDCG@k, plus the
mean latency. These are the numbers to look at when choosing between models,
fusion methods and reranking on a real corpus.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import ValidationError
from .retrieval import SearchRequest

if TYPE_CHECKING:
    from .engine import Engine


@dataclass
class StrategySpec:
    label: str
    strategy: str
    models: list[str] | None = None
    fusion: str = ""
    rerank: bool = False

    @classmethod
    def parse(cls, text: str) -> StrategySpec:
        """``hybrid``, ``dense:BAAI/bge-small-en-v1.5``, ``hybrid+rerank``, ``hybrid:weighted``."""
        label = text.strip()
        rerank = label.endswith("+rerank")
        core = label[:-7] if rerank else label
        strategy, _, rest = core.partition(":")
        if strategy not in ("lexical", "dense", "hybrid"):
            raise ValidationError(
                f"unknown strategy in {text!r}; use lexical, dense, hybrid, optionally :model or :weighted and +rerank"
            )
        models = None
        fusion = ""
        if rest:
            if rest in ("rrf", "weighted"):
                fusion = rest
            else:
                models = [rest]
        return cls(label, strategy, models, fusion, rerank)


@dataclass
class QueryResult:
    query: str
    relevant: list[str]
    hit_rank: int | None
    latency_ms: float
    top: list[str] = field(default_factory=list)


@dataclass
class StrategyReport:
    label: str
    k: int
    results: list[QueryResult]
    warnings: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return sum(1 for r in self.results if r.hit_rank is not None) / len(self.results) if self.results else 0.0

    @property
    def mrr(self) -> float:
        return sum(1.0 / r.hit_rank for r in self.results if r.hit_rank) / len(self.results) if self.results else 0.0

    @property
    def ndcg(self) -> float:
        # Binary relevance with one ideal hit at rank 1, so the ideal DCG is 1.
        return (
            sum(1.0 / math.log2(r.hit_rank + 1) for r in self.results if r.hit_rank) / len(self.results)
            if self.results
            else 0.0
        )

    @property
    def latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.label,
            "k": self.k,
            "queries": len(self.results),
            "recall_at_k": round(self.recall, 4),
            "mrr_at_k": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg, 4),
            "mean_latency_ms": round(self.latency_ms, 1),
            "misses": [r.query for r in self.results if r.hit_rank is None],
            "warnings": sorted(set(self.warnings)),
        }


def load_labels(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = data.get("queries", data) if isinstance(data, dict) else data
    out = []
    for item in queries:
        if not isinstance(item, dict) or not item.get("query") or not item.get("relevant"):
            raise ValidationError("every entry needs a 'query' and a non-empty 'relevant' list")
        out.append({"query": str(item["query"]), "relevant": [str(r) for r in item["relevant"]]})
    if not out:
        raise ValidationError("the labels file has no queries")
    return out


def _matches(result: dict[str, Any], label: str) -> bool:
    doc, _, page = label.partition("#")
    relpath = (result["document"]["relpath"] or "").lower()
    filename = (result["document"]["filename"] or "").lower()
    doc = doc.strip().lower()
    if doc not in relpath and doc != filename:
        return False
    if page:
        pages = [result["page"]] + [a["page"] for a in result.get("also_matched", [])]
        return int(page) in pages
    return True


def evaluate(
    engine: Engine, labels: list[dict[str, Any]], specs: list[StrategySpec], k: int = 10
) -> list[StrategyReport]:
    reports = []
    for spec in specs:
        results: list[QueryResult] = []
        warnings: list[str] = []
        # Every strategy embeds its queries afresh, so the numbers are comparable, and one
        # untimed query per strategy keeps model loading out of the latency column.
        engine.query_cache.clear()
        engine.search(
            SearchRequest(
                query=labels[0]["query"],
                strategy=spec.strategy,
                models=spec.models,
                fusion=spec.fusion,
                rerank=spec.rerank,
                top_k=k,
                explain=False,
            )
        )
        for item in labels:
            request = SearchRequest(
                query=item["query"],
                strategy=spec.strategy,
                models=spec.models,
                fusion=spec.fusion,
                rerank=spec.rerank,
                top_k=k,
                explain=False,
            )
            started = time.perf_counter()
            response = engine.search(request).to_dict()
            latency = (time.perf_counter() - started) * 1000
            warnings += response["warnings"]
            hit_rank = None
            for r in response["results"]:
                if any(_matches(r, label) for label in item["relevant"]):
                    hit_rank = r["rank"]
                    break
            results.append(
                QueryResult(
                    item["query"],
                    item["relevant"],
                    hit_rank,
                    latency,
                    [r["document"]["relpath"] for r in response["results"][:3]],
                )
            )
        reports.append(StrategyReport(spec.label, k, results, warnings))
    return reports


def format_table(reports: list[StrategyReport]) -> str:
    width = max(len(r.label) for r in reports) if reports else 10
    lines = [f"{'strategy':<{width}}  recall@k   mrr@k   ndcg@k   latency", "-" * (width + 40)]
    for r in reports:
        lines.append(f"{r.label:<{width}}  {r.recall:8.3f} {r.mrr:7.3f} {r.ndcg:8.3f}   {r.latency_ms:6.1f} ms")
    return "\n".join(lines)
