"""The search pipeline: retrieve, fuse, rerank, explain.

Steps for one request:

1. Run the retrievers the strategy asks for: FTS5 for ``lexical``, one dense
   retriever per model for ``dense``, both for ``hybrid``.
2. Fuse the ranked lists (RRF by default).
3. Optionally rerank the top fused candidates with a cross-encoder.
4. Load the chunks and documents, build a highlighted snippet for each hit,
   compute a confidence and a label, and group chunks by document.

Every hit carries its evidence: which retrievers found it and at what rank
and score, which query terms matched, the exact page and character span in
the source, and the stage timings for the request.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..errors import ModelError, ValidationError
from ..observability import Trace, get_logger, metrics
from ..store import HL_CLOSE, HL_OPEN
from .fusion import FusedCandidate, fuse
from .lexical import LexicalRetriever, covered_terms, matched_terms, parse_query
from .types import RetrieverResult

if TYPE_CHECKING:
    from ..config import Settings
    from ..rerank import Reranker
    from ..store import ChunkRecord, Document, Store
    from .dense import DenseRetriever

log = get_logger("contextsearch.search")

_MARK = re.compile(f"{HL_OPEN}|{HL_CLOSE}")


@dataclass
class SearchRequest:
    query: str
    strategy: str = ""
    models: list[str] | None = None
    top_k: int = 0
    candidates: int = 0
    fusion: str = ""
    rrf_k: int = 0
    alpha: float | None = None
    rerank: bool | None = None
    rerank_model: str = ""
    rerank_candidates: int = 0
    group_by_document: bool | None = None
    prefix_last: bool = False
    source_id: str | None = None
    ext: str | None = None
    document_id: str | None = None
    snippet_chars: int = 0
    min_confidence: float = 0.0
    include_text: bool = True
    explain: bool = True
    request_id: str = ""

    def resolved(self, settings: Settings) -> SearchRequest:
        """Fill blanks from settings and validate ranges."""
        query = (self.query or "").strip()
        if not query:
            raise ValidationError("query must not be empty")
        if len(query) > 2000:
            raise ValidationError("query is too long (2000 characters max)")
        strategy = self.strategy or settings.default_strategy
        if strategy not in ("lexical", "dense", "hybrid"):
            raise ValidationError(f"strategy must be lexical, dense or hybrid, got {strategy!r}")
        fusion = self.fusion or settings.fusion
        if fusion not in ("rrf", "weighted"):
            raise ValidationError(f"fusion must be rrf or weighted, got {fusion!r}")
        top_k = self.top_k or settings.default_top_k
        if not 1 <= top_k <= settings.max_top_k:
            raise ValidationError(f"top_k must be between 1 and {settings.max_top_k}")
        alpha = settings.hybrid_alpha if self.alpha is None else self.alpha
        if not 0.0 <= alpha <= 1.0:
            raise ValidationError("alpha must be between 0 and 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValidationError("min_confidence must be between 0 and 1")
        return SearchRequest(
            query=query,
            strategy=strategy,
            models=self.models,
            top_k=top_k,
            candidates=max(top_k, self.candidates or settings.candidates),
            fusion=fusion,
            rrf_k=self.rrf_k or settings.rrf_k,
            alpha=alpha,
            rerank=settings.rerank if self.rerank is None else self.rerank,
            rerank_model=self.rerank_model or settings.rerank_model,
            rerank_candidates=self.rerank_candidates or settings.rerank_candidates,
            group_by_document=settings.group_by_document if self.group_by_document is None else self.group_by_document,
            prefix_last=self.prefix_last,
            source_id=self.source_id or None,
            ext=(self.ext or "").lower().lstrip(".") or None,
            document_id=self.document_id or None,
            snippet_chars=self.snippet_chars or settings.snippet_chars,
            min_confidence=self.min_confidence,
            include_text=self.include_text,
            explain=self.explain,
            request_id=self.request_id,
        )


@dataclass
class Hit:
    rank: int
    chunk: ChunkRecord
    document: Document
    score: float
    confidence: float
    confidence_label: str
    snippet: str
    highlights: list[list[int]]
    evidence: dict[str, Any]
    also_matched: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_text: bool = True, explain: bool = True) -> dict[str, Any]:
        doc = self.document
        out: dict[str, Any] = {
            "rank": self.rank,
            "score": round(self.score, 6),
            "confidence": round(self.confidence, 4),
            "confidence_label": self.confidence_label,
            "document": {
                "id": doc.id,
                "filename": doc.filename,
                "relpath": doc.relpath,
                "title": doc.title,
                "ext": doc.ext,
                "source_id": doc.source_id,
                "revision": doc.revision,
                "pages": doc.pages,
                "indexed_at": doc.indexed_at,
            },
            "page": self.chunk.page,
            "chunk_id": self.chunk.id,
            "chunk_index": self.chunk.chunk_index,
            "char_start": self.chunk.char_start,
            "char_end": self.chunk.char_end,
            "snippet": self.snippet,
            "highlights": self.highlights,
            "also_matched": self.also_matched,
        }
        if include_text:
            out["text"] = self.chunk.text
        if explain:
            out["evidence"] = self.evidence
        return out


@dataclass
class SearchResponse:
    request: SearchRequest
    hits: list[Hit]
    retrievers: list[RetrieverResult]
    total_candidates: int
    trace: Trace
    models_used: list[str]
    reranked: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        req = self.request
        return {
            "query": req.query,
            "strategy": req.strategy,
            "models": self.models_used,
            "fusion": req.fusion if req.strategy == "hybrid" or len(self.models_used) > 1 else None,
            "rerank": self.reranked,
            "rerank_model": req.rerank_model if self.reranked else None,
            "top_k": req.top_k,
            "min_confidence": req.min_confidence,
            "grouped": req.group_by_document,
            "total_candidates": self.total_candidates,
            "results": [h.to_dict(req.include_text, req.explain) for h in self.hits],
            "retrievers": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "hits": len(r.hits),
                    "duration_ms": round(r.duration_ms, 2),
                    "details": r.details,
                }
                for r in self.retrievers
            ]
            if req.explain
            else None,
            "timings": self.trace.timings(),
            "warnings": self.warnings,
            "request_id": self.trace.request_id,
            "trace": self.trace.to_dict() if req.explain else None,
        }


class Searcher:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        dense_retriever: Callable[[str], DenseRetriever | None],
        available_models: Callable[[], list[str]],
        reranker: Callable[[str], Reranker],
    ):
        self.store = store
        self.settings = settings
        self.lexical = LexicalRetriever(store, drop_stopwords=settings.lexical_stopwords)
        self._dense_retriever = dense_retriever
        self._available_models = available_models
        self._reranker = reranker
        self._logged = 0

    # ---------------------------------------------------------------- main entry
    def search(self, request: SearchRequest) -> SearchResponse:
        req = request.resolved(self.settings)
        trace = Trace(req.request_id or None)
        warnings: list[str] = []
        results: list[RetrieverResult] = []
        filters = {"source_id": req.source_id, "ext": req.ext, "document_id": req.document_id}

        if req.strategy in ("lexical", "hybrid"):
            with trace.span("lexical") as span:
                lexical = self.lexical.search(req.query, req.candidates, prefix_last=req.prefix_last, **filters)
                span.attributes.update({"hits": len(lexical.hits), **lexical.details})
            results.append(lexical)

        models_used: list[str] = []
        if req.strategy in ("dense", "hybrid"):
            models = self._pick_models(req.models, warnings)
            for model in models:
                retriever = self._dense_retriever(model)
                if retriever is None:
                    warnings.append(f"no dense index for model {model}; skipped")
                    continue
                with trace.span(f"dense:{model}") as span:
                    try:
                        dense = retriever.search(req.query, req.candidates, self.store, **filters)
                    except ModelError as exc:
                        warnings.append(f"dense retrieval failed for {model}: {exc.message}")
                        span.attributes["error"] = exc.message
                        continue
                    span.attributes.update({"hits": len(dense.hits), **dense.details})
                results.append(dense)
                models_used.append(model)
            if not models_used:
                if req.strategy == "dense":
                    warnings.append("no dense index is available; nothing to search")
                elif not warnings:
                    warnings.append("no dense index is available; hybrid ran as keyword search")

        with trace.span("fusion", method=req.fusion) as span:
            candidates = self._fuse(results, req)
            span.attributes["candidates"] = len(candidates)

        reranked = False
        if req.rerank and candidates:
            reranked = self._rerank(req, candidates, trace, warnings)

        with trace.span("assemble") as span:
            hits = self._assemble(req, results, candidates, reranked)
            span.attributes["results"] = len(hits)

        trace.finish()
        response = SearchResponse(req, hits, results, len(candidates), trace, models_used, reranked, warnings)
        self._record(response)
        return response

    # ---------------------------------------------------------------- pieces
    def _pick_models(self, requested: list[str] | None, warnings: list[str]) -> list[str]:
        available = self._available_models()
        if not requested:
            return available
        picked = []
        for model in requested:
            if model in available:
                picked.append(model)
            else:
                warnings.append(f"model {model} has no dense index; available: {', '.join(available) or 'none'}")
        return picked

    def _fuse(self, results: list[RetrieverResult], req: SearchRequest) -> list[FusedCandidate]:
        if len(results) == 1:
            # One list: keep its order and raw scores, no fusion needed.
            r = results[0]
            return [
                FusedCandidate(h.chunk_id, h.score, {r.name: h.score}, {r.name: h.rank}, {r.name: h.score})
                for h in r.hits
            ]
        return fuse(results, req.fusion, req.rrf_k, req.alpha or 0.5)

    def _rerank(self, req: SearchRequest, candidates: list[FusedCandidate], trace: Trace, warnings: list[str]) -> bool:
        top = candidates[: req.rerank_candidates]
        chunks = self.store.get_chunks([c.chunk_id for c in top])
        passages = [chunks[c.chunk_id].text if c.chunk_id in chunks else "" for c in top]
        with trace.span("rerank", model=req.rerank_model, candidates=len(top)) as span:
            try:
                scores = self._reranker(req.rerank_model).score(req.query, passages)
            except ModelError as exc:
                warnings.append(f"reranking skipped: {exc.message}")
                span.attributes["error"] = exc.message
                return False
        for position, (cand, score) in enumerate(zip(top, scores, strict=True), start=1):
            cand.raw_scores["rerank"] = float(score)
            cand.ranks["rerank_before"] = position
        top.sort(key=lambda c: (-c.raw_scores["rerank"], c.chunk_id))
        for position, cand in enumerate(top, start=1):
            cand.ranks["rerank"] = position
        candidates[: len(top)] = top
        return True

    def _assemble(
        self, req: SearchRequest, results: list[RetrieverResult], candidates: list[FusedCandidate], reranked: bool
    ) -> list[Hit]:
        if not candidates:
            return []
        limit = req.top_k if not req.group_by_document else req.candidates * max(1, len(results))
        chosen = candidates[:limit]
        chunks = self.store.get_chunks([c.chunk_id for c in chosen])
        docs = self.store.get_documents(list({chunks[c.chunk_id].document_id for c in chosen if c.chunk_id in chunks}))
        parsed = parse_query(req.query, self.settings.lexical_stopwords)
        lexical_by_chunk = {}
        for r in results:
            if r.kind == "lexical":
                lexical_by_chunk = r.by_chunk()
        total_terms = len(parsed.units)

        hits: list[Hit] = []
        groups: dict[str, Hit] = {}
        for cand in chosen:
            chunk = chunks.get(cand.chunk_id)
            if chunk is None:  # deleted between retrieval and assembly
                continue
            doc = docs.get(chunk.document_id)
            if doc is None or doc.status != "indexed":
                continue
            lexical_hit = lexical_by_chunk.get(cand.chunk_id)
            terms = matched_terms(lexical_hit.highlight) if lexical_hit and lexical_hit.highlight else []
            covered = covered_terms(parsed, terms, chunk.text) if terms else []
            snippet, spans = build_snippet(
                chunk.text, lexical_hit.highlight if lexical_hit else None, parsed.units, req.snippet_chars
            )
            dense_scores = {k.split(":", 1)[1]: v for k, v in cand.raw_scores.items() if k.startswith("dense:")}
            best_cos = max(dense_scores.values()) if dense_scores else None
            rerank_score = cand.raw_scores.get("rerank") if reranked else None
            conf, label, parts = confidence(self.settings, rerank_score, total_terms, len(covered), best_cos)
            if conf < req.min_confidence:
                continue
            evidence = {
                "found_by": [k for k in cand.ranks if not k.startswith("rerank")],
                "agreement": sum(1 for k in cand.ranks if not k.startswith("rerank")) >= 2,
                "fused_score": round(cand.score, 6),
                "fusion": req.fusion if len(results) > 1 else None,
                "lexical": {
                    "rank": lexical_hit.rank,
                    "bm25": round(lexical_hit.score, 4),
                    "matched_terms": terms,
                    "covered_terms": covered,
                    "query_terms": total_terms,
                }
                if lexical_hit
                else None,
                "dense": {
                    model: {"rank": cand.ranks.get(f"dense:{model}"), "cosine": round(score, 4)}
                    for model, score in dense_scores.items()
                }
                or None,
                "rerank": {
                    "score": round(rerank_score, 4),
                    "rank_before": cand.ranks.get("rerank_before"),
                    "rank_after": cand.ranks.get("rerank"),
                }
                if rerank_score is not None
                else None,
                "confidence_parts": parts,
                "locator": {
                    "page": chunk.page,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "chunk_index": chunk.chunk_index,
                },
            }
            score = rerank_score if rerank_score is not None else cand.score
            hit = Hit(0, chunk, doc, float(score), conf, label, snippet, spans, evidence)
            if req.group_by_document:
                head = groups.get(doc.id)
                if head is not None:
                    if len(head.also_matched) < 3:
                        head.also_matched.append(
                            {
                                "chunk_id": chunk.id,
                                "page": chunk.page,
                                "chunk_index": chunk.chunk_index,
                                "char_start": chunk.char_start,
                                "char_end": chunk.char_end,
                                "score": round(float(score), 6),
                                "confidence": round(conf, 4),
                                "snippet": snippet,
                                "highlights": spans,
                                "found_by": evidence["found_by"],
                            }
                        )
                    continue
                groups[doc.id] = hit
            hits.append(hit)
            # Grouped results keep scanning so documents already listed collect
            # their other matching passages; ungrouped ones stop at top_k.
            if len(hits) >= req.top_k and not req.group_by_document:
                break
        hits = hits[: req.top_k]
        for i, hit in enumerate(hits, start=1):
            hit.rank = i
        return hits

    def _record(self, response: SearchResponse) -> None:
        req = response.request
        timings = response.trace.timings()
        metrics.inc("contextsearch_searches", strategy=req.strategy)
        metrics.observe("contextsearch_search_latency_ms", timings["total"], strategy=req.strategy)
        log.info(
            "search",
            extra_fields={
                "query": req.query[:200],
                "strategy": req.strategy,
                "models": ",".join(response.models_used),
                "rerank": response.reranked,
                "results": len(response.hits),
                "candidates": response.total_candidates,
                "ms": timings["total"],
            },
        )
        if self.settings.query_log:
            try:
                self._logged += 1
                if self._logged % 200 == 0:
                    self.store.trim_queries(self.settings.query_log_keep)
                self.store.log_query(
                    response.trace.request_id,
                    req.query,
                    req.strategy,
                    response.models_used,
                    response.reranked,
                    req.top_k,
                    len(response.hits),
                    timings["total"],
                    timings,
                )
            except Exception as exc:  # the query log must never break a search
                log.warning("query log write failed", extra_fields={"error": str(exc)})


# ---------------------------------------------------------------------- snippets


def build_snippet(text: str, highlighted: str | None, terms: list[str], max_chars: int) -> tuple[str, list[list[int]]]:
    """A window of the chunk around the matches, with highlight spans relative to the window.

    ``highlighted`` is FTS5's marked copy of the chunk when the lexical
    retriever found it. Otherwise the query terms are matched by prefix, which
    approximates the stemming FTS5 does.
    """
    if highlighted:
        plain, spans = _strip_markers(highlighted)
        if plain != text:
            # highlight() works on the tokenised text and should reproduce the
            # original exactly; if it ever does not, fall back to term matching.
            plain, spans = text, _term_spans(text, terms)
    else:
        plain, spans = text, _term_spans(text, terms)

    # Line breaks become spaces (same length, so the spans stay valid); the
    # snippet is meant to read as one paragraph.
    plain = plain.replace("\r", " ").replace("\n", " ")
    if len(plain) <= max_chars:
        return plain, [list(s) for s in spans]

    start = 0
    if spans:
        # Put the window where the most spans fall, starting a little before the first of them.
        best_start, best_count = 0, -1
        for s, _ in spans:
            window_start = max(0, s - max_chars // 3)
            count = sum(1 for a, _ in spans if window_start <= a < window_start + max_chars)
            if count > best_count:
                best_start, best_count = window_start, count
        start = best_start
    start = _snap_left(plain, start)
    end = min(len(plain), start + max_chars)
    end = _snap_right(plain, end)
    window = plain[start:end]
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(plain) else ""
    shifted = [[a - start + len(prefix), b - start + len(prefix)] for a, b in spans if a >= start and b <= end]
    return prefix + window + suffix, shifted


def _strip_markers(marked: str) -> tuple[str, list[tuple[int, int]]]:
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    length = 0
    open_at: int | None = None
    for m in _MARK.finditer(marked):
        out.append(marked[pos : m.start()])
        length += m.start() - pos
        pos = m.end()
        if m.group() == HL_OPEN:
            open_at = length
        elif open_at is not None:
            spans.append((open_at, length))
            open_at = None
    out.append(marked[pos:])
    return "".join(out), spans


def _term_spans(text: str, terms: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for term in terms:
        term = term.strip()
        if len(term) < 2:
            continue
        words = [w for w in re.findall(r"\w+", term) if len(w) >= 2]
        if not words:
            continue
        if len(words) == 1:
            pattern = re.compile(r"\b" + re.escape(words[0]) + r"\w{0,4}", re.IGNORECASE)
        else:
            pattern = re.compile(r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\w*", re.IGNORECASE)
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
            if len(spans) > 200:
                break
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _snap_left(text: str, pos: int) -> int:
    while pos > 0 and not text[pos - 1].isspace():
        pos -= 1
    return pos


def _snap_right(text: str, pos: int) -> int:
    while pos < len(text) and not text[pos].isspace():
        pos += 1
    return pos


# ---------------------------------------------------------------------- confidence


def confidence(
    settings: Settings, rerank_score: float | None, total_terms: int, matched: int, best_cosine: float | None
) -> tuple[float, str, dict[str, Any]]:
    """A 0..1 number and a label for one hit, with the pieces it came from.

    With a reranker on, the reranker's probability is the confidence. Without
    one it comes from two signals: how much of the query the keywords matched
    (``lexical_coverage``) and where the best cosine similarity sits between
    ``confidence_dense_low`` and ``confidence_dense_high``
    (``dense_similarity``). When both retrievers found the hit the two are
    averaged with a small bonus for agreement. A hit only one retriever found
    is discounted by 15 percent. It is a comparability score, not a probability;
    docs/SEARCH.md explains how to calibrate the thresholds for a model.
    """
    parts: dict[str, Any] = {}
    lexical_part = None
    if total_terms and matched:
        lexical_part = min(1.0, matched / total_terms)
        parts["lexical_coverage"] = round(lexical_part, 3)
    dense_part = None
    if best_cosine is not None:
        lo, hi = settings.confidence_dense_low, settings.confidence_dense_high
        dense_part = max(0.0, min(1.0, (best_cosine - lo) / (hi - lo)))
        parts["dense_similarity"] = round(dense_part, 3)
    if rerank_score is not None:
        value = float(rerank_score)
        parts["source"] = "reranker"
    elif lexical_part is not None and dense_part is not None:
        # Average the two signals and add a little for agreement, so a hit both
        # retrievers rate highly ends up above one that only the keywords like.
        value = 0.5 * lexical_part + 0.5 * dense_part + 0.1 * min(lexical_part, dense_part)
        parts["source"] = "lexical+dense"
    elif dense_part is not None:
        value = 0.85 * dense_part
        parts["source"] = "dense only"
    elif lexical_part is not None:
        value = 0.85 * lexical_part
        parts["source"] = "lexical only"
    else:
        value = 0.0
        parts["source"] = "none"
    value = max(0.0, min(1.0, value))
    if value >= settings.confidence_strong:
        label = "strong"
    elif value >= settings.confidence_likely:
        label = "likely"
    else:
        label = "weak"
    return value, label, parts
