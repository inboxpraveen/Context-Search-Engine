"""Query parsing, fusion maths, snippets and confidence: the pieces that decide ranking."""

from __future__ import annotations

import pytest

from contextsearch.retrieval import RetrieverHit, RetrieverResult, build_snippet, confidence, parse_query, rrf, weighted
from contextsearch.retrieval.lexical import covered_terms, matched_terms
from contextsearch.store import HL_CLOSE, HL_OPEN
from tests.conftest import make_settings


def test_parse_query_phrases_terms_and_stopwords():
    parsed = parse_query('the "original payment method" for a refund and Refunds')
    assert parsed.phrases == ["original payment method"]
    assert parsed.terms == ["refund", "refunds"]
    assert parsed.dropped == ["the", "for", "a", "and"]
    assert parsed.expression("AND") == '"original payment method" AND "refund" AND "refunds"'
    assert parsed.expression("OR", prefix_last=True).endswith('"refunds"*')
    assert parse_query("to be or not to be").terms == ["to", "be", "or", "not"]
    assert parse_query('say "hi"there"').phrases == ["hi"]
    assert parse_query("").empty


def test_matched_and_covered_terms():
    marked = f"The {HL_OPEN}refunds{HL_CLOSE} for {HL_OPEN}late{HL_CLOSE} {HL_OPEN}deliveries{HL_CLOSE} and {HL_OPEN}refunds{HL_CLOSE}"
    forms = matched_terms(marked)
    assert forms == ["refunds", "late", "deliveries"]
    parsed = parse_query('refund late delivery policy "late deliveries"')
    covered = covered_terms(parsed, forms, "The refunds for late deliveries and refunds")
    assert covered == ["late deliveries", "refund", "late", "delivery"]


def _result(name: str, kind: str, ids: list[int], scores: list[float] | None = None) -> RetrieverResult:
    scores = scores or [float(len(ids) - i) for i in range(len(ids))]
    return RetrieverResult(
        name, kind, [RetrieverHit(cid, s, rank) for rank, (cid, s) in enumerate(zip(ids, scores, strict=True), start=1)]
    )


def test_rrf_rewards_agreement():
    lexical = _result("lexical", "lexical", [1, 2, 3])
    dense = _result("dense:m", "dense", [3, 4, 1])
    fused = rrf([lexical, dense], k=60)
    order = [c.chunk_id for c in fused]
    assert order[:2] == [1, 3], "chunks found by both retrievers come first"
    top = fused[0]
    assert top.score == pytest.approx(1 / 61 + 1 / 63)
    assert top.ranks == {"lexical": 1, "dense:m": 3}
    assert fused[-1].found_by in (["lexical"], ["dense:m"])


def test_weighted_fusion_respects_alpha():
    lexical = _result("lexical", "lexical", [1, 2], [10.0, 5.0])
    dense = _result("dense:m", "dense", [2, 1], [0.9, 0.5])
    keyword_first = weighted([lexical, dense], alpha=0.0)
    assert keyword_first[0].chunk_id == 1
    semantic_first = weighted([lexical, dense], alpha=1.0)
    assert semantic_first[0].chunk_id == 2
    assert weighted([lexical], alpha=0.5)[0].score == pytest.approx(1.0)


def test_confidence_blend_and_labels(tmp_path):
    settings = make_settings(tmp_path)
    value, label, parts = confidence(settings, None, total_terms=3, matched=3, best_cosine=0.80)
    assert value == pytest.approx(1.0) and label == "strong" and parts["source"] == "lexical+dense"
    value, label, _ = confidence(settings, None, total_terms=3, matched=1, best_cosine=0.40)
    assert label == "weak"
    lexical_only, _, parts = confidence(settings, None, total_terms=2, matched=2, best_cosine=None)
    assert lexical_only == pytest.approx(0.85) and parts["source"] == "lexical only"
    dense_only, _, _ = confidence(settings, None, total_terms=0, matched=0, best_cosine=0.575)
    assert dense_only == pytest.approx(0.85 * 0.5)
    reranked, label, parts = confidence(settings, 0.91, 3, 0, None)
    assert reranked == 0.91 and label == "strong" and parts["source"] == "reranker"
    assert confidence(settings, None, 0, 0, None)[1] == "weak"


def test_snippet_window_keeps_highlight_offsets():
    text = ("filler words " * 40).strip() + " The refund arrives late. " + ("more filler " * 40).strip()
    marked = text.replace("refund", f"{HL_OPEN}refund{HL_CLOSE}").replace("late", f"{HL_OPEN}late{HL_CLOSE}")
    snippet, spans = build_snippet(text, marked, ["refund", "late"], max_chars=120)
    assert len(snippet) <= 120 + 8 + 12, "window plus ellipses, snapped to a word boundary"
    assert snippet.startswith("... ") and snippet.endswith(" ...")
    assert [snippet[a:b] for a, b in spans] == ["refund", "late"]


def test_snippet_without_fts_highlight_matches_terms_by_prefix():
    text = "Refunds are issued to the original payment method within five business days."
    snippet, spans = build_snippet(text, None, ["refund", "payment method"], max_chars=500)
    assert snippet == text
    assert [snippet[a:b] for a, b in spans] == ["Refunds", "payment method"]


def test_snippet_flattens_line_breaks():
    text = "Line one\n\nLine two"
    snippet, _ = build_snippet(text, None, [], max_chars=100)
    assert snippet == "Line one  Line two"
