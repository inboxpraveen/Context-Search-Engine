"""Dense retrieval with a real model. Marked slow: needs sentence-transformers and the default model on disk."""

from __future__ import annotations

import pytest

from contextsearch.embedding import dense_available
from tests.conftest import make_settings, wait_idle, write_text

pytestmark = pytest.mark.slow


@pytest.fixture
def dense_engine(tmp_path):
    ok, reason = dense_available()
    if not ok:
        pytest.skip(f"dense extra not installed: {reason}")
    from contextsearch import Engine

    settings = make_settings(tmp_path, embedding_models="BAAI/bge-small-en-v1.5")
    eng = Engine(settings, configure_logs=False)
    eng.start(warm=False)
    wait_idle(eng, 300)
    yield eng
    eng.close()


def test_semantic_search_finds_paraphrases(dense_engine, tmp_path):
    docs = tmp_path / "docs"
    write_text(
        docs,
        "refunds.md",
        "Items can be returned within thirty days of delivery for a full refund to the original payment method.",
    )
    write_text(docs, "coffee.md", "Ethiopia Guji is a washed lot with floral and bergamot notes, roasted light.")
    write_text(
        docs, "safety.md", "Never use water on a roaster fire; close the drum air, cut the gas and smother the drum."
    )
    _, job = dense_engine.add_folder(str(docs))
    assert job.wait(300) and job.status == "done"
    wait_idle(dense_engine, 300)

    # no keyword overlap with "returned", "refund" or "payment"
    hits = dense_engine.search("how do I get my money back", strategy="dense").hits
    assert hits[0].document.filename == "refunds.md"
    assert hits[0].evidence["dense"]["BAAI/bge-small-en-v1.5"]["cosine"] > 0.5

    hybrid = dense_engine.search("what to do when the roaster is burning", strategy="hybrid", rerank=False).hits
    assert hybrid[0].document.filename == "safety.md"
    assert dense_engine.indexes["BAAI/bge-small-en-v1.5"].size == 3


def test_reranker_scores_are_probabilities(dense_engine, tmp_path):
    docs = tmp_path / "docs"
    write_text(docs, "a.md", "Refunds are issued within five business days of the parcel reaching our warehouse.")
    write_text(docs, "b.md", "The tasting room is open on Tuesdays and Thursdays at ten.")
    _, job = dense_engine.add_folder(str(docs))
    assert job.wait(300)
    wait_idle(dense_engine, 300)
    hits = dense_engine.search("when do I get my refund", strategy="hybrid", rerank=True).hits
    assert hits[0].document.filename == "a.md"
    assert 0.0 <= hits[0].evidence["rerank"]["score"] <= 1.0
    assert hits[0].confidence > hits[-1].confidence
