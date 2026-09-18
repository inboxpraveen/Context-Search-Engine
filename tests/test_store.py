"""Storage: revisions replace chunks atomically, FTS stays in sync, vectors follow chunks."""

from __future__ import annotations

import numpy as np
import pytest

from contextsearch.errors import ConflictError, NotFoundError
from contextsearch.store import Store
from contextsearch.vectors import VectorIndex


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def _index(store: Store, source, path: str, pages: list[str], chunks_per_page: int = 1):
    doc = store.upsert_pending(source, path, path.split("/")[-1], size=10, mtime=1.0)
    rows = []
    for page_no, text in enumerate(pages, start=1):
        step = max(1, len(text) // chunks_per_page)
        for i in range(chunks_per_page):
            start, end = i * step, (len(text) if i == chunks_per_page - 1 else (i + 1) * step)
            rows.append((page_no, len(rows), start, end, len(text[start:end].split()), text[start:end]))
    return store.store_indexed(
        doc.id,
        title="t",
        size=10,
        mtime=1.0,
        content_hash="h" + str(len(rows)),
        reader="text",
        pages=[(n, t, False) for n, t in enumerate(pages, start=1)],
        chunks=rows,
        chunking_signature="sig",
    )


def test_index_search_and_revision_replace(store):
    source = store.add_source("folder", "docs", "/docs")
    doc, chunk_ids = _index(store, source, "/docs/a.txt", ["The refund policy allows returns within thirty days."])
    assert doc.revision == 1 and doc.status == "indexed" and doc.chunks == 1
    hits = store.search_fts('"refund"', limit=5)
    assert [h[0] for h in hits] == chunk_ids
    assert "refund" in hits[0][2]

    doc2, new_ids = _index(store, source, "/docs/a.txt", ["Shipping takes two days."])
    assert doc2.id == doc.id and doc2.revision == 2
    assert store.search_fts('"refund"', limit=5) == []
    assert [h[0] for h in store.search_fts('"shipping"', limit=5)] == new_ids
    assert set(new_ids).isdisjoint(chunk_ids)
    assert [r["revision"] for r in store.revisions(doc.id)] == [2, 1]
    kinds = [e["kind"] for e in store.list_events(5)]
    assert kinds[0] == "updated" and "added" in kinds


def test_removed_documents_keep_history_but_lose_content(store):
    source = store.add_source("folder", "docs", "/docs")
    doc, _ = _index(store, source, "/docs/gone.txt", ["Some searchable words here."])
    store.mark_removed(doc.id)
    removed = store.get_document(doc.id)
    assert removed.status == "removed" and removed.removed_at is not None
    assert store.search_fts('"searchable"', limit=5) == []
    assert store.chunks_for_document(doc.id) == []
    docs, total = store.list_documents(status="removed")
    assert total == 1 and docs[0].id == doc.id
    assert store.snapshot_for_source(source.id) == {}


def test_source_removal_cascades(store):
    source = store.add_source("folder", "docs", "/docs")
    _index(store, source, "/docs/a.txt", ["alpha beta"])
    _index(store, source, "/docs/b.txt", ["gamma delta"])
    assert store.stats()["documents"] == 2
    store.remove_source(source.id)
    assert store.stats()["documents"] == 0 and store.stats()["chunks"] == 0
    with pytest.raises(NotFoundError):
        store.get_source(source.id)
    with pytest.raises(ConflictError):
        store.add_source("folder", "x", "/docs")
        store.add_source("folder", "y", "/docs")


def test_vectors_follow_chunks_and_index_search(store):
    source = store.add_source("folder", "docs", "/docs")
    _, ids = _index(
        store, source, "/docs/v.txt", ["one two three four five six seven eight nine ten"], chunks_per_page=3
    )
    store.register_model("m", 4)
    vectors = np.eye(4, dtype=np.float32)[:3]
    store.store_vectors("m", ids, vectors)
    assert store.count_vectors("m") == 3
    assert store.chunks_missing_vectors("m") == []

    index = VectorIndex("m", 4)
    index.load(store)
    top_ids, scores = index.search(np.array([0, 1, 0, 0], dtype=np.float32), k=2)
    assert top_ids[0] == ids[1] and scores[0] == pytest.approx(1.0)

    # a filter restricts the candidates
    top_ids, _ = index.search(np.array([0, 1, 0, 0], dtype=np.float32), k=2, allowed_ids=np.array([ids[0], ids[2]]))
    assert ids[1] not in top_ids

    # re-indexing the document drops the old vectors from the database
    _, new_ids = _index(store, source, "/docs/v.txt", ["fresh text"], chunks_per_page=1)
    assert store.count_vectors("m") == 0
    assert [c.id for c in store.chunks_missing_vectors("m")] == new_ids
    assert index.remove(ids) == 3 and index.size == 0

    with pytest.raises(ConflictError):
        store.register_model("m", 8)


def test_vector_index_replaces_duplicates_and_compacts():
    index = VectorIndex("m", 2, compact_ratio=0.5)
    index.add([1, 2, 3], np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    index.add([2], np.array([[1, 0]], dtype=np.float32))  # re-embedded chunk 2
    ids, _ = index.search(np.array([1, 0], dtype=np.float32), k=3)
    assert sorted(ids.tolist()) == [1, 2, 3]
    assert index.size == 3
    index.remove([1, 3])
    ids, scores = index.search(np.array([1, 0], dtype=np.float32), k=5)
    assert ids.tolist() == [2] and scores[0] == pytest.approx(1.0)


def test_query_log_and_trim(store):
    for i in range(5):
        store.log_query("r", f"q{i}", "hybrid", ["m"], False, 10, 1, 3.0, {"total": 3.0})
    store.trim_queries(keep=2)
    assert [q["query"] for q in store.recent_queries(10)] == ["q4", "q3"]
