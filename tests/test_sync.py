"""Folder sync: new, changed, unchanged, removed and failed files end up in the right state."""

from __future__ import annotations

import os
import time

from contextsearch.ingest import is_excluded, plan_sync, scan_folder
from contextsearch.store import ScanSnapshot
from tests.conftest import make_pdf, wait_idle, write_text


def _sync(engine, source_id):
    job = engine.sync_source(source_id)
    assert job.wait(60), "sync did not finish"
    return job


def test_scan_filters_and_exclusions(tmp_path):
    write_text(tmp_path, "a.md", "one")
    write_text(tmp_path, "sub/b.txt", "two")
    write_text(tmp_path, "sub/.git/c.txt", "three")
    write_text(tmp_path, "~$lock.docx", "four")
    write_text(tmp_path, "image.bin", "five")
    result = scan_folder(str(tmp_path), True, {"md", "txt", "docx"}, ["**/.git/**", "**/~$*"], max_bytes=10_000)
    assert sorted(f.relpath for f in result.files) == ["a.md", "sub/b.txt"]
    assert result.ignored == 2, "the lock file and the .bin; the pruned .git folder is never walked"
    assert is_excluded("deep/nested/.git/x", ["**/.git/**"]) and not is_excluded("git/x", ["**/.git/**"])
    flat = scan_folder(str(tmp_path), False, {"md", "txt"}, [], 10_000)
    assert [f.relpath for f in flat.files] == ["a.md"]


def test_plan_sync_buckets(tmp_path):
    write_text(tmp_path, "new.txt", "n")
    write_text(tmp_path, "same.txt", "s")
    write_text(tmp_path, "changed.txt", "c")
    write_text(tmp_path, "failed.txt", "f")
    scan = scan_folder(str(tmp_path), True, {"txt"}, [], 10_000)
    by_name = {f.relpath: f for f in scan.files}
    snapshot = {
        by_name["same.txt"].path: ScanSnapshot(
            "1", by_name["same.txt"].size, by_name["same.txt"].mtime, "h", "indexed", "sig"
        ),
        by_name["changed.txt"].path: ScanSnapshot("2", 999, by_name["changed.txt"].mtime, "h", "indexed", "sig"),
        by_name["failed.txt"].path: ScanSnapshot(
            "3", by_name["failed.txt"].size, by_name["failed.txt"].mtime, "", "failed", "sig"
        ),
        os.path.join(str(tmp_path), "deleted.txt"): ScanSnapshot("4", 1, 1.0, "h", "indexed", "sig"),
    }
    plan = plan_sync(None, snapshot, scan, "sig")
    assert [f.relpath for f in plan.new] == ["new.txt"]
    assert [f.relpath for f in plan.changed] == ["changed.txt"]
    assert [f.relpath for f in plan.retry] == ["failed.txt"]
    assert [os.path.basename(p) for p, _ in plan.removed] == ["deleted.txt"]
    assert plan.unchanged == 1

    # a different chunking signature means every indexed file is re-cut
    plan = plan_sync(None, snapshot, scan, "other")
    assert sorted(f.relpath for f in plan.changed) == ["changed.txt", "same.txt"]

    # an unmounted folder must not look like a mass deletion
    missing = scan_folder(str(tmp_path / "nope"), True, {"txt"}, [], 10_000)
    assert missing.root_missing and plan_sync(None, snapshot, missing, "sig").removed == []


def test_engine_sync_lifecycle(engine, tmp_path):
    docs = tmp_path / "docs"
    write_text(docs, "policies/refunds.md", "# Refunds\n\nItems can be returned within thirty days of delivery.")
    write_text(docs, "notes.txt", "Onboarding: new hires get a laptop on day one.")
    (docs / "scan.pdf").write_bytes(make_pdf(["Shipping takes two business days.", "Tracking numbers are emailed."]))

    source, job = engine.add_folder(str(docs), name="docs")
    assert job.wait(60) and job.status == "done"
    assert job.result["added"] == 3 and job.result["failed"] == 0
    wait_idle(engine)
    assert engine.stats()["documents"] == 3

    hits = engine.search("returned within thirty days", strategy="lexical").hits
    assert hits and hits[0].document.relpath == "policies/refunds.md"
    assert hits[0].chunk.page == 1 and hits[0].evidence["lexical"]["matched_terms"]

    pdf_hits = engine.search("tracking numbers", strategy="hybrid").hits
    assert pdf_hits[0].document.filename == "scan.pdf" and pdf_hits[0].chunk.page == 2

    # unchanged rescan touches nothing
    assert _sync(engine, source.id).result["unchanged"] == 3

    # an edit becomes a new revision; the old text is gone
    time.sleep(0.05)
    write_text(docs, "policies/refunds.md", "# Refunds\n\nRefunds go back to the original payment method.")
    os.utime(docs / "policies/refunds.md", None)
    assert _sync(engine, source.id).result["updated"] == 1
    doc = engine.store.get_document_by_path(str(docs / "policies" / "refunds.md"))
    assert doc.revision == 2
    assert not engine.search('"thirty days"', strategy="lexical").hits
    assert engine.search("original payment method", strategy="lexical").hits[0].document.id == doc.id

    # same content rewritten with a new mtime is not re-indexed
    os.utime(docs / "notes.txt", (time.time() + 5, time.time() + 5))
    job = _sync(engine, source.id)
    assert job.result["unchanged"] == 3 and engine.store.get_document_by_path(str(docs / "notes.txt")).revision == 1

    # a deleted file leaves the index but keeps its history row
    (docs / "notes.txt").unlink()
    assert _sync(engine, source.id).result["removed"] == 1
    assert not engine.search("laptop", strategy="lexical").hits
    removed, _ = engine.store.list_documents(status="removed")
    assert [d.filename for d in removed] == ["notes.txt"]

    # a broken file is reported, not fatal, and retried on the next scan
    (docs / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
    job = _sync(engine, source.id)
    assert job.result["failed"] == 1 and job.errors[0]["path"] == "bad.pdf"
    failed, _ = engine.store.list_documents(status="failed")
    assert failed[0].error
    (docs / "bad.pdf").write_bytes(make_pdf(["Now readable."]))
    job = _sync(engine, source.id)
    assert job.result["added"] == 1 and engine.search("readable", strategy="lexical").hits

    # removing the source drops its documents and vectors
    engine.remove_source(source.id)
    assert engine.stats()["documents"] == 0
    assert engine.indexes["test/hashing-64"].size == 0


def test_folder_guards(engine, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "inner").mkdir()
    engine.add_folder(str(docs))
    import pytest

    from contextsearch.errors import ConflictError, ValidationError

    with pytest.raises(ConflictError):
        engine.add_folder(str(docs / "inner"))
    with pytest.raises(ValidationError):
        engine.add_folder(str(tmp_path / "missing"))
    with pytest.raises(ValidationError):
        engine.add_folder("   ")
