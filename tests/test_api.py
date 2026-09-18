"""HTTP API: auth, error shape, upload, search evidence, safe file serving. Keyword-only, no model download."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from contextsearch.api import create_app
from tests.conftest import make_pdf, make_settings, write_text


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path, api_key="secret", embedding_models="")
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.headers.update({"X-API-Key": "secret"})
        yield c


def test_auth_and_error_shape(client):
    anonymous = TestClient(client.app)
    assert anonymous.get("/api/v1/health").json() == {
        "status": "ok",
        "version": client.get("/api/v1/health").json()["version"],
    }
    denied = anonymous.get("/api/v1/search", params={"q": "x"})
    assert denied.status_code == 401
    body = denied.json()
    assert body["code"] == "unauthorized" and body["request_id"] == denied.headers["X-Request-ID"]
    assert anonymous.get("/api/v1/system/stats", headers={"Authorization": "Bearer secret"}).status_code == 200
    missing = client.get("/api/v1/documents/nope")
    assert missing.status_code == 404 and missing.json()["code"] == "not_found"
    bad = client.post("/api/v1/search", json={"query": "   "})
    assert bad.status_code == 400 and bad.json()["code"] == "validation_error"
    schema = client.post("/api/v1/search", json={"top_k": 3})
    assert schema.status_code == 422 and schema.json()["details"]["problems"][0]["field"].endswith("query")
    assert client.get("/api/v1/system/config").json()["settings"]["api_key"] == "***"


def test_upload_search_and_file_access(client, tmp_path):
    files = [
        (
            "files",
            (
                "refunds.md",
                io.BytesIO(b"# Refund policy\n\nItems can be returned within 30 days of delivery.\n"),
                "text/markdown",
            ),
        ),
        (
            "files",
            (
                "guide.pdf",
                io.BytesIO(make_pdf(["Page one is about shipping.", "Page two mentions tracking numbers."])),
                "application/pdf",
            ),
        ),
        ("files", ("virus.exe", io.BytesIO(b"nope"), "application/octet-stream")),
    ]
    res = client.post("/api/v1/documents/upload", files=files)
    assert res.status_code == 201
    data = res.json()
    assert sorted(d["filename"] for d in data["documents"]) == ["guide.pdf", "refunds.md"]
    assert data["errors"][0]["file"] == "virus.exe"
    assert all(d["status"] == "indexed" for d in data["documents"])

    res = client.get("/api/v1/search", params={"q": "tracking numbers", "strategy": "lexical"})
    assert res.status_code == 200
    body = res.json()
    hit = body["results"][0]
    assert hit["document"]["filename"] == "guide.pdf" and hit["page"] == 2
    assert hit["evidence"]["lexical"]["matched_terms"] and hit["evidence"]["locator"]["page"] == 2
    assert hit["highlights"] and body["timings"]["total"] > 0 and body["trace"]["spans"]
    assert "hybrid ran as keyword search" not in " ".join(body["warnings"])  # strategy was lexical

    hybrid = client.get("/api/v1/search", params={"q": "tracking", "strategy": "hybrid"}).json()
    assert hybrid["warnings"] and hybrid["results"], (
        "hybrid without a dense index degrades to keyword search with a warning"
    )

    doc_id = hit["document"]["id"]
    detail = client.get(f"/api/v1/documents/{doc_id}").json()
    assert detail["pages"] == 2 and detail["revisions"][0]["revision"] == 1 and "path" not in detail
    page = client.get(f"/api/v1/documents/{doc_id}/pages/2").json()
    assert "tracking" in page["text"].lower() and page["chunks"][0]["char_start"] == 0

    served = client.get(f"/api/v1/documents/{doc_id}/file")
    assert served.status_code == 200 and served.content.startswith(b"%PDF")
    assert (
        TestClient(client.app).get(f"/api/v1/documents/{doc_id}/file", params={"api_key": "secret"}).status_code == 200
    )
    assert TestClient(client.app).get(f"/api/v1/documents/{doc_id}/file").status_code == 401

    listed = client.get("/api/v1/documents").json()
    assert listed["total"] == 2 and all("path" not in d for d in listed["documents"])

    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 200
    assert client.get("/api/v1/documents").json()["total"] == 1
    assert client.get("/api/v1/search/recent").json()["stats"]["count"] == 0  # query log is off in the testing profile


def test_sources_documents_are_not_deletable_and_compare_works(client, tmp_path):
    docs = tmp_path / "docs"
    write_text(docs, "a.txt", "The chaff collector must be emptied after every roasting session.")
    res = client.post("/api/v1/sources", json={"path": str(docs), "name": "docs"})
    assert res.status_code == 201
    job_id = res.json()["job"]["id"]
    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{job_id}").json()["job"]
        if job["status"] in ("done", "failed"):
            break
    assert job["status"] == "done" and job["result"]["added"] == 1
    assert client.post("/api/v1/sources", json={"path": str(docs)}).status_code == 409
    assert client.post("/api/v1/sources", json={"path": ""}).status_code == 422

    doc = client.get("/api/v1/documents").json()["documents"][0]
    assert doc["path"].endswith("a.txt")
    assert client.delete(f"/api/v1/documents/{doc['id']}").status_code == 409

    cmp = client.post(
        "/api/v1/search/compare",
        json={"query": "chaff collector", "variants": [{"strategy": "lexical"}, {"strategy": "lexical", "top_k": 1}]},
    )
    assert cmp.status_code == 200
    assert [v["label"] for v in cmp.json()["variants"]] == ["lexical", "lexical"]
    assert cmp.json()["common_documents"] == [doc["id"]]

    events = client.get(f"/api/v1/sources/{res.json()['source']['id']}/events").json()["events"]
    assert {e["kind"] for e in events} >= {"scan", "added", "source_added"}
    assert client.get("/api/v1/system/metrics").text.startswith("contextsearch_")
    assert client.delete(f"/api/v1/sources/{res.json()['source']['id']}").status_code == 200
