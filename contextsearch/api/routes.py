"""All API routes, grouped by area under ``/api/v1``.

Handlers are plain ``def`` functions on purpose: FastAPI runs them in a
thread pool, so a slow embedding call never blocks the event loop that
serves the UI and the health endpoint.
"""

from __future__ import annotations

import mimetypes
from typing import Any

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .. import __version__
from ..embedding import PRESETS, dense_available, pick_device
from ..engine import Engine
from ..errors import ValidationError
from ..observability import metrics
from ..readers import supported_extensions
from ..rerank import RERANKER_PRESETS
from ..retrieval import SearchRequest
from .deps import get_engine, has_valid_key, require_api_key
from .schemas import AddSourceBody, BuildModelBody, CompareBody, PatchSourceBody, ReindexBody, SearchBody

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
open_router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------- search


def _run_search(engine: Engine, request: Request, **kwargs: Any) -> dict[str, Any]:
    kwargs["request_id"] = request.state.request_id
    return engine.search(SearchRequest(**kwargs)).to_dict()


@router.get("/search", tags=["search"], summary="Search with query parameters")
def search_get(
    request: Request,
    q: str = Query(..., min_length=1, max_length=2000, description="The query. Quote a phrase to require it."),
    strategy: str = Query("", description="lexical, dense or hybrid"),
    models: str = Query("", description="Comma-separated embedding models for dense retrieval"),
    top_k: int = Query(0, ge=0, le=1000),
    candidates: int = Query(0, ge=0, le=5000),
    fusion: str = Query("", description="rrf or weighted"),
    alpha: float | None = Query(None, ge=0.0, le=1.0),
    rrf_k: int = Query(0, ge=0),
    rerank: bool | None = Query(None),
    rerank_model: str = Query(""),
    group: bool | None = Query(None, description="Group chunks by document"),
    prefix_last: bool = Query(False, description="Treat the last word as a prefix (search-as-you-type)"),
    source: str | None = Query(None, description="Restrict to one source id"),
    ext: str | None = Query(None, description="Restrict to one file extension"),
    document: str | None = Query(None, description="Restrict to one document id"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    text: bool = Query(True, description="Include full chunk text"),
    explain: bool = Query(True, description="Include evidence and trace"),
    engine: Engine = Depends(get_engine),
) -> dict[str, Any]:
    return _run_search(
        engine,
        request,
        query=q,
        strategy=strategy,
        models=[m.strip() for m in models.split(",") if m.strip()] or None,
        top_k=top_k,
        candidates=candidates,
        fusion=fusion,
        alpha=alpha,
        rrf_k=rrf_k,
        rerank=rerank,
        rerank_model=rerank_model,
        group_by_document=group,
        prefix_last=prefix_last,
        source_id=source,
        ext=ext,
        document_id=document,
        min_confidence=min_confidence,
        include_text=text,
        explain=explain,
    )


@router.post("/search", tags=["search"], summary="Search with a JSON body")
def search_post(body: SearchBody, request: Request, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return _run_search(engine, request, **body.to_kwargs())


@router.post("/search/compare", tags=["search"], summary="Run one query under several configurations")
def search_compare(body: CompareBody, request: Request, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    variants = []
    for v in body.variants:
        kwargs = v.model_dump(exclude={"label"})
        kwargs.update({"source_id": body.source_id, "ext": body.ext, "request_id": request.state.request_id})
        variants.append(kwargs)
    responses = engine.compare(body.query, variants)
    out = []
    for v, resp in zip(body.variants, responses, strict=True):
        data = resp.to_dict()
        data["label"] = v.label or _variant_label(data)
        out.append(data)
    # Which documents every variant agrees on: a quick read of how much the strategies differ.
    doc_sets = [{r["document"]["id"] for r in d["results"]} for d in out]
    common = set.intersection(*doc_sets) if doc_sets else set()
    return {
        "query": body.query,
        "variants": out,
        "common_documents": sorted(common),
        "request_id": request.state.request_id,
    }


def _variant_label(data: dict[str, Any]) -> str:
    label = data["strategy"]
    if data["strategy"] != "lexical" and data["models"]:
        label += " (" + ", ".join(m.split("/")[-1] for m in data["models"]) + ")"
    if data["rerank"]:
        label += " + rerank"
    return label


@router.get("/search/recent", tags=["search"], summary="Recent searches from the query log")
def search_recent(limit: int = Query(50, ge=1, le=500), engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"queries": engine.store.recent_queries(limit), "stats": engine.store.query_stats()}


# ---------------------------------------------------------------------- sources


@router.get("/sources", tags=["sources"], summary="List sources")
def list_sources(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"sources": [s.to_dict() for s in engine.store.list_sources()], "watched": engine.watcher.watched()}


@router.post("/sources", tags=["sources"], status_code=201, summary="Add a folder and start indexing it")
def add_source(body: AddSourceBody, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    source, job = engine.add_folder(body.path, body.name, body.recursive, body.include, body.exclude)
    return {"source": source.to_dict(), "job": job.to_dict()}


@router.get("/sources/{source_id}", tags=["sources"], summary="One source with its last scan summary")
def get_source(source_id: str, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    source = engine.store.get_source(source_id)
    active = engine.jobs.active_for("sync", source_id)
    return {"source": source.to_dict(), "active_job": active.to_dict() if active else None}


@router.patch("/sources/{source_id}", tags=["sources"], summary="Enable or pause a source")
def patch_source(source_id: str, body: PatchSourceBody, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"source": engine.set_source_enabled(source_id, body.enabled).to_dict()}


@router.delete("/sources/{source_id}", tags=["sources"], summary="Remove a source and everything indexed from it")
def delete_source(source_id: str, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"removed": engine.remove_source(source_id).to_dict()}


@router.post("/sources/{source_id}/sync", tags=["sources"], status_code=202, summary="Scan the folder now")
def sync_source(source_id: str, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"job": engine.sync_source(source_id).to_dict()}


@router.get("/sources/{source_id}/events", tags=["sources"], summary="What happened to files in this source")
def source_events(
    source_id: str, limit: int = Query(100, ge=1, le=1000), engine: Engine = Depends(get_engine)
) -> dict[str, Any]:
    engine.store.get_source(source_id)
    return {"events": engine.store.list_events(limit, source_id=source_id)}


# ---------------------------------------------------------------------- documents


@router.get("/documents", tags=["documents"], summary="List documents")
def list_documents(
    q: str | None = Query(None, description="Filter by file name, title or path"),
    source: str | None = None,
    status: str = Query("indexed", description="indexed, failed, removed, pending or all"),
    ext: str | None = None,
    sort: str = Query("indexed_at", description="indexed_at, updated_at, filename, size, pages, chunks or revision"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    engine: Engine = Depends(get_engine),
) -> dict[str, Any]:
    sources = {s.id: s for s in engine.store.list_sources()}
    docs, total = engine.store.list_documents(source, status, q, ext, sort, order == "desc", limit, offset)
    items = []
    for d in docs:
        src = sources.get(d.source_id)
        item = d.to_dict(include_path=bool(src and src.kind == "folder"))
        item["source_name"] = src.name if src else ""
        items.append(item)
    return {"documents": items, "total": total, "limit": limit, "offset": offset}


@router.post("/documents/upload", tags=["documents"], status_code=201, summary="Upload files into the uploads source")
def upload_documents(
    files: list[UploadFile] = File(..., description="One or more files"),
    wait: bool = Query(True, description="Wait for indexing to finish before responding"),
    engine: Engine = Depends(get_engine),
) -> dict[str, Any]:
    documents, jobs, errors = [], [], []
    for upload in files:
        name = upload.filename or "upload"
        try:
            document, job = engine.save_upload(name, upload.file, wait=120.0 if wait else None)
        except ValidationError as exc:
            errors.append({"file": name, "error": exc.message})
            continue
        jobs.append(job.to_dict())
        if document is not None:
            documents.append(document.to_dict(include_path=False))
    return {"documents": documents, "jobs": jobs, "errors": errors}


@router.get("/documents/{document_id}", tags=["documents"], summary="Document details, pages and revision history")
def get_document(document_id: str, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return engine.document_detail(document_id)


@router.get(
    "/documents/{document_id}/pages/{page}", tags=["documents"], summary="Text of one page with chunk boundaries"
)
def get_page(document_id: str, page: int, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return engine.document_page(document_id, page)


@router.get("/documents/{document_id}/file", tags=["documents"], summary="Download or open the original file")
def get_file(document_id: str, download: bool = Query(False), engine: Engine = Depends(get_engine)) -> FileResponse:
    doc, path = engine.document_file(document_id)
    media_type = mimetypes.guess_type(doc.filename)[0] or "application/octet-stream"
    disposition = "attachment" if download else "inline"
    return FileResponse(path, media_type=media_type, filename=doc.filename, content_disposition_type=disposition)


@router.delete("/documents/{document_id}", tags=["documents"], summary="Delete an uploaded document")
def delete_document(document_id: str, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"deleted": engine.delete_document(document_id).to_dict(include_path=False)}


# ---------------------------------------------------------------------- jobs


@router.get("/jobs", tags=["jobs"], summary="Recent background jobs")
def list_jobs(
    limit: int = Query(50, ge=1, le=100), status: str | None = None, engine: Engine = Depends(get_engine)
) -> dict[str, Any]:
    return {"jobs": [j.to_dict() for j in engine.jobs.list(limit, status)], "queued": engine.worker.queued}


@router.get("/jobs/{job_id}", tags=["jobs"], summary="One job with progress")
def get_job(job_id: str, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"job": engine.jobs.get(job_id).to_dict()}


@router.post("/jobs/{job_id}/cancel", tags=["jobs"], summary="Ask a running job to stop")
def cancel_job(job_id: str, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"job": engine.jobs.cancel(job_id).to_dict()}


# ---------------------------------------------------------------------- models


@router.get("/models", tags=["models"], summary="Indexed embedding models, presets and reranker options")
def list_models(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    ok, reason = dense_available()
    indexed = engine.models_info()
    indexed_names = {m["name"] for m in indexed}
    return {
        "dense_available": ok,
        "reason": reason or None,
        "device": engine.settings.embedding_device or (pick_device() if ok else None),
        "indexed": indexed,
        "configured": engine.settings.embedding_model_list,
        "presets": [
            {
                "name": p.name,
                "dimension": p.dimension,
                "note": p.note,
                "multilingual": p.multilingual,
                "size_mb": p.size_mb,
                "indexed": p.name in indexed_names,
            }
            for p in PRESETS.values()
        ],
        "rerankers": [
            {"name": n, "note": note, "default": n == engine.settings.rerank_model}
            for n, note in RERANKER_PRESETS.items()
        ],
        "rerank_default": engine.settings.rerank,
        "rerank_model": engine.settings.rerank_model,
    }


@router.post("/models", tags=["models"], status_code=202, summary="Build a dense index for a model")
def build_model(body: BuildModelBody, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"job": engine.build_dense_index(body.name.strip()).to_dict()}


@router.delete("/models/{name:path}", tags=["models"], summary="Drop a model's dense index")
def drop_model(name: str, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    engine.drop_dense_index(name)
    return {"dropped": name}


@router.post("/reindex", tags=["models"], status_code=202, summary="Re-chunk documents with the current settings")
def reindex(body: ReindexBody | None = None, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {"job": engine.reindex(force=bool(body and body.force)).to_dict()}


# ---------------------------------------------------------------------- system


@open_router.get("/health", tags=["system"], summary="Liveness and component status")
def health(request: Request, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    # Load balancers poll this without a key; they get the short form.
    if not has_valid_key(request, engine):
        return {"status": "ok", "version": __version__}
    return engine.health()


@router.get("/system/stats", tags=["system"], summary="Corpus, index and query statistics")
def stats(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return engine.stats()


@router.get("/system/config", tags=["system"], summary="Effective settings, secrets masked")
def config(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
    return {
        "settings": engine.settings.to_dict(),
        "chunking_signature": engine.settings.chunking_signature(),
        "readers": sorted(supported_extensions()),
        "version": __version__,
    }


@router.get("/system/metrics", tags=["system"], summary="Prometheus text metrics", response_class=PlainTextResponse)
def prometheus_metrics(engine: Engine = Depends(get_engine)) -> PlainTextResponse:
    s = engine.store.stats()
    metrics.gauge("contextsearch_documents", s["documents"])
    metrics.gauge("contextsearch_chunks", s["chunks"])
    metrics.gauge("contextsearch_jobs_queued", engine.worker.queued)
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@router.get("/system/events", tags=["system"], summary="Recent index events across all sources")
def events(
    limit: int = Query(100, ge=1, le=1000), kinds: str = Query(""), engine: Engine = Depends(get_engine)
) -> dict[str, Any]:
    kind_list = [k.strip() for k in kinds.split(",") if k.strip()] or None
    return {"events": engine.store.list_events(limit, kind_list)}
