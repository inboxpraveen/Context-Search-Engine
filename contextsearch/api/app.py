"""FastAPI application factory.

``create_app()`` builds the app around an :class:`Engine`. The engine starts
with the server (lifespan) and stops with it. Every request gets an id that
appears in the ``X-Request-ID`` response header, in the JSON error body and
in the log line for that request.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Settings
from ..engine import Engine
from ..observability import get_logger, metrics, new_request_id, set_request_id
from .errors import install_error_handlers
from .routes import open_router, router

log = get_logger("contextsearch.api")
UI_DIR = Path(__file__).resolve().parent.parent / "ui"

DESCRIPTION = """
Hybrid document search you can run yourself: keyword (BM25) and semantic (embedding) retrieval
fused into one ranked list, with the evidence for every result.

Authentication: when the server has `CSE_API_KEY` set, send it as `X-API-Key: <key>` or
`Authorization: Bearer <key>` on every `/api` call. `/api/v1/health` answers without a key.

Errors are JSON: `{"error", "code", "status", "details", "request_id"}`. The `request_id`
matches the `X-Request-ID` header and the server log line.
"""


def create_app(engine: Engine | None = None, settings: Settings | None = None) -> FastAPI:
    engine = engine or Engine(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        if engine.settings.is_production and not engine.settings.allowed_roots:
            log.warning("CSE_ALLOWED_ROOTS is empty: folder sources cannot be added through the API in production")
        if not engine.settings.api_key:
            log.warning("no CSE_API_KEY set: the API is open to anyone who can reach this port")
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(
        title="contextsearch",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "search", "description": "Search, compare strategies, query log."},
            {"name": "sources", "description": "Watched folders."},
            {"name": "documents", "description": "Indexed documents, pages, files, uploads."},
            {"name": "jobs", "description": "Background indexing jobs."},
            {"name": "models", "description": "Dense indexes per embedding model, reranking, reindexing."},
            {"name": "system", "description": "Health, stats, configuration, metrics, events."},
        ],
    )
    app.state.engine = engine
    install_error_handlers(app)

    if engine.settings.cors_origin_list:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=engine.settings.cors_origin_list,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        rid = (request.headers.get("x-request-id") or "").strip()[:64] or new_request_id()
        request.state.request_id = rid
        set_request_id(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            set_request_id("")
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            metrics.observe("contextsearch_http_latency_ms", elapsed, method=request.method)
            metrics.inc("contextsearch_http_requests", method=request.method, status=str(response.status_code))
            if request.url.path != "/api/v1/health":
                set_request_id(rid)
                log.info(
                    "http",
                    extra_fields={
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "ms": round(elapsed, 1),
                    },
                )
                set_request_id("")
        return response

    app.include_router(open_router)
    app.include_router(router)

    if UI_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
        # The asset URLs carry the version, so a browser that cached 3.0's app.js
        # fetches 3.1's after an upgrade instead of running old UI against a new API.
        index_html = (
            (UI_DIR / "index.html")
            .read_text(encoding="utf-8")
            .replace('href="/ui/app.css"', f'href="/ui/app.css?v={__version__}"')
            .replace('src="/ui/app.js"', f'src="/ui/app.js?v={__version__}"')
        )

        @app.get("/", include_in_schema=False)
        async def index() -> HTMLResponse:
            return HTMLResponse(index_html, headers={"Cache-Control": "no-cache"})

    return app
