"""Every error leaves the API as JSON with the same shape.

    {"error": "human readable message", "code": "not_found", "status": 404,
     "details": {...}, "request_id": "..."}

The request id matches the ``X-Request-ID`` response header and the server
log line, so a client can quote it when asking what went wrong.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..errors import ContextSearchError
from ..observability import current_request_id, get_logger

log = get_logger("contextsearch.api")


def _payload(message: str, code: str, status: int, details: dict | None = None) -> dict:
    return {
        "error": message,
        "code": code,
        "status": status,
        "details": details or {},
        "request_id": current_request_id(),
    }


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ContextSearchError)
    async def _domain_error(request: Request, exc: ContextSearchError) -> JSONResponse:
        if exc.status >= 500:
            log.error("request failed", extra_fields={"code": exc.code, "error": exc.message, "path": request.url.path})
        return JSONResponse(status_code=exc.status, content=_payload(exc.message, exc.code, exc.status, exc.details))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            {"field": ".".join(str(p) for p in e.get("loc", [])), "problem": e.get("msg", "")} for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_payload("request validation failed", "validation_error", 422, {"problems": problems}),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            405: "method_not_allowed",
            413: "too_large",
        }.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(str(exc.detail), code, exc.status_code),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra_fields={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content=_payload("internal error; check the server log with this request id", "internal_error", 500),
        )
