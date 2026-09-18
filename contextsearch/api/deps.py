"""Dependencies shared by the routers: the engine and the API key check."""

from __future__ import annotations

import hmac

from fastapi import Depends, Request

from ..engine import Engine
from ..errors import AuthError


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


def require_api_key(request: Request, engine: Engine = Depends(get_engine)) -> None:
    """Reject the request when an API key is configured and not presented.

    The key can come as ``X-API-Key``, ``Authorization: Bearer <key>`` or,
    for links a browser opens directly (file downloads), ``?api_key=``.
    """
    expected = engine.settings.api_key
    if not expected:
        return
    presented = request.headers.get("x-api-key") or ""
    if not presented:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
    if not presented:
        presented = request.query_params.get("api_key") or ""
    if not presented or not hmac.compare_digest(presented, expected):
        raise AuthError("a valid API key is required (X-API-Key header or Authorization: Bearer)")


def has_valid_key(request: Request, engine: Engine) -> bool:
    try:
        require_api_key(request, engine)
    except AuthError:
        return False
    return True
