"""API key authentication middleware."""

from __future__ import annotations

import hmac
from typing import List, Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from utils.logger import get_logger

logger = get_logger(__name__)

_SKIP_PATHS: Set[str] = {"/docs", "/redoc", "/openapi.json", "/api/v1/health", "/"}
_SKIP_PREFIXES = ("/api/v1/results/",)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests missing a valid ``X-API-Key`` header.

    Configured via ``API_KEYS`` env var (comma-separated list of accepted keys).
    When the key list is empty, auth is disabled and all requests pass through.
    """

    def __init__(self, app: ASGIApp, api_keys: list[str] | None = None) -> None:
        super().__init__(app)
        self._api_keys: List[str] = [k for k in (api_keys or []) if k]

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._api_keys:
            return await call_next(request)

        # Allow CORS preflight through without auth
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in _SKIP_PATHS or any(path.startswith(p) for p in _SKIP_PREFIXES):
            return await call_next(request)

        key = request.headers.get("X-API-Key", "")
        if not self._check_key(key):
            return Response(
                content='{"error":"unauthorized","message":"Invalid or missing API key."}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)

    def _check_key(self, candidate: str) -> bool:
        """Constant-time comparison against all accepted keys."""
        if not candidate:
            return False
        return any(
            hmac.compare_digest(candidate, valid_key)
            for valid_key in self._api_keys
        )
