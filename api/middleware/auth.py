"""API key authentication middleware."""

from __future__ import annotations

from typing import Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from utils.logger import get_logger

logger = get_logger(__name__)

_SKIP_PATHS: Set[str] = {"/docs", "/redoc", "/openapi.json", "/api/v1/health", "/"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests missing a valid ``X-API-Key`` header.

    Configured via ``API_KEYS`` env var (comma-separated list of accepted keys).
    When the key list is empty, auth is disabled and all requests pass through.
    """

    def __init__(self, app: ASGIApp, api_keys: list[str] | None = None) -> None:
        super().__init__(app)
        self.api_keys: Set[str] = set(k for k in (api_keys or []) if k)

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.api_keys:
            return await call_next(request)

        path = request.url.path
        if path in _SKIP_PATHS or path.startswith("/api/v1/results"):
            return await call_next(request)

        key = request.headers.get("X-API-Key", "")
        if key not in self.api_keys:
            return Response(
                content='{"error":"unauthorized","message":"Invalid or missing API key."}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)
