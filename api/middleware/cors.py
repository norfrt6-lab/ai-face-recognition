# ============================================================
# AI Face Recognition & Face Swap
# api/middleware/cors.py
# ============================================================
# CORS configuration + request-ID injection middleware.
#
# Provides:
#   - configure_cors(app)  — attach CORSMiddleware with settings
#   - RequestIDMiddleware  — stamps every request/response with a
#                            unique X-Request-ID header for tracing
#   - RateLimitMiddleware  — simple in-memory sliding-window rate
#                            limiter keyed on client IP
# ============================================================

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Dict, Deque

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# CORS
# ============================================================

def configure_cors(app: FastAPI) -> None:
    """
    Attach CORSMiddleware to *app* using values from settings.

    Falls back to sensible development defaults if the settings
    module cannot be imported.

    Args:
        app: The FastAPI application instance.
    """
    try:
        from config.settings import settings  # noqa: PLC0415
        origins      = settings.api.cors_origins
        max_age      = 600
    except Exception:
        origins      = ["http://localhost:8501", "http://127.0.0.1:8501"]
        max_age      = 600

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            "X-API-Key",
            "Accept",
            "Origin",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Faces-Swapped",
            "X-Processing-Ms",
            "X-Enhanced",
            "X-Watermarked",
        ],
        max_age=max_age,
    )
    logger.info(f"CORS configured | allowed origins: {origins}")


# ============================================================
# Request-ID Middleware
# ============================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Stamps every HTTP request and response with a unique UUID and
    records processing time.

    - Reads ``X-Request-ID`` from the incoming request headers.
      If present, uses the client-supplied value; otherwise generates
      a new ``uuid4`` string.
    - Stores the ID in ``request.state.request_id`` so route handlers
      and other middleware can reference it for structured logging.
    - Echoes the ID back in the ``X-Request-ID`` response header.
    - Records wall-clock processing time in the ``X-Processing-Ms``
      response header.

    Usage::

        app.add_middleware(RequestIDMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Honour client-supplied ID or generate a new one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Process the request and record timing
        t_start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Echo back on response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Ms"] = f"{elapsed_ms:.1f}"
        return response


# ============================================================
# Rate Limit Middleware
# ============================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple sliding-window in-memory rate limiter keyed on client IP.

    Limits each IP to *max_requests* requests per *window_seconds* window.
    Requests exceeding the limit receive HTTP 429 Too Many Requests with a
    ``Retry-After`` header.

    This implementation uses a ``deque`` per IP to store timestamps of
    recent requests, making it O(1) for most operations.

    ⚠️  This is a single-process in-memory limiter and does NOT share
    state across multiple Uvicorn workers.  For multi-worker / multi-node
    deployments, replace with a Redis-backed limiter such as
    ``slowapi`` or ``fastapi-limiter``.

    Args:
        app:            The ASGI app to wrap.
        max_requests:   Maximum allowed requests per window per IP.
        window_seconds: Sliding window size in seconds.
        exclude_paths:  Set of path prefixes that bypass rate limiting
                        (e.g. ``{"/api/v1/health"}``).

    Usage::

        app.add_middleware(
            RateLimitMiddleware,
            max_requests=60,
            window_seconds=60,
            exclude_paths={"/api/v1/health", "/docs", "/openapi.json"},
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        max_requests:   int   = 60,
        window_seconds: float = 60.0,
        exclude_paths:  set   = frozenset({"/api/v1/health", "/docs", "/openapi.json", "/redoc"}),
    ) -> None:
        super().__init__(app)
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self.exclude_paths  = set(exclude_paths)

        # ip → deque of request timestamps (float, seconds since epoch)
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for excluded paths
        path = request.url.path
        if any(path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        # Resolve client IP
        client_ip = self._get_client_ip(request)
        now       = time.time()
        window    = self._windows[client_ip]

        # Evict timestamps outside the current window
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

        # Check limit
        if len(window) >= self.max_requests:
            oldest      = window[0]
            retry_after = int(self.window_seconds - (now - oldest)) + 1
            logger.warning(
                f"Rate limit exceeded | ip={client_ip} | "
                f"requests={len(window)}/{self.max_requests} | "
                f"path={path}"
            )
            return Response(
                content=(
                    '{"error":"rate_limit_exceeded",'
                    f'"message":"Too many requests. Retry after {retry_after}s.",'
                    f'"retry_after":{retry_after}}}'
                ),
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(retry_after)},
            )

        # Record this request
        window.append(now)
        return await call_next(request)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Extract the real client IP from the request.

        Prefers ``X-Forwarded-For`` (set by reverse proxies / load
        balancers) over the direct connection IP.

        Args:
            request: FastAPI/Starlette Request object.

        Returns:
            Client IP address string.
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For may be a comma-separated list; take the first
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        if request.client:
            return request.client.host

        return "unknown"


# ============================================================
# Convenience setup function
# ============================================================

def configure_middleware(app: FastAPI) -> None:
    """
    Attach all middleware to *app* in the correct order.

    Middleware is applied in reverse registration order by Starlette,
    so the first-registered middleware is outermost (runs first on
    request, last on response).

    Order (outermost → innermost):
      1. RequestIDMiddleware  — stamp every request with a unique ID
      2. RateLimitMiddleware  — enforce per-IP request rate limits
      3. CORSMiddleware       — handle preflight + CORS headers

    Args:
        app: The FastAPI application instance.
    """
    # 1. Request ID (outermost — always runs)
    app.add_middleware(RequestIDMiddleware)

    # 2. Rate limiter
    try:
        from config.settings import settings  # noqa: PLC0415
        rpm = settings.api.rate_limit_per_minute
        workers = settings.api.workers
    except Exception:
        rpm = 60
        workers = 1

    if workers > 1:
        logger.warning(
            f"In-memory rate limiter is active with API_WORKERS={workers}. "
            "Each worker maintains its own counter, so effective limit is "
            f"{rpm}×{workers}={rpm * workers} requests/min per IP. "
            "For accurate multi-worker rate limiting, use slowapi + Redis."
        )

    app.add_middleware(
        RateLimitMiddleware,
        max_requests=rpm,
        window_seconds=60.0,
        exclude_paths={"/api/v1/health", "/docs", "/openapi.json", "/redoc"},
    )

    # 3. CORS (innermost — runs closest to the route handler)
    configure_cors(app)

    logger.info(
        f"Middleware configured | "
        f"RequestID=on | RateLimit={rpm}/min | CORS=on"
    )
