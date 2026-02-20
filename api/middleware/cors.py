# CORS configuration + request-ID injection middleware.
#
# Provides:
#   - configure_cors(app)  — attach CORSMiddleware with settings
#   - RequestIDMiddleware  — stamps every request/response with a
#                            unique X-Request-ID header for tracing
#   - RateLimitMiddleware  — simple in-memory sliding-window rate
#                            limiter keyed on client IP

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from api.metrics import ACTIVE_REQUESTS, REQUEST_COUNT, REQUEST_LATENCY
from utils.logger import get_logger

logger = get_logger(__name__)


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

        origins = settings.api.cors_origins
        max_age = 600
    except Exception:
        origins = ["http://localhost:8501", "http://127.0.0.1:8501"]
        max_age = 600

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
        max_requests: int = 60,
        window_seconds: float = 60.0,
        exclude_paths: set = frozenset({"/api/v1/health", "/docs", "/openapi.json", "/redoc"}),
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exclude_paths = set(exclude_paths)

        # ip → deque of request timestamps (float, seconds since epoch)
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._last_cleanup: float = 0.0
        self._cleanup_interval: float = 300.0  # evict stale IPs every 5 min
        self._max_tracked_ips: int = 10_000

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for excluded paths
        path = request.url.path
        if any(path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        # Resolve client IP
        client_ip = self._get_client_ip(request)
        now = time.time()

        # Periodic eviction of stale IP entries to bound memory
        if now - self._last_cleanup > self._cleanup_interval:
            self._last_cleanup = now
            cutoff_evict = now - self.window_seconds
            stale = [ip for ip, dq in self._windows.items() if not dq or dq[-1] <= cutoff_evict]
            for ip in stale:
                del self._windows[ip]

        # Hard cap on tracked IPs to prevent memory exhaustion
        if len(self._windows) >= self._max_tracked_ips and client_ip not in self._windows:
            logger.warning(
                "Rate limiter: max tracked IPs "
                f"({self._max_tracked_ips}) reached, rejecting new IP."
            )
            return Response(
                content=(
                    '{"error":"rate_limit_exceeded",'
                    '"message":"Server under heavy load. Try again later."}'
                ),
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        window = self._windows[client_ip]

        # Evict timestamps outside the current window
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

        # Check limit
        if len(window) >= self.max_requests:
            oldest = window[0]
            retry_after = max(1, int(self.window_seconds - (now - oldest)) + 1)
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

        Uses the direct connection IP to prevent spoofing via
        ``X-Forwarded-For``. Deploy behind a reverse proxy that sets
        ``X-Real-IP`` or use Starlette's ``ProxyHeadersMiddleware``
        with explicit trusted hosts for proxy-aware IP resolution.
        """
        if request.client:
            return request.client.host
        return "unknown"


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request Prometheus counters and latency histogram."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        ACTIVE_REQUESTS.inc()
        t_start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = time.perf_counter() - t_start
            route = request.scope.get("route")
            endpoint = getattr(route, "path", request.url.path) if route else request.url.path
            method = request.method
            sc = str(response.status_code) if response else "500"
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=sc).inc()
            ACTIVE_REQUESTS.dec()


def configure_middleware(app: FastAPI) -> None:
    """Attach all middleware to *app* in the correct order.

    Starlette applies middleware in **reverse** registration order, so
    the **last** registered middleware is the outermost (runs first on
    request). We register from innermost to outermost.
    """
    try:
        from config.settings import settings  # noqa: PLC0415

        rpm = settings.api.rate_limit_per_minute
        workers = settings.api.workers
        api_keys = settings.api.api_keys
    except Exception:
        rpm = 60
        workers = 1
        api_keys = []

    # 1. CORS (innermost — runs closest to the route handler)
    configure_cors(app)

    # 2. Rate limiter
    if workers > 1:
        logger.warning(
            f"In-memory rate limiter is active with API_WORKERS={workers}. "
            "Each worker maintains its own counter, so effective limit is "
            f"{rpm}x{workers}={rpm * workers} requests/min per IP. "
            "For accurate multi-worker rate limiting, use slowapi + Redis."
        )

    app.add_middleware(
        RateLimitMiddleware,
        max_requests=rpm,
        window_seconds=60.0,
        exclude_paths={"/api/v1/health", "/docs", "/openapi.json", "/redoc"},
    )

    # 3. API key auth (no-op when api_keys is empty)
    from api.middleware.auth import APIKeyMiddleware  # noqa: PLC0415

    app.add_middleware(APIKeyMiddleware, api_keys=api_keys)

    # 4. Prometheus metrics
    app.add_middleware(MetricsMiddleware)

    # 5. Request ID (outermost — always runs first)
    app.add_middleware(RequestIDMiddleware)

    from api.metrics import METRICS_AVAILABLE  # noqa: PLC0415

    auth_status = "on" if api_keys else "off"
    metrics_status = "on" if METRICS_AVAILABLE else "off (install prometheus-client)"
    logger.info(
        f"Middleware configured | "
        f"RequestID=on | Auth={auth_status} | Metrics={metrics_status} | "
        f"RateLimit={rpm}/min | CORS=on"
    )
