"""In-memory sliding-window rate limiter for FastAPI / Starlette.

Provides:
  - ``RateLimiter``          — pure-Python, thread-safe rate limiting logic.
  - ``RateLimitMiddleware``  — Starlette middleware that enforces per-IP limits.

Usage (standalone)::

    limiter = RateLimiter(max_calls=60, window_seconds=60.0)
    if limiter.is_allowed("192.168.1.1"):
        # process request
    else:
        # return 429

Usage (FastAPI middleware)::

    from utils.rate_limiter import RateLimitMiddleware
    app.add_middleware(RateLimitMiddleware, max_calls=60, window_seconds=60.0)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Tracks call timestamps per key in a ``collections.deque``.  On each call
    to ``is_allowed()`` the deque is trimmed to remove timestamps that have
    fallen outside the current window.

    Args:
        max_calls:       Maximum number of calls permitted within *window_seconds*.
        window_seconds:  Length of the sliding time window in seconds.
    """

    def __init__(self, max_calls: int = 60, window_seconds: float = 60.0) -> None:
        if max_calls < 1:
            raise ValueError(f"max_calls must be >= 1, got {max_calls}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        self._max_calls = max_calls
        self._window = window_seconds
        # Per-key deque of call timestamps (monotonic)
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_allowed(self, key: str) -> bool:
        """Check whether *key* is within the rate limit.

        Removes timestamps outside the current window, then either records a
        new call (if under the limit) or rejects it.

        Args:
            key: Arbitrary string identifying the caller (e.g. an IP address).

        Returns:
            ``True`` if the call is permitted; ``False`` if the limit is
            exceeded and the request should be rejected (429).
        """
        now = time.monotonic()
        cutoff = now - self._window

        with self._lock:
            dq = self._windows[key]
            # Evict expired timestamps (oldest are at the left of the deque)
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) < self._max_calls:
                dq.append(now)
                return True
            return False

    def reset(self, key: str) -> None:
        """Clear all recorded timestamps for *key*.

        Args:
            key: The key whose history should be cleared.
        """
        with self._lock:
            self._windows.pop(key, None)

    def reset_all(self) -> None:
        """Clear all rate-limit state (all keys)."""
        with self._lock:
            self._windows.clear()

    @property
    def max_calls(self) -> int:
        """Maximum calls allowed per window."""
        return self._max_calls

    @property
    def window_seconds(self) -> float:
        """Sliding window duration in seconds."""
        return self._window

    def __repr__(self) -> str:
        with self._lock:
            n_keys = len(self._windows)
        return (
            f"RateLimiter("
            f"max_calls={self._max_calls}, "
            f"window={self._window}s, "
            f"tracked_keys={n_keys})"
        )


# ---------------------------------------------------------------------------
# Starlette / FastAPI middleware
# ---------------------------------------------------------------------------


class RateLimitMiddleware:
    """Starlette ASGI middleware that enforces per-IP rate limits.

    Reads the client IP from the ``X-Forwarded-For`` header (first entry),
    falling back to ``request.client.host``.  Requests that exceed the
    configured limit receive a ``429 Too Many Requests`` JSON response.

    Args:
        app:            The next ASGI application in the stack.
        max_calls:      Allowed calls per *window_seconds* per IP.
        window_seconds: Sliding window duration in seconds.
    """

    def __init__(
        self,
        app,
        max_calls: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        self._app = app
        self._limiter = RateLimiter(max_calls=max_calls, window_seconds=window_seconds)

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[override]
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        # Extract client IP
        client_ip = self._get_client_ip(scope)

        if not self._limiter.is_allowed(client_ip):
            logger.warning(
                f"[rate_limiter] Rate limit exceeded for IP={client_ip} "
                f"(max={self._limiter.max_calls} per {self._limiter.window_seconds}s)"
            )
            await self._send_429(send)
            return

        await self._app(scope, receive, send)

    @staticmethod
    def _get_client_ip(scope: dict) -> str:
        """Extract the client IP address from the ASGI scope.

        Prefers the first entry of the ``X-Forwarded-For`` header (set by
        reverse proxies), falling back to the direct client host.

        Args:
            scope: ASGI connection scope dict.

        Returns:
            IP address string, or 'unknown' if it cannot be determined.
        """
        # Check X-Forwarded-For header (list of (name, value) byte pairs)
        headers: dict = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        forwarded_for = headers.get("x-forwarded-for", "")
        if forwarded_for:
            # Take only the first (client) IP from the comma-separated list
            return forwarded_for.split(",")[0].strip()

        # Fall back to direct client host
        client = scope.get("client")
        if client:
            return client[0]

        return "unknown"

    @staticmethod
    async def _send_429(send: Callable) -> None:
        """Send an HTTP 429 Too Many Requests JSON response."""
        import json

        body = json.dumps(
            {
                "error": "too_many_requests",
                "message": "Rate limit exceeded. Please try again later.",
            }
        ).encode("utf-8")

        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )
