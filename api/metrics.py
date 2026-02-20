"""Prometheus metrics for the face swap API."""

from __future__ import annotations

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "Request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    ACTIVE_REQUESTS = Gauge(
        "http_requests_in_progress",
        "Number of requests currently being processed",
    )
    SWAP_FACE_COUNT = Counter(
        "swap_faces_total",
        "Total faces swapped",
        ["status"],
    )
    RECOGNITION_COUNT = Counter(
        "recognition_requests_total",
        "Total recognition requests",
    )

    METRICS_AVAILABLE = True

except ImportError:
    METRICS_AVAILABLE = False

    # Provide no-op stubs so callers don't need to check availability
    class _NoOp:
        def labels(self, *a, **kw):
            return self

        def inc(self, *a, **kw):
            pass

        def dec(self, *a, **kw):
            pass

        def observe(self, *a, **kw):
            pass

    REQUEST_COUNT = _NoOp()
    REQUEST_LATENCY = _NoOp()
    ACTIVE_REQUESTS = _NoOp()
    SWAP_FACE_COUNT = _NoOp()
    RECOGNITION_COUNT = _NoOp()

    def generate_latest():  # noqa: F811
        return b""

    CONTENT_TYPE_LATEST = "text/plain"
