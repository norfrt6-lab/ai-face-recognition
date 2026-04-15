"""Unit tests for utils.rate_limiter.RateLimiter.

Tests the rate limiting logic directly — no HTTP server or FastAPI app needed.
"""

from __future__ import annotations

import threading
import time

import pytest

from utils.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_window(limiter: RateLimiter, key: str, n: int) -> None:
    """Call is_allowed *n* times for *key* (all expected to succeed)."""
    for _ in range(n):
        assert limiter.is_allowed(key) is True


# ---------------------------------------------------------------------------
# Basic allow / deny
# ---------------------------------------------------------------------------


class TestRateLimiterAllowDeny:
    def test_allows_calls_within_limit(self):
        limiter = RateLimiter(max_calls=5, window_seconds=60.0)
        for _ in range(5):
            assert limiter.is_allowed("client1") is True

    def test_denies_call_exceeding_limit(self):
        limiter = RateLimiter(max_calls=3, window_seconds=60.0)
        _fill_window(limiter, "client1", 3)
        assert limiter.is_allowed("client1") is False

    def test_different_keys_are_independent(self):
        limiter = RateLimiter(max_calls=2, window_seconds=60.0)
        _fill_window(limiter, "a", 2)
        # "a" is exhausted but "b" should still be allowed
        assert limiter.is_allowed("a") is False
        assert limiter.is_allowed("b") is True

    def test_max_calls_one_allows_first_denies_second(self):
        limiter = RateLimiter(max_calls=1, window_seconds=60.0)
        assert limiter.is_allowed("x") is True
        assert limiter.is_allowed("x") is False

    def test_invalid_max_calls_raises(self):
        with pytest.raises(ValueError, match="max_calls"):
            RateLimiter(max_calls=0)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window_seconds"):
            RateLimiter(max_calls=1, window_seconds=0.0)


# ---------------------------------------------------------------------------
# Window expiry
# ---------------------------------------------------------------------------


class TestRateLimiterWindowExpiry:
    def test_calls_allowed_after_window_expires(self):
        """After the window elapses the counter should reset."""
        limiter = RateLimiter(max_calls=2, window_seconds=0.1)
        _fill_window(limiter, "client", 2)
        assert limiter.is_allowed("client") is False
        time.sleep(0.15)  # wait for window to expire
        # Old timestamps have expired → should be allowed again
        assert limiter.is_allowed("client") is True

    def test_sliding_window_counts_only_recent_calls(self):
        """Calls from before the window should not count against the limit."""
        limiter = RateLimiter(max_calls=3, window_seconds=0.2)
        # Make 2 calls and let them age out
        limiter.is_allowed("c")
        limiter.is_allowed("c")
        time.sleep(0.25)
        # Now make 3 more calls; first two are expired → all 3 should pass
        for _ in range(3):
            assert limiter.is_allowed("c") is True

    def test_reset_clears_state_for_key(self):
        limiter = RateLimiter(max_calls=1, window_seconds=60.0)
        assert limiter.is_allowed("k") is True
        assert limiter.is_allowed("k") is False
        limiter.reset("k")
        assert limiter.is_allowed("k") is True

    def test_reset_all_clears_all_keys(self):
        limiter = RateLimiter(max_calls=1, window_seconds=60.0)
        limiter.is_allowed("a")
        limiter.is_allowed("b")
        assert limiter.is_allowed("a") is False
        limiter.reset_all()
        assert limiter.is_allowed("a") is True
        assert limiter.is_allowed("b") is True


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestRateLimiterProperties:
    def test_max_calls_property(self):
        limiter = RateLimiter(max_calls=42, window_seconds=30.0)
        assert limiter.max_calls == 42

    def test_window_seconds_property(self):
        limiter = RateLimiter(max_calls=10, window_seconds=5.5)
        assert limiter.window_seconds == pytest.approx(5.5)

    def test_repr_contains_config(self):
        limiter = RateLimiter(max_calls=10, window_seconds=30.0)
        r = repr(limiter)
        assert "RateLimiter" in r
        assert "10" in r
        assert "30" in r


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestRateLimiterThreadSafety:
    def test_concurrent_calls_respect_limit(self):
        """Even under heavy concurrency the limit must never be exceeded."""
        limit = 50
        limiter = RateLimiter(max_calls=limit, window_seconds=60.0)
        allowed: list = []
        errors: list = []

        def worker():
            try:
                for _ in range(10):
                    if limiter.is_allowed("shared"):
                        allowed.append(1)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(allowed) <= limit

    def test_no_race_on_reset(self):
        """reset() and is_allowed() called concurrently should not crash."""
        limiter = RateLimiter(max_calls=5, window_seconds=60.0)
        errors: list = []

        def caller():
            try:
                for _ in range(30):
                    limiter.is_allowed("r")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def resetter():
            try:
                for _ in range(10):
                    limiter.reset("r")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = (
            [threading.Thread(target=caller) for _ in range(4)]
            + [threading.Thread(target=resetter) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
