"""Simple circuit breaker for model inference calls.

States:
  CLOSED     — normal operation, calls pass through.
  OPEN       — too many consecutive failures, calls rejected fast.
  HALF_OPEN  — recovery probe: allow one call through to test health.
"""

from __future__ import annotations

import time
import threading
from enum import Enum
from typing import Optional


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is OPEN."""

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Retry after {retry_after:.1f}s."
        )


class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    Args:
        name:             Human-readable label (e.g. 'detector', 'swapper').
        failure_threshold: Consecutive failures before opening the circuit.
        recovery_timeout:  Seconds to wait in OPEN state before probing.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._effective_state()

    def check(self) -> None:
        """Raise CircuitOpenError if the circuit is OPEN and not ready to probe."""
        with self._lock:
            st = self._effective_state()
            if st == CircuitState.OPEN:
                retry_after = self.recovery_timeout - (
                    time.monotonic() - (self._last_failure_time or 0)
                )
                raise CircuitOpenError(self.name, max(0.0, retry_after))

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def _effective_state(self) -> CircuitState:
        """Determine real state (transitions OPEN → HALF_OPEN after timeout)."""
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state
