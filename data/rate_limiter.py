"""Thread-safe token bucket rate limiter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Condition, Lock

__all__ = ["TokenBucketRateLimiter"]


@dataclass(slots=True)
class _BucketState:
    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """Token bucket rate limiter with blocking acquire.

    This implementation is thread-safe and suitable for sharing across all
    adapters that must respect a global request budget (e.g., Polygon free tier).
    """

    def __init__(self, calls_per_minute: float, bucket_size: int | None = None) -> None:
        """Create a token bucket limiter.

        Args:
            calls_per_minute: Steady-state refill rate.
            bucket_size: Maximum burst size (defaults to ``calls_per_minute``).
        """

        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive.")
        if bucket_size is None:
            bucket_size = int(max(1, round(calls_per_minute)))
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive.")

        self._calls_per_minute = float(calls_per_minute)
        self._bucket_size = int(bucket_size)
        self._refill_rate = self._calls_per_minute / 60.0

        lock = Lock()
        self._lock = lock
        self._condition = Condition(lock)
        now = time.monotonic()
        self._state = _BucketState(tokens=float(self._bucket_size), last_refill=now)

    @property
    def calls_per_minute(self) -> float:
        return self._calls_per_minute

    @property
    def bucket_size(self) -> int:
        return self._bucket_size

    def reset(self) -> None:
        """Reset the bucket to a full state."""

        with self._condition:
            self._state.tokens = float(self._bucket_size)
            self._state.last_refill = time.monotonic()
            self._condition.notify_all()

    def acquire(self, timeout_seconds: float | None = None) -> bool:
        """Wait for a token and consume it when available."""

        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative or None.")

        deadline = None if timeout_seconds is None else (time.monotonic() + timeout_seconds)
        with self._condition:
            while True:
                self._refill_locked()
                if self._state.tokens >= 1.0:
                    self._state.tokens -= 1.0
                    return True

                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    return False

                remaining = None if deadline is None else max(0.0, deadline - now)
                needed = 1.0 - self._state.tokens
                wait_for = needed / self._refill_rate if self._refill_rate > 0 else 0.1
                if remaining is not None:
                    wait_for = min(wait_for, remaining)
                self._condition.wait(timeout=wait_for)

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._state.last_refill
        if elapsed <= 0:
            return
        self._state.last_refill = now
        self._state.tokens = min(
            float(self._bucket_size),
            self._state.tokens + elapsed * self._refill_rate,
        )
