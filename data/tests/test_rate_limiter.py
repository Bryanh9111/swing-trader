from __future__ import annotations

import threading
import time as time_module

import pytest

from data.rate_limiter import TokenBucketRateLimiter


def test_token_bucket_initialization_defaults() -> None:
    limiter = TokenBucketRateLimiter(calls_per_minute=12)
    assert limiter.calls_per_minute == 12.0
    assert limiter.bucket_size == 12


@pytest.mark.parametrize("calls_per_minute", [0, -1])
def test_token_bucket_initialization_rejects_non_positive_calls(calls_per_minute: float) -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(calls_per_minute=calls_per_minute)


@pytest.mark.parametrize("bucket_size", [0, -10])
def test_token_bucket_initialization_rejects_non_positive_bucket(bucket_size: int) -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(calls_per_minute=10, bucket_size=bucket_size)


def test_acquire_success_path_consumes_token() -> None:
    limiter = TokenBucketRateLimiter(calls_per_minute=60, bucket_size=1)
    assert limiter.acquire(timeout_seconds=0.1) is True
    assert limiter.acquire(timeout_seconds=0.01) is False


def test_acquire_timeout_when_rate_limited() -> None:
    limiter = TokenBucketRateLimiter(calls_per_minute=1, bucket_size=1)
    assert limiter.acquire(timeout_seconds=0.1) is True
    assert limiter.acquire(timeout_seconds=0.01) is False


@pytest.mark.parametrize("timeout_seconds", [-0.1, -10.0])
def test_acquire_rejects_negative_timeout(timeout_seconds: float) -> None:
    limiter = TokenBucketRateLimiter(calls_per_minute=60, bucket_size=1)
    with pytest.raises(ValueError):
        limiter.acquire(timeout_seconds=timeout_seconds)


def test_reset_refills_bucket() -> None:
    limiter = TokenBucketRateLimiter(calls_per_minute=60, bucket_size=1)
    assert limiter.acquire(timeout_seconds=0.1) is True
    assert limiter.acquire(timeout_seconds=0.01) is False
    limiter.reset()
    assert limiter.acquire(timeout_seconds=0.1) is True


def test_acquire_thread_safety_multiple_threads() -> None:
    threads = 20
    limiter = TokenBucketRateLimiter(calls_per_minute=60, bucket_size=threads)
    barrier = threading.Barrier(threads)
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        ok = limiter.acquire(timeout_seconds=0.5)
        with results_lock:
            results.append(ok)

    running = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for thread in running:
        thread.start()
    for thread in running:
        thread.join(timeout=2.0)

    assert len(results) == threads
    assert all(results)
    assert 0.0 <= limiter._state.tokens <= limiter.bucket_size  # noqa: SLF001 - invariant check


def test_refill_locked_no_elapsed_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time_module, "monotonic", lambda: 1.0)
    limiter = TokenBucketRateLimiter(calls_per_minute=60, bucket_size=1)
    assert limiter.acquire(timeout_seconds=0.01) is True
