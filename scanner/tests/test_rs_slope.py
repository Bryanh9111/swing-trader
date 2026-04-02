"""Tests for compute_rs_slope pure function."""

from __future__ import annotations

import pytest

from scanner.gates.relative_strength import compute_rs_slope


def _rising(start: float, step: float, n: int) -> list[float]:
    """Generate a steadily rising price series."""
    return [start + step * i for i in range(n)]


def _flat(price: float, n: int) -> list[float]:
    """Generate a flat price series."""
    return [price] * n


class TestComputeRsSlope:
    """compute_rs_slope: pure function, no gate logic."""

    def test_both_rising_asset_faster_positive_slope(self) -> None:
        """Asset rising faster than benchmark → positive RS slope."""
        n = 60
        asset = _rising(100.0, 1.0, n)       # +1/day
        bench = _rising(100.0, 0.3, n)        # +0.3/day
        result = compute_rs_slope(asset, bench)
        assert result is not None
        assert result > 0.0

    def test_asset_flat_benchmark_rising_negative_slope(self) -> None:
        """Asset flat, benchmark rising → negative RS slope."""
        n = 60
        asset = _flat(100.0, n)
        bench = _rising(100.0, 1.0, n)
        result = compute_rs_slope(asset, bench)
        assert result is not None
        assert result < 0.0

    def test_insufficient_bars_returns_none(self) -> None:
        """Too few bars → None."""
        asset = _flat(100.0, 10)
        bench = _flat(100.0, 10)
        result = compute_rs_slope(asset, bench)
        assert result is None

    def test_exact_minimum_bars(self) -> None:
        """Exactly min_len bars should compute (not None)."""
        # min_len = lookback(20) + ema_period(10) + 5 = 35
        n = 35
        asset = _rising(100.0, 0.5, n)
        bench = _rising(100.0, 0.5, n)
        result = compute_rs_slope(asset, bench)
        assert result is not None

    def test_one_below_minimum_bars_returns_none(self) -> None:
        """One bar below min_len → None."""
        n = 34
        asset = _rising(100.0, 0.5, n)
        bench = _rising(100.0, 0.5, n)
        result = compute_rs_slope(asset, bench)
        assert result is None

    def test_equal_performance_near_zero_slope(self) -> None:
        """Asset and benchmark identical → slope near zero."""
        n = 60
        prices = _rising(100.0, 0.5, n)
        result = compute_rs_slope(prices, prices.copy())
        assert result is not None
        assert abs(result) < 1e-6

    def test_custom_lookback_and_ema(self) -> None:
        """Custom parameters should work without error."""
        n = 80
        asset = _rising(50.0, 0.8, n)
        bench = _rising(50.0, 0.2, n)
        result = compute_rs_slope(asset, bench, lookback=30, ema_period=15)
        assert result is not None
        assert result > 0.0

    def test_different_length_arrays_aligned(self) -> None:
        """Different-length inputs are right-aligned before computation."""
        asset = _rising(100.0, 1.0, 100)   # longer
        bench = _rising(100.0, 0.3, 60)    # shorter
        result = compute_rs_slope(asset, bench)
        assert result is not None
        assert result > 0.0
