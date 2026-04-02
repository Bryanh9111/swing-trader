"""Tests for KDJ indicator implementation."""

from __future__ import annotations

import pytest

from data.interface import PriceBar
from indicators.interface import KDJResult, compute_kdj_last, compute_kdj_series


def _bar(*, high: float, low: float, close: float, ts: int = 0) -> PriceBar:
    """Create a PriceBar with specified H/L/C values."""
    return PriceBar(timestamp=ts, open=close, high=high, low=low, close=close, volume=1000)


def _bars_trending_up(n: int = 20) -> list[PriceBar]:
    """Create bars with upward trend (close near highs)."""
    return [
        _bar(high=100.0 + i * 2, low=98.0 + i * 2, close=99.5 + i * 2, ts=i)
        for i in range(n)
    ]


def _bars_trending_down(n: int = 20) -> list[PriceBar]:
    """Create bars with downward trend (close near lows)."""
    return [
        _bar(high=140.0 - i * 2, low=138.0 - i * 2, close=138.5 - i * 2, ts=i)
        for i in range(n)
    ]


def _bars_flat(n: int = 20) -> list[PriceBar]:
    """Create bars with flat price (no trend)."""
    return [
        _bar(high=100.0, low=100.0, close=100.0, ts=i)
        for i in range(n)
    ]


class TestComputeKdjLast:
    """Tests for compute_kdj_last function."""

    def test_basic_calculation_uptrend(self) -> None:
        """KDJ should show high values (overbought) in strong uptrend."""
        bars = _bars_trending_up(20)
        result = compute_kdj_last(bars)

        assert result is not None
        assert isinstance(result, KDJResult)
        # In uptrend, K/D should be high (close near period high)
        assert result.k > 50.0
        assert result.d > 50.0
        # J = 3K - 2D, should also be high in uptrend
        assert result.j > 50.0

    def test_basic_calculation_downtrend(self) -> None:
        """KDJ should show low values (oversold) in strong downtrend."""
        bars = _bars_trending_down(20)
        result = compute_kdj_last(bars)

        assert result is not None
        # In downtrend, K/D should be low (close near period low)
        assert result.k < 50.0
        assert result.d < 50.0
        # J should also be low in downtrend
        assert result.j < 50.0

    def test_flat_price_returns_neutral(self) -> None:
        """KDJ should return neutral values (50) when price is flat."""
        bars = _bars_flat(20)
        result = compute_kdj_last(bars)

        assert result is not None
        # When high == low, RSV should be 50 (neutral)
        assert result.k == pytest.approx(50.0, rel=0.1)
        assert result.d == pytest.approx(50.0, rel=0.1)
        assert result.j == pytest.approx(50.0, rel=0.1)

    def test_insufficient_data_returns_none(self) -> None:
        """KDJ should return None when insufficient data."""
        bars = _bars_trending_up(5)  # Less than default n_period=9
        result = compute_kdj_last(bars)

        assert result is None

    def test_empty_input_returns_none(self) -> None:
        """KDJ should return None for empty input."""
        result = compute_kdj_last([])
        assert result is None

    def test_custom_periods(self) -> None:
        """KDJ should respect custom period parameters."""
        bars = _bars_trending_up(15)
        result = compute_kdj_last(bars, n_period=5, k_period=2, d_period=2)

        assert result is not None
        assert result.k > 50.0  # Still uptrend

    def test_invalid_period_raises(self) -> None:
        """KDJ should raise ValueError for invalid periods."""
        bars = _bars_trending_up(20)

        with pytest.raises(ValueError):
            compute_kdj_last(bars, n_period=0)

        with pytest.raises(ValueError):
            compute_kdj_last(bars, k_period=-1)

        with pytest.raises(ValueError):
            compute_kdj_last(bars, d_period=0)

    def test_j_formula_correctness(self) -> None:
        """J should equal 3*K - 2*D."""
        bars = _bars_trending_up(20)
        result = compute_kdj_last(bars)

        assert result is not None
        expected_j = 3.0 * result.k - 2.0 * result.d
        assert result.j == pytest.approx(expected_j, rel=1e-9)


class TestComputeKdjSeries:
    """Tests for compute_kdj_series function."""

    def test_series_length_matches_input(self) -> None:
        """KDJ series should have same length as input."""
        bars = _bars_trending_up(20)
        result = compute_kdj_series(bars)

        assert len(result) == len(bars)

    def test_prefix_values_are_none(self) -> None:
        """Insufficient prefix values should be None."""
        bars = _bars_trending_up(20)
        result = compute_kdj_series(bars, n_period=9)

        # First 8 values (indices 0-7) should be None
        for i in range(8):
            assert result[i] is None

        # Index 8 onwards should have values
        assert result[8] is not None

    def test_series_values_are_valid(self) -> None:
        """Non-None series values should be valid KDJResult."""
        bars = _bars_trending_up(20)
        result = compute_kdj_series(bars)

        for kdj in result:
            if kdj is not None:
                assert isinstance(kdj, KDJResult)
                assert 0.0 <= kdj.k <= 100.0
                assert 0.0 <= kdj.d <= 100.0
                # J can exceed 0-100 range (it's 3K - 2D)

    def test_last_value_matches_last_function(self) -> None:
        """Last series value should match compute_kdj_last result."""
        bars = _bars_trending_up(20)
        series = compute_kdj_series(bars)
        last = compute_kdj_last(bars)

        assert series[-1] is not None
        assert last is not None
        assert series[-1].k == pytest.approx(last.k, rel=1e-9)
        assert series[-1].d == pytest.approx(last.d, rel=1e-9)
        assert series[-1].j == pytest.approx(last.j, rel=1e-9)

    def test_insufficient_data_returns_all_none(self) -> None:
        """Series with insufficient data should be all None."""
        bars = _bars_trending_up(5)
        result = compute_kdj_series(bars, n_period=9)

        assert len(result) == 5
        assert all(v is None for v in result)


class TestKdjEdgeCases:
    """Edge case tests for KDJ indicator."""

    def test_close_at_high(self) -> None:
        """When close equals high, RSV should be 100."""
        bars = [
            _bar(high=110.0, low=100.0, close=110.0, ts=i)
            for i in range(15)
        ]
        result = compute_kdj_last(bars, n_period=9)

        assert result is not None
        # K should approach 100 (close at high)
        assert result.k > 90.0

    def test_close_at_low(self) -> None:
        """When close equals low, RSV should be 0."""
        bars = [
            _bar(high=110.0, low=100.0, close=100.0, ts=i)
            for i in range(15)
        ]
        result = compute_kdj_last(bars, n_period=9)

        assert result is not None
        # K should approach 0 (close at low)
        assert result.k < 10.0

    def test_non_pricebar_input_returns_none(self) -> None:
        """KDJ requires PriceBar input (H/L/C data)."""
        # Float list input should return None
        result = compute_kdj_last([100.0, 101.0, 102.0, 103.0, 104.0] * 5)
        assert result is None
