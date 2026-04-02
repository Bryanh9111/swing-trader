from __future__ import annotations

import math

import pytest

from data.interface import PriceBar
from indicators import (
    MACDResult,
    compute_atr_last,
    compute_ema_last,
    compute_ema_series,
    compute_macd_last,
    compute_macd_series,
    compute_rsi_last,
    compute_rsi_series,
    compute_sma_last,
    compute_sma_series,
)
from indicators.interface import compute_adx_last


def _make_bars(closes: list[float]) -> list[PriceBar]:
    """Helper: create PriceBar sequence from close prices."""

    return [
        PriceBar(
            timestamp=int(i * 1e9),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1000,
        )
        for i, c in enumerate(closes)
    ]


class TestRSI:
    """Tests for RSI indicator."""

    def test_rsi_last_insufficient_data(self) -> None:
        """RSI returns None when data is insufficient."""

        prices = [100.0, 101.0, 102.0]  # len=3, period=14
        assert compute_rsi_last(prices, period=14) is None

    def test_rsi_last_valid_data(self) -> None:
        """RSI returns valid value for sufficient data."""

        prices = [100.0 + i for i in range(20)]  # len=20, period=14
        rsi = compute_rsi_last(prices, period=14)
        assert rsi is not None
        assert 0 <= rsi <= 100
        assert math.isfinite(rsi)

    def test_rsi_last_upward_trend_is_100(self) -> None:
        """RSI is 100 for sustained upward trend (avg_loss == 0)."""

        prices = [100.0 + i * 2 for i in range(30)]
        rsi = compute_rsi_last(prices, period=14)
        assert rsi == 100.0

    def test_rsi_last_downward_trend_is_0(self) -> None:
        """RSI is 0 for sustained downward trend (avg_gain == 0)."""

        prices = [200.0 - i * 2 for i in range(30)]
        rsi = compute_rsi_last(prices, period=14)
        assert rsi == 0.0

    def test_rsi_last_constant_prices(self) -> None:
        """RSI handles constant prices (avg_loss == 0)."""

        prices = [100.0] * 20
        rsi = compute_rsi_last(prices, period=14)
        assert rsi == 100.0

    def test_rsi_series_length_matches_input(self) -> None:
        """RSI series matches input length."""

        prices = [100.0 + i for i in range(20)]
        series = compute_rsi_series(prices, period=14)
        assert len(series) == 20

    def test_rsi_series_prefix_is_none(self) -> None:
        """RSI series has None prefix for insufficient data."""

        prices = [100.0 + i for i in range(20)]
        series = compute_rsi_series(prices, period=14)
        assert series[:14] == [None] * 14
        assert series[14] is not None

    def test_rsi_series_all_values_finite_and_in_range(self) -> None:
        prices = [100.0 + i * 0.25 for i in range(100)]
        series = compute_rsi_series(prices, period=14)
        for value in series:
            if value is None:
                continue
            assert 0 <= value <= 100
            assert math.isfinite(value)

    def test_rsi_series_insufficient_data_returns_all_none(self) -> None:
        prices = [100.0 + i for i in range(14)]  # len == period
        assert compute_rsi_series(prices, period=14) == [None] * 14

    def test_rsi_last_with_pricebar_input(self) -> None:
        """RSI works with PriceBar input."""

        bars = _make_bars([100.0 + i for i in range(20)])
        rsi = compute_rsi_last(bars, period=14)
        assert rsi is not None
        assert 0 <= rsi <= 100
        assert math.isfinite(rsi)

    def test_rsi_invalid_period(self) -> None:
        """RSI raises ValueError for invalid period."""

        prices = [100.0, 101.0]
        with pytest.raises(ValueError, match="period must be > 0"):
            compute_rsi_last(prices, period=0)
        with pytest.raises(ValueError, match="period must be > 0"):
            compute_rsi_series(prices, period=0)


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_last_insufficient_data(self) -> None:
        """MACD returns None when data is insufficient."""

        prices = [100.0, 101.0, 102.0]  # len=3, slow=26
        assert compute_macd_last(prices, fast=12, slow=26, signal=9) is None

    def test_macd_last_valid_data(self) -> None:
        """MACD returns valid result for sufficient data."""

        prices = [100.0 + i * 0.5 for i in range(30)]
        result = compute_macd_last(prices, fast=12, slow=26, signal=9)
        assert result is not None
        assert isinstance(result, MACDResult)
        assert math.isfinite(result.macd)
        assert math.isfinite(result.signal)
        assert math.isfinite(result.histogram)

    def test_macd_histogram_equals_difference(self) -> None:
        """MACD histogram equals macd - signal."""

        prices = [100.0 + i * 0.5 for i in range(30)]
        result = compute_macd_last(prices, fast=12, slow=26, signal=9)
        assert result is not None
        assert abs(result.histogram - (result.macd - result.signal)) < 1e-9

    def test_macd_series_length_matches_input(self) -> None:
        """MACD series matches input length."""

        prices = [100.0 + i * 0.5 for i in range(30)]
        series = compute_macd_series(prices, fast=12, slow=26, signal=9)
        assert len(series) == 30

    def test_macd_series_prefix_is_none(self) -> None:
        """MACD series has None prefix for insufficient data."""

        prices = [100.0 + i * 0.5 for i in range(30)]
        series = compute_macd_series(prices, fast=12, slow=26, signal=9)
        assert series[:25] == [None] * 25
        assert series[25] is not None

    def test_macd_series_histogram_equals_difference(self) -> None:
        prices = [100.0 + i * 0.5 for i in range(60)]
        series = compute_macd_series(prices, fast=12, slow=26, signal=9)
        for item in series:
            if item is None:
                continue
            assert abs(item.histogram - (item.macd - item.signal)) < 1e-9
            assert math.isfinite(item.macd)
            assert math.isfinite(item.signal)
            assert math.isfinite(item.histogram)

    def test_macd_last_with_pricebar_input(self) -> None:
        """MACD works with PriceBar input."""

        bars = _make_bars([100.0 + i * 0.5 for i in range(30)])
        result = compute_macd_last(bars, fast=12, slow=26, signal=9)
        assert result is not None

    def test_macd_invalid_fast_slow(self) -> None:
        """MACD raises ValueError when fast >= slow."""

        prices = [100.0, 101.0]
        with pytest.raises(ValueError, match="fast .* must be < slow"):
            compute_macd_last(prices, fast=26, slow=12, signal=9)
        with pytest.raises(ValueError, match="fast .* must be < slow"):
            compute_macd_series(prices, fast=26, slow=12, signal=9)

    def test_macd_invalid_signal(self) -> None:
        """MACD raises ValueError for invalid signal period."""

        prices = [100.0, 101.0]
        with pytest.raises(ValueError, match="must be > 0"):
            compute_macd_last(prices, fast=12, slow=26, signal=0)
        with pytest.raises(ValueError, match="must be > 0"):
            compute_macd_series(prices, fast=12, slow=26, signal=0)

    def test_macd_series_insufficient_data_returns_all_none(self) -> None:
        prices = [100.0 + i for i in range(10)]  # len < slow
        assert compute_macd_series(prices, fast=12, slow=26, signal=9) == [None] * 10


class TestEMA:
    """Tests for EMA indicator."""

    def test_ema_last_insufficient_data(self) -> None:
        """EMA returns None when data is insufficient."""

        prices = [100.0, 101.0, 102.0]  # len=3, period=10
        assert compute_ema_last(prices, period=10) is None

    def test_ema_last_valid_data(self) -> None:
        """EMA returns valid value for sufficient data."""

        prices = [100.0 + i for i in range(15)]
        ema = compute_ema_last(prices, period=10)
        assert ema is not None
        assert isinstance(ema, float)
        assert math.isfinite(ema)

    def test_ema_last_constant_prices(self) -> None:
        """EMA of constant sequence equals the constant."""

        prices = [100.0] * 15
        ema = compute_ema_last(prices, period=10)
        assert ema == 100.0

    def test_ema_series_length_matches_input(self) -> None:
        """EMA series matches input length."""

        prices = [100.0 + i for i in range(15)]
        series = compute_ema_series(prices, period=10)
        assert len(series) == 15

    def test_ema_series_prefix_is_none(self) -> None:
        """EMA series has None prefix for insufficient data."""

        prices = [100.0 + i for i in range(15)]
        series = compute_ema_series(prices, period=10)
        assert series[:9] == [None] * 9
        assert series[9] is not None

    def test_ema_series_constant_prices(self) -> None:
        prices = [100.0] * 15
        series = compute_ema_series(prices, period=10)
        assert series[-1] == 100.0
        for value in series:
            if value is None:
                continue
            assert value == 100.0
            assert math.isfinite(value)

    def test_ema_last_with_pricebar_input(self) -> None:
        """EMA works with PriceBar input."""

        bars = _make_bars([100.0 + i for i in range(15)])
        ema = compute_ema_last(bars, period=10)
        assert ema is not None

    def test_ema_invalid_period(self) -> None:
        """EMA raises ValueError for invalid period."""

        prices = [100.0, 101.0]
        with pytest.raises(ValueError, match="period must be > 0"):
            compute_ema_last(prices, period=0)
        with pytest.raises(ValueError, match="period must be > 0"):
            compute_ema_series(prices, period=0)


class TestSMA:
    """Tests for SMA indicator."""

    def test_sma_last_insufficient_data(self) -> None:
        """SMA returns None when data is insufficient."""

        prices = [100.0, 101.0, 102.0]  # len=3, period=10
        assert compute_sma_last(prices, period=10) is None

    def test_sma_last_valid_data(self) -> None:
        """SMA returns valid value for sufficient data."""

        prices = [100.0 + i for i in range(15)]
        sma = compute_sma_last(prices, period=10)
        assert sma is not None
        expected = sum([100.0 + i for i in range(5, 15)]) / 10
        assert abs(sma - expected) < 1e-9
        assert math.isfinite(sma)

    def test_sma_last_constant_prices(self) -> None:
        """SMA of constant sequence equals the constant."""

        prices = [100.0] * 15
        sma = compute_sma_last(prices, period=10)
        assert sma == 100.0

    def test_sma_series_length_matches_input(self) -> None:
        """SMA series matches input length."""

        prices = [100.0 + i for i in range(15)]
        series = compute_sma_series(prices, period=10)
        assert len(series) == 15

    def test_sma_series_prefix_is_none(self) -> None:
        """SMA series has None prefix for insufficient data."""

        prices = [100.0 + i for i in range(15)]
        series = compute_sma_series(prices, period=10)
        assert series[:9] == [None] * 9
        assert series[9] is not None

    def test_sma_series_constant_prices(self) -> None:
        prices = [100.0] * 15
        series = compute_sma_series(prices, period=10)
        assert series[-1] == 100.0
        for value in series:
            if value is None:
                continue
            assert value == 100.0
            assert math.isfinite(value)

    def test_sma_last_with_pricebar_input(self) -> None:
        """SMA works with PriceBar input."""

        bars = _make_bars([100.0 + i for i in range(15)])
        sma = compute_sma_last(bars, period=10)
        assert sma is not None

    def test_sma_invalid_period(self) -> None:
        """SMA raises ValueError for invalid period."""

        prices = [100.0, 101.0]
        with pytest.raises(ValueError, match="period must be > 0"):
            compute_sma_last(prices, period=0)
        with pytest.raises(ValueError, match="period must be > 0"):
            compute_sma_series(prices, period=0)


class TestATR:
    """Tests for ATR indicator."""

    def test_atr_last_insufficient_data(self) -> None:
        bars = [
            PriceBar(
                timestamp=i,
                open=10.0 + i,
                high=11.0 + i,
                low=9.0 + i,
                close=10.0 + i,
                volume=1000,
            )
            for i in range(14)
        ]
        assert compute_atr_last(bars, period=14, percentage=True, reference_price=100.0) is None

    def test_atr_last_percentage_true(self) -> None:
        closes = [10.0 + i for i in range(15)]
        bars = [
            PriceBar(
                timestamp=i,
                open=c,
                high=c + 1.0,
                low=c - 1.0,
                close=c,
                volume=1000,
            )
            for i, c in enumerate(closes)
        ]
        atr_pct = compute_atr_last(bars, period=14, percentage=True)
        assert atr_pct is not None
        assert math.isclose(atr_pct, 2.0 / closes[-1], rel_tol=0.0, abs_tol=1e-12)

    def test_atr_last_percentage_false(self) -> None:
        closes = [10.0 + i for i in range(15)]
        bars = [
            PriceBar(
                timestamp=i,
                open=c,
                high=c + 1.0,
                low=c - 1.0,
                close=c,
                volume=1000,
            )
            for i, c in enumerate(closes)
        ]
        atr_abs = compute_atr_last(bars, period=14, percentage=False)
        assert atr_abs is not None
        assert math.isclose(atr_abs, 2.0, rel_tol=0.0, abs_tol=1e-12)

    def test_atr_last_custom_reference_price(self) -> None:
        closes = [10.0 + i for i in range(15)]
        bars = [
            PriceBar(
                timestamp=i,
                open=c,
                high=c + 1.0,
                low=c - 1.0,
                close=c,
                volume=1000,
            )
            for i, c in enumerate(closes)
        ]
        atr_pct = compute_atr_last(bars, period=14, percentage=True, reference_price=20.0)
        assert atr_pct is not None
        assert math.isclose(atr_pct, 2.0 / 20.0, rel_tol=0.0, abs_tol=1e-12)

    def test_atr_invalid_period(self) -> None:
        bars = _make_bars([100.0, 101.0])
        with pytest.raises(ValueError, match="period must be > 0"):
            compute_atr_last(bars, period=0)

    def test_atr_reference_price_zero_returns_none(self) -> None:
        bars = _make_bars([100.0 + i for i in range(20)])
        assert compute_atr_last(bars, period=14, percentage=True, reference_price=0.0) is None


class TestEdgeCases:
    """Edge cases and boundary tests."""

    def test_empty_input(self) -> None:
        """All indicators handle empty input gracefully."""

        assert compute_rsi_last([], period=14) is None
        assert compute_macd_last([], fast=12, slow=26, signal=9) is None
        assert compute_ema_last([], period=10) is None
        assert compute_sma_last([], period=10) is None
        assert compute_atr_last([], period=14) is None

        assert compute_rsi_series([], period=14) == []
        assert compute_macd_series([], fast=12, slow=26, signal=9) == []
        assert compute_ema_series([], period=10) == []
        assert compute_sma_series([], period=10) == []

    def test_single_price(self) -> None:
        """All indicators handle single price input."""

        assert compute_rsi_last([100.0], period=14) is None
        assert compute_macd_last([100.0], fast=12, slow=26, signal=9) is None
        assert compute_ema_last([100.0], period=10) is None
        assert compute_sma_last([100.0], period=10) is None
        assert compute_atr_last([100.0], period=14) is None

        assert compute_rsi_series([100.0], period=14) == [None]
        assert compute_macd_series([100.0], fast=12, slow=26, signal=9) == [None]
        assert compute_ema_series([100.0], period=10) == [None]
        assert compute_sma_series([100.0], period=10) == [None]


class TestADX:
    """Tests for ADX indicator."""

    def test_adx_insufficient_data(self) -> None:
        highs = [10.0, 11.0]
        lows = [9.0, 10.0]
        closes = [9.5, 10.5]
        assert compute_adx_last(highs, lows, closes, period=14) is None

    def test_adx_invalid_period(self) -> None:
        highs = [10.0, 11.0, 12.0]
        lows = [9.0, 10.0, 11.0]
        closes = [9.5, 10.5, 11.5]
        with pytest.raises(ValueError, match="period must be > 0"):
            compute_adx_last(highs, lows, closes, period=0)

    def test_adx_uptrend_is_finite(self) -> None:
        n = 40
        highs = [101.0 + i for i in range(n)]
        lows = [99.0 + i for i in range(n)]
        closes = [100.0 + i for i in range(n)]
        adx = compute_adx_last(highs, lows, closes, period=14)
        assert adx is not None
        assert 0.0 <= adx <= 100.0
        assert math.isfinite(adx)
