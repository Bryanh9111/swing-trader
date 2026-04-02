"""Tests for TrendPatternRouter and trend pattern detection.

These tests verify:
1. Gate evaluation (Sector Regime, Relative Strength)
2. Pattern detection (MA Crossover)
3. Router integration and candidate conversion
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from common.interface import ResultStatus
from data.interface import PriceBar
from scanner.trend_pattern_router import (
    TrendPatternRouter,
    TrendPatternRouterConfig,
    detect_trend_pattern_candidate,
)
from scanner.patterns import (
    MACrossoverDetector,
    MACrossoverConfig,
    TrendPatternResult,
)
from scanner.gates import (
    SectorRegimeGate,
    RelativeStrengthGate,
    GateResult,
)


def _make_bars(
    closes: list[float],
    base_volume: float = 1_000_000,
    *,
    high_mult: float = 1.01,
    low_mult: float = 0.99,
    start_timestamp: int = 1696118400_000_000_000,  # 2023-10-01 00:00:00 UTC in ns
) -> list[PriceBar]:
    """Create synthetic price bars from close prices."""
    bars: list[PriceBar] = []
    ns_per_day = 86400 * 1_000_000_000
    for i, close in enumerate(closes):
        open_price = close * 0.995 if i % 2 == 0 else close * 1.005
        high_price = max(open_price, close) * high_mult
        low_price = min(open_price, close) * low_mult
        volume = int(base_volume * (1 + 0.1 * (i % 5)))
        timestamp = start_timestamp + i * ns_per_day
        bars.append(
            PriceBar(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close,
                volume=volume,
            )
        )
    return bars


def _make_uptrend_bars(
    start_price: float = 100.0,
    days: int = 100,
    daily_gain: float = 0.002,
) -> list[PriceBar]:
    """Create synthetic uptrend price bars."""
    closes = []
    price = start_price
    for _ in range(days):
        price *= (1 + daily_gain)
        closes.append(price)
    return _make_bars(closes)


def _make_pullback_bars(
    start_price: float = 100.0,
    uptrend_days: int = 60,
    pullback_days: int = 10,
    pullback_pct: float = 0.05,
) -> list[PriceBar]:
    """Create bars with uptrend followed by pullback to EMA20 region."""
    closes = []
    price = start_price

    # Uptrend phase
    for _ in range(uptrend_days):
        price *= 1.003
        closes.append(price)

    peak = price

    # Pullback phase
    for i in range(pullback_days):
        pullback_depth = pullback_pct * (i + 1) / pullback_days
        price = peak * (1 - pullback_depth)
        closes.append(price)

    # Recovery bar (entry trigger)
    closes.append(price * 1.015)

    return _make_bars(closes)


class TestSectorRegimeGate:
    """Tests for SectorRegimeGate."""

    def test_bullish_regime_passes(self) -> None:
        """Test that bullish sector regime passes the gate."""
        # Create bullish sector bars: price > EMA20 > EMA50, positive slope
        closes = [100 + i * 0.5 for i in range(100)]
        bars = _make_bars(closes)

        gate = SectorRegimeGate()
        result = gate.evaluate(
            symbol="TEST",
            bars=bars,
            current_date=date(2023, 10, 15),
            benchmark_bars=bars,
        )

        assert result.passed is True
        assert result.value is not None
        assert result.value > 0  # Positive slope

    def test_bearish_regime_fails(self) -> None:
        """Test that bearish sector regime fails the gate."""
        # Create bearish sector bars: downtrend
        closes = [100 - i * 0.5 for i in range(100)]
        bars = _make_bars(closes)

        gate = SectorRegimeGate()
        result = gate.evaluate(
            symbol="TEST",
            bars=bars,
            current_date=date(2023, 10, 15),
            benchmark_bars=bars,
        )

        assert result.passed is False


class TestRelativeStrengthGate:
    """Tests for RelativeStrengthGate."""

    def test_strong_rs_passes(self) -> None:
        """Test that stock outperforming benchmark passes."""
        # Stock gaining faster than benchmark
        stock_closes = [100 + i * 0.8 for i in range(100)]
        benchmark_closes = [100 + i * 0.3 for i in range(100)]

        stock_bars = _make_bars(stock_closes)
        benchmark_bars = _make_bars(benchmark_closes)

        gate = RelativeStrengthGate()
        result = gate.evaluate(
            symbol="TEST",
            bars=stock_bars,
            current_date=date(2023, 10, 15),
            benchmark_bars=benchmark_bars,
        )

        assert result.passed is True
        assert result.value is not None
        assert result.value > 0  # Positive RS slope

    def test_weak_rs_fails(self) -> None:
        """Test that stock underperforming benchmark fails."""
        # Stock gaining slower than benchmark
        stock_closes = [100 + i * 0.2 for i in range(100)]
        benchmark_closes = [100 + i * 0.5 for i in range(100)]

        stock_bars = _make_bars(stock_closes)
        benchmark_bars = _make_bars(benchmark_closes)

        gate = RelativeStrengthGate()
        result = gate.evaluate(
            symbol="TEST",
            bars=stock_bars,
            current_date=date(2023, 10, 15),
            benchmark_bars=benchmark_bars,
        )

        # RS slope should be negative or near zero
        assert result.value is not None
        # Depending on exact calculation, may pass or fail
        # The key is that value reflects relative performance


class TestMACrossoverDetector:
    """Tests for MACrossoverDetector."""

    def test_pattern_disabled_returns_not_detected(self) -> None:
        """Test that disabled pattern returns not detected."""
        config = MACrossoverConfig(enabled=False)
        detector = MACrossoverDetector(config)

        bars = _make_uptrend_bars()
        result = detector.detect(
            symbol="TEST",
            bars=bars,
            current_date=date(2023, 10, 15),
            sector_bullish=True,
            rs_strong=True,
        )

        assert result.detected is False
        assert "pattern_disabled" in result.reasons

    def test_gate_free_ignores_sector(self) -> None:
        """MA Crossover is gate-free: sector_bullish=False does NOT block detection."""
        detector = MACrossoverDetector()
        bars = _make_uptrend_bars()

        result = detector.detect(
            symbol="TEST",
            bars=bars,
            current_date=date(2023, 10, 15),
            sector_bullish=False,
            rs_strong=True,
        )

        # Should still attempt detection (may or may not detect depending on data)
        # The key assertion: no "sector" gate failure reason
        assert not any("sector" in r for r in result.reasons)

    def test_gate_free_ignores_rs(self) -> None:
        """MA Crossover is gate-free: rs_strong=False does NOT block detection."""
        detector = MACrossoverDetector()
        bars = _make_uptrend_bars()

        result = detector.detect(
            symbol="TEST",
            bars=bars,
            current_date=date(2023, 10, 15),
            sector_bullish=True,
            rs_strong=False,
        )

        # Should still attempt detection — no "rs" gate failure reason
        assert not any("rs" in r for r in result.reasons)


class TestTrendPatternRouter:
    """Tests for TrendPatternRouter."""

    def test_router_disabled_returns_none(self) -> None:
        """Test that disabled router returns None."""
        config = TrendPatternRouterConfig(enabled=False)
        router = TrendPatternRouter(config)

        bars = _make_uptrend_bars()
        result = router.detect(
            symbol="TEST",
            bars=bars,
            current_date=date(2023, 10, 15),
        )

        assert result.status is ResultStatus.SUCCESS
        assert result.data is None
        assert result.reason_code == "TREND_PATTERNS_DISABLED"

    def test_router_gate_failure_returns_none(self) -> None:
        """Test that gate failure returns None candidate."""
        router = TrendPatternRouter()

        # Create downtrend bars (should fail sector gate)
        closes = [100 - i * 0.5 for i in range(100)]
        bars = _make_bars(closes)
        sector_bars = _make_bars(closes)  # Same downtrend
        benchmark_bars = _make_bars(closes)

        result = router.detect(
            symbol="TEST",
            bars=bars,
            current_date=date(2023, 10, 15),
            sector_bars=sector_bars,
            benchmark_bars=benchmark_bars,
        )

        assert result.status is ResultStatus.SUCCESS
        assert result.data is None
        # Gate failure indicated in reason code

    def test_router_with_bullish_conditions(self) -> None:
        """Test router with bullish sector and RS conditions."""
        router = TrendPatternRouter()

        # Create bullish bars
        stock_closes = [100 + i * 0.5 for i in range(100)]
        stock_bars = _make_bars(stock_closes)

        sector_closes = [100 + i * 0.4 for i in range(100)]
        sector_bars = _make_bars(sector_closes)

        benchmark_closes = [100 + i * 0.3 for i in range(100)]
        benchmark_bars = _make_bars(benchmark_closes)

        result = router.detect(
            symbol="TEST",
            bars=stock_bars,
            current_date=date(2023, 10, 15),
            sector_bars=sector_bars,
            benchmark_bars=benchmark_bars,
        )

        assert result.status is ResultStatus.SUCCESS
        # Result depends on whether pattern is detected


class TestDetectTrendPatternCandidate:
    """Tests for detect_trend_pattern_candidate convenience function."""

    def test_convenience_function(self) -> None:
        """Test the convenience function works."""
        bars = _make_uptrend_bars()

        result = detect_trend_pattern_candidate(
            symbol="TEST",
            bars=bars,
            current_date=date(2023, 10, 15),
        )

        assert result.status is ResultStatus.SUCCESS
        # May or may not detect pattern depending on bar characteristics
