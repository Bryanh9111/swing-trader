"""Sector Regime Gate for trend pattern detectors.

This gate evaluates whether the tech sector (XLK, SOXX, etc.) is in a bullish
regime, which is a prerequisite for trend continuation patterns on growth stocks.

Gate criteria:
1. Sector ETF close > EMA20 (short-term bullish)
2. EMA20 > EMA50 (bullish structure)
3. EMA20 slope is positive (trending up)

Usage:
    gate = SectorRegimeGate(SectorRegimeConfig())
    result = gate.evaluate(symbol, stock_bars, date, benchmark_bars=xlk_bars)
    if result.passed:
        # Sector is bullish, proceed with trend pattern detection
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import msgspec

from data.interface import PriceBar
from .interface import GateConfig, GateResult


class SectorRegimeConfig(GateConfig, frozen=True, kw_only=True):
    """Configuration for Sector Regime Gate.

    Attributes:
        ema_short_period: Short EMA period for trend (default 20).
        ema_long_period: Long EMA period for structure (default 50).
        slope_lookback_days: Days to calculate EMA slope (default 10).
        min_slope: Minimum slope to be considered bullish (default 0.001 = 0.1%/day).
        require_price_above_ema: Require price > EMA20.
        require_ema_stack: Require EMA20 > EMA50.
    """

    ema_short_period: int = 20
    ema_long_period: int = 50
    slope_lookback_days: int = 10
    min_slope: float = 0.001  # 0.1% per day
    require_price_above_ema: bool = True
    require_ema_stack: bool = True


@dataclass(slots=True)
class SectorRegimeGate:
    """Sector Regime Gate implementation.

    Evaluates whether the sector (via benchmark ETF like XLK) is bullish.
    """

    config: SectorRegimeConfig

    def __init__(self, config: SectorRegimeConfig | None = None) -> None:
        self.config = config or SectorRegimeConfig()

    @property
    def gate_name(self) -> str:
        return "sector_regime"

    def evaluate(
        self,
        symbol: str,
        bars: list[PriceBar],
        current_date: date,
        *,
        benchmark_bars: list[PriceBar] | None = None,
    ) -> GateResult:
        """Evaluate sector regime gate.

        Args:
            symbol: Stock symbol (not used, for interface compatibility).
            bars: Stock price bars (not used, we use benchmark_bars).
            current_date: Current simulation date.
            benchmark_bars: Sector ETF bars (XLK, SOXX, etc.). Required.

        Returns:
            GateResult indicating whether sector is bullish.
        """
        if not self.config.enabled:
            return GateResult(passed=True, reason="gate_disabled")

        if not benchmark_bars or len(benchmark_bars) < self.config.ema_long_period + 10:
            return GateResult(
                passed=False,
                reason=f"insufficient_benchmark_bars:{len(benchmark_bars) if benchmark_bars else 0}",
            )

        closes = [bar.close for bar in benchmark_bars]
        current_close = closes[-1]

        # Calculate EMAs
        ema_short = self._calculate_ema(closes, self.config.ema_short_period)
        ema_long = self._calculate_ema(closes, self.config.ema_long_period)

        if len(ema_short) < 2 or len(ema_long) < 2:
            return GateResult(passed=False, reason="ema_calculation_failed")

        ema20_current = ema_short[-1]
        ema50_current = ema_long[-1]

        reasons_failed: list[str] = []
        meta: dict[str, Any] = {
            "current_close": current_close,
            "ema20": ema20_current,
            "ema50": ema50_current,
        }

        # Check 1: Price > EMA20
        price_above_ema = current_close > ema20_current
        if self.config.require_price_above_ema and not price_above_ema:
            reasons_failed.append(f"price_below_ema20:{current_close:.2f}<{ema20_current:.2f}")

        # Check 2: EMA20 > EMA50 (bullish stack)
        ema_stack_bullish = ema20_current > ema50_current
        if self.config.require_ema_stack and not ema_stack_bullish:
            reasons_failed.append(f"no_bullish_stack:ema20={ema20_current:.2f}<=ema50={ema50_current:.2f}")

        # Check 3: EMA20 slope is positive
        slope = self._calculate_slope(ema_short, self.config.slope_lookback_days)
        meta["ema20_slope"] = slope

        slope_positive = slope >= self.config.min_slope
        if not slope_positive:
            reasons_failed.append(f"negative_slope:{slope:.4f}<{self.config.min_slope}")

        if reasons_failed:
            return GateResult(
                passed=False,
                reason="; ".join(reasons_failed),
                value=slope,
                meta=meta,
            )

        return GateResult(
            passed=True,
            reason=f"sector_bullish:price>{ema20_current:.0f},stack_ok,slope={slope:.4f}",
            value=slope,
            meta=meta,
        )

    def _calculate_ema(self, values: list[float], period: int) -> list[float]:
        """Calculate EMA for a series of values."""
        if len(values) < period:
            return []

        multiplier = 2 / (period + 1)
        ema_values: list[float] = []

        # Start with SMA
        sma = sum(values[:period]) / period
        ema_values.append(sma)

        for i in range(period, len(values)):
            ema = (values[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)

        return ema_values

    def _calculate_slope(self, ema_values: list[float], lookback: int) -> float:
        """Calculate slope of EMA over lookback period (normalized)."""
        if len(ema_values) < lookback + 1:
            return 0.0

        recent = ema_values[-lookback:]
        if len(recent) < 2:
            return 0.0

        # Simple slope: (end - start) / start / days
        start_val = recent[0]
        end_val = recent[-1]
        if start_val <= 0:
            return 0.0

        return (end_val - start_val) / start_val / len(recent)
