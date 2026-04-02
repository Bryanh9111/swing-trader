"""Relative Strength Gate for trend pattern detectors.

This gate evaluates whether a stock is outperforming its benchmark (QQQ/SPY),
which indicates institutional interest and trend leadership.

Gate criteria:
1. RS line (stock/benchmark) is rising over lookback period
2. RS line slope is positive

RS = Stock_Close / Benchmark_Close

Usage:
    gate = RelativeStrengthGate(RelativeStrengthConfig())
    result = gate.evaluate(symbol, stock_bars, date, benchmark_bars=qqq_bars)
    if result.passed:
        # Stock has strong RS, proceed with trend pattern detection
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from collections.abc import Sequence
from typing import Any

import msgspec

from data.interface import PriceBar
from .interface import GateConfig, GateResult


class RelativeStrengthConfig(GateConfig, frozen=True, kw_only=True):
    """Configuration for Relative Strength Gate.

    Attributes:
        rs_slope_lookback: Days to calculate RS slope (default 20).
        min_rs_slope: Minimum RS slope to pass (default 0.0 = non-negative).
        rs_ema_period: EMA period for smoothing RS line (default 10).
        require_rs_new_high: Require RS at or near 20-day high (optional).
        rs_high_tolerance_pct: How close to 20D high counts as "near" (default 5%).
    """

    rs_slope_lookback: int = 20
    min_rs_slope: float = 0.0  # Non-negative slope = outperforming
    rs_ema_period: int = 10
    require_rs_new_high: bool = False
    rs_high_tolerance_pct: float = 0.05


@dataclass(slots=True)
class RelativeStrengthGate:
    """Relative Strength Gate implementation.

    Evaluates whether a stock is outperforming its benchmark.
    """

    config: RelativeStrengthConfig

    def __init__(self, config: RelativeStrengthConfig | None = None) -> None:
        self.config = config or RelativeStrengthConfig()

    @property
    def gate_name(self) -> str:
        return "relative_strength"

    def evaluate(
        self,
        symbol: str,
        bars: list[PriceBar],
        current_date: date,
        *,
        benchmark_bars: list[PriceBar] | None = None,
    ) -> GateResult:
        """Evaluate relative strength gate.

        Args:
            symbol: Stock symbol.
            bars: Stock price bars. Required.
            current_date: Current simulation date.
            benchmark_bars: Benchmark (QQQ/SPY) bars. Required.

        Returns:
            GateResult indicating whether RS is strong.
        """
        if not self.config.enabled:
            return GateResult(passed=True, reason="gate_disabled")

        if not bars or not benchmark_bars:
            return GateResult(
                passed=False,
                reason="missing_bars",
                meta={"stock_bars": len(bars) if bars else 0, "benchmark_bars": len(benchmark_bars) if benchmark_bars else 0},
            )

        min_bars = self.config.rs_slope_lookback + self.config.rs_ema_period + 5
        if len(bars) < min_bars or len(benchmark_bars) < min_bars:
            return GateResult(
                passed=False,
                reason=f"insufficient_bars:{min(len(bars), len(benchmark_bars))}<{min_bars}",
            )

        # Calculate RS line: stock_close / benchmark_close
        # Align bars by using the same length (most recent)
        align_len = min(len(bars), len(benchmark_bars))
        stock_closes = [bar.close for bar in bars[-align_len:]]
        bench_closes = [bar.close for bar in benchmark_bars[-align_len:]]

        rs_line: list[float] = []
        for sc, bc in zip(stock_closes, bench_closes):
            if bc > 0:
                rs_line.append(sc / bc)
            else:
                rs_line.append(0.0)

        if not rs_line or len(rs_line) < self.config.rs_slope_lookback:
            return GateResult(passed=False, reason="rs_calculation_failed")

        # Smooth RS with EMA
        rs_smoothed = self._calculate_ema(rs_line, self.config.rs_ema_period)
        if len(rs_smoothed) < self.config.rs_slope_lookback:
            return GateResult(passed=False, reason="rs_ema_calculation_failed")

        # Calculate RS slope
        rs_slope = self._calculate_slope(rs_smoothed, self.config.rs_slope_lookback)
        rs_current = rs_smoothed[-1]

        meta: dict[str, Any] = {
            "rs_current": rs_current,
            "rs_slope": rs_slope,
            "stock_close": stock_closes[-1],
            "benchmark_close": bench_closes[-1],
        }

        reasons_failed: list[str] = []

        # Check RS slope
        if rs_slope < self.config.min_rs_slope:
            reasons_failed.append(f"rs_slope_negative:{rs_slope:.4f}<{self.config.min_rs_slope}")

        # Optional: Check if RS near new high
        if self.config.require_rs_new_high:
            rs_20d_high = max(rs_smoothed[-20:]) if len(rs_smoothed) >= 20 else max(rs_smoothed)
            distance_from_high = (rs_20d_high - rs_current) / rs_20d_high if rs_20d_high > 0 else 1.0
            meta["rs_20d_high"] = rs_20d_high
            meta["distance_from_high_pct"] = distance_from_high

            if distance_from_high > self.config.rs_high_tolerance_pct:
                reasons_failed.append(f"rs_not_near_high:{distance_from_high*100:.1f}%>{self.config.rs_high_tolerance_pct*100:.0f}%")

        if reasons_failed:
            return GateResult(
                passed=False,
                reason="; ".join(reasons_failed),
                value=rs_slope,
                meta=meta,
            )

        return GateResult(
            passed=True,
            reason=f"rs_strong:slope={rs_slope:.4f},rs={rs_current:.4f}",
            value=rs_slope,
            meta=meta,
        )

    def _calculate_ema(self, values: list[float], period: int) -> list[float]:
        """Calculate EMA for a series of values."""
        if len(values) < period:
            return values.copy()

        multiplier = 2 / (period + 1)
        ema_values: list[float] = []

        sma = sum(values[:period]) / period
        ema_values.append(sma)

        for i in range(period, len(values)):
            ema = (values[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)

        return ema_values

    def _calculate_slope(self, values: list[float], lookback: int) -> float:
        """Calculate slope over lookback period (normalized)."""
        if len(values) < lookback:
            return 0.0

        recent = values[-lookback:]
        if len(recent) < 2:
            return 0.0

        start_val = recent[0]
        end_val = recent[-1]
        if start_val <= 0:
            return 0.0

        # Slope per day, normalized
        return (end_val - start_val) / start_val / len(recent)


# ---------------------------------------------------------------------------
# Public pure function — shared by platform detector & trend router
# ---------------------------------------------------------------------------

def _ema(values: list[float], period: int) -> list[float]:
    """Calculate EMA for a series of values (standalone, no class dependency)."""
    if len(values) < period:
        return values.copy()
    multiplier = 2 / (period + 1)
    result: list[float] = [sum(values[:period]) / period]
    for i in range(period, len(values)):
        result.append(values[i] * multiplier + result[-1] * (1 - multiplier))
    return result


def _slope(values: list[float], lookback: int) -> float:
    """Normalized slope over lookback period (standalone, no class dependency)."""
    if len(values) < lookback or lookback < 2:
        return 0.0
    recent = values[-lookback:]
    start_val, end_val = recent[0], recent[-1]
    if start_val <= 0:
        return 0.0
    return (end_val - start_val) / start_val / len(recent)


def compute_rs_slope(
    asset_closes: Sequence[float],
    benchmark_closes: Sequence[float],
    lookback: int = 20,
    ema_period: int = 10,
) -> float | None:
    """Compute relative-strength slope between an asset and a benchmark.

    Pure function — no side effects, no gate logic, no pass/fail decision.
    Used by both platform detector and trend pattern router.

    Args:
        asset_closes: Chronological close prices of the stock.
        benchmark_closes: Chronological close prices of the benchmark (e.g. QQQ).
        lookback: Days over which to measure the RS slope.
        ema_period: EMA smoothing period for the RS line.

    Returns:
        Normalised slope (change per day / starting value), or *None* when
        the input data is insufficient.
    """
    min_len = lookback + ema_period + 5
    if len(asset_closes) < min_len or len(benchmark_closes) < min_len:
        return None

    align_len = min(len(asset_closes), len(benchmark_closes))
    ac = list(asset_closes[-align_len:])
    bc = list(benchmark_closes[-align_len:])

    rs_line = [a / b if b > 0 else 0.0 for a, b in zip(ac, bc)]
    if len(rs_line) < lookback:
        return None

    rs_smoothed = _ema(rs_line, ema_period)
    if len(rs_smoothed) < lookback:
        return None

    return _slope(rs_smoothed, lookback)
