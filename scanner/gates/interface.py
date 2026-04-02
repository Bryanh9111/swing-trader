"""Interface definitions for pattern gate conditions.

Gates control when trend continuation patterns are allowed to fire.
They prevent the trend pattern detectors from generating signals
during unfavorable market conditions (e.g., sector rotation, weak RS).

Two gates are required for trend patterns:
1. Sector Regime Gate: XLK/SOXX must be bullish
2. Relative Strength Gate: Stock must outperform benchmark (QQQ/SPY)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from datetime import date

import msgspec

from data.interface import PriceBar


class GateConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Base configuration for gate conditions.

    Attributes:
        enabled: Master switch for this gate.
    """

    enabled: bool = True


class GateResult(msgspec.Struct, frozen=True, kw_only=True):
    """Result from a gate evaluation.

    Attributes:
        passed: Whether the gate condition is met.
        reason: Human-readable reason for the result.
        value: Numeric value used for evaluation (e.g., RS slope, sector score).
        meta: Additional metadata for debugging.
    """

    passed: bool
    reason: str = ""
    value: float | None = None
    meta: dict[str, Any] = msgspec.field(default_factory=dict)


@runtime_checkable
class Gate(Protocol):
    """Protocol for gate conditions.

    Gates evaluate market conditions and return whether the condition
    is favorable for trend pattern detection.
    """

    def evaluate(
        self,
        symbol: str,
        bars: list[PriceBar],
        current_date: date,
        *,
        benchmark_bars: list[PriceBar] | None = None,
    ) -> GateResult:
        """Evaluate gate condition.

        Args:
            symbol: Stock symbol being evaluated.
            bars: Historical price bars for the symbol (oldest first).
            current_date: Current simulation/detection date.
            benchmark_bars: Benchmark price bars (e.g., QQQ, XLK) for comparison.

        Returns:
            GateResult indicating whether the gate condition is met.
        """
        ...

    @property
    def gate_name(self) -> str:
        """Return the gate type name."""
        ...
