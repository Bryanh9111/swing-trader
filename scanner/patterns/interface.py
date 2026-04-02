"""Interface definitions for trend continuation pattern detectors.

These detectors are designed to capture growth/momentum stocks that exhibit
trend continuation patterns rather than box consolidation patterns.

The framework ships with a single demo pattern (MA Crossover).
Add your own detectors by implementing the ``TrendPatternDetector`` protocol.

Gate requirements:
- Sector must be bullish (XLK/SOXX in uptrend)
- Relative strength must be positive (stock outperforming QQQ/SPY)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from datetime import date

import msgspec

from data.interface import PriceBar


class TrendPatternConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Base configuration for trend pattern detectors.

    Attributes:
        enabled: Master switch for this pattern detector.
        min_score: Minimum pattern score (0-1) to emit a candidate.
        atr_period: ATR calculation period for volatility-based levels.
    """

    enabled: bool = True
    min_score: float = 0.6
    atr_period: int = 14


class TrendPatternResult(msgspec.Struct, frozen=True, kw_only=True):
    """Result from a trend pattern detection.

    Attributes:
        detected: Whether the pattern was detected.
        pattern_type: Type of pattern detected (e.g., "ma_crossover").
        score: Pattern quality score (0-1).
        entry_price: Suggested entry price.
        stop_loss: Suggested stop loss level.
        target_price: Suggested target price.
        reasons: Human-readable reasons for detection.
        meta: Additional metadata for debugging.
    """

    detected: bool
    pattern_type: str
    score: float = 0.0
    entry_price: float | None = None
    stop_loss: float | None = None
    target_price: float | None = None
    reasons: list[str] = msgspec.field(default_factory=list)
    meta: dict[str, Any] = msgspec.field(default_factory=dict)


@runtime_checkable
class TrendPatternDetector(Protocol):
    """Protocol for trend continuation pattern detectors.

    Each detector analyzes price bars and returns a TrendPatternResult
    indicating whether the pattern was detected and with what confidence.

    Gate parameters (sector_bullish, rs_strong) allow the detector to
    short-circuit when market conditions are unfavorable.
    """

    def detect(
        self,
        symbol: str,
        bars: list[PriceBar],
        current_date: date,
        *,
        sector_bullish: bool = False,
        rs_strong: bool = False,
    ) -> TrendPatternResult:
        """Detect trend continuation pattern.

        Args:
            symbol: Stock symbol.
            bars: Historical price bars (oldest first).
            current_date: Current simulation/detection date.
            sector_bullish: Whether the sector is in bullish regime (gate).
            rs_strong: Whether relative strength is positive (gate).

        Returns:
            TrendPatternResult with detection outcome.
        """
        ...

    @property
    def pattern_name(self) -> str:
        """Return the pattern type name."""
        ...
