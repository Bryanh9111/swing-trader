"""ADX entry filter (trend confirmation).

Bear/choppy markets can generate many false breakouts. This filter blocks entry
when the trend strength (ADX) is below a configurable threshold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, TYPE_CHECKING

from common.interface import Result
from indicators.interface import compute_adx_last
from scanner.interface import ScannerConfig

from .base import BaseFilter, FilterResult

if TYPE_CHECKING:
    from data.interface import PriceBar
else:  # pragma: no cover - type-only import may not exist in early phases.
    PriceBar = Any  # type: ignore[assignment]

__all__ = ["ADXEntryFilter"]


class ADXEntryFilter(BaseFilter):
    """Reject entry signals when ADX is below the configured threshold."""

    name: ClassVar[str] = "adx_entry_filter"

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        adx_cfg = getattr(config, "adx_entry_filter", None)
        if adx_cfg is None:
            return Result.success(
                FilterResult(passed=True, reason="ADX_CONFIG_MISSING", score=1.0, metadata={"enabled": False}),
                reason_code="ADX_CONFIG_MISSING",
            )

        min_adx = float(getattr(adx_cfg, "min_adx", 20.0))
        period = int(getattr(adx_cfg, "period", 14))

        if period <= 0:
            return Result.failed(ValueError("adx_entry_filter.period must be > 0"), "INVALID_ADX_PERIOD")

        required = period + 2
        if len(bars) < required:
            return Result.success(
                FilterResult(
                    passed=False,
                    reason="INSUFFICIENT_BARS_FOR_ADX",
                    score=0.0,
                    metadata={"required": required, "actual": len(bars), "period": period},
                ),
                reason_code="INSUFFICIENT_BARS_FOR_ADX",
            )

        highs = [float(getattr(bar, "high", float("nan"))) for bar in bars]
        lows = [float(getattr(bar, "low", float("nan"))) for bar in bars]
        closes = [float(getattr(bar, "close", float("nan"))) for bar in bars]
        if any(not math.isfinite(v) for v in highs + lows + closes):
            return Result.success(
                FilterResult(
                    passed=False,
                    reason="NONFINITE_OHLC",
                    score=0.0,
                    metadata={"period": period},
                ),
                reason_code="NONFINITE_OHLC",
            )

        adx = compute_adx_last(highs, lows, closes, period=period)
        if adx is None or not math.isfinite(adx):
            return Result.success(
                FilterResult(
                    passed=False,
                    reason="ADX_UNAVAILABLE",
                    score=0.0,
                    metadata={"period": period},
                ),
                reason_code="ADX_UNAVAILABLE",
            )

        passed = bool(adx >= min_adx)
        reason_code = "OK" if passed else "ADX_BELOW_THRESHOLD"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=1.0 if passed else 0.0,
                features={"adx": float(adx)},
                metadata={"thresholds": {"min_adx": min_adx}, "period": period},
            ),
            reason_code=reason_code,
        )

