"""MA200 trend filter (long-term trend filtering).

Pass when the latest close is above the long-term simple moving average (SMA),
optionally allowing a small tolerance band around the SMA.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, TYPE_CHECKING

from common.interface import Result
from indicators.interface import compute_sma_last
from scanner.interface import ScannerConfig

from .base import BaseFilter, FilterResult

if TYPE_CHECKING:
    from data.interface import PriceBar
else:  # pragma: no cover - type-only import may not exist in early phases.
    PriceBar = Any  # type: ignore[assignment]

__all__ = ["MA200TrendFilter"]


class MA200TrendFilter(BaseFilter):
    """Filter candidates by requiring price to be above the SMA trend line."""

    name: ClassVar[str] = "ma200_trend_filter"

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        ma_cfg = getattr(config, "ma200_trend_filter", None)
        if ma_cfg is None:
            return Result.success(
                FilterResult(passed=True, reason="OK", score=1.0, metadata={"enabled": False}),
                reason_code="OK",
            )

        period = int(getattr(ma_cfg, "period", 200))
        fallback_periods = getattr(ma_cfg, "fallback_periods", None)
        require_above = bool(getattr(ma_cfg, "require_above", True))
        tolerance_pct = float(getattr(ma_cfg, "tolerance_pct", 0.02))

        if period <= 0:
            return Result.failed(ValueError("ma200_trend_filter.period must be > 0"), "INVALID_MA_PERIOD")
        if not math.isfinite(tolerance_pct) or tolerance_pct < 0:
            return Result.failed(
                ValueError("ma200_trend_filter.tolerance_pct must be finite and >= 0"),
                "INVALID_TOLERANCE_PCT",
            )

        close = float(getattr(bars[-1], "close", float("nan")))
        if not math.isfinite(close) or close <= 0:
            return Result.failed(ValueError("last close invalid"), "INVALID_LAST_CLOSE")

        if fallback_periods is None:
            configured_fallbacks: list[int] = [period, period // 2]
        else:
            try:
                configured_fallbacks = [int(p) for p in fallback_periods]
            except Exception as exc:  # noqa: BLE001 - tolerate non-typed configs.
                error = ValueError("ma200_trend_filter.fallback_periods must be a list[int]")
                error.__cause__ = exc
                return Result.failed(error, "INVALID_MA_PERIOD")

        if not configured_fallbacks:
            configured_fallbacks = [period]

        periods: list[int] = []
        for value in configured_fallbacks:
            if value <= 0:
                return Result.failed(ValueError("ma200_trend_filter.fallback_periods must contain positive ints"), "INVALID_MA_PERIOD")
            if value not in periods:
                periods.append(value)

        invalid_periods: list[int] = []
        for ma_period in periods:
            if len(bars) < ma_period:
                continue

            sma = compute_sma_last(bars, period=ma_period)
            if sma is None:
                continue

            ma_value = float(sma)
            if not math.isfinite(ma_value) or ma_value <= 0:
                invalid_periods.append(ma_period)
                continue

            diff_pct = float((close - ma_value) / ma_value)

            if require_above:
                threshold = float(ma_value * (1.0 - tolerance_pct))
                passed = bool(close >= threshold)
            else:
                threshold = float(ma_value * (1.0 + tolerance_pct))
                passed = bool(close <= threshold)

            reason_code = "OK" if passed else "MA_BELOW_TREND"
            return Result.success(
                FilterResult(
                    passed=passed,
                    reason=reason_code,
                    score=1.0 if passed else 0.0,
                    features={"ma": ma_value, "close": close, "diff_pct": diff_pct},
                    metadata={
                        "thresholds": {
                            "require_above": require_above,
                            "tolerance_pct": tolerance_pct,
                            "close_threshold": threshold,
                        },
                        "period": ma_period,
                        "fallback_periods": list(periods),
                        "configured_period": period,
                    },
                ),
                reason_code=reason_code,
            )

        return Result.success(
            FilterResult(
                passed=False,
                reason="INSUFFICIENT_BARS",
                score=0.0,
                features={"ma": float("nan"), "close": close, "diff_pct": float("nan")},
                metadata={
                    "required_periods": list(periods),
                    "actual": len(bars),
                    "fallback_periods": list(periods),
                    "configured_period": period,
                    "invalid_periods": list(invalid_periods),
                },
            ),
            reason_code="INSUFFICIENT_BARS",
        )
