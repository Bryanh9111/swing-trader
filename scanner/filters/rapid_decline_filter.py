"""Rapid decline filter (rapid drop detection).

This filter measures the maximum peak-to-trough decline that could have started
on any day within the most recent ``rapid_decline_days`` bars.

Phase 1 algorithm (adapted from the reference project's decline analyzer):
For the last N bars:
    For each start index i:
        high_day = high[i]
        min_low_after = min(low[i:])
        max_drop_from_day = (high_day - min_low_after) / high_day
    max_rapid_decline = max(max_drop_from_day over i)

Pass condition:
    max_rapid_decline >= config.rapid_decline_threshold

Edge cases:
    - Insufficient bars -> FAILED with a clear reason_code.
    - Non-finite or non-positive highs/lows -> FAILED.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

from common.interface import Result, ResultStatus
from scanner.interface import ScannerConfig

from .base import BaseFilter, FilterResult

if TYPE_CHECKING:
    from data.interface import PriceBar
else:  # pragma: no cover - type-only import may not exist in early phases.
    PriceBar = Any  # type: ignore[assignment]

__all__ = ["RapidDeclineFilter"]


def _as_finite_float_array(values: Sequence[Any], *, name: str) -> Result[np.ndarray]:
    try:
        array = np.asarray(list(values), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return Result.failed(exc, f"INVALID_{name.upper()}_ARRAY")

    if array.size == 0:
        return Result.failed(ValueError(f"{name} is empty"), f"EMPTY_{name.upper()}")
    if not np.all(np.isfinite(array)):
        return Result.failed(ValueError(f"{name} contains non-finite values"), f"NONFINITE_{name.upper()}")

    return Result.success(array)


class RapidDeclineFilter(BaseFilter):
    """Filter for detecting a rapid decline within a recent rolling window."""

    name: ClassVar[str] = "rapid_decline"

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        window = int(config.rapid_decline_days)
        if window <= 1:
            return Result.failed(ValueError("rapid_decline_days must be > 1"), "INVALID_RAPID_DECLINE_DAYS")
        if len(bars) < window:
            return Result.failed(
                ValueError(f"Need >= {window} bars, got {len(bars)}"),
                "INSUFFICIENT_BARS",
            )

        highs_result = _as_finite_float_array([getattr(bar, "high", float("nan")) for bar in bars], name="high")
        if highs_result.status is ResultStatus.FAILED:
            return Result.failed(
                highs_result.error or ValueError("high extraction failed"),
                highs_result.reason_code or "HIGH_EXTRACTION_FAILED",
            )

        lows_result = _as_finite_float_array([getattr(bar, "low", float("nan")) for bar in bars], name="low")
        if lows_result.status is ResultStatus.FAILED:
            return Result.failed(
                lows_result.error or ValueError("low extraction failed"),
                lows_result.reason_code or "LOW_EXTRACTION_FAILED",
            )

        high_all = highs_result.data
        low_all = lows_result.data
        if np.any(high_all <= 0) or np.any(low_all <= 0):
            return Result.failed(ValueError("high/low must be positive"), "NONPOSITIVE_PRICE")
        if np.any(low_all > high_all):
            return Result.failed(ValueError("low cannot exceed high"), "INVALID_OHLC_RANGE")

        high = high_all[-window:]
        low = low_all[-window:]
        offset = int(high_all.size - window)

        if np.any(high == 0):
            return Result.failed(ValueError("high contains zeros"), "ZERO_HIGH_VALUE")

        suffix_min_low = np.minimum.accumulate(low[::-1])[::-1]
        drops = (high - suffix_min_low) / high
        if not np.all(np.isfinite(drops)):
            return Result.failed(ValueError("drop series contains non-finite values"), "NONFINITE_DROP_SERIES")

        max_idx = int(np.argmax(drops))
        max_rapid_decline = float(drops[max_idx])
        if not math.isfinite(max_rapid_decline) or max_rapid_decline < 0:
            return Result.failed(ValueError("max_rapid_decline invalid"), "INVALID_MAX_RAPID_DECLINE")

        decline_start_idx = int(offset + max_idx)
        passed = bool(max_rapid_decline >= config.rapid_decline_threshold)
        reason_code = "OK" if passed else "RAPID_DECLINE_FILTERED"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=1.0 if passed else 0.0,
                features={
                    "max_rapid_decline": max_rapid_decline,
                    "rapid_decline_days": window,
                    "decline_start_idx": decline_start_idx,
                },
                metadata={
                    "thresholds": {
                        "rapid_decline_threshold": config.rapid_decline_threshold,
                        "rapid_decline_days": config.rapid_decline_days,
                    }
                },
            ),
            reason_code=reason_code,
        )

