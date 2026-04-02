"""Low position filter (low-price analysis).

This filter identifies symbols trading sufficiently below their historical high
and ensures the high point occurred far enough in the past.

Phase 1 algorithm (adapted from the reference project's position analyzer):
1) Compute the historical high over *all* available bars (max close).
2) Compute decline percentage from the historical high to the latest close:
      decline_pct = (historical_high - current_close) / historical_high
3) Compute how many bars have elapsed since that historical high:
      periods_from_high = (len(bars) - 1) - idx_high

Pass conditions:
    - decline_pct >= config.decline_threshold
    - periods_from_high >= config.min_periods

Edge cases:
    - Insufficient bars -> FAILED with a clear reason_code.
    - Non-finite or non-positive prices -> FAILED.
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

__all__ = ["LowPositionFilter"]


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


class LowPositionFilter(BaseFilter):
    """Filter for detecting "low position" setups after a meaningful decline."""

    name: ClassVar[str] = "low_position"

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        min_required = max(2, int(config.min_periods) + 1)
        if len(bars) < min_required:
            return Result.failed(
                ValueError(f"Need >= {min_required} bars, got {len(bars)}"),
                "INSUFFICIENT_BARS",
            )

        closes_result = _as_finite_float_array([getattr(bar, "close", float("nan")) for bar in bars], name="close")
        if closes_result.status is ResultStatus.FAILED:
            return Result.failed(
                closes_result.error or ValueError("close extraction failed"),
                closes_result.reason_code or "CLOSE_EXTRACTION_FAILED",
            )

        close = closes_result.data
        if np.any(close <= 0):
            return Result.failed(ValueError("close must be positive"), "NONPOSITIVE_CLOSE")

        current_close = float(close[-1])
        historical_high = float(np.max(close))
        if not math.isfinite(historical_high) or historical_high <= 0:
            return Result.failed(ValueError("historical_high invalid"), "INVALID_HISTORICAL_HIGH")

        idx_high = int(np.argmax(close))
        periods_from_high = int((close.size - 1) - idx_high)
        decline_pct = float((historical_high - current_close) / historical_high)
        if not math.isfinite(decline_pct) or decline_pct < 0:
            return Result.failed(ValueError("decline_pct invalid"), "INVALID_DECLINE_PCT")

        passed = bool(decline_pct >= config.decline_threshold and periods_from_high >= config.min_periods)
        reason_code = "OK" if passed else "LOW_POSITION_FILTERED"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=1.0 if passed else 0.0,
                features={
                    "historical_high": historical_high,
                    "current_close": current_close,
                    "decline_pct": decline_pct,
                    "periods_from_high": periods_from_high,
                },
                metadata={
                    "high_index": idx_high,
                    "thresholds": {
                        "decline_threshold": config.decline_threshold,
                        "min_periods": config.min_periods,
                    },
                },
            ),
            reason_code=reason_code,
        )

