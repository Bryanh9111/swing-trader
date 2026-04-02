"""Breakthrough potential filter (breakout precursor, Phase 1 simplified).

The reference project uses multiple technical indicators (MACD/RSI/KDJ/BBands)
to confirm breakthrough signals. For Phase 1 in AST we implement a simplified,
high-signal filter that remains independent and cheap to compute:

1) Near resistance:
    - Compute a simple resistance proxy as the highest high over the most
      recent lookback window (default: min(20, len(bars))).
    - Mark ``near_resistance`` when the latest close is within
      ``config.resistance_proximity_pct`` of this resistance level.

2) Volume increasing:
    - Compare recent average volume (last 5 bars) with prior average volume
      (previous 10 bars).
    - Mark ``volume_increasing`` when:
          recent_avg >= prior_avg * config.volume_increase_ratio

3) Breakthrough score:
    - A 0~1 heuristic score that rewards proximity to resistance and volume
      acceleration. This is a Phase 1 aid for ranking/debugging; the pass/fail
      rule remains strictly boolean.

Pass condition:
    - When ``config.require_breakout_confirmation`` is False (legacy behavior):
        near_resistance AND volume_increasing

    - When ``config.require_breakout_confirmation`` is True:
        actual_breakout AND volume_increasing

      Where ``actual_breakout`` requires the most recent
      ``config.breakout_confirmation_days`` closes to be above a resistance
      level computed from history excluding those confirmation days:

          resistance = max(high[:-confirmation_days])

      Optionally (when ``config.require_breakout_volume_spike`` is True), the
      breakout day volume must also exceed ``config.breakout_volume_ratio``
      times the 20-day average volume before the breakout day.

Edge cases:
    - Insufficient bars -> FAILED with a clear reason_code.
    - Non-finite or non-positive price/volume -> FAILED.
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

__all__ = ["BreakthroughFilter"]


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


class BreakthroughFilter(BaseFilter):
    """Filter for detecting "breakthrough potential" via resistance + volume."""

    name: ClassVar[str] = "breakthrough_potential"

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        if len(bars) < 15:
            return Result.failed(
                ValueError(f"Need >= 15 bars for volume trend check, got {len(bars)}"),
                "INSUFFICIENT_BARS",
            )

        highs_result = _as_finite_float_array([getattr(bar, "high", float("nan")) for bar in bars], name="high")
        if highs_result.status is ResultStatus.FAILED:
            return Result.failed(
                highs_result.error or ValueError("high extraction failed"),
                highs_result.reason_code or "HIGH_EXTRACTION_FAILED",
            )

        closes_result = _as_finite_float_array([getattr(bar, "close", float("nan")) for bar in bars], name="close")
        if closes_result.status is ResultStatus.FAILED:
            return Result.failed(
                closes_result.error or ValueError("close extraction failed"),
                closes_result.reason_code or "CLOSE_EXTRACTION_FAILED",
            )

        volumes_result = _as_finite_float_array(
            [getattr(bar, "volume", float("nan")) for bar in bars], name="volume"
        )
        if volumes_result.status is ResultStatus.FAILED:
            return Result.failed(
                volumes_result.error or ValueError("volume extraction failed"),
                volumes_result.reason_code or "VOLUME_EXTRACTION_FAILED",
            )

        high = highs_result.data
        close = closes_result.data
        volume = volumes_result.data

        if np.any(high <= 0) or np.any(close <= 0):
            return Result.failed(ValueError("prices must be positive"), "NONPOSITIVE_PRICE")
        if np.any(volume < 0):
            return Result.failed(ValueError("volume cannot be negative"), "NEGATIVE_VOLUME")

        proximity_pct = float(config.resistance_proximity_pct)
        if not math.isfinite(proximity_pct) or proximity_pct <= 0:
            return Result.failed(ValueError("resistance_proximity_pct must be > 0"), "INVALID_PROXIMITY_PCT")

        lookback = int(min(20, len(bars)))
        resistance = float(np.max(high[-lookback:]))
        current_close = float(close[-1])
        if not math.isfinite(resistance) or resistance <= 0:
            return Result.failed(ValueError("resistance invalid"), "INVALID_RESISTANCE")

        price_distance_pct = float(abs(resistance - current_close) / resistance)
        near_resistance = bool(price_distance_pct <= proximity_pct)

        recent_avg = float(np.mean(volume[-5:]))
        prior_avg = float(np.mean(volume[-15:-5]))
        if not math.isfinite(recent_avg) or not math.isfinite(prior_avg):
            return Result.failed(ValueError("volume averages not finite"), "INVALID_VOLUME_AVG")
        if prior_avg <= 0:
            return Result.failed(ValueError("prior_avg volume must be > 0"), "ZERO_PRIOR_VOLUME")

        required_ratio = float(config.volume_increase_ratio)
        if not math.isfinite(required_ratio) or required_ratio <= 0:
            return Result.failed(ValueError("volume_increase_ratio must be > 0"), "INVALID_VOLUME_INCREASE_RATIO")

        volume_ratio = float(recent_avg / prior_avg)
        volume_increasing = bool(volume_ratio >= required_ratio)

        require_breakout_confirmation = bool(config.require_breakout_confirmation)
        confirmation_days = int(config.breakout_confirmation_days)
        if require_breakout_confirmation and confirmation_days <= 0:
            return Result.failed(
                ValueError("breakout_confirmation_days must be >= 1"),
                "INVALID_BREAKOUT_CONFIRMATION_DAYS",
            )

        resistance_excluding_confirmation = float("nan")
        actual_breakout = False
        breakout_day_index: int | None = None
        breakout_run_length = 0

        if confirmation_days > 0 and len(high) > confirmation_days:
            resistance_excluding_confirmation = float(np.max(high[:-confirmation_days]))
            if math.isfinite(resistance_excluding_confirmation) and resistance_excluding_confirmation > 0:
                above_resistance = close > resistance_excluding_confirmation
                actual_breakout = bool(np.all(above_resistance[-confirmation_days:]))

                if actual_breakout and bool(above_resistance[-1]):
                    false_indices = np.flatnonzero(~above_resistance)
                    if false_indices.size == 0:
                        start_idx = 0
                    else:
                        start_idx = int(false_indices[-1] + 1)
                    if 0 <= start_idx < len(bars) and bool(above_resistance[start_idx]):
                        breakout_day_index = start_idx
                        breakout_run_length = int(len(bars) - start_idx)
        else:
            resistance_excluding_confirmation = float("nan")
            actual_breakout = False

        require_breakout_volume_spike = bool(config.require_breakout_volume_spike)
        breakout_volume_confirmed = False
        breakout_volume_ratio = float("nan")
        breakout_day_volume = float("nan")
        avg_volume_before_breakout = float("nan")
        breakout_volume_days = 20

        volume_ratio_threshold = float(config.breakout_volume_ratio)
        if require_breakout_volume_spike and require_breakout_confirmation:
            if not math.isfinite(volume_ratio_threshold) or volume_ratio_threshold <= 0:
                return Result.failed(
                    ValueError("breakout_volume_ratio must be > 0"),
                    "INVALID_BREAKOUT_VOLUME_RATIO",
                )

        if require_breakout_volume_spike and require_breakout_confirmation and breakout_day_index is not None:
            breakout_day_volume = float(volume[breakout_day_index])
            start_idx = int(breakout_day_index - breakout_volume_days)
            if start_idx >= 0:
                avg_volume_before_breakout = float(np.mean(volume[start_idx:breakout_day_index]))
                if math.isfinite(avg_volume_before_breakout) and avg_volume_before_breakout > 0:
                    breakout_volume_ratio = float(breakout_day_volume / avg_volume_before_breakout)
                    breakout_volume_confirmed = bool(breakout_volume_ratio >= volume_ratio_threshold)

        price_component = float(max(0.0, 1.0 - (price_distance_pct / proximity_pct)))
        volume_component = float(max(0.0, min(1.0, volume_ratio / required_ratio)))
        breakthrough_score = float(max(0.0, min(1.0, 0.6 * price_component + 0.4 * volume_component)))

        if require_breakout_confirmation:
            passed = bool(actual_breakout and volume_increasing)
            if require_breakout_volume_spike:
                passed = bool(passed and breakout_volume_confirmed)
        else:
            passed = bool(near_resistance and volume_increasing)
        reason_code = "OK" if passed else "BREAKTHROUGH_FILTERED"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=breakthrough_score,
                features={
                    "near_resistance": near_resistance,
                    "volume_increasing": volume_increasing,
                    "actual_breakout": actual_breakout,
                    "breakout_volume_confirmed": breakout_volume_confirmed,
                    "breakthrough_score": breakthrough_score,
                },
                metadata={
                    "resistance": resistance,
                    "resistance_excluding_confirmation_days": resistance_excluding_confirmation,
                    "current_close": current_close,
                    "price_distance_pct": price_distance_pct,
                    "resistance_lookback": lookback,
                    "recent_volume_avg": recent_avg,
                    "prior_volume_avg": prior_avg,
                    "volume_ratio": volume_ratio,
                    "breakout": {
                        "require_breakout_confirmation": require_breakout_confirmation,
                        "breakout_confirmation_days": confirmation_days,
                        "confirmation_closes": [float(val) for val in close[-confirmation_days:]]
                        if (require_breakout_confirmation and confirmation_days > 0)
                        else [],
                        "confirmation_passed": actual_breakout,
                        "breakout_day_index": breakout_day_index,
                        "breakout_run_length": breakout_run_length,
                        "require_breakout_volume_spike": require_breakout_volume_spike,
                        "breakout_day_volume": breakout_day_volume,
                        "avg_volume_before_breakout": avg_volume_before_breakout,
                        "breakout_volume_days": breakout_volume_days,
                        "breakout_volume_ratio": breakout_volume_ratio,
                        "breakout_volume_ratio_threshold": volume_ratio_threshold,
                        "breakout_volume_confirmed": breakout_volume_confirmed,
                    },
                    "thresholds": {
                        "resistance_proximity_pct": config.resistance_proximity_pct,
                        "volume_increase_ratio": config.volume_increase_ratio,
                        "require_breakout_confirmation": config.require_breakout_confirmation,
                        "breakout_confirmation_days": config.breakout_confirmation_days,
                        "require_breakout_volume_spike": config.require_breakout_volume_spike,
                        "breakout_volume_ratio": config.breakout_volume_ratio,
                    },
                },
            ),
            reason_code=reason_code,
        )
