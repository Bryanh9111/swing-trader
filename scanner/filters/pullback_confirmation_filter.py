"""Pullback confirmation filter (breakout + pullback + hold).

This filter delays entry after a resistance breakout until:
1) Breakthrough: within the recent lookback window, some day's high > resistance
2) Pullback: after the breakthrough, price revisits resistance within a tolerance band
3) Hold: on/after the pullback, a close > resistance confirms support

Resistance is computed as the highest high *before* the lookback window.
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

__all__ = ["PullbackConfirmationFilter"]


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


class PullbackConfirmationFilter(BaseFilter):
    """Filter requiring breakout + pullback + re-claim above resistance."""

    name: ClassVar[str] = "pullback_confirmation"

    def __init__(
        self,
        *,
        lookback_days: int | None = None,
        pullback_tolerance_pct: float | None = None,
        require_volume_increase: bool | None = None,
        volume_increase_ratio: float | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self._lookback_days = lookback_days
        self._pullback_tolerance_pct = pullback_tolerance_pct
        self._require_volume_increase = require_volume_increase
        self._volume_increase_ratio = volume_increase_ratio

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        cfg = getattr(config, "pullback_confirmation", None)

        lookback_days = int(
            self._lookback_days
            if self._lookback_days is not None
            else getattr(cfg, "lookback_days", 5)
        )
        tolerance = float(
            self._pullback_tolerance_pct
            if self._pullback_tolerance_pct is not None
            else getattr(cfg, "pullback_tolerance_pct", 0.02)
        )
        require_volume_increase = bool(
            self._require_volume_increase
            if self._require_volume_increase is not None
            else getattr(cfg, "require_volume_increase", False)
        )
        volume_increase_ratio = float(
            self._volume_increase_ratio
            if self._volume_increase_ratio is not None
            else getattr(cfg, "volume_increase_ratio", 1.5)
        )

        if lookback_days < 2:
            return Result.failed(ValueError("lookback_days must be >= 2"), "INVALID_LOOKBACK_DAYS")
        if len(bars) < lookback_days + 1:
            return Result.failed(
                ValueError(f"Need >= {lookback_days + 1} bars, got {len(bars)}"),
                "INSUFFICIENT_BARS",
            )

        if not math.isfinite(tolerance) or tolerance <= 0:
            return Result.failed(ValueError("pullback_tolerance_pct must be > 0"), "INVALID_TOLERANCE_PCT")
        if not math.isfinite(volume_increase_ratio) or volume_increase_ratio <= 0:
            return Result.failed(
                ValueError("volume_increase_ratio must be > 0"),
                "INVALID_VOLUME_INCREASE_RATIO",
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
        low = lows_result.data
        close = closes_result.data
        volume = volumes_result.data

        if np.any(high <= 0) or np.any(low <= 0) or np.any(close <= 0):
            return Result.failed(ValueError("prices must be positive"), "NONPOSITIVE_PRICE")
        if np.any(volume < 0):
            return Result.failed(ValueError("volume cannot be negative"), "NEGATIVE_VOLUME")

        window_start = len(bars) - lookback_days
        if window_start <= 0:
            return Result.failed(ValueError("insufficient pre-window bars for resistance"), "INSUFFICIENT_RESISTANCE_BARS")

        resistance = float(np.max(high[:window_start]))
        if not math.isfinite(resistance) or resistance <= 0:
            return Result.failed(ValueError("resistance invalid"), "INVALID_RESISTANCE")

        lower = float(resistance * (1.0 - tolerance))
        upper = float(resistance * (1.0 + tolerance))

        # 1) Breakthrough: any high within lookback window exceeds resistance.
        breakthrough_idx: int | None = None
        for idx in range(window_start, len(bars)):
            if float(high[idx]) > resistance:
                breakthrough_idx = idx
                break

        if breakthrough_idx is None:
            return Result.success(
                FilterResult(
                    passed=False,
                    reason="NO_BREAKTHROUGH",
                    score=0.0,
                    features={"resistance": resistance},
                    metadata={"lookback_days": lookback_days, "band": {"lower": lower, "upper": upper}},
                ),
                reason_code="NO_BREAKTHROUGH",
            )

        # 2) Pullback: after breakthrough, price revisits resistance band.
        pullback_idx: int | None = None
        for idx in range(breakthrough_idx + 1, len(bars)):
            if float(low[idx]) <= upper and float(high[idx]) >= lower:
                pullback_idx = idx
                break

        if pullback_idx is None:
            return Result.success(
                FilterResult(
                    passed=False,
                    reason="NO_PULLBACK",
                    score=0.0,
                    features={"resistance": resistance, "breakthrough_idx": breakthrough_idx},
                    metadata={"lookback_days": lookback_days, "band": {"lower": lower, "upper": upper}},
                ),
                reason_code="NO_PULLBACK",
            )

        # 3) Hold: on/after pullback, a close above resistance confirms support.
        hold_idx: int | None = None
        for idx in range(pullback_idx, len(bars)):
            if float(close[idx]) > resistance:
                hold_idx = idx
                break

        if hold_idx is None:
            return Result.success(
                FilterResult(
                    passed=False,
                    reason="NOT_HOLDING_ABOVE",
                    score=0.0,
                    features={
                        "resistance": resistance,
                        "breakthrough_idx": breakthrough_idx,
                        "pullback_idx": pullback_idx,
                    },
                    metadata={"lookback_days": lookback_days, "band": {"lower": lower, "upper": upper}},
                ),
                reason_code="NOT_HOLDING_ABOVE",
            )

        volume_ok = True
        baseline_avg: float | None = None
        hold_volume: float | None = None
        if require_volume_increase:
            baseline_start = max(0, breakthrough_idx - 10)
            baseline = volume[baseline_start:breakthrough_idx]
            if baseline.size < 5:
                return Result.failed(
                    ValueError("Need >=5 baseline bars before breakthrough for volume check"),
                    "INSUFFICIENT_VOLUME_BASELINE",
                )
            baseline_avg = float(np.mean(baseline))
            hold_volume = float(volume[hold_idx])
            if not math.isfinite(baseline_avg) or baseline_avg <= 0:
                return Result.failed(ValueError("baseline volume average invalid"), "INVALID_BASELINE_VOLUME_AVG")
            if not math.isfinite(hold_volume) or hold_volume < 0:
                return Result.failed(ValueError("hold volume invalid"), "INVALID_HOLD_VOLUME")
            volume_ok = bool(hold_volume >= baseline_avg * volume_increase_ratio)

        passed = bool(volume_ok)
        reason_code = "OK" if passed else "VOLUME_NOT_INCREASING"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=1.0 if passed else 0.0,
                features={
                    "resistance": resistance,
                    "breakthrough_idx": breakthrough_idx,
                    "pullback_idx": pullback_idx,
                    "hold_idx": hold_idx,
                    "band_lower": lower,
                    "band_upper": upper,
                    "require_volume_increase": require_volume_increase,
                    "volume_ok": volume_ok,
                },
                metadata={
                    "lookback_days": lookback_days,
                    "thresholds": {
                        "pullback_tolerance_pct": tolerance,
                        "volume_increase_ratio": volume_increase_ratio,
                    },
                    "volume": {
                        "baseline_avg": baseline_avg,
                        "hold_volume": hold_volume,
                    },
                },
            ),
            reason_code=reason_code,
        )

