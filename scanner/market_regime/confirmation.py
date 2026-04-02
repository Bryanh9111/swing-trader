"""Multi-day breakthrough confirmation helper."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import msgspec

from common.interface import Result

try:  # Prefer the real data layer interfaces when available.
    from data.interface import PriceBar
except Exception:  # pragma: no cover
    PriceBar = Any  # type: ignore[assignment]

__all__ = ["BreakthroughConfirmationConfig", "BreakthroughConfirmation"]


class BreakthroughConfirmationConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration controlling multi-day breakthrough confirmation."""

    require_confirmation_days: int = 1
    confirmation_close_above_resistance: bool = True
    require_volume_confirmation: bool = False


@dataclass(frozen=True, slots=True)
class BreakthroughConfirmation:
    """Confirm breakouts by requiring N consecutive closes above resistance."""

    config: BreakthroughConfirmationConfig

    def check_confirmation(
        self,
        bars: Sequence[PriceBar],
        *,
        resistance: float,
        volume_increase_ratio: float = 1.5,
        volume_lookback: int = 10,
    ) -> Result[bool]:
        """Check whether the last N bars confirm the breakthrough.

        Args:
            bars: Full bar history; last N bars are evaluated.
            resistance: Resistance level to confirm against.
            volume_increase_ratio: Required ratio for volume confirmation.
            volume_lookback: Bars used to compute baseline volume (excluding confirmation window).
        """

        n = int(self.config.require_confirmation_days)
        if n <= 0:
            return Result.failed(ValueError("require_confirmation_days must be > 0"), "INVALID_CONFIRMATION_DAYS")
        if not math.isfinite(resistance) or resistance <= 0:
            return Result.failed(ValueError("resistance must be positive and finite"), "INVALID_RESISTANCE")
        if len(bars) < n:
            return Result.failed(ValueError(f"Need >= {n} bars, got {len(bars)}"), "INSUFFICIENT_BARS")

        closes = [float(getattr(bar, "close", float("nan"))) for bar in bars[-n:]]
        if any(not math.isfinite(c) for c in closes):
            return Result.failed(ValueError("non-finite closes in confirmation window"), "NONFINITE_CLOSE")

        if self.config.confirmation_close_above_resistance:
            if not all(c > resistance for c in closes):
                return Result.success(False, reason_code="CONFIRMATION_CLOSE_NOT_ABOVE")

        if self.config.require_volume_confirmation:
            if volume_lookback <= 0:
                return Result.failed(ValueError("volume_lookback must be > 0"), "INVALID_VOLUME_LOOKBACK")
            if volume_increase_ratio <= 0 or not math.isfinite(volume_increase_ratio):
                return Result.failed(ValueError("volume_increase_ratio must be > 0"), "INVALID_VOLUME_INCREASE_RATIO")
            if len(bars) < n + volume_lookback:
                return Result.failed(
                    ValueError("insufficient bars for volume baseline"),
                    "INSUFFICIENT_BARS_FOR_VOLUME",
                )

            baseline_bars = bars[-(n + volume_lookback) : -n]
            baseline_volumes = [float(getattr(bar, "volume", float("nan"))) for bar in baseline_bars]
            confirm_volumes = [float(getattr(bar, "volume", float("nan"))) for bar in bars[-n:]]
            if any(not math.isfinite(v) or v < 0 for v in baseline_volumes + confirm_volumes):
                return Result.failed(ValueError("invalid volume values"), "INVALID_VOLUME")

            baseline_avg = sum(baseline_volumes) / float(len(baseline_volumes))
            if baseline_avg <= 0:
                return Result.failed(ValueError("baseline_avg volume must be > 0"), "ZERO_BASELINE_VOLUME")

            required = baseline_avg * volume_increase_ratio
            if not all(v >= required for v in confirm_volumes):
                return Result.success(False, reason_code="CONFIRMATION_VOLUME_NOT_MET")

        return Result.success(True)

