"""Platform-days filter.

Computes the number of consecutive bars (ending at the latest bar) whose close
remains within the detected platform range, and enforces ``min_platform_days``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, TYPE_CHECKING

from common.interface import Result, ResultStatus
from scanner import detector
from scanner.interface import ScannerConfig

from .base import BaseFilter, FilterResult

if TYPE_CHECKING:
    from data.interface import PriceBar
else:  # pragma: no cover - type-only import may not exist in early phases.
    PriceBar = Any  # type: ignore[assignment]


def calculate_platform_days(bars: Sequence[PriceBar], *, box_low: float, box_high: float) -> int:
    """Calculate consecutive days within platform range (latest bar backwards)."""

    platform_days = 0
    for bar in reversed(bars):
        close = float(getattr(bar, "close"))
        if box_low <= close <= box_high:
            platform_days += 1
        else:
            break
    return platform_days


class PlatformDaysFilter(BaseFilter):
    name: ClassVar[str] = "platform_days"

    def __init__(self, *, window: int, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._window = int(window)

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        window_bars = detector._window_slice(bars, window=self._window)
        if window_bars.status is ResultStatus.FAILED:
            return Result.failed(
                window_bars.error or ValueError("window slice failed"),
                window_bars.reason_code or "WINDOW_SLICE_FAILED",
            )

        low_values = [float(getattr(bar, "low")) for bar in window_bars.data]
        high_values = [float(getattr(bar, "high")) for bar in window_bars.data]
        if not low_values or not high_values:
            return Result.failed(ValueError("empty window bars"), "WINDOW_EMPTY")

        box_low = min(low_values)
        box_high = max(high_values)
        platform_days = calculate_platform_days(window_bars.data, box_low=box_low, box_high=box_high)

        min_platform_days = int(getattr(config, "min_platform_days", 10))
        passed = platform_days >= min_platform_days
        reason_code = "OK" if passed else "PLATFORM_DAYS_FILTERED"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=1.0 if passed else 0.0,
                features={
                    "platform_days": int(platform_days),
                    "box_low": float(box_low),
                    "box_high": float(box_high),
                },
                metadata={
                    "window": self._window,
                    "thresholds": {"min_platform_days": min_platform_days},
                },
            ),
            reason_code=reason_code,
        )

