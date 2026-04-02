"""Box quality filter wrapping :func:`scanner.detector.detect_support_resistance`."""

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


class BoxQualityFilter(BaseFilter):
    name: ClassVar[str] = "box_quality"

    def __init__(
        self,
        *,
        window: int,
        tolerance: float = detector.DEFAULT_TOLERANCE,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self._window = int(window)
        self._tolerance = float(tolerance)

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        window_bars = detector._window_slice(bars, window=self._window)
        if window_bars.status is ResultStatus.FAILED:
            return Result.failed(
                window_bars.error or ValueError("window slice failed"),
                window_bars.reason_code or "WINDOW_SLICE_FAILED",
            )

        box = detector.detect_support_resistance(list(window_bars.data), tolerance=self._tolerance)
        if box.status is ResultStatus.FAILED:
            return Result.failed(
                box.error or ValueError("support/resistance detector failed"),
                box.reason_code or "BOX_DETECTION_FAILED",
            )

        data = box.data
        support = float(data.get("support_level") or 0.0)
        resistance = float(data.get("resistance_level") or 0.0)
        box_quality = float(data.get("box_quality") or 0.0)
        touch_score = float(data.get("touch_score") or 0.0)
        support_score = float(data.get("support_score") or 0.0)
        resistance_score = float(data.get("resistance_score") or 0.0)
        containment = float(data.get("containment") or 0.0)
        box_tightness = float(data.get("box_tightness") or 0.0)
        box_range_pct = float(data.get("box_range_pct") or 0.0)
        distribution_score = float(data.get("distribution_score") or 0.0)

        passed = box_quality >= config.min_box_quality
        reason_code = "OK" if passed else "BOX_QUALITY_FILTERED"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=float(max(0.0, min(1.0, box_quality))),
                features={
                    "support": support,
                    "resistance": resistance,
                    "box_quality": box_quality,
                    "touch_score": touch_score,
                    "support_score": support_score,
                    "resistance_score": resistance_score,
                    "containment": containment,
                    "box_tightness": box_tightness,
                    "box_range_pct": box_range_pct,
                    "distribution_score": distribution_score,
                },
                metadata={
                    "window": self._window,
                    "tolerance": self._tolerance,
                    "thresholds": {"min_box_quality": config.min_box_quality},
                },
            ),
            reason_code=reason_code,
        )
