"""Volume platform filter wrapping :func:`scanner.detector.detect_volume_platform`."""

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


class VolumePlatformFilter(BaseFilter):
    name: ClassVar[str] = "volume_platform"

    def __init__(self, *, window: int, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._window = int(window)

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        detection = detector.detect_volume_platform(list(bars), window=self._window, config=config)
        if detection.status is ResultStatus.FAILED:
            return Result.failed(
                detection.error or ValueError("volume platform detector failed"),
                detection.reason_code or "VOLUME_PLATFORM_FAILED",
            )

        volume_ok, raw_features = detection.data
        volume_change_ratio = float(raw_features.get("volume_change_ratio", float("nan")))
        volume_stability = float(raw_features.get("volume_stability", float("nan")))
        volume_stability_robust = float(raw_features.get("volume_stability_robust", float("nan")))
        volume_trend = float(raw_features.get("volume_trend", float("nan")))
        trend_score = float(raw_features.get("trend_score", float("nan")))
        volume_quality = float(raw_features.get("volume_quality", float("nan")))
        avg_dollar_volume = float(raw_features.get("avg_dollar_volume", float("nan")))

        passed = bool(volume_ok)
        reason_code = detection.reason_code if not passed else "OK"
        if not reason_code:
            reason_code = "VOLUME_RULES_NOT_MET" if not passed else "OK"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=1.0 if passed else 0.0,
                features={
                    "volume_change_ratio": volume_change_ratio,
                    "volume_stability": volume_stability,
                    "volume_stability_robust": volume_stability_robust,
                    "volume_trend": volume_trend,
                    "trend_score": trend_score,
                    "volume_quality": volume_quality,
                    "avg_dollar_volume": avg_dollar_volume,
                },
                metadata={
                    "window": self._window,
                    "thresholds": {
                        "volume_change_threshold": config.volume_change_threshold,
                        "volume_stability_threshold": config.volume_stability_threshold,
                    },
                },
            ),
            reason_code=reason_code,
        )
