"""Price platform filter wrapping :func:`scanner.detector.detect_price_platform`."""

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


class PricePlatformFilter(BaseFilter):
    name: ClassVar[str] = "price_platform"

    def __init__(self, *, window: int, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._window = int(window)

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        detection = detector.detect_price_platform(list(bars), window=self._window, config=config)
        if detection.status is ResultStatus.FAILED:
            return Result.failed(
                detection.error or ValueError("price platform detector failed"),
                detection.reason_code or "PRICE_PLATFORM_FAILED",
            )

        is_platform, raw_features = detection.data
        box_range = float(raw_features.get("box_range", float("nan")))
        box_low = float(raw_features.get("box_low", float("nan")))
        box_high = float(raw_features.get("box_high", float("nan")))
        ma_diff = float(raw_features.get("ma_diff", float("nan")))
        volatility = float(raw_features.get("volatility", float("nan")))
        ma_values = raw_features.get("ma_values", {})

        passed = (
            box_range <= config.box_threshold
            and ma_diff <= config.ma_diff_threshold
            and volatility <= config.volatility_threshold
        )

        reason_code = detection.reason_code if not passed else "OK"
        if not reason_code:
            reason_code = "PRICE_RULES_NOT_MET" if not passed else "OK"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=1.0 if passed else 0.0,
                features={
                    "box_range": box_range,
                    "box_low": box_low,
                    "box_high": box_high,
                    "ma_diff": ma_diff,
                    "volatility": volatility,
                    "ma_values": ma_values,
                },
                metadata={
                    "window": self._window,
                    "thresholds": {
                        "box_threshold": config.box_threshold,
                        "ma_diff_threshold": config.ma_diff_threshold,
                        "volatility_threshold": config.volatility_threshold,
                    },
                },
            ),
            reason_code=reason_code,
        )
