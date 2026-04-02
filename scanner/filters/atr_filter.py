"""ATR range filter wrapping :func:`scanner.detector.calculate_atr`."""

from __future__ import annotations

import math
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


class ATRFilter(BaseFilter):
    name: ClassVar[str] = "atr_range"

    def __init__(self, *, period: int = detector.ATR_DEFAULT_PERIOD, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._period = int(period)

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        if not bars:
            return Result.failed(ValueError("bars is empty"), "INSUFFICIENT_ATR_BARS")

        close_price = float(getattr(bars[-1], "close", float("nan")))
        if not math.isfinite(close_price) or close_price <= 0:
            return Result.failed(ValueError("last close invalid"), "INVALID_LAST_CLOSE")

        atr = detector.calculate_atr(list(bars), period=self._period)
        if atr.status is ResultStatus.FAILED:
            return Result.failed(
                atr.error or ValueError("ATR calculation failed"),
                atr.reason_code or "ATR_FAILED",
            )

        atr_value = float(atr.data)
        atr_pct = float(atr_value / close_price)
        if not math.isfinite(atr_pct) or atr_pct <= 0:
            return Result.failed(ValueError("atr_pct invalid"), "INVALID_ATR_PCT")

        passed = config.min_atr_pct <= atr_pct <= config.max_atr_pct
        reason_code = "OK" if passed else "ATR_FILTERED"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=1.0 if passed else 0.0,
                features={
                    "atr_value": atr_value,
                    "atr_pct": atr_pct,
                    "close_price": close_price,
                },
                metadata={
                    "period": self._period,
                    "thresholds": {
                        "min_atr_pct": config.min_atr_pct,
                        "max_atr_pct": config.max_atr_pct,
                    },
                },
            ),
            reason_code=reason_code,
        )

