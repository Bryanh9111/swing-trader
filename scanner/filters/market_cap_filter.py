"""Market cap stratification filter for US equities.

Applies different ATR and liquidity thresholds based on market capitalization:
- Large-cap ($10B+): Lower volatility tolerance, higher liquidity requirement
- Mid-cap ($2B-$10B): Medium thresholds
- Small-cap ($300M-$2B): Higher volatility acceptable, lower liquidity ok

This filter expects ``market_cap`` (USD) to be provided via filter context:
``context={"meta": {"market_cap": <float>}}``.
"""

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


class MarketCapFilter(BaseFilter):
    """Filter candidates based on market cap stratification."""

    name: ClassVar[str] = "market_cap"

    def __init__(self, *, window: int, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._window = int(window)

    def apply(
        self,
        bars: Sequence[PriceBar],
        config: ScannerConfig,
        *,
        context: dict[str, Any] | None = None,
    ) -> Result[FilterResult]:
        if not self.enabled:
            return super().apply(bars, config, context=context)

        meta = (context or {}).get("meta")
        if not isinstance(meta, dict):
            meta = {}

        market_cap_raw = meta.get("market_cap", 0)
        try:
            market_cap = float(market_cap_raw or 0.0)
        except Exception:  # noqa: BLE001
            market_cap = 0.0

        if not math.isfinite(market_cap) or market_cap <= 0:
            return Result.success(
                FilterResult(
                    passed=True,
                    reason="MARKET_CAP_UNKNOWN",
                    score=1.0,
                    features={},
                    metadata={"market_cap": None, "tier": "UNKNOWN"},
                ),
                reason_code="MARKET_CAP_UNKNOWN",
            )

        if market_cap >= 10_000_000_000:  # $10B+
            tier = "LARGE_CAP"
            max_atr_pct = 0.015  # 1.5%
            min_dollar_volume = 50_000_000  # $50M
        elif market_cap >= 2_000_000_000:  # $2B-$10B
            tier = "MID_CAP"
            max_atr_pct = 0.03  # 3%
            min_dollar_volume = 10_000_000  # $10M
        elif market_cap >= 300_000_000:  # $300M-$2B
            tier = "SMALL_CAP"
            max_atr_pct = 0.05  # 5%
            min_dollar_volume = 5_000_000  # $5M
        else:
            return Result.success(
                FilterResult(
                    passed=False,
                    reason="MICRO_CAP_EXCLUDED",
                    score=0.0,
                    features={},
                    metadata={"market_cap": market_cap, "tier": "MICRO_CAP", "threshold": 300_000_000},
                ),
                reason_code="MICRO_CAP_EXCLUDED",
            )

        window = max(1, self._window)
        if len(bars) < window:
            return Result.success(
                FilterResult(
                    passed=True,
                    reason="INSUFFICIENT_BARS_FOR_MARKET_CAP_FILTER",
                    score=1.0,
                    features={},
                    metadata={"market_cap": market_cap, "tier": tier, "window": window, "bars": len(bars)},
                ),
                reason_code="INSUFFICIENT_BARS_FOR_MARKET_CAP_FILTER",
            )

        avg_dollar_volume = _avg_dollar_volume(bars[-window:])
        if not math.isfinite(avg_dollar_volume):
            avg_dollar_volume = 0.0

        atr_pct = _atr_pct(bars)
        if atr_pct is not None and atr_pct > max_atr_pct:
            return Result.success(
                FilterResult(
                    passed=False,
                    reason="ATR_TOO_HIGH_FOR_MARKET_CAP",
                    score=0.0,
                    features={"atr_pct": atr_pct},
                    metadata={
                        "market_cap": market_cap,
                        "tier": tier,
                        "max_atr_pct": max_atr_pct,
                        "avg_dollar_volume": avg_dollar_volume,
                    },
                ),
                reason_code="ATR_TOO_HIGH_FOR_MARKET_CAP",
            )

        if avg_dollar_volume < min_dollar_volume:
            return Result.success(
                FilterResult(
                    passed=False,
                    reason="INSUFFICIENT_LIQUIDITY_FOR_MARKET_CAP",
                    score=0.0,
                    features={"avg_dollar_volume": avg_dollar_volume},
                    metadata={
                        "market_cap": market_cap,
                        "tier": tier,
                        "min_dollar_volume": min_dollar_volume,
                        "atr_pct": atr_pct,
                    },
                ),
                reason_code="INSUFFICIENT_LIQUIDITY_FOR_MARKET_CAP",
            )

        return Result.success(
            FilterResult(
                passed=True,
                reason="OK",
                score=1.0,
                features={"atr_pct": atr_pct, "avg_dollar_volume": avg_dollar_volume},
                metadata={"market_cap": market_cap, "tier": tier},
            ),
            reason_code="OK",
        )

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        return Result.failed(
            NotImplementedError("MarketCapFilter requires context (market_cap) and should be called via apply(..., context=...)"),
            "MARKET_CAP_FILTER_CONTEXT_REQUIRED",
        )


def _avg_dollar_volume(bars: Sequence[PriceBar]) -> float:
    if not bars:
        return 0.0
    total = 0.0
    for bar in bars:
        close = float(getattr(bar, "close", 0.0))
        volume = float(getattr(bar, "volume", 0.0))
        total += close * volume
    return float(total / len(bars))


def _atr_pct(bars: Sequence[PriceBar]) -> float | None:
    if not bars:
        return None
    close_price = float(getattr(bars[-1], "close", float("nan")))
    if not math.isfinite(close_price) or close_price <= 0:
        return None

    atr = detector.calculate_atr(list(bars), period=detector.ATR_DEFAULT_PERIOD)
    if atr.status is ResultStatus.FAILED or atr.data is None:
        return None

    atr_value = float(atr.data)
    atr_pct = float(atr_value / close_price)
    if not math.isfinite(atr_pct) or atr_pct <= 0:
        return None
    return atr_pct
