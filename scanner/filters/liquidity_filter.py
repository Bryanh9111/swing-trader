"""Liquidity monitoring filter (soft filter).

Computes liquidity metrics from daily OHLCV bars and emits a 0-1 liquidity score
that can be used for ranking and position sizing decisions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, TYPE_CHECKING, Literal

import msgspec

from common.interface import Result
from scanner.interface import LiquidityFilterConfig, ScannerConfig

from .base import BaseFilter, FilterResult

if TYPE_CHECKING:
    from data.interface import PriceBar
else:  # pragma: no cover - type-only import may not exist in early phases.
    PriceBar = Any  # type: ignore[assignment]

VolumeTrend = Literal["improving", "stable", "deteriorating"]


class LiquidityMetrics(msgspec.Struct, frozen=True, kw_only=True):
    avg_dollar_volume_20d: float
    avg_dollar_volume_60d: float
    volume_trend: VolumeTrend
    liquidity_score: float
    spread_proxy: float
    unusual_activity: bool


def _clamp(value: float, lower: float, upper: float) -> float:
    if value != value:  # NaN
        return lower
    return max(lower, min(upper, value))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001 - tolerate schema drift
        return float("nan")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    total = 0.0
    for value in values:
        total += float(value)
    return total / float(len(values))


def _resolve_config(config: ScannerConfig) -> LiquidityFilterConfig:
    cfg = getattr(config, "liquidity_filter", None)
    if isinstance(cfg, LiquidityFilterConfig):
        return cfg
    try:
        return LiquidityFilterConfig(**(msgspec.to_builtins(cfg) if cfg is not None else {}))
    except Exception:  # noqa: BLE001 - fall back to defaults
        return LiquidityFilterConfig()


def _volume_trend(avg_volume_20d: float, avg_volume_60d: float) -> tuple[VolumeTrend, float]:
    if not (math.isfinite(avg_volume_20d) and math.isfinite(avg_volume_60d)) or avg_volume_60d <= 0:
        return ("stable", 0.5)

    if avg_volume_20d > avg_volume_60d * 1.1:
        return ("improving", 1.0)
    if avg_volume_20d < avg_volume_60d * 0.8:
        return ("deteriorating", 0.0)
    return ("stable", 0.5)


class LiquidityFilter(BaseFilter):
    name: ClassVar[str] = "liquidity"

    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        cfg = _resolve_config(config)

        if len(bars) < 2:
            return Result.success(
                FilterResult(
                    passed=True,
                    reason="INSUFFICIENT_BARS",
                    score=1.0,
                    metadata={"skipped": True, "bars": len(bars)},
                ),
                reason_code="INSUFFICIENT_BARS",
            )

        window_20 = min(20, len(bars))
        window_60 = min(60, len(bars))
        bars_20 = list(bars[-window_20:])
        bars_60 = list(bars[-window_60:])

        dollar_volumes_20: list[float] = []
        dollar_volumes_60: list[float] = []
        volumes_20: list[float] = []
        volumes_60: list[float] = []
        range_pcts_20: list[float] = []

        for bar in bars_20:
            close = _safe_float(getattr(bar, "close", float("nan")))
            high = _safe_float(getattr(bar, "high", float("nan")))
            low = _safe_float(getattr(bar, "low", float("nan")))
            volume = _safe_float(getattr(bar, "volume", float("nan")))

            if not (math.isfinite(close) and math.isfinite(high) and math.isfinite(low) and math.isfinite(volume)):
                continue
            if close <= 0 or volume < 0:
                continue

            dollar_volumes_20.append(close * volume)
            volumes_20.append(volume)
            range_pcts_20.append((high - low) / close)

        for bar in bars_60:
            close = _safe_float(getattr(bar, "close", float("nan")))
            volume = _safe_float(getattr(bar, "volume", float("nan")))
            if not (math.isfinite(close) and math.isfinite(volume)):
                continue
            if close <= 0 or volume < 0:
                continue
            dollar_volumes_60.append(close * volume)
            volumes_60.append(volume)

        avg_dollar_volume_20d = float(_mean(dollar_volumes_20))
        avg_dollar_volume_60d = float(_mean(dollar_volumes_60))
        avg_volume_20d = float(_mean(volumes_20))
        avg_volume_60d = float(_mean(volumes_60))
        spread_proxy = float(_mean(range_pcts_20))

        trend, volume_trend_score = _volume_trend(avg_volume_20d, avg_volume_60d)

        min_required_volume = float(cfg.min_dollar_volume) if cfg.min_dollar_volume > 0 else 1.0
        max_spread = float(cfg.max_spread_proxy) if cfg.max_spread_proxy > 0 else 1e-9

        score_raw = (
            0.4 * (avg_dollar_volume_20d / min_required_volume)
            + 0.3 * (1.0 - (spread_proxy / max_spread))
            + 0.3 * float(volume_trend_score)
        )
        liquidity_score = float(_clamp(score_raw, 0.0, 1.0))

        # Unusual activity: compare latest bar vs average of previous bars (up to 20D).
        unusual_activity = False
        baseline_bars = bars_20[:-1] if len(bars_20) > 1 else []
        if baseline_bars:
            baseline_ranges: list[float] = []
            baseline_volumes: list[float] = []
            for bar in baseline_bars:
                close = _safe_float(getattr(bar, "close", float("nan")))
                high = _safe_float(getattr(bar, "high", float("nan")))
                low = _safe_float(getattr(bar, "low", float("nan")))
                volume = _safe_float(getattr(bar, "volume", float("nan")))
                if not (math.isfinite(close) and math.isfinite(high) and math.isfinite(low) and math.isfinite(volume)):
                    continue
                if close <= 0 or volume < 0:
                    continue
                baseline_ranges.append((high - low) / close)
                baseline_volumes.append(volume)

            avg_range = float(_mean(baseline_ranges))
            avg_volume = float(_mean(baseline_volumes))
            latest = bars_20[-1]
            close = _safe_float(getattr(latest, "close", float("nan")))
            high = _safe_float(getattr(latest, "high", float("nan")))
            low = _safe_float(getattr(latest, "low", float("nan")))
            volume = _safe_float(getattr(latest, "volume", float("nan")))
            if (
                math.isfinite(avg_range)
                and math.isfinite(avg_volume)
                and math.isfinite(close)
                and math.isfinite(high)
                and math.isfinite(low)
                and math.isfinite(volume)
                and close > 0
                and avg_range > 0
                and avg_volume > 0
            ):
                latest_range = (high - low) / close
                unusual_activity = bool(latest_range > 2.0 * avg_range and volume > 2.0 * avg_volume)

        metrics = LiquidityMetrics(
            avg_dollar_volume_20d=float(avg_dollar_volume_20d),
            avg_dollar_volume_60d=float(avg_dollar_volume_60d),
            volume_trend=trend,
            liquidity_score=float(liquidity_score),
            spread_proxy=float(spread_proxy),
            unusual_activity=bool(unusual_activity),
        )

        passed = bool(liquidity_score >= float(cfg.min_liquidity_score))
        reason_code = "OK" if passed else "LOW_LIQUIDITY"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=float(liquidity_score),
                features={
                    "liquidity_metrics": msgspec.to_builtins(metrics),
                },
                metadata={
                    "windows": {"20d": window_20, "60d": window_60},
                    "thresholds": msgspec.to_builtins(cfg),
                    "volume_trend_score": float(volume_trend_score),
                },
            ),
            reason_code=reason_code,
        )

