"""Pluggable indicator scoring module (Phase 2 Step 2).

This module computes an optional 0-1 indicator-based score derived from
RSI/MACD/ATR signals. It is intentionally independent from the scanner's
feature detection logic and only depends on ``indicators.interface``.
"""

from __future__ import annotations

from math import exp, log
from typing import Any, Sequence

from data.interface import PriceBar
from indicators.interface import (
    compute_atr_last,
    compute_bbands_last,
    compute_kdj_last,
    compute_macd_last,
    compute_obv_series,
    compute_rsi_last,
    compute_volume_price_divergence,
)
from scanner.interface import IndicatorScoringConfig

__all__ = [
    "compute_indicator_score",
]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score_rsi(rsi_value: float, optimal_range: tuple[float, float]) -> float:
    """RSI scoring: 1.0 inside optimal_range, linear decay outside."""

    low, high = optimal_range
    if low <= rsi_value <= high:
        return 1.0
    if rsi_value < low:
        if low <= 0:
            return 0.0
        distance = (low - rsi_value) / low
        return _clamp01(1.0 - distance)

    # rsi_value > high
    if high >= 100.0:
        return 0.0
    distance = (rsi_value - high) / (100.0 - high)
    return _clamp01(1.0 - distance)


def _score_macd(histogram: float, threshold: float) -> float:
    """MACD scoring: 1.0 if histogram >= threshold, else 0.5 (conservative penalty)."""

    return 1.0 if float(histogram) >= float(threshold) else 0.5


def _score_atr(atr_pct: float, optimal_range: tuple[float, float]) -> float:
    """ATR% scoring: 1.0 inside optimal_range, linear decay outside."""

    low, high = optimal_range
    if low <= atr_pct <= high:
        return 1.0
    if atr_pct < low:
        if low <= 0:
            return 0.0
        distance = (low - atr_pct) / low
        return _clamp01(1.0 - distance)

    # atr_pct > high
    if high <= 0:
        return 0.0
    distance = (atr_pct - high) / high
    return _clamp01(1.0 - distance)


def _score_bbands(percent_b: float) -> float:
    """Bollinger Bands %B scoring (bullish bias).

    Rules:
        - percent_b < 0.2 => oversold => high bullish score
        - percent_b > 0.8 => overbought => low bullish score

    Between 0.2 and 0.8, score decays linearly from 1.0 to 0.0.
    """

    pb = float(percent_b)
    if pb <= 0.2:
        return 1.0
    if pb >= 0.8:
        return 0.0
    return _clamp01(1.0 - (pb - 0.2) / 0.6)


def _score_kdj(j_value: float, *, oversold: float = 20.0, overbought: float = 80.0) -> float:
    """KDJ J-line scoring (bullish bias).

    The J line (3*K - 2*D) is the most sensitive KDJ component and is used
    primarily for overbought/oversold detection.

    Rules:
        - J <= oversold (20) => oversold => high bullish score (1.0)
        - J >= overbought (80) => overbought => low bullish score (0.0)
        - Between: linear interpolation from 1.0 to 0.0

    Args:
        j_value: J line value from KDJ indicator.
        oversold: J threshold for oversold condition (default 20).
        overbought: J threshold for overbought condition (default 80).

    Returns:
        Score in [0, 1] where higher is more bullish.
    """
    j = float(j_value)
    oversold_f = float(oversold)
    overbought_f = float(overbought)

    if j <= oversold_f:
        return 1.0
    if j >= overbought_f:
        return 0.0

    # Linear decay from oversold to overbought
    range_width = overbought_f - oversold_f
    if range_width <= 0.0:
        return 0.5
    return _clamp01(1.0 - (j - oversold_f) / range_width)


def _combine_scores_weighted_avg(items: Sequence[tuple[str, float, float]]) -> float:
    total_weight = sum(weight for _, _, weight in items if weight > 0.0)
    if total_weight <= 0.0:
        return 1.0
    weighted_sum = sum(score * weight for _, score, weight in items if weight > 0.0)
    return _clamp01(weighted_sum / total_weight)


def _combine_scores_multiply(items: Sequence[tuple[str, float, float]]) -> float:
    """Weighted geometric mean (multiplicative aggregation)."""

    total_weight = sum(weight for _, _, weight in items if weight > 0.0)
    if total_weight <= 0.0:
        return 1.0

    normalized = [(name, _clamp01(score), weight / total_weight) for name, score, weight in items if weight > 0.0]
    if any(score == 0.0 for _, score, _ in normalized):
        return 0.0

    # log/exp keeps numerical stability for many indicators.
    geo_log_sum = sum(norm_w * log(score) for _, score, norm_w in normalized)
    return _clamp01(exp(geo_log_sum))


def _score_obv_trend(delta: float) -> float:
    """OBV trend scoring: rising => high, falling => low, flat => neutral."""

    if delta > 0.0:
        return 1.0
    if delta < 0.0:
        return 0.0
    return 0.5


def _score_divergence(divergence_type: str, strength: float) -> float:
    """Divergence scoring with neutral baseline at 0.5."""

    s = _clamp01(float(strength))
    if divergence_type == "bullish":
        return _clamp01(0.5 + 0.5 * s)
    if divergence_type == "bearish":
        return _clamp01(0.5 - 0.5 * s)
    return 0.5


def _compute_price_momentum_pct(bars: Sequence[PriceBar], *, lookback: int = 10) -> float | None:
    lookback = int(lookback)
    if lookback <= 0 or len(bars) <= lookback:
        return None
    start = float(bars[-lookback - 1].close)
    end = float(bars[-1].close)
    if start <= 0.0:
        return None
    return (end - start) / start


def _score_short_volume_ratio(short_volume_ratio: float) -> float:
    r = _clamp01(float(short_volume_ratio))
    if r >= 0.8:
        return 0.0
    if r >= 0.4:
        # 0.4 => 0.5, 0.8 => 0.0
        return _clamp01(0.5 - 0.5 * (r - 0.4) / 0.4)
    # 0.0 => 0.7, 0.4 => 0.5
    return _clamp01(0.7 - 0.2 * (r / 0.4))


def _score_short_interest(short_percent: float, *, squeeze_threshold: float, price_momentum_pct: float | None) -> tuple[float, bool]:
    sp = _clamp01(float(short_percent))
    threshold = _clamp01(float(squeeze_threshold))
    improving = (price_momentum_pct or 0.0) > 0.0

    if sp < threshold:
        return 0.5, False

    intensity = _clamp01((sp - threshold) / 0.30)
    if improving:
        return _clamp01(0.5 + 0.5 * intensity), True
    return _clamp01(0.5 - 0.3 * intensity), False


def compute_indicator_score(
    bars: Sequence[PriceBar],
    config: IndicatorScoringConfig,
    *,
    symbol: str | None = None,
    short_data_provider: Any | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute combined indicator score (0-1) for a price series.

    Args:
        bars: Price bars (most recent bar last).
        config: Indicator scoring configuration.

    Returns:
        (final_score, metadata)
    """

    if not config.enabled:
        return 1.0, {}

    scores: list[tuple[str, float, float]] = []
    metadata: dict[str, Any] = {"combination_mode": config.combination_mode, "indicators": {}, "skipped": {}}

    def _maybe_add_score(*, name: str, weight: float, value: Any, score: float) -> None:
        scores.append((name, _clamp01(score), float(weight)))
        metadata["indicators"][name] = {"value": value, "score": _clamp01(score), "weight": float(weight)}

    def _skip(*, name: str, reason: str) -> None:
        metadata["skipped"][name] = reason

    if config.rsi_enabled and config.rsi_weight > 0.0:
        try:
            rsi_value = compute_rsi_last(bars)
            if rsi_value is None:
                _skip(name="rsi", reason="insufficient_data")
            else:
                _maybe_add_score(
                    name="rsi",
                    weight=config.rsi_weight,
                    value=float(rsi_value),
                    score=_score_rsi(float(rsi_value), config.rsi_optimal_range),
                )
        except Exception as exc:  # pragma: no cover - defensive
            _skip(name="rsi", reason=f"error:{type(exc).__name__}")

    if config.macd_enabled and config.macd_weight > 0.0:
        try:
            macd = compute_macd_last(bars)
            if macd is None:
                _skip(name="macd", reason="insufficient_data")
            else:
                _maybe_add_score(
                    name="macd",
                    weight=config.macd_weight,
                    value={"histogram": float(macd.histogram), "threshold": float(config.macd_histogram_threshold)},
                    score=_score_macd(float(macd.histogram), float(config.macd_histogram_threshold)),
                )
        except Exception as exc:  # pragma: no cover - defensive
            _skip(name="macd", reason=f"error:{type(exc).__name__}")

    if config.atr_enabled and config.atr_weight > 0.0:
        try:
            atr_pct = compute_atr_last(bars, percentage=True)
            if atr_pct is None:
                _skip(name="atr", reason="insufficient_data")
            else:
                _maybe_add_score(
                    name="atr",
                    weight=config.atr_weight,
                    value=float(atr_pct),
                    score=_score_atr(float(atr_pct), config.atr_optimal_range),
                )
        except Exception as exc:  # pragma: no cover - defensive
            _skip(name="atr", reason=f"error:{type(exc).__name__}")

    if config.bbands_enabled and config.bbands_weight > 0.0:
        try:
            bbands = compute_bbands_last(
                bars,
                period=int(config.bbands_period),
                std_dev=float(config.bbands_std_dev),
            )
            if bbands is None:
                _skip(name="bbands", reason="insufficient_data")
            else:
                _maybe_add_score(
                    name="bbands",
                    weight=config.bbands_weight,
                    value={"percent_b": float(bbands.percent_b)},
                    score=_score_bbands(float(bbands.percent_b)),
                )
        except Exception as exc:  # pragma: no cover - defensive
            _skip(name="bbands", reason=f"error:{type(exc).__name__}")

    if config.obv_enabled and config.obv_weight > 0.0:
        try:
            obv_series = compute_obv_series(bars)
            obv_last = obv_series[-1] if obv_series else None
            if obv_last is None:
                _skip(name="obv", reason="insufficient_data")
            else:
                lookback = max(2, int(config.divergence_lookback))
                if len(bars) < lookback:
                    _skip(name="obv", reason="insufficient_data")
                else:
                    obv_window = obv_series[-lookback:]
                    obv_start: float | None = None
                    for v in obv_window:
                        if v is None:
                            continue
                        obv_start = float(v)
                        break
                    if obv_start is None:
                        _skip(name="obv", reason="insufficient_data")
                    else:
                        delta = float(obv_last) - float(obv_start)
                        obv_trend_score = _score_obv_trend(float(delta))

                        divergence = compute_volume_price_divergence(bars, lookback=lookback)
                        if divergence is None:
                            divergence_score = 0.5
                            divergence_meta: Any = None
                        else:
                            divergence_score = _score_divergence(
                                str(divergence.divergence_type),
                                float(divergence.strength),
                            )
                            divergence_meta = {
                                "type": str(divergence.divergence_type),
                                "price_trend": str(divergence.price_trend),
                                "obv_trend": str(divergence.obv_trend),
                                "strength": float(divergence.strength),
                            }

                        # One weight for volume-price relationships; combine internal signals.
                        combined_obv_score = _clamp01(0.6 * obv_trend_score + 0.4 * divergence_score)
                        _maybe_add_score(
                            name="obv",
                            weight=config.obv_weight,
                            value={
                                "obv_last": float(obv_last),
                                "obv_delta": float(delta),
                                "divergence": divergence_meta,
                            },
                            score=float(combined_obv_score),
                        )
        except Exception as exc:  # pragma: no cover - defensive
            _skip(name="obv", reason=f"error:{type(exc).__name__}")

    if getattr(config, "short_enabled", False) and float(getattr(config, "short_weight", 0.0)) > 0.0:
        if not symbol:
            _skip(name="short", reason="missing_symbol")
        else:
            try:
                from data.short_data_provider import ShortDataProvider

                provider = short_data_provider or ShortDataProvider()
                if not isinstance(provider, ShortDataProvider) and not hasattr(provider, "fetch_short_interest"):
                    _skip(name="short", reason="invalid_provider")
                else:
                    interest = provider.fetch_short_interest(symbol)
                    volumes = provider.fetch_short_volume(symbol, days=30)

                    price_momentum_pct = _compute_price_momentum_pct(bars, lookback=10)
                    squeeze_threshold = float(getattr(config, "short_squeeze_threshold", 0.20))

                    short_interest_pct: float | None = None
                    short_ratio: float | None = None
                    settlement_date: str | None = None
                    squeeze_potential = False
                    interest_score: float | None = None

                    if interest is not None:
                        short_interest_pct = float(getattr(interest, "short_percent", 0.0))
                        short_ratio = float(getattr(interest, "short_ratio", 0.0))
                        settlement_date = str(getattr(interest, "settlement_date", "") or "")
                        interest_score, squeeze_potential = _score_short_interest(
                            float(short_interest_pct),
                            squeeze_threshold=squeeze_threshold,
                            price_momentum_pct=price_momentum_pct,
                        )

                    recent_short_volume_ratio: float | None = None
                    volume_score: float | None = None
                    if volumes:
                        ratios = [float(getattr(row, "short_volume_ratio", 0.0)) for row in volumes[-5:]]
                        if ratios:
                            recent_short_volume_ratio = float(sum(ratios) / len(ratios))
                            volume_score = _score_short_volume_ratio(float(recent_short_volume_ratio))

                    if interest_score is None and volume_score is None:
                        _skip(name="short", reason="insufficient_data")
                    else:
                        if interest_score is None:
                            short_score = float(volume_score)
                        elif volume_score is None:
                            short_score = float(interest_score)
                        else:
                            short_score = float(_clamp01(0.6 * float(interest_score) + 0.4 * float(volume_score)))

                        short_meta = {
                            "short_interest_pct": short_interest_pct,
                            "short_ratio": short_ratio,
                            "short_volume_ratio": recent_short_volume_ratio,
                            "short_squeeze_potential": bool(squeeze_potential),
                            "short_squeeze_threshold": float(squeeze_threshold),
                            "settlement_date": settlement_date,
                            "price_momentum_pct": price_momentum_pct,
                        }
                        metadata["short_data"] = short_meta
                        _maybe_add_score(
                            name="short",
                            weight=float(getattr(config, "short_weight", 0.2)),
                            value=short_meta,
                            score=float(short_score),
                        )
            except Exception as exc:  # pragma: no cover - defensive
                _skip(name="short", reason=f"error:{type(exc).__name__}")

    if getattr(config, "kdj_enabled", False) and float(getattr(config, "kdj_weight", 0.0)) > 0.0:
        try:
            n_period = int(getattr(config, "kdj_n_period", 9))
            k_period = int(getattr(config, "kdj_k_period", 3))
            d_period = int(getattr(config, "kdj_d_period", 3))
            kdj = compute_kdj_last(bars, n_period=n_period, k_period=k_period, d_period=d_period)
            if kdj is None:
                _skip(name="kdj", reason="insufficient_data")
            else:
                oversold = float(getattr(config, "kdj_oversold", 20.0))
                overbought = float(getattr(config, "kdj_overbought", 80.0))
                kdj_score = _score_kdj(float(kdj.j), oversold=oversold, overbought=overbought)
                _maybe_add_score(
                    name="kdj",
                    weight=float(getattr(config, "kdj_weight", 0.3)),
                    value={"k": float(kdj.k), "d": float(kdj.d), "j": float(kdj.j)},
                    score=float(kdj_score),
                )
        except Exception as exc:  # pragma: no cover - defensive
            _skip(name="kdj", reason=f"error:{type(exc).__name__}")

    if not scores:
        # No usable indicator values -> neutral score, but keep metadata for observability.
        return 1.0, metadata

    if config.combination_mode == "multiply":
        combined = _combine_scores_multiply(scores)
    else:  # default to weighted average for now (includes "weighted_avg")
        combined = _combine_scores_weighted_avg(scores)

    metadata["combined"] = float(combined)
    return float(combined), metadata
