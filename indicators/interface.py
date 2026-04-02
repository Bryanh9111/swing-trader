"""Public interfaces for technical indicators.

This module defines:
- Shared input type aliases for price series (``PriceBar`` or raw close values).
- Structured output types (e.g. ``MACDResult``) using ``msgspec.Struct``.
- A lightweight ``IndicatorProtocol`` for future plugin implementations.
- Public compute APIs following the project patterns:
  - ``compute_*_last``: compute only the latest value (stream/decision friendly).
  - ``compute_*_series``: compute the full aligned series (debug/plot/feature friendly).

Implementation lives in ``indicators.core``; functions here delegate to that module.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import msgspec

from data.interface import PriceBar

InputPrices = Sequence[PriceBar] | Sequence[float]


class MACDResult(msgspec.Struct, frozen=True, kw_only=True):
    """MACD indicator output (single timestamp).

    Attributes:
        macd: MACD line (fast EMA - slow EMA)
        signal: Signal line (EMA of MACD)
        histogram: Histogram (MACD - Signal)
    """

    macd: float
    signal: float
    histogram: float


class BollingerBandsResult(msgspec.Struct, frozen=True, kw_only=True):
    """Bollinger Bands output (single timestamp).

    Attributes:
        upper: Upper band
        middle: Middle band (SMA)
        lower: Lower band
        bandwidth: (upper - lower) / middle
        percent_b: (close - lower) / (upper - lower)
    """

    upper: float
    middle: float
    lower: float
    bandwidth: float
    percent_b: float


class VolumePriceDivergence(msgspec.Struct, frozen=True, kw_only=True):
    """Volume-price divergence detection output.

    Attributes:
        divergence_type: "bullish" | "bearish" | "none"
        price_trend: "up" | "down" | "flat"
        obv_trend: "up" | "down" | "flat"
        strength: Divergence strength in [0, 1]
    """

    divergence_type: str
    price_trend: str
    obv_trend: str
    strength: float


class KDJResult(msgspec.Struct, frozen=True, kw_only=True):
    """KDJ indicator output (single timestamp).

    KDJ is a momentum indicator derived from the Stochastic Oscillator,
    commonly used in Asian markets (especially China).

    Attributes:
        k: K line (fast stochastic, smoothed RSV)
        d: D line (slow stochastic, smoothed K)
        j: J line = 3*K - 2*D (more sensitive overbought/oversold signal)
    """

    k: float
    d: float
    j: float


@runtime_checkable
class IndicatorProtocol(Protocol):
    """Technical indicator computation protocol."""

    def compute_last(self, prices: InputPrices) -> float | None:
        """Compute the latest indicator value.

        Args:
            prices: Price sequence (``PriceBar`` or ``float`` close values).

        Returns:
            Latest indicator value, or ``None`` when there is insufficient data.
        """

    def compute_series(self, prices: InputPrices) -> list[float | None]:
        """Compute the full indicator series aligned to the input.

        Args:
            prices: Price sequence.

        Returns:
            Indicator series with the same length as the input; insufficient
            prefix values are ``None``.
        """


def compute_rsi_last(prices: InputPrices, period: int = 14) -> float | None:
    """Compute latest RSI value (0-100).

    Args:
        prices: Price sequence (``PriceBar`` or ``float`` close values).
        period: RSI lookback window size.

    Returns:
        Latest RSI value, or ``None`` when there is insufficient data.
    """

    from indicators.core import compute_rsi_last as _impl

    return _impl(prices=prices, period=period)


def compute_rsi_series(prices: InputPrices, period: int = 14) -> list[float | None]:
    """Compute RSI values for the entire series.

    Args:
        prices: Price sequence.
        period: RSI lookback window size.

    Returns:
        RSI series aligned to the input; insufficient prefix values are ``None``.
    """

    from indicators.core import compute_rsi_series as _impl

    return _impl(prices=prices, period=period)


def compute_macd_last(
    prices: InputPrices,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult | None:
    """Compute latest MACD result.

    Args:
        prices: Price sequence.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal EMA period (applied to MACD line).

    Returns:
        Latest MACD result, or ``None`` when there is insufficient data.
    """

    from indicators.core import compute_macd_last as _impl

    return _impl(prices=prices, fast=fast, slow=slow, signal=signal)


def compute_macd_series(
    prices: InputPrices,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> list[MACDResult | None]:
    """Compute MACD results for the entire series.

    Args:
        prices: Price sequence.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal EMA period (applied to MACD line).

    Returns:
        MACD series aligned to the input; insufficient prefix values are ``None``.
    """

    from indicators.core import compute_macd_series as _impl

    return _impl(prices=prices, fast=fast, slow=slow, signal=signal)


def compute_ema_last(prices: InputPrices, period: int) -> float | None:
    """Compute latest EMA value.

    Args:
        prices: Price sequence.
        period: EMA period.

    Returns:
        Latest EMA value, or ``None`` when there is insufficient data.
    """

    from indicators.core import compute_ema_last as _impl

    return _impl(prices=prices, period=period)


def compute_ema_series(prices: InputPrices, period: int) -> list[float | None]:
    """Compute EMA values for the entire series.

    Args:
        prices: Price sequence.
        period: EMA period.

    Returns:
        EMA series aligned to the input; insufficient prefix values are ``None``.
    """

    from indicators.core import compute_ema_series as _impl

    return _impl(prices=prices, period=period)


def compute_sma_last(prices: InputPrices, period: int) -> float | None:
    """Compute latest SMA value.

    Args:
        prices: Price sequence.
        period: SMA period.

    Returns:
        Latest SMA value, or ``None`` when there is insufficient data.
    """

    from indicators.core import compute_sma_last as _impl

    return _impl(prices=prices, period=period)


def compute_sma_series(prices: InputPrices, period: int) -> list[float | None]:
    """Compute SMA values for the entire series.

    Args:
        prices: Price sequence.
        period: SMA period.

    Returns:
        SMA series aligned to the input; insufficient prefix values are ``None``.
    """

    from indicators.core import compute_sma_series as _impl

    return _impl(prices=prices, period=period)


def compute_adx_last(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Compute latest ADX value (0-100).

    Args:
        highs: High prices.
        lows: Low prices.
        closes: Close prices.
        period: ADX period.

    Returns:
        Latest ADX value, or ``None`` when there is insufficient data.
    """

    from indicators.core import compute_adx_last as _impl

    return _impl(highs=highs, lows=lows, closes=closes, period=period)


def compute_atr_last(
    prices: InputPrices,
    period: int = 14,
    percentage: bool = True,
    reference_price: float | None = None,
) -> float | None:
    """Compute the latest ATR (Average True Range) value.

    Args:
        prices: Sequence of PriceBar or close prices.
        period: ATR averaging period (default 14).
        percentage: If True, return ATR as percentage of reference_price.
        reference_price: Reference price for percentage calculation (default: latest close).

    Returns:
        float | None: Latest ATR value (absolute or percentage), or None if insufficient data.
    """

    from indicators.core import compute_atr_last as _impl

    return _impl(prices, period, percentage, reference_price)


def compute_bbands_last(
    prices: InputPrices,
    period: int = 20,
    std_dev: float = 2.0,
) -> BollingerBandsResult | None:
    """Compute latest Bollinger Bands result."""

    from indicators.core import compute_bbands_last as _impl

    return _impl(prices=prices, period=period, std_dev=std_dev)


def compute_bbands_series(
    prices: InputPrices,
    period: int = 20,
    std_dev: float = 2.0,
) -> list[BollingerBandsResult | None]:
    """Compute Bollinger Bands results for the entire series."""

    from indicators.core import compute_bbands_series as _impl

    return _impl(prices=prices, period=period, std_dev=std_dev)


def compute_obv_last(bars: Sequence[PriceBar]) -> float | None:
    """Compute latest OBV (On-Balance Volume)."""

    from indicators.core import compute_obv_last as _impl

    return _impl(bars)


def compute_obv_series(bars: Sequence[PriceBar]) -> list[float | None]:
    """Compute OBV series aligned to the input bars."""

    from indicators.core import compute_obv_series as _impl

    return _impl(bars)


def compute_vpt_last(bars: Sequence[PriceBar]) -> float | None:
    """Compute latest VPT (Volume Price Trend)."""

    from indicators.core import compute_vpt_last as _impl

    return _impl(bars)


def compute_vpt_series(bars: Sequence[PriceBar]) -> list[float | None]:
    """Compute VPT series aligned to the input bars."""

    from indicators.core import compute_vpt_series as _impl

    return _impl(bars)


def compute_volume_price_divergence(
    bars: Sequence[PriceBar],
    lookback: int = 20,
) -> VolumePriceDivergence | None:
    """Detect volume-price divergence based on price and OBV extremes."""

    from indicators.core import compute_volume_price_divergence as _impl

    return _impl(bars, lookback=lookback)


def compute_kdj_last(
    prices: InputPrices,
    *,
    n_period: int = 9,
    k_period: int = 3,
    d_period: int = 3,
) -> KDJResult | None:
    """Compute latest KDJ indicator result.

    KDJ formula:
        RSV = (Close - Lowest_N) / (Highest_N - Lowest_N) * 100
        K = EMA(RSV, k_period)
        D = EMA(K, d_period)
        J = 3*K - 2*D

    Args:
        prices: Price sequence (PriceBar or float close values).
        n_period: Lookback period for highest high / lowest low (default 9).
        k_period: Smoothing period for K line (default 3).
        d_period: Smoothing period for D line (default 3).

    Returns:
        KDJResult with K, D, J values, or None if insufficient data.
    """

    from indicators.core import compute_kdj_last as _impl

    return _impl(prices, n_period=n_period, k_period=k_period, d_period=d_period)


def compute_kdj_series(
    prices: InputPrices,
    *,
    n_period: int = 9,
    k_period: int = 3,
    d_period: int = 3,
) -> list[KDJResult | None]:
    """Compute KDJ indicator series for the entire price sequence.

    Args:
        prices: Price sequence.
        n_period: Lookback period for highest high / lowest low (default 9).
        k_period: Smoothing period for K line (default 3).
        d_period: Smoothing period for D line (default 3).

    Returns:
        KDJ series aligned to the input; insufficient prefix values are None.
    """

    from indicators.core import compute_kdj_series as _impl

    return _impl(prices, n_period=n_period, k_period=k_period, d_period=d_period)


__all__ = [
    "InputPrices",
    "MACDResult",
    "BollingerBandsResult",
    "VolumePriceDivergence",
    "KDJResult",
    "IndicatorProtocol",
    "compute_rsi_last",
    "compute_rsi_series",
    "compute_macd_last",
    "compute_macd_series",
    "compute_ema_last",
    "compute_ema_series",
    "compute_sma_last",
    "compute_sma_series",
    "compute_atr_last",
    "compute_bbands_last",
    "compute_bbands_series",
    "compute_obv_last",
    "compute_obv_series",
    "compute_vpt_last",
    "compute_vpt_series",
    "compute_volume_price_divergence",
    "compute_kdj_last",
    "compute_kdj_series",
]
