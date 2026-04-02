"""Core computation logic for technical indicators.

This module contains pure-Python implementations of common technical
indicators without relying on pandas/numpy. Public APIs are exposed via
``indicators.interface`` and delegate into this file.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from indicators.interface import BollingerBandsResult, InputPrices, MACDResult, VolumePriceDivergence


def _extract_close_prices(prices: InputPrices) -> list[float]:
    """Extract close prices from ``PriceBar`` sequence or raw float sequence."""

    from data.interface import PriceBar

    if not prices:
        return []

    first = prices[0]
    if isinstance(first, PriceBar) or hasattr(first, "close"):
        return [float(bar.close) for bar in prices]

    return [float(p) for p in prices]


def _normalize_to_bars(prices: InputPrices) -> list["PriceBar"]:
    """Normalize inputs to a ``PriceBar`` list for OHLC-based indicators.

    For raw close sequences, creates synthetic bars with ``open=high=low=close``.
    """

    from data.interface import PriceBar

    if not prices:
        return []

    first = prices[0]
    if isinstance(first, PriceBar):
        return list(prices)

    if hasattr(first, "close"):
        bars: list[PriceBar] = []
        for i, bar in enumerate(prices):
            close = float(getattr(bar, "close"))
            bars.append(
                PriceBar(
                    timestamp=int(getattr(bar, "timestamp", i)),
                    open=float(getattr(bar, "open", close)),
                    high=float(getattr(bar, "high", close)),
                    low=float(getattr(bar, "low", close)),
                    close=close,
                    volume=int(getattr(bar, "volume", 0)),
                )
            )
        return bars

    bars: list[PriceBar] = []
    for i, close in enumerate(prices):
        c = float(close)
        bars.append(
            PriceBar(
                timestamp=i,
                open=c,
                high=c,
                low=c,
                close=c,
                volume=0,
            ),
        )
    return bars


def compute_atr_last(
    prices: InputPrices,
    period: int = 14,
    percentage: bool = True,
    reference_price: float | None = None,
) -> float | None:
    """Compute latest ATR using True Range averaging.

    Notes:
        Uses a simple mean of the most recent ``period`` True Range values (no
        Wilder smoothing) to match the Strategy Engine fallback implementation.
    """

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    bars = _normalize_to_bars(prices)
    if len(bars) < period + 1:
        return None  # Need period+1 bars for period TR values

    tr_values: list[float] = []
    for idx in range(len(bars) - period, len(bars)):
        if idx == 0:
            continue  # Need previous close for TR calculation

        prev_close = float(bars[idx - 1].close)
        high = float(bars[idx].high)
        low = float(bars[idx].low)

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        tr_values.append(float(tr))

    if not tr_values:
        return None

    atr = sum(tr_values) / float(len(tr_values))

    if not percentage:
        return float(atr)

    ref_price = reference_price if reference_price is not None else float(bars[-1].close)
    if ref_price <= 0:
        return None

    atr_pct = atr / float(ref_price)
    return float(atr_pct)


def compute_adx_last(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Compute latest ADX value (0-100) using Wilder smoothing."""

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    n = len(closes)
    if n < period + 2:
        return None

    tr: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        high = float(highs[i])
        low = float(lows[i])
        prev_close = float(closes[i - 1])
        if not (math.isfinite(high) and math.isfinite(low) and math.isfinite(prev_close)):
            return None

        true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr.append(float(true_range))

        up_move = float(highs[i]) - float(highs[i - 1])
        down_move = float(lows[i - 1]) - float(lows[i])
        plus = up_move if up_move > down_move and up_move > 0 else 0.0
        minus = down_move if down_move > up_move and down_move > 0 else 0.0
        plus_dm.append(float(plus))
        minus_dm.append(float(minus))

    tr14 = sum(tr[:period])
    plus14 = sum(plus_dm[:period])
    minus14 = sum(minus_dm[:period])
    if tr14 <= 0 or not math.isfinite(tr14):
        return None

    dx_values: list[float] = []
    for i in range(period, len(tr)):
        if i != period:
            tr14 = tr14 - (tr14 / period) + tr[i]
            plus14 = plus14 - (plus14 / period) + plus_dm[i]
            minus14 = minus14 - (minus14 / period) + minus_dm[i]

        if tr14 <= 0:
            return None

        plus_di = 100.0 * (plus14 / tr14)
        minus_di = 100.0 * (minus14 / tr14)
        denom = plus_di + minus_di
        if denom == 0:
            dx = 0.0
        else:
            dx = 100.0 * (abs(plus_di - minus_di) / denom)
        dx_values.append(float(dx))

    if len(dx_values) < period:
        return None

    adx = sum(dx_values[:period]) / float(period)
    for dx in dx_values[period:]:
        adx = (adx * (period - 1) + dx) / float(period)

    return float(adx) if math.isfinite(adx) else None


def compute_rsi_last(prices: InputPrices, period: int = 14) -> float | None:
    """Compute latest RSI value (0-100) using Wilder's smoothing.

    Args:
        prices: Price sequence (``PriceBar`` or raw close floats).
        period: RSI lookback window size.

    Returns:
        Latest RSI value, or ``None`` when there is insufficient data.
    """

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    closes = _extract_close_prices(prices)
    if len(closes) <= period:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [max(-delta, 0.0) for delta in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_rsi_series(prices: InputPrices, period: int = 14) -> list[float | None]:
    """Compute RSI values for the entire series using Wilder's smoothing.

    Args:
        prices: Price sequence (``PriceBar`` or raw close floats).
        period: RSI lookback window size.

    Returns:
        RSI series aligned to the input; insufficient prefix values are ``None``.
    """

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    closes = _extract_close_prices(prices)
    n = len(closes)
    if n <= period:
        return [None] * n

    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [max(-delta, 0.0) for delta in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    result: list[float | None] = [None] * n

    def _rsi_from_avgs(gain: float, loss: float) -> float:
        if loss == 0:
            return 100.0
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))

    result[period] = _rsi_from_avgs(avg_gain, avg_loss)

    for delta_index in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[delta_index]) / period
        avg_loss = (avg_loss * (period - 1) + losses[delta_index]) / period
        price_index = delta_index + 1
        result[price_index] = _rsi_from_avgs(avg_gain, avg_loss)

    return result


def compute_macd_last(
    prices: InputPrices,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult | None:
    """Compute latest MACD result.

    Args:
        prices: Price sequence (``PriceBar`` or raw close floats).
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal EMA period (applied to MACD line).

    Returns:
        Latest MACD result, or ``None`` when there is insufficient data.
    """

    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("fast, slow, signal must be > 0")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    closes = _extract_close_prices(prices)
    if len(closes) < slow:
        return None

    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)

    ema_fast = closes[0]
    ema_slow = closes[0]
    macd_series: list[float] = [ema_fast - ema_slow]
    for price in closes[1:]:
        ema_fast = alpha_fast * price + (1 - alpha_fast) * ema_fast
        ema_slow = alpha_slow * price + (1 - alpha_slow) * ema_slow
        macd_series.append(ema_fast - ema_slow)

    if len(macd_series) < signal:
        return None

    alpha_signal = 2.0 / (signal + 1)
    signal_ema = macd_series[0]
    for macd_val in macd_series[1:]:
        signal_ema = alpha_signal * macd_val + (1 - alpha_signal) * signal_ema

    macd_line = macd_series[-1]
    histogram = macd_line - signal_ema

    from indicators.interface import MACDResult

    return MACDResult(macd=macd_line, signal=signal_ema, histogram=histogram)


def compute_macd_series(
    prices: InputPrices,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> list[MACDResult | None]:
    """Compute MACD results for the entire series.

    Args:
        prices: Price sequence (``PriceBar`` or raw close floats).
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal EMA period (applied to MACD line).

    Returns:
        MACD series aligned to the input; insufficient prefix values are ``None``.
    """

    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("fast, slow, signal must be > 0")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    closes = _extract_close_prices(prices)
    n = len(closes)
    if n < slow:
        return [None] * n

    from indicators.interface import MACDResult

    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)

    ema_fast = closes[0]
    ema_slow = closes[0]
    macd_series: list[float] = [ema_fast - ema_slow]
    for price in closes[1:]:
        ema_fast = alpha_fast * price + (1 - alpha_fast) * ema_fast
        ema_slow = alpha_slow * price + (1 - alpha_slow) * ema_slow
        macd_series.append(ema_fast - ema_slow)

    alpha_signal = 2.0 / (signal + 1)
    signal_ema = macd_series[0]
    signal_series: list[float] = [signal_ema]
    for macd_val in macd_series[1:]:
        signal_ema = alpha_signal * macd_val + (1 - alpha_signal) * signal_ema
        signal_series.append(signal_ema)

    result: list[MACDResult | None] = []
    for i in range(n):
        if i < slow - 1:
            result.append(None)
            continue
        macd_val = macd_series[i]
        signal_val = signal_series[i]
        result.append(MACDResult(macd=macd_val, signal=signal_val, histogram=macd_val - signal_val))

    return result


def compute_ema_last(prices: InputPrices, period: int) -> float | None:
    """Compute latest EMA value.

    EMA uses ``alpha = 2/(period+1)`` with ``adjust=False`` style recursion and
    the first price as the initial EMA value.

    Args:
        prices: Price sequence (``PriceBar`` or raw close floats).
        period: EMA period.

    Returns:
        Latest EMA value, or ``None`` when there is insufficient data.
    """

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    closes = _extract_close_prices(prices)
    if len(closes) < period:
        return None

    alpha = 2.0 / (period + 1)
    ema = closes[0]
    for price in closes[1:]:
        ema = alpha * price + (1 - alpha) * ema

    return ema


def compute_ema_series(prices: InputPrices, period: int) -> list[float | None]:
    """Compute EMA values for the entire series.

    Values before index ``period-1`` are returned as ``None`` to indicate
    insufficient lookback.

    Args:
        prices: Price sequence (``PriceBar`` or raw close floats).
        period: EMA period.

    Returns:
        EMA series aligned to the input; insufficient prefix values are ``None``.
    """

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    closes = _extract_close_prices(prices)
    n = len(closes)
    if n < period:
        return [None] * n

    alpha = 2.0 / (period + 1)
    result: list[float | None] = [None] * n

    ema = closes[0]
    for i, price in enumerate(closes):
        if i == 0:
            ema = price
        else:
            ema = alpha * price + (1 - alpha) * ema
        if i >= period - 1:
            result[i] = ema

    return result


def compute_sma_last(prices: InputPrices, period: int) -> float | None:
    """Compute latest SMA value.

    Args:
        prices: Price sequence (``PriceBar`` or raw close floats).
        period: SMA period.

    Returns:
        Latest SMA value, or ``None`` when there is insufficient data.
    """

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    closes = _extract_close_prices(prices)
    if len(closes) < period:
        return None

    window = closes[-period:]
    return sum(window) / period


def compute_sma_series(prices: InputPrices, period: int) -> list[float | None]:
    """Compute SMA values for the entire series.

    Args:
        prices: Price sequence (``PriceBar`` or raw close floats).
        period: SMA period.

    Returns:
        SMA series aligned to the input; insufficient prefix values are ``None``.
    """

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    closes = _extract_close_prices(prices)
    n = len(closes)
    result: list[float | None] = [None] * n

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        result[i] = sum(window) / period

    return result


def compute_bbands_last(
    prices: InputPrices,
    period: int = 20,
    std_dev: float = 2.0,
) -> "BollingerBandsResult" | None:
    """Compute the latest Bollinger Bands result.

    Formula:
        middle = SMA(close, period)
        upper = middle + std_dev * stdev(close, period)
        lower = middle - std_dev * stdev(close, period)
        bandwidth = (upper - lower) / middle
        percent_b = (close - lower) / (upper - lower)
    """

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    closes = _extract_close_prices(prices)
    if len(closes) < period:
        return None

    from indicators.interface import BollingerBandsResult

    window = closes[-period:]
    middle = sum(window) / float(period)
    if middle == 0.0:
        return None

    mean = middle
    var = sum((x - mean) ** 2 for x in window) / float(period)
    std = math.sqrt(var)

    upper = mean + float(std_dev) * std
    lower = mean - float(std_dev) * std

    bandwidth = (upper - lower) / mean

    denom = upper - lower
    if denom == 0.0:
        percent_b = 0.5
    else:
        percent_b = (float(closes[-1]) - lower) / denom

    return BollingerBandsResult(
        upper=float(upper),
        middle=float(mean),
        lower=float(lower),
        bandwidth=float(bandwidth),
        percent_b=float(percent_b),
    )


def compute_bbands_series(
    prices: InputPrices,
    period: int = 20,
    std_dev: float = 2.0,
) -> list["BollingerBandsResult" | None]:
    """Compute Bollinger Bands results for the entire series."""

    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    closes = _extract_close_prices(prices)
    n = len(closes)
    result: list["BollingerBandsResult" | None] = [None] * n
    if n < period:
        return result

    from indicators.interface import BollingerBandsResult

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        middle = sum(window) / float(period)
        if middle == 0.0:
            result[i] = None
            continue

        mean = middle
        var = sum((x - mean) ** 2 for x in window) / float(period)
        std = math.sqrt(var)

        upper = mean + float(std_dev) * std
        lower = mean - float(std_dev) * std
        bandwidth = (upper - lower) / mean

        denom = upper - lower
        if denom == 0.0:
            percent_b = 0.5
        else:
            percent_b = (float(closes[i]) - lower) / denom

        result[i] = BollingerBandsResult(
            upper=float(upper),
            middle=float(mean),
            lower=float(lower),
            bandwidth=float(bandwidth),
            percent_b=float(percent_b),
        )

    return result


def compute_obv_last(bars: Sequence["PriceBar"]) -> float | None:
    """Compute latest OBV (On-Balance Volume).

    Algorithm:
        - close > prev_close: OBV += volume
        - close < prev_close: OBV -= volume
        - close == prev_close: unchanged
    """

    series = compute_obv_series(bars)
    if not series:
        return None
    return series[-1]


def compute_obv_series(bars: Sequence["PriceBar"]) -> list[float | None]:
    """Compute OBV series aligned to input bars."""

    if not bars:
        return []

    result: list[float | None] = [None] * len(bars)
    if len(bars) < 2:
        return result

    obv = 0.0
    for i in range(1, len(bars)):
        close = float(bars[i].close)
        prev_close = float(bars[i - 1].close)
        if not (math.isfinite(close) and math.isfinite(prev_close)):
            return [None] * len(bars)

        if close > prev_close:
            obv += float(bars[i].volume)
        elif close < prev_close:
            obv -= float(bars[i].volume)
        result[i] = float(obv)

    return result


def compute_vpt_last(bars: Sequence["PriceBar"]) -> float | None:
    """Compute latest VPT (Volume Price Trend)."""

    series = compute_vpt_series(bars)
    if not series:
        return None
    return series[-1]


def compute_vpt_series(bars: Sequence["PriceBar"]) -> list[float | None]:
    """Compute VPT series aligned to input bars.

    Algorithm:
        VPT += volume * (close - prev_close) / prev_close
    """

    if not bars:
        return []

    result: list[float | None] = [None] * len(bars)
    if len(bars) < 2:
        return result

    vpt = 0.0
    for i in range(1, len(bars)):
        close = float(bars[i].close)
        prev_close = float(bars[i - 1].close)
        if not (math.isfinite(close) and math.isfinite(prev_close)):
            return [None] * len(bars)
        if prev_close <= 0.0:
            # Price should not be <= 0.0; treat as invalid input.
            return result

        vpt += float(bars[i].volume) * ((close - prev_close) / prev_close)
        result[i] = float(vpt)

    return result


def _trend_label(start: float, end: float, *, eps: float = 1e-12) -> str:
    delta = float(end) - float(start)
    tol = float(eps) * max(1.0, abs(float(start)), abs(float(end)))
    if abs(delta) <= tol:
        return "flat"
    return "up" if delta > 0 else "down"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_volume_price_divergence(
    bars: Sequence["PriceBar"],
    lookback: int = 20,
) -> "VolumePriceDivergence" | None:
    """Detect volume-price divergence using price extremes vs OBV extremes.

    Rules:
        - Price new high + OBV not new high => bearish divergence
        - Price new low + OBV not new low => bullish divergence
    """

    if lookback <= 1:
        raise ValueError(f"lookback must be > 1, got {lookback}")

    if len(bars) < lookback or len(bars) < 2:
        return None

    from indicators.interface import VolumePriceDivergence

    closes = [float(bar.close) for bar in bars[-lookback:]]
    if any(not math.isfinite(c) for c in closes):
        return None

    obv_series = compute_obv_series(bars)
    if not obv_series:
        return None

    obv_window = obv_series[-lookback:]
    obv_last = obv_window[-1]
    if obv_last is None or not math.isfinite(float(obv_last)):
        return None

    prev_closes = closes[:-1]
    prev_max_price = max(prev_closes)
    prev_min_price = min(prev_closes)
    close_last = closes[-1]

    price_new_high = close_last > prev_max_price
    price_new_low = close_last < prev_min_price

    prev_obv_values = [float(v) for v in obv_window[:-1] if v is not None and math.isfinite(float(v))]
    if not prev_obv_values:
        return None

    prev_max_obv = max(prev_obv_values)
    prev_min_obv = min(prev_obv_values)
    obv_last_f = float(obv_last)

    obv_new_high = obv_last_f > prev_max_obv
    obv_new_low = obv_last_f < prev_min_obv

    if price_new_high and not obv_new_high:
        divergence_type = "bearish"
        price_excess = (close_last - prev_max_price) / max(abs(prev_max_price), 1e-12)
        obv_gap = (prev_max_obv - obv_last_f) / max(max(abs(prev_max_obv), abs(obv_last_f)), 1.0)
    elif price_new_low and not obv_new_low:
        divergence_type = "bullish"
        price_excess = (prev_min_price - close_last) / max(abs(prev_min_price), 1e-12)
        obv_gap = (obv_last_f - prev_min_obv) / max(max(abs(prev_min_obv), abs(obv_last_f)), 1.0)
    else:
        divergence_type = "none"
        price_excess = 0.0
        obv_gap = 0.0

    # Strength is a bounded blend of price "excess" and OBV gap.
    if divergence_type == "none":
        strength = 0.0
    else:
        price_component = 1.0 - math.exp(-max(0.0, float(price_excess)) / 0.02)  # ~2% scale
        obv_component = 1.0 - math.exp(-max(0.0, float(obv_gap)) / 0.10)  # relative gap scale
        strength = _clamp01(0.5 * price_component + 0.5 * obv_component)

    price_trend = _trend_label(closes[0], closes[-1])

    obv_start: float | None = None
    for v in obv_window:
        if v is None:
            continue
        obv_start = float(v)
        break
    if obv_start is None:
        return None

    obv_trend = _trend_label(obv_start, obv_last_f)

    return VolumePriceDivergence(
        divergence_type=divergence_type,
        price_trend=price_trend,
        obv_trend=obv_trend,
        strength=float(strength),
    )


def _extract_hlc(prices: InputPrices) -> tuple[list[float], list[float], list[float]] | None:
    """Extract high, low, close prices from price bars.

    Returns:
        Tuple of (highs, lows, closes) lists, or None if input is not PriceBar.
    """
    if not prices:
        return None

    from data.interface import PriceBar

    if isinstance(prices[0], PriceBar):
        bars = cast(Sequence["PriceBar"], prices)
        highs = [float(bar.high) for bar in bars]
        lows = [float(bar.low) for bar in bars]
        closes = [float(bar.close) for bar in bars]
        return highs, lows, closes

    return None


def compute_kdj_last(
    prices: InputPrices,
    *,
    n_period: int = 9,
    k_period: int = 3,
    d_period: int = 3,
) -> "KDJResult" | None:
    """Compute latest KDJ indicator result.

    KDJ formula:
        RSV = (Close - Lowest_N) / (Highest_N - Lowest_N) * 100
        K = EMA(RSV, k_period)
        D = EMA(K, d_period)
        J = 3*K - 2*D

    Args:
        prices: Price sequence (PriceBar required for H/L/C data).
        n_period: Lookback period for highest high / lowest low (default 9).
        k_period: Smoothing period for K line (default 3).
        d_period: Smoothing period for D line (default 3).

    Returns:
        KDJResult with K, D, J values, or None if insufficient data.
    """
    if n_period <= 0 or k_period <= 0 or d_period <= 0:
        raise ValueError(f"All periods must be > 0, got n={n_period}, k={k_period}, d={d_period}")

    hlc = _extract_hlc(prices)
    if hlc is None:
        return None

    highs, lows, closes = hlc
    n = len(closes)

    # Need enough data for n_period + smoothing warmup
    min_required = n_period
    if n < min_required:
        return None

    from indicators.interface import KDJResult

    # EMA multipliers
    k_mult = 2.0 / (float(k_period) + 1.0)
    d_mult = 2.0 / (float(d_period) + 1.0)

    # Initialize K and D at 50 (neutral)
    k_val = 50.0
    d_val = 50.0

    # Process all data points to get the final KDJ values
    for i in range(n_period - 1, n):
        # Get N-period window
        high_n = max(highs[i - n_period + 1 : i + 1])
        low_n = min(lows[i - n_period + 1 : i + 1])
        close_i = closes[i]

        # Compute RSV
        denom = high_n - low_n
        if denom == 0.0 or not math.isfinite(denom):
            rsv = 50.0  # Neutral when price is flat
        else:
            rsv = (close_i - low_n) / denom * 100.0

        # Clamp RSV to [0, 100]
        rsv = max(0.0, min(100.0, rsv))

        # EMA smoothing for K
        k_val = k_mult * rsv + (1.0 - k_mult) * k_val

        # EMA smoothing for D
        d_val = d_mult * k_val + (1.0 - d_mult) * d_val

    # J = 3K - 2D
    j_val = 3.0 * k_val - 2.0 * d_val

    return KDJResult(k=float(k_val), d=float(d_val), j=float(j_val))


def compute_kdj_series(
    prices: InputPrices,
    *,
    n_period: int = 9,
    k_period: int = 3,
    d_period: int = 3,
) -> list["KDJResult" | None]:
    """Compute KDJ indicator series for the entire price sequence.

    Args:
        prices: Price sequence (PriceBar required for H/L/C data).
        n_period: Lookback period for highest high / lowest low (default 9).
        k_period: Smoothing period for K line (default 3).
        d_period: Smoothing period for D line (default 3).

    Returns:
        KDJ series aligned to the input; insufficient prefix values are None.
    """
    if n_period <= 0 or k_period <= 0 or d_period <= 0:
        raise ValueError(f"All periods must be > 0, got n={n_period}, k={k_period}, d={d_period}")

    hlc = _extract_hlc(prices)
    if hlc is None:
        return [None] * len(prices) if prices else []

    highs, lows, closes = hlc
    n = len(closes)
    result: list["KDJResult" | None] = [None] * n

    if n < n_period:
        return result

    from indicators.interface import KDJResult

    # EMA multipliers
    k_mult = 2.0 / (float(k_period) + 1.0)
    d_mult = 2.0 / (float(d_period) + 1.0)

    # Initialize K and D at 50 (neutral)
    k_val = 50.0
    d_val = 50.0

    for i in range(n_period - 1, n):
        # Get N-period window
        high_n = max(highs[i - n_period + 1 : i + 1])
        low_n = min(lows[i - n_period + 1 : i + 1])
        close_i = closes[i]

        # Compute RSV
        denom = high_n - low_n
        if denom == 0.0 or not math.isfinite(denom):
            rsv = 50.0  # Neutral when price is flat
        else:
            rsv = (close_i - low_n) / denom * 100.0

        # Clamp RSV to [0, 100]
        rsv = max(0.0, min(100.0, rsv))

        # EMA smoothing for K
        k_val = k_mult * rsv + (1.0 - k_mult) * k_val

        # EMA smoothing for D
        d_val = d_mult * k_val + (1.0 - d_mult) * d_val

        # J = 3K - 2D
        j_val = 3.0 * k_val - 2.0 * d_val

        result[i] = KDJResult(k=float(k_val), d=float(d_val), j=float(j_val))

    return result


__all__ = [
    "compute_atr_last",
    "compute_rsi_last",
    "compute_rsi_series",
    "compute_macd_last",
    "compute_macd_series",
    "compute_ema_last",
    "compute_ema_series",
    "compute_sma_last",
    "compute_sma_series",
    "compute_bbands_last",
    "compute_bbands_series",
    "compute_obv_last",
    "compute_obv_series",
    "compute_vpt_last",
    "compute_vpt_series",
    "compute_volume_price_divergence",
]
