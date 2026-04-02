"""Market regime detection based on broad market ETFs (e.g., SPY/QQQ)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import structlog

from common.interface import Result

from .interface import MarketRegime, RegimeDetectionResult
from .regime_tracker import RegimeTransitionTracker

try:  # Prefer the real data layer interfaces when available.
    from data.interface import PriceBar, PriceSeriesSnapshot
    from data.vix_provider import VIXData, VIXProvider
except Exception:  # pragma: no cover
    PriceBar = Any  # type: ignore[assignment]
    PriceSeriesSnapshot = Any  # type: ignore[assignment]
    VIXData = Any  # type: ignore[assignment]
    VIXProvider = Any  # type: ignore[assignment]

__all__ = ["MarketRegimeDetector"]


def _extract_bars(snapshot: PriceSeriesSnapshot) -> list[PriceBar]:
    try:
        return list(snapshot.bars)
    except Exception:  # noqa: BLE001
        return []


def _get_float_attr(bar: Any, attr: str) -> float:
    value = getattr(bar, attr, float("nan"))
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


def _get_int_attr(bar: Any, attr: str) -> int:
    value = getattr(bar, attr, 0)
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return 0


def _pct_changes(values: Sequence[float]) -> list[float]:
    changes: list[float] = []
    for prev, cur in zip(values, values[1:]):
        if prev == 0:
            changes.append(float("nan"))
        else:
            changes.append((cur - prev) / prev)
    return changes


def _std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    mean = sum(values) / float(len(values))
    var = sum((x - mean) ** 2 for x in values) / float(len(values))
    return math.sqrt(var)


def _compute_volatility_last(closes: Sequence[float], lookback: int = 20) -> float | None:
    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    if len(closes) < lookback + 1:
        return None
    window = list(closes[-(lookback + 1) :])
    returns = _pct_changes(window)
    finite = [r for r in returns if math.isfinite(r)]
    if len(finite) < lookback:
        return None
    vol = _std(finite)
    return float(vol) if math.isfinite(vol) else None


def _compute_ma50_slope(closes: Sequence[float], slope_days: int = 10) -> float | None:
    ma_period = 50
    if slope_days <= 0:
        raise ValueError("slope_days must be > 0")
    required = ma_period + slope_days
    if len(closes) < required:
        return None

    end_ma = sum(closes[-ma_period:]) / float(ma_period)
    start_ma = sum(closes[-(ma_period + slope_days) : -slope_days]) / float(ma_period)
    if start_ma == 0 or not math.isfinite(start_ma) or not math.isfinite(end_ma):
        return None
    slope = (end_ma - start_ma) / start_ma
    return float(slope) if math.isfinite(slope) else None


def _compute_adx_last(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    n = len(closes)
    if n < period + 2:
        return None

    tr: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        if not (math.isfinite(high) and math.isfinite(low) and math.isfinite(prev_close)):
            return None

        true_range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        tr.append(float(true_range))

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus = up_move if up_move > down_move and up_move > 0 else 0.0
        minus = down_move if down_move > up_move and down_move > 0 else 0.0
        plus_dm.append(float(plus))
        minus_dm.append(float(minus))

    # Wilder smoothing
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


def _compute_daily_change(bars: list[Any]) -> float | None:
    """Compute the most recent daily percentage change from OHLC bars."""
    if len(bars) < 2:
        return None
    prev_close = _get_float_attr(bars[-2], "close")
    curr_close = _get_float_attr(bars[-1], "close")
    if not math.isfinite(prev_close) or not math.isfinite(curr_close) or prev_close == 0:
        return None
    return (curr_close - prev_close) / prev_close


@dataclass(frozen=True, slots=True)
class MarketRegimeDetector:
    """Detect broad market regime using trend, direction, stability, and VIX signals."""

    index_symbols: tuple[str, ...] = ("SPY", "QQQ")
    adx_period: int = 14
    stable_volatility_threshold: float = 0.02
    adx_trend_threshold: float = 28.0
    adx_choppy_threshold: float = 18.0
    ma_slope_days: int = 10
    ma_slope_bull_threshold: float = 0.01
    ma_slope_bear_threshold: float = -0.01

    tracker: RegimeTransitionTracker | None = None

    # Real VIX configuration (preferred)
    vix_enabled: bool = False
    vix_extreme_threshold: float = 35.0
    vix_elevated_threshold: float = 25.0
    vix_spike_threshold: float = 0.20  # 20% daily increase triggers spike
    vix_provider: Any | None = None

    # VIX proxy configuration
    vix_proxy_symbols: tuple[str, ...] = ("VIXY", "VXX", "UVXY")
    vix_spike_forces_choppy: bool = True  # VIX spike overrides to choppy regime

    def __post_init__(self) -> None:
        if float(self.ma_slope_bear_threshold) >= float(self.ma_slope_bull_threshold):
            raise ValueError("ma_slope_bear_threshold must be < ma_slope_bull_threshold")

    def detect(self, market_data: dict[str, PriceSeriesSnapshot]) -> Result[RegimeDetectionResult]:
        """Detect the current regime from provided market index snapshots."""

        logger = structlog.get_logger(__name__).bind(module="market_regime.detector")

        symbol = self._select_symbol(market_data)
        if symbol is None:
            return Result.failed(ValueError("No market index data provided"), "NO_MARKET_INDEX_DATA")

        snapshot = market_data[symbol]
        bars = _extract_bars(snapshot)
        if len(bars) < (self.adx_period + 2):
            return Result.failed(ValueError(f"Need more bars for ADX, got {len(bars)}"), "INSUFFICIENT_BARS")

        highs = [_get_float_attr(bar, "high") for bar in bars]
        lows = [_get_float_attr(bar, "low") for bar in bars]
        closes = [_get_float_attr(bar, "close") for bar in bars]
        if any(not math.isfinite(v) for v in highs + lows + closes):
            return Result.failed(ValueError("Non-finite OHLC values"), "NONFINITE_OHLC")

        adx = _compute_adx_last(highs, lows, closes, period=self.adx_period)
        if adx is None:
            return Result.failed(ValueError("ADX computation failed"), "ADX_FAILED")

        ma50_slope = _compute_ma50_slope(closes, slope_days=self.ma_slope_days)
        volatility = _compute_volatility_last(closes, lookback=20)

        is_trending = bool(adx >= self.adx_trend_threshold)
        is_choppy = bool(adx <= self.adx_choppy_threshold)
        is_stable = bool(volatility is not None and volatility <= self.stable_volatility_threshold)

        # Real VIX detection (preferred, optional)
        vix_data: VIXData | None = None
        vix_spike_real = False
        if self.vix_enabled:
            try:
                provider = self.vix_provider if self.vix_provider is not None else VIXProvider()
                vix_data = provider.fetch_vix_current()
                if vix_data is None:
                    logger.warning("vix.unavailable")
                else:
                    vix_spike_real = bool(vix_data.change_pct >= self.vix_spike_threshold)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vix.fetch_failed", error=str(exc)[:200])
                vix_data = None
                vix_spike_real = False

        # VIX proxy detection (auxiliary signal; also used as a soft fallback)
        vix_symbol, vix_change, vix_spike_proxy = self._detect_vix_signal(market_data)
        vix_spike = vix_spike_real if vix_data is not None else vix_spike_proxy

        # Determine baseline regime first (price-only signals)
        # Priority: MA slope (trend direction) > ADX (trend strength)
        # This allows "narrow bull markets" (low ADX but clear uptrend) to be recognized
        detected_regime = MarketRegime.UNKNOWN
        if ma50_slope is not None and ma50_slope > self.ma_slope_bull_threshold:
            # MA slope strongly positive → BULL even if ADX is low (narrow bull market)
            detected_regime = MarketRegime.BULL
        elif ma50_slope is not None and ma50_slope < self.ma_slope_bear_threshold:
            # MA slope strongly negative → BEAR
            detected_regime = MarketRegime.BEAR
        elif is_choppy or (not is_trending and not is_stable):
            # ADX low + MA slope flat → true CHOP
            detected_regime = MarketRegime.CHOPPY
        elif ma50_slope is None:
            detected_regime = MarketRegime.UNKNOWN
        else:
            # MA slope in neutral zone, ADX moderate
            detected_regime = MarketRegime.CHOPPY

        regime = detected_regime

        # Apply VIX adjustments (real VIX takes precedence when available).
        if vix_data is not None:
            if vix_data.value > self.vix_extreme_threshold:
                regime = MarketRegime.BEAR
            elif vix_spike and self.vix_spike_forces_choppy:
                regime = MarketRegime.CHOPPY
            elif vix_data.value > self.vix_elevated_threshold and regime in (MarketRegime.BULL, MarketRegime.UNKNOWN):
                regime = MarketRegime.CHOPPY
        elif vix_spike and self.vix_spike_forces_choppy:
            regime = MarketRegime.CHOPPY

        details = {
            "close_last": closes[-1],
            "volume_last": _get_int_attr(bars[-1], "volume"),
            "adx_thresholds": {
                "trend": self.adx_trend_threshold,
                "choppy": self.adx_choppy_threshold,
            },
            "stable_volatility_threshold": self.stable_volatility_threshold,
            "ma_slope_thresholds": {
                "bull": self.ma_slope_bull_threshold,
                "bear": self.ma_slope_bear_threshold,
            },
            "vix_enabled": self.vix_enabled,
            "vix_extreme_threshold": self.vix_extreme_threshold,
            "vix_elevated_threshold": self.vix_elevated_threshold,
            "vix_spike_threshold": self.vix_spike_threshold,
            "vix_spike_forces_choppy": self.vix_spike_forces_choppy,
        }

        if self.tracker is not None:
            confirmed_regime = self.tracker.update(regime)
            details["detected_regime"] = regime.value
            details["confirmed_regime"] = confirmed_regime.value
            details["confirmation"] = {
                "confirmation_days": int(self.tracker.config.confirmation_days),
                "reset_on_opposite": bool(self.tracker.config.reset_on_opposite),
                "pending_regime": self.tracker.pending_regime.value if self.tracker.pending_regime is not None else None,
                "pending_days": int(self.tracker.pending_days),
            }
            regime = confirmed_regime

        return Result.success(
            RegimeDetectionResult(
                regime=regime,
                market_symbol=symbol,
                adx=float(adx),
                ma50_slope=float(ma50_slope) if ma50_slope is not None else None,
                volatility=float(volatility) if volatility is not None else None,
                is_trending=is_trending,
                is_stable=is_stable,
                vix_data=vix_data,
                vix_proxy_symbol=vix_symbol,
                vix_proxy_change=vix_change,
                vix_spike_detected=vix_spike,
                details=details,
            )
        )

    def _detect_vix_signal(
        self, market_data: dict[str, PriceSeriesSnapshot]
    ) -> tuple[str | None, float | None, bool]:
        """Detect VIX proxy signal from available ETF data.

        Returns:
            Tuple of (vix_symbol, daily_change, spike_detected)
        """
        for symbol in self.vix_proxy_symbols:
            if symbol not in market_data:
                continue
            snapshot = market_data[symbol]
            bars = _extract_bars(snapshot)
            if len(bars) < 2:
                # VIX proxy data unavailable or insufficient
                continue
            daily_change = _compute_daily_change(bars)
            if daily_change is None:
                continue
            spike_detected = daily_change >= self.vix_spike_threshold
            return symbol, daily_change, spike_detected
        return None, None, False

    def _select_symbol(self, market_data: dict[str, PriceSeriesSnapshot]) -> str | None:
        for symbol in self.index_symbols:
            if symbol in market_data:
                return symbol
        if market_data:
            return next(iter(market_data.keys()))
        return None
