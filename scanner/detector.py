"""Core platform-period detection algorithms for the Phase 2.3 scanner.

This module implements the "Reference scanning result" consolidation/platform
pattern detector described in the task prompt. The detector operates on a list
of OHLCV bars and produces structured :class:`~scanner.interface.PlatformCandidate`
objects when a price/volume/structure platform is detected.

Design notes
------------
- All public functions return :class:`~common.interface.Result` to propagate
  success/failure/degraded outcomes across the scanning pipeline.
- The implementation is intentionally dependency-light: numpy is used for
  vectorised numeric operations; scipy is not required.
- The algorithms are "window-last": each detector evaluates the most recent
  ``window`` bars from the supplied series.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from typing import Any

import msgspec
import numpy as np

from common.interface import Result, ResultStatus
from scanner.market_regime.confirmation import BreakthroughConfirmation, BreakthroughConfirmationConfig
from scanner.reversal_detector import ReversalCandidate, ReversalConfig, ReversalDetector

_logger = logging.getLogger(__name__)

try:  # Prefer the real data layer interfaces when available.
    from data.interface import PriceBar, PriceSeriesSnapshot
except Exception:  # pragma: no cover - early phases may not ship the data layer yet.
    PriceBar = Any  # type: ignore[assignment]
    PriceSeriesSnapshot = Any  # type: ignore[assignment]

from .interface import PlatformCandidate, PlatformFeatures, ScannerConfig

__all__ = [
    "detect_price_platform",
    "detect_volume_platform",
    "calculate_atr",
    "detect_support_resistance",
    "detect_platform_candidate",
]


MA_PERIODS: tuple[int, ...] = (5, 10, 20, 30)
ATR_DEFAULT_PERIOD: int = 14
DEFAULT_TOLERANCE: float = 0.02
INVALIDATION_BUFFER: float = 0.99
TARGET_BUFFER: float = 1.01


def _snapshot_to_bars(snapshot: PriceSeriesSnapshot) -> list[PriceBar]:
    """Best-effort extraction of ``bars`` from a data-layer snapshot."""

    try:
        return list(snapshot.bars)
    except Exception:  # noqa: BLE001 - tolerate schema drift in early phases.
        return []


def _as_float_array(values: Iterable[Any], *, name: str) -> Result[np.ndarray]:
    """Convert an iterable to a finite float numpy array.

    Args:
        values: Input values (numbers or number-like).
        name: Field label used for error messaging.

    Returns:
        Result[np.ndarray]: Success with a 1D float array; failure if the data
        is empty, non-numeric, or contains non-finite values.
    """

    try:
        array = np.asarray(list(values), dtype=float)
    except Exception as exc:  # noqa: BLE001 - surface conversion error as Result.
        return Result.failed(exc, f"INVALID_{name.upper()}_ARRAY")

    if array.size == 0:
        return Result.failed(ValueError(f"{name} is empty"), f"EMPTY_{name.upper()}_ARRAY")

    if not np.all(np.isfinite(array)):
        return Result.failed(
            ValueError(f"{name} contains non-finite values"), f"NONFINITE_{name.upper()}"
        )

    return Result.success(array)


def _pct_change(values: np.ndarray) -> Result[np.ndarray]:
    """Compute percentage change series for an array of prices."""

    if values.size < 2:
        return Result.failed(ValueError("Need >=2 values for pct_change"), "INSUFFICIENT_VALUES")

    prev = values[:-1]
    if np.any(prev == 0):
        return Result.failed(ValueError("pct_change division by zero"), "ZERO_PREV_VALUE")

    return Result.success(np.diff(values) / prev)


def _safe_mean(values: np.ndarray, *, name: str) -> Result[float]:
    """Return mean(values) ensuring finiteness and non-zero when needed."""

    if values.size == 0:
        return Result.failed(ValueError(f"{name} is empty"), f"EMPTY_{name.upper()}")
    if not np.all(np.isfinite(values)):
        return Result.failed(
            ValueError(f"{name} contains non-finite values"), f"NONFINITE_{name.upper()}_MEAN"
        )

    count = float(values.size)
    mean_value = float(np.sum(values / count))
    if not math.isfinite(mean_value):
        return Result.failed(
            ValueError(f"{name} mean is not finite"), f"NONFINITE_{name.upper()}_MEAN"
        )
    return Result.success(mean_value)


def _coefficient_of_variation(values: np.ndarray, *, mean_value: float, name: str) -> Result[float]:
    """Compute coefficient of variation: std(values) / mean_value."""

    if mean_value == 0:
        return Result.failed(ValueError(f"{name} mean is zero"), f"ZERO_{name.upper()}_MEAN")
    scaled = values / mean_value
    std_value = float(np.std(scaled))
    if not math.isfinite(std_value):
        return Result.failed(
            ValueError(f"{name} std is not finite"), f"NONFINITE_{name.upper()}_STD"
        )
    return Result.success(std_value * math.copysign(1.0, mean_value))


def _window_slice(bars: Sequence[PriceBar], window: int) -> Result[list[PriceBar]]:
    """Return the last ``window`` bars with validation."""

    if window <= 1:
        return Result.failed(ValueError("window must be > 1"), "INVALID_WINDOW")
    if len(bars) < window:
        return Result.failed(
            ValueError(f"Need >= {window} bars, got {len(bars)}"),
            "INSUFFICIENT_BARS",
        )
    return Result.success(list(bars[-window:]))


def _extract_ohlcv(bars: Sequence[PriceBar]) -> Result[dict[str, np.ndarray]]:
    """Extract OHLCV arrays from a list of PriceBar objects.

    The detector expects the data layer schema to provide attributes:
    ``open``, ``high``, ``low``, ``close``, ``volume``.
    """

    open_result = _as_float_array((bar.open for bar in bars), name="open")
    if open_result.status is ResultStatus.FAILED:
        return Result.failed(
            open_result.error or ValueError("open extraction failed"), "OPEN_EXTRACTION_FAILED"
        )

    high_result = _as_float_array((bar.high for bar in bars), name="high")
    if high_result.status is ResultStatus.FAILED:
        return Result.failed(
            high_result.error or ValueError("high extraction failed"), "HIGH_EXTRACTION_FAILED"
        )

    low_result = _as_float_array((bar.low for bar in bars), name="low")
    if low_result.status is ResultStatus.FAILED:
        return Result.failed(
            low_result.error or ValueError("low extraction failed"), "LOW_EXTRACTION_FAILED"
        )

    close_result = _as_float_array((bar.close for bar in bars), name="close")
    if close_result.status is ResultStatus.FAILED:
        return Result.failed(
            close_result.error or ValueError("close extraction failed"), "CLOSE_EXTRACTION_FAILED"
        )

    volume_result = _as_float_array((bar.volume for bar in bars), name="volume")
    if volume_result.status is ResultStatus.FAILED:
        return Result.failed(
            volume_result.error or ValueError("volume extraction failed"),
            "VOLUME_EXTRACTION_FAILED",
        )

    return Result.success(
        {
            "open": open_result.data,
            "high": high_result.data,
            "low": low_result.data,
            "close": close_result.data,
            "volume": volume_result.data,
        }
    )


def _compute_price_platform_features(
    window_bars: Sequence[PriceBar],
    config: ScannerConfig,
) -> Result[tuple[bool, dict[str, Any]]]:
    ohlcv_result = _extract_ohlcv(window_bars)
    if ohlcv_result.status is ResultStatus.FAILED:
        return Result.failed(
            ohlcv_result.error or ValueError("OHLCV extraction failed"), "PRICE_OHLCV_FAILED"
        )
    high = ohlcv_result.data["high"]
    low = ohlcv_result.data["low"]
    close = ohlcv_result.data["close"]

    box_low = float(np.min(low))
    if box_low <= 0 or not math.isfinite(box_low):
        return Result.failed(ValueError("box_low must be positive"), "INVALID_BOX_LOW")
    box_high = float(np.max(high))
    if not math.isfinite(box_high) or box_high <= 0:
        return Result.failed(ValueError("box_high must be positive"), "INVALID_BOX_HIGH")

    box_range = (box_high - box_low) / box_low
    if not math.isfinite(box_range) or box_range < 0:
        return Result.failed(ValueError("box_range invalid"), "INVALID_BOX_RANGE")

    available_periods = [p for p in MA_PERIODS if p <= close.size]
    ma_values = {str(period): float(np.mean(close[-period:])) for period in available_periods}
    mas = np.asarray(list(ma_values.values()), dtype=float)
    ma_mean_result = _safe_mean(mas, name="moving_averages")
    if ma_mean_result.status is ResultStatus.FAILED:
        return Result.failed(ma_mean_result.error or ValueError("MA mean failed"), "MA_MEAN_FAILED")

    ma_diff_result = _coefficient_of_variation(
        mas,
        mean_value=ma_mean_result.data,
        name="moving_averages",
    )
    if ma_diff_result.status is ResultStatus.FAILED:
        return Result.failed(ma_diff_result.error or ValueError("MA diff failed"), "MA_DIFF_FAILED")

    returns_result = _pct_change(close)
    if returns_result.status is ResultStatus.FAILED:
        return Result.failed(
            returns_result.error or ValueError("returns failed"), "RETURN_SERIES_FAILED"
        )
    volatility = float(np.std(returns_result.data))
    if not math.isfinite(volatility) or volatility < 0:
        return Result.failed(ValueError("volatility invalid"), "INVALID_VOLATILITY")

    is_platform = (
        box_range <= config.box_threshold
        and ma_diff_result.data <= config.ma_diff_threshold
        and volatility <= config.volatility_threshold
    )

    platform = bool(is_platform)
    return Result.success(
        (
            platform,
            {
                "box_range": float(box_range),
                "box_low": box_low,
                "box_high": box_high,
                "ma_diff": float(ma_diff_result.data),
                "volatility": float(volatility),
                "ma_values": ma_values,
            },
        ),
        reason_code="OK" if platform else "PRICE_RULES_NOT_MET",
    )


def detect_price_platform(
    bars: list[PriceBar],
    window: int,
    config: ScannerConfig,
) -> Result[tuple[bool, dict[str, Any]]]:
    """Detect price-based platform characteristics over a rolling window.

    The "platform period" price conditions follow the reference algorithm:
    1) The high-low range (box height) is small relative to price.
    2) Multiple moving averages (5/10/20/30) converge within the window.
    3) Return volatility remains low.

    Args:
        bars: Full price series ordered oldest -> newest.
        window: Lookback window (in bars) evaluated at the end of ``bars``.
        config: Threshold configuration controlling platform qualification.

    Returns:
        Result[(is_platform, features)] where:
        - is_platform: True if all three price conditions are satisfied.
        - features: Dict with:
            box_range: (box_high - box_low) / box_low
            box_low: min(low) within window
            box_high: max(high) within window
            ma_diff: std(MAs)/mean(MAs) for available MA periods
            volatility: std(pct_change(close)) within window
    """

    slice_result = _window_slice(bars, window)
    if slice_result.status is ResultStatus.FAILED:
        return Result.failed(
            slice_result.error or ValueError("window slice failed"), "INSUFFICIENT_PRICE_BARS"
        )
    return _compute_price_platform_features(slice_result.data, config)


def detect_volume_platform(
    bars: list[PriceBar],
    window: int,
    config: ScannerConfig,
) -> Result[tuple[bool, dict[str, float]]]:
    """Detect volume-based platform characteristics using a two-window compare.

    The reference algorithm compares the most recent window to the prior window
    of equal length to estimate volume contraction and stability.

    Args:
        bars: Full price series ordered oldest -> newest.
        window: Lookback window (in bars). Requires at least ``2*window`` bars.
        config: Threshold configuration for contraction and stability.

    Returns:
        Result[(is_volume_ok, features)] where features contains:
        - volume_change_ratio: recent_avg / previous_avg
        - volume_stability: std(recent_volume) / recent_avg (coefficient of variation)
        - volume_stability_robust: MAD(recent_volume) / median(recent_volume)
        - volume_trend: (recent_avg - previous_avg) / previous_avg
        - trend_score: Higher when volume is contracting
        - volume_quality: Composite score combining stability + trend
        - avg_dollar_volume: mean(recent_close * recent_volume)
    """

    required = 2 * window
    if window <= 1:
        return Result.failed(ValueError("window must be > 1"), "INVALID_WINDOW")
    if len(bars) < required:
        return Result.failed(
            ValueError(f"Need >= {required} bars, got {len(bars)}"),
            "INSUFFICIENT_VOLUME_BARS",
        )

    recent = list(bars[-window:])
    previous = list(bars[-required:-window])
    return _compute_volume_platform_features(recent, previous, config)


def _compute_volume_platform_features(
    recent_bars: Sequence[PriceBar],
    previous_bars: Sequence[PriceBar],
    config: ScannerConfig,
) -> Result[tuple[bool, dict[str, float]]]:
    recent_ohlcv = _extract_ohlcv(recent_bars)
    if recent_ohlcv.status is ResultStatus.FAILED:
        return Result.failed(
            recent_ohlcv.error or ValueError("recent OHLCV failed"), "RECENT_OHLCV_FAILED"
        )
    prev_ohlcv = _extract_ohlcv(previous_bars)
    if prev_ohlcv.status is ResultStatus.FAILED:
        return Result.failed(
            prev_ohlcv.error or ValueError("previous OHLCV failed"), "PREVIOUS_OHLCV_FAILED"
        )

    recent_volume = recent_ohlcv.data["volume"]
    prev_volume = prev_ohlcv.data["volume"]
    recent_close = recent_ohlcv.data["close"]

    recent_avg_result = _safe_mean(recent_volume, name="recent_volume")
    if recent_avg_result.status is ResultStatus.FAILED:
        return Result.failed(
            recent_avg_result.error or ValueError("recent volume mean failed"), "RECENT_MEAN_FAILED"
        )
    recent_avg = recent_avg_result.data

    prev_avg_result = _safe_mean(prev_volume, name="previous_volume")
    if prev_avg_result.status is ResultStatus.FAILED:
        return Result.failed(
            prev_avg_result.error or ValueError("previous volume mean failed"),
            "PREVIOUS_MEAN_FAILED",
        )
    prev_avg = prev_avg_result.data
    if prev_avg == 0:
        return Result.failed(ValueError("previous_avg volume is zero"), "ZERO_PREVIOUS_VOLUME_MEAN")

    volume_change_ratio = float(recent_avg / prev_avg)
    if not math.isfinite(volume_change_ratio) or volume_change_ratio < 0:
        return Result.failed(ValueError("volume_change_ratio invalid"), "INVALID_VOLUME_RATIO")

    stability_result = _coefficient_of_variation(
        recent_volume,
        mean_value=recent_avg,
        name="volume",
    )
    if stability_result.status is ResultStatus.FAILED:
        return Result.failed(
            stability_result.error or ValueError("volume stability failed"),
            "VOLUME_STABILITY_FAILED",
        )
    volume_stability = stability_result.data

    median_vol = float(np.median(recent_volume))
    if median_vol > 0:
        mad = float(np.median(np.abs(recent_volume - median_vol)))
        volume_stability_robust = mad / median_vol
    else:
        volume_stability_robust = float("inf")

    volume_trend = float((recent_avg - prev_avg) / prev_avg)
    trend_score = float(max(0.0, min(1.0, 0.5 - volume_trend)))

    stability_score = float(max(0.0, min(1.0, 1.0 - volume_stability / 0.5)))
    volume_quality = float(0.6 * stability_score + 0.4 * trend_score)

    dollar_volume = recent_close * recent_volume
    avg_dollar_volume_result = _safe_mean(dollar_volume, name="dollar_volume")
    if avg_dollar_volume_result.status is ResultStatus.FAILED:
        return Result.failed(
            avg_dollar_volume_result.error or ValueError("avg dollar volume failed"),
            "DOLLAR_VOLUME_FAILED",
        )
    avg_dollar_volume = avg_dollar_volume_result.data

    is_volume_ok = (
        volume_change_ratio <= config.volume_change_threshold
        and volume_stability <= config.volume_stability_threshold
    )

    volume_ok = bool(is_volume_ok)
    return Result.success(
        (
            volume_ok,
            {
                "volume_change_ratio": float(volume_change_ratio),
                "volume_stability": float(volume_stability),
                "volume_stability_robust": float(volume_stability_robust),
                "volume_trend": float(volume_trend),
                "trend_score": float(trend_score),
                "volume_quality": float(volume_quality),
                "avg_dollar_volume": float(avg_dollar_volume),
            },
        )
        ,
        reason_code="OK" if volume_ok else "VOLUME_RULES_NOT_MET",
    )


def calculate_atr(bars: list[PriceBar], period: int = ATR_DEFAULT_PERIOD) -> Result[float]:
    """Calculate ATR (Average True Range) over the most recent period.

    ATR is computed as the simple mean of True Range (TR) over ``period`` bars.

    TR for bar i is defined as::

        TR_i = max(
            high_i - low_i,
            abs(high_i - close_{i-1}),
            abs(low_i  - close_{i-1}),
        )

    Args:
        bars: Full price series ordered oldest -> newest.
        period: ATR lookback length (default: 14). Requires at least ``period+1`` bars.

    Returns:
        Result[float]: ATR value (same units as price).
    """

    if period <= 1:
        return Result.failed(ValueError("period must be > 1"), "INVALID_ATR_PERIOD")
    if len(bars) < period + 1:
        return Result.failed(
            ValueError(f"Need >= {period + 1} bars for ATR, got {len(bars)}"),
            "INSUFFICIENT_ATR_BARS",
        )

    window = list(bars[-(period + 1) :])
    ohlcv_result = _extract_ohlcv(window)
    if ohlcv_result.status is ResultStatus.FAILED:
        return Result.failed(
            ohlcv_result.error or ValueError("OHLCV extraction failed"), "ATR_OHLCV_FAILED"
        )
    high = ohlcv_result.data["high"]
    low = ohlcv_result.data["low"]
    close = ohlcv_result.data["close"]

    prev_close = close[:-1]
    curr_high = high[1:]
    curr_low = low[1:]

    tr1 = curr_high - curr_low
    tr2 = np.abs(curr_high - prev_close)
    tr3 = np.abs(curr_low - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = float(np.mean(true_range))
    if not math.isfinite(atr) or atr <= 0:
        return Result.failed(ValueError("ATR is invalid"), "INVALID_ATR")
    return Result.success(atr)


def detect_support_resistance(
    bars: list[PriceBar], tolerance: float = DEFAULT_TOLERANCE
) -> Result[dict[str, float | None]]:
    """Detect support/resistance levels and a simplified box quality score.

    This is a simplified, dependency-light approximation suitable for scanner
    ranking and invalidation/target calculations.

    Algorithm (simplified per prompt):
    1) Compute raw extremes: support ~= min(low), resistance ~= max(high).
    2) "Cluster" touch points within ``tolerance`` of each level.
    3) Compute a box_quality score (0~1) combining:
       - touch intensity at support/resistance
       - containment ratio of closes staying within the box

    Args:
        bars: Window bars ordered oldest -> newest.
        tolerance: Relative tolerance for a "touch" (e.g., 0.02 => ±2%).

    Returns:
        Result[dict] with keys:
            support_level: float | None
            resistance_level: float | None
            box_quality: float | None  (0~1)
    """

    if tolerance <= 0 or tolerance >= 0.2:
        return Result.failed(ValueError("tolerance must be within (0, 0.2)"), "INVALID_TOLERANCE")
    if len(bars) < 5:
        return Result.failed(ValueError("Need >=5 bars for box structure"), "INSUFFICIENT_BOX_BARS")

    ohlcv_result = _extract_ohlcv(bars)
    if ohlcv_result.status is ResultStatus.FAILED:
        return Result.failed(
            ohlcv_result.error or ValueError("OHLCV extraction failed"), "BOX_OHLCV_FAILED"
        )

    high = ohlcv_result.data["high"]
    low = ohlcv_result.data["low"]
    close = ohlcv_result.data["close"]

    support = float(np.min(low))
    resistance = float(np.max(high))
    if not (math.isfinite(support) and math.isfinite(resistance)) or support <= 0:
        return Result.failed(ValueError("support/resistance invalid"), "INVALID_SUPPORT_RESISTANCE")
    if resistance <= support:
        return Result.failed(ValueError("resistance must exceed support"), "INVALID_BOX_LEVELS")

    support_band = support * tolerance
    resistance_band = resistance * tolerance
    support_touches = int(np.sum(np.abs(low - support) <= support_band))
    resistance_touches = int(np.sum(np.abs(high - resistance) <= resistance_band))

    upper = resistance * (1 + tolerance)
    lower = support * (1 - tolerance)
    containment = float(np.mean((close >= lower) & (close <= upper)))

    support_score = min(1.0, support_touches / 3.0)
    resistance_score = min(1.0, resistance_touches / 3.0)
    touch_score = 0.5 * support_score + 0.5 * resistance_score

    # Box tightness: narrower box (relative to support) scores higher.
    # A box_range of 6% of support price maps to tightness=1.0; 20% maps to ~0.3.
    box_range_pct = (resistance - support) / support
    # Normalize: 0% range -> 1.0, 20% range -> 0.0, clamp to [0, 1]
    box_tightness = float(max(0.0, min(1.0, 1.0 - box_range_pct / 0.20)))

    # Distribution quality: prefer closes centered between support and resistance.
    box_range = resistance - support
    if box_range > 0:
        mean_position = float(np.mean((close - support) / box_range))
        distribution_score = float(max(0.0, 1.0 - abs(mean_position - 0.5) * 2))
    else:
        distribution_score = 0.0

    box_quality = (
        0.25 * containment
        + 0.35 * touch_score
        + 0.25 * box_tightness
        + 0.15 * distribution_score
    )
    box_quality = float(max(0.0, min(1.0, box_quality)))

    return Result.success(
        {
            "support_level": support,
            "resistance_level": resistance,
            "box_quality": box_quality,
            "touch_score": float(touch_score),
            "support_score": float(support_score),
            "resistance_score": float(resistance_score),
            "containment": float(containment),
            "box_tightness": float(box_tightness),
            "box_range_pct": float(box_range_pct),
            "distribution_score": float(distribution_score),
        }
    )


def _atr_pct_for_series(bars: Sequence[PriceBar], config: ScannerConfig) -> Result[float | None]:
    last_close = float(bars[-1].close)
    if not math.isfinite(last_close) or last_close <= 0:
        return Result.failed(ValueError("last close invalid"), "INVALID_LAST_CLOSE")

    atr_result = calculate_atr(list(bars), period=ATR_DEFAULT_PERIOD)
    if atr_result.status is ResultStatus.FAILED:
        return Result.failed(atr_result.error or ValueError("ATR failed"), "ATR_FAILED")

    atr_pct = float(atr_result.data / last_close)
    if not math.isfinite(atr_pct) or atr_pct <= 0:
        return Result.failed(ValueError("atr_pct invalid"), "INVALID_ATR_PCT")

    if atr_pct < config.min_atr_pct or atr_pct > config.max_atr_pct:
        return Result.success(None, reason_code="ATR_FILTERED")

    return Result.success(atr_pct)


def _box_structure_for_window(
    window_bars: Sequence[PriceBar],
    config: ScannerConfig,
    *,
    box_low_fallback: float,
    box_high_fallback: float,
) -> Result[tuple[float, float, float] | None]:
    box_result = detect_support_resistance(list(window_bars), tolerance=DEFAULT_TOLERANCE)
    if box_result.status is ResultStatus.FAILED:
        return Result.failed(
            box_result.error or ValueError("box structure failed"), "BOX_STRUCTURE_FAILED"
        )

    box = box_result.data
    support = float(box.get("support_level") or box_low_fallback)
    resistance = float(box.get("resistance_level") or box_high_fallback)
    box_quality = float(box.get("box_quality") or 0.0)

    if support <= 0 or resistance <= 0 or resistance <= support:
        return Result.failed(ValueError("support/resistance invalid"), "INVALID_BOX_LEVELS")
    if box_quality < config.min_box_quality:
        return Result.success(None, reason_code="BOX_QUALITY_FILTERED")

    return Result.success((support, resistance, float(max(0.0, min(1.0, box_quality)))))


def _target_and_invalidation(support: float, resistance: float) -> Result[tuple[float, float]]:
    invalidation_level = float(support * INVALIDATION_BUFFER)
    if not math.isfinite(invalidation_level) or invalidation_level <= 0:
        return Result.failed(ValueError("invalidation_level invalid"), "INVALID_INVALIDATION_LEVEL")

    measured_move = float(resistance + (resistance - support))
    buffer_target = float(resistance * TARGET_BUFFER)
    target_level = float(max(buffer_target, measured_move))
    if not math.isfinite(target_level) or target_level <= 0:
        return Result.failed(ValueError("target_level invalid"), "INVALID_TARGET_LEVEL")

    return Result.success((invalidation_level, target_level))


def _candidate_meta(config: ScannerConfig) -> dict[str, Any]:
    return {
        "thresholds": {
            "box_threshold": config.box_threshold,
            "ma_diff_threshold": config.ma_diff_threshold,
            "volatility_threshold": config.volatility_threshold,
            "volume_change_threshold": config.volume_change_threshold,
            "volume_stability_threshold": config.volume_stability_threshold,
            "min_box_quality": config.min_box_quality,
            "min_atr_pct": config.min_atr_pct,
            "max_atr_pct": config.max_atr_pct,
            "min_dollar_volume": config.min_dollar_volume,
        },
        "ma_periods": list(MA_PERIODS),
        "config_snapshot": msgspec.to_builtins(config),
    }


def _require_price_platform(
    bars: Sequence[PriceBar],
    window: int,
    config: ScannerConfig,
) -> Result[dict[str, float] | None]:
    price_result = detect_price_platform(list(bars), window, config)
    if price_result.status is ResultStatus.FAILED:
        return Result.failed(
            price_result.error or ValueError("price detection failed"), "PRICE_DETECTION_FAILED"
        )
    price_ok, price_features = price_result.data
    if not price_ok:
        return Result.success(None, reason_code="PRICE_RULES_NOT_MET")
    return Result.success(price_features)


def _require_volume_platform(
    bars: Sequence[PriceBar],
    window: int,
    config: ScannerConfig,
) -> Result[dict[str, float] | None]:
    volume_result = detect_volume_platform(list(bars), window, config)
    if volume_result.status is ResultStatus.FAILED:
        return Result.failed(
            volume_result.error or ValueError("volume detection failed"), "VOLUME_DETECTION_FAILED"
        )
    volume_ok, volume_features = volume_result.data
    if not volume_ok:
        return Result.success(None, reason_code="VOLUME_RULES_NOT_MET")
    return Result.success(volume_features)


def _build_platform_features(
    *,
    price_features: dict[str, float],
    volume_features: dict[str, float],
    atr_pct: float | None,
    support: float,
    resistance: float,
    box_quality: float,
) -> PlatformFeatures:
    return PlatformFeatures(
        box_range=float(price_features["box_range"]),
        box_low=float(price_features["box_low"]),
        box_high=float(price_features["box_high"]),
        ma_diff=float(price_features["ma_diff"]),
        volatility=float(price_features["volatility"]),
        atr_pct=float(atr_pct) if atr_pct is not None else None,
        volume_change_ratio=float(volume_features["volume_change_ratio"]),
        volume_stability=float(volume_features["volume_stability"]),
        avg_dollar_volume=float(volume_features["avg_dollar_volume"]),
        box_quality=float(box_quality),
        support_level=float(support),
        resistance_level=float(resistance),
    )


def _build_candidate(
    *,
    symbol: str,
    detected_at: int,
    window: int,
    score: float,
    features: PlatformFeatures,
    invalidation_level: float,
    target_level: float,
    config: ScannerConfig,
    extra_meta: dict[str, Any] | None = None,
) -> PlatformCandidate:
    return PlatformCandidate(
        symbol=symbol,
        detected_at=detected_at,
        window=window,
        score=score,
        features=features,
        invalidation_level=invalidation_level,
        target_level=target_level,
        reasons=["price_platform", "volume_platform", "atr_ok", "box_structure"],
        meta={**_candidate_meta(config), **dict(extra_meta or {})},
    )


class _CandidateInputs(msgspec.Struct, frozen=True, kw_only=True):
    price_features: dict[str, float]
    volume_features: dict[str, float]
    atr_pct: float
    support: float
    resistance: float
    box_quality: float


def _gather_candidate_inputs(
    bars: Sequence[PriceBar],
    window: int,
    config: ScannerConfig,
) -> Result[_CandidateInputs | None]:
    if len(bars) < 2 * window:
        # Skip stocks with insufficient historical data (common in backtesting)
        # This prevents backtest failures when stocks haven't been listed long enough
        return Result.success(None, reason_code="INSUFFICIENT_BARS")

    price_gate = _require_price_platform(bars, window, config)
    if price_gate.status is ResultStatus.FAILED:
        return Result.failed(
            price_gate.error or ValueError("price gate failed"), "PRICE_GATE_FAILED"
        )
    if price_gate.data is None:
        return Result.success(None, reason_code=price_gate.reason_code)

    volume_gate = _require_volume_platform(bars, window, config)
    if volume_gate.status is ResultStatus.FAILED:
        return Result.failed(
            volume_gate.error or ValueError("volume gate failed"), "VOLUME_GATE_FAILED"
        )
    if volume_gate.data is None:
        return Result.success(None, reason_code=volume_gate.reason_code)

    atr_gate = _atr_pct_for_series(bars, config)
    if atr_gate.status is ResultStatus.FAILED:
        return Result.failed(atr_gate.error or ValueError("ATR gate failed"), "ATR_GATE_FAILED")
    if atr_gate.data is None:
        return Result.success(None, reason_code=atr_gate.reason_code)

    window_bars = _window_slice(bars, window)
    if window_bars.status is ResultStatus.FAILED:
        return Result.failed(
            window_bars.error or ValueError("window slice failed"), "WINDOW_SLICE_FAILED"
        )

    box_gate = _box_structure_for_window(
        window_bars.data,
        config,
        box_low_fallback=price_gate.data["box_low"],
        box_high_fallback=price_gate.data["box_high"],
    )
    if box_gate.status is ResultStatus.FAILED:
        return Result.failed(
            box_gate.error or ValueError("box structure failed"), "BOX_GATE_FAILED"
        )
    if box_gate.data is None:
        return Result.success(None, reason_code=box_gate.reason_code)

    support, resistance, box_quality = box_gate.data
    return Result.success(
        _CandidateInputs(
            price_features=price_gate.data,
            volume_features=volume_gate.data,
            atr_pct=float(atr_gate.data),
            support=support,
            resistance=resistance,
            box_quality=box_quality,
        )
    )


def detect_platform_candidate(
    symbol: str,
    bars: list[PriceBar],
    window: int,
    config: ScannerConfig,
    detected_at: int,
    regime: str | None = None,
    *,
    meta: dict[str, Any] | None = None,
    event_constraints: dict[str, Any] | None = None,
    breakthrough_config: dict[str, Any] | None = None,
) -> Result[PlatformCandidate | None]:
    """Detect a platform-period candidate using the latest ``window`` bars.

    Args:
        symbol: Ticker/symbol identifier.
        bars: Full price series ordered oldest -> newest.
        window: Lookback window used for the candidate (also used for volume compare).
        config: Threshold configuration.
        detected_at: Detection timestamp (ns since epoch).
        regime: Optional market regime label used for supplemental detectors.

    Returns:
        ``Result.success(candidate)`` when detected, ``Result.success(None)`` when filtered,
        or ``Result.failed(...)`` on insufficient/invalid data.
    """

    def _apply_indicator_scoring(
        candidate_result: Result[PlatformCandidate | None],
    ) -> Result[PlatformCandidate | None]:
        if candidate_result.status is ResultStatus.FAILED or candidate_result.data is None:
            return candidate_result

        indicator_config = getattr(config, "indicator_scoring", None)
        if not getattr(indicator_config, "enabled", False):
            return candidate_result

        base_candidate = candidate_result.data
        base_score = float(base_candidate.score)

        try:
            from scanner.indicator_scoring import compute_indicator_score

            indicator_score, indicator_metadata = compute_indicator_score(
                bars=bars,
                config=indicator_config,
                symbol=symbol,
            )

            mode = str(getattr(indicator_config, "combination_mode", "multiply"))
            if mode == "multiply":
                adjusted_score = base_score * float(indicator_score)
            elif mode == "weighted_avg":
                base_weight = float(getattr(indicator_config, "base_weight", 0.7))
                base_weight = max(0.0, min(1.0, base_weight))
                adjusted_score = base_score * base_weight + float(indicator_score) * (1.0 - base_weight)
            else:  # "min"
                adjusted_score = min(base_score, float(indicator_score))

            final_score = float(max(0.0, min(1.0, adjusted_score)))

            new_meta = dict(base_candidate.meta or {})
            new_meta["indicator_scoring"] = {
                "enabled": True,
                "base_score": base_score,
                "indicator_score": float(indicator_score),
                "adjusted_score": final_score,
                "combination_mode": mode,
                "details": indicator_metadata,
            }
            short_data = indicator_metadata.get("short_data") if isinstance(indicator_metadata, dict) else None
            if isinstance(short_data, dict):
                new_meta["short_data"] = short_data

            updated = msgspec.structs.replace(base_candidate, score=final_score, meta=new_meta)
            return Result(
                status=candidate_result.status,
                data=updated,
                error=candidate_result.error,
                reason_code=candidate_result.reason_code,
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, keep candidate.
            new_meta = dict(base_candidate.meta or {})
            new_meta["indicator_scoring"] = {
                "enabled": True,
                "base_score": base_score,
                "error": f"{type(exc).__name__}",
            }
            updated = msgspec.structs.replace(base_candidate, meta=new_meta)
            return Result(
                status=candidate_result.status,
                data=updated,
                error=candidate_result.error,
                reason_code=candidate_result.reason_code,
            )

    def _apply_breakthrough_confirmation(
        candidate_result: Result[PlatformCandidate | None],
    ) -> Result[PlatformCandidate | None]:
        if not breakthrough_config:
            return candidate_result

        config_dict = dict(breakthrough_config)
        require_days = int(config_dict.get("require_confirmation_days", 1) or 1)
        if require_days <= 1:
            return candidate_result

        if candidate_result.status is not ResultStatus.SUCCESS or candidate_result.data is None:
            return candidate_result

        try:
            confirmation_config = msgspec.convert(config_dict, type=BreakthroughConfirmationConfig)
        except Exception as exc:  # noqa: BLE001
            return Result.failed(exc, "BREAKTHROUGH_CONFIG_INVALID")

        resistance = None
        features = getattr(candidate_result.data, "features", None)
        if features is not None:
            resistance = getattr(features, "resistance_level", None) or getattr(features, "box_high", None)

        if resistance is None or not math.isfinite(float(resistance)) or float(resistance) <= 0:
            return candidate_result

        confirmation = BreakthroughConfirmation(config=confirmation_config)
        check = confirmation.check_confirmation(bars, resistance=float(resistance))
        if check.status is ResultStatus.FAILED:
            return Result.failed(
                check.error or ValueError("breakthrough confirmation failed"),
                check.reason_code or "BREAKTHROUGH_CONFIRMATION_FAILED",
            )
        if not check.data:
            return Result.success(None, reason_code="BREAKTHROUGH_NOT_CONFIRMED")
        return candidate_result

    if not getattr(config, "use_filter_chain_detector", True):
        legacy = _detect_platform_candidate_legacy(symbol, bars, window, config, detected_at)
        legacy = _apply_indicator_scoring(legacy)
        legacy = _apply_breakthrough_confirmation(legacy)
        filtered_result = _apply_event_guard_filter(
            symbol,
            legacy,
            detected_at=detected_at,
            config=config,
            event_constraints=event_constraints,
        )
        return _maybe_emit_reversal_candidate(
            symbol,
            bars,
            window,
            config,
            detected_at,
            regime,
            previous_result=filtered_result,
            meta=meta,
        )

    via_chain = _detect_platform_candidate_via_chain(symbol, bars, window, config, detected_at, meta=meta)
    if via_chain.status is ResultStatus.FAILED:
        return Result.failed(
            via_chain.error or ValueError("candidate chain failed"),
            via_chain.reason_code or "CANDIDATE_CHAIN_FAILED",
        )
    if via_chain.status is ResultStatus.DEGRADED:
        return Result.degraded(
            via_chain.data,
            via_chain.error or ValueError("candidate chain degraded"),
            via_chain.reason_code or "CANDIDATE_CHAIN_DEGRADED",
        )

    chain_result = _apply_indicator_scoring(
        Result.success(via_chain.data, reason_code=via_chain.reason_code),
    )
    chain_result = _apply_breakthrough_confirmation(chain_result)
    filtered_result = _apply_event_guard_filter(
        symbol,
        chain_result,
        detected_at=detected_at,
        config=config,
        event_constraints=event_constraints,
    )
    return _maybe_emit_reversal_candidate(
        symbol,
        bars,
        window,
        config,
        detected_at,
        regime,
        previous_result=filtered_result,
        meta=meta,
    )


def _detect_platform_candidate_legacy(
    symbol: str,
    bars: list[PriceBar],
    window: int,
    config: ScannerConfig,
    detected_at: int,
) -> Result[PlatformCandidate | None]:
    inputs_result = _gather_candidate_inputs(bars, window, config)
    if inputs_result.status is ResultStatus.FAILED:
        return Result.failed(
            inputs_result.error or ValueError("candidate gating failed"), "CANDIDATE_GATING_FAILED"
        )
    if inputs_result.data is None:
        return Result.success(None, reason_code=inputs_result.reason_code)

    inputs = inputs_result.data
    levels = _target_and_invalidation(inputs.support, inputs.resistance)
    if levels.status is ResultStatus.FAILED:
        return Result.failed(levels.error or ValueError("levels failed"), "LEVELS_FAILED")

    score = float(max(0.0, min(1.0, 0.8 + 0.2 * inputs.box_quality)))
    features = _build_platform_features(
        price_features=inputs.price_features,
        volume_features=inputs.volume_features,
        atr_pct=inputs.atr_pct,
        support=inputs.support,
        resistance=inputs.resistance,
        box_quality=inputs.box_quality,
    )

    liquidity_metrics = _liquidity_metrics_for_series(bars, config)
    invalidation_level, target_level = levels.data
    candidate = _build_candidate(
        symbol=symbol,
        detected_at=detected_at,
        window=window,
        score=score,
        features=features,
        invalidation_level=invalidation_level,
        target_level=target_level,
        config=config,
        extra_meta={"liquidity_metrics": liquidity_metrics} if liquidity_metrics is not None else None,
    )

    return Result.success(candidate)


def _liquidity_metrics_for_series(
    bars: Sequence[PriceBar],
    config: ScannerConfig,
) -> dict[str, Any] | None:
    try:  # local import avoids detector<->filters import cycles
        from .filters import LiquidityFilter
    except Exception:  # pragma: no cover
        return None

    liq_cfg = getattr(config, "liquidity_filter", None)
    enabled = bool(getattr(liq_cfg, "enabled", True)) if liq_cfg is not None else True
    result = LiquidityFilter(enabled=enabled).apply(bars, config)
    if result.status is not ResultStatus.SUCCESS or result.data is None:
        return None
    return result.data.features.get("liquidity_metrics")


def _detect_platform_candidate_via_chain(
    symbol: str,
    bars: list[PriceBar],
    window: int,
    config: ScannerConfig,
    detected_at: int,
    *,
    meta: dict[str, Any] | None = None,
) -> Result[PlatformCandidate | None]:
    if len(bars) < 2 * window:
        # Skip stocks with insufficient historical data (common in backtesting)
        # Log at debug level to avoid cluttering backtest output
        _logger.debug(
            "Skipping %s: insufficient bars (%d < %d required)",
            symbol,
            len(bars),
            2 * window,
            extra={"symbol": symbol, "bars_count": len(bars), "required": 2 * window},
        )
        return Result.success(None, reason_code="INSUFFICIENT_BARS")

    from .filters import (  # local import to avoid detector<->filters import cycles
        ADXEntryFilter,
        ATRFilter,
        BoxQualityFilter,
        BreakthroughFilter,
        FilterChain,
        LiquidityFilter,
        LowPositionFilter,
        MA200TrendFilter,
        MarketCapFilter,
        PricePlatformFilter,
        PullbackConfirmationFilter,
        RapidDeclineFilter,
        VolumePlatformFilter,
    )

    chain = FilterChain(logic="AND")
    chain.add_filter(PricePlatformFilter(window=window))
    chain.add_filter(VolumePlatformFilter(window=window))
    liq_cfg = getattr(config, "liquidity_filter", None)
    chain.add_filter(LiquidityFilter(enabled=bool(getattr(liq_cfg, "enabled", True)) if liq_cfg is not None else True))
    chain.add_filter(ATRFilter())
    chain.add_filter(ADXEntryFilter(enabled=bool(getattr(getattr(config, "adx_entry_filter", None), "enabled", False))))
    ma200_cfg = getattr(config, "ma200_trend_filter", None)
    chain.add_filter(MA200TrendFilter(enabled=bool(getattr(ma200_cfg, "enabled", False)) if ma200_cfg else False))
    if getattr(config, "use_market_cap_filter", False):
        chain.add_filter(MarketCapFilter(window=window))
    chain.add_filter(BoxQualityFilter(window=window))

    if getattr(config, "use_low_position_filter", False):
        chain.add_filter(LowPositionFilter())
    if getattr(config, "use_rapid_decline_filter", False):
        chain.add_filter(RapidDeclineFilter())
    if getattr(config, "use_breakthrough_filter", False):
        chain.add_filter(BreakthroughFilter())
    if getattr(config, "use_pullback_confirmation_filter", False):
        pullback_cfg = getattr(config, "pullback_confirmation", None)
        chain.add_filter(
            PullbackConfirmationFilter(
                enabled=bool(getattr(pullback_cfg, "enabled", True)) if pullback_cfg is not None else True
            )
        )

    chain_result = chain.execute(bars, config, context={"symbol": symbol, "meta": dict(meta or {})})
    if chain_result.status is ResultStatus.FAILED:
        return Result.failed(
            chain_result.error or ValueError("filter chain failed"),
            chain_result.reason_code or "CHAIN_FAILED",
        )
    if chain_result.status is ResultStatus.DEGRADED:
        if chain_result.data is not None:
            degraded_results = chain_result.data.filter_results
            box_quality_result = degraded_results.get("box_quality")
            box_quality_value = (
                float(box_quality_result.features.get("box_quality", float("nan")))
                if box_quality_result is not None
                else float("nan")
            )
            volume_platform_result = degraded_results.get("volume_platform")
            volume_stability_value = (
                float(volume_platform_result.features.get("volume_stability", float("nan")))
                if volume_platform_result is not None
                else float("nan")
            )
            passed_filters = [
                name
                for name, filter_result in degraded_results.items()
                if getattr(filter_result, "passed", False)
            ]
            _logger.debug(
                "filter_stats symbol=%s box_quality=%.3f volume_stability=%.3f passed_filters=%s",
                symbol,
                box_quality_value,
                volume_stability_value,
                passed_filters,
            )
        return Result.degraded(
            None,
            chain_result.error or ValueError("filter chain degraded"),
            chain_result.reason_code or "CHAIN_DEGRADED",
        )
    if chain_result.data is None:
        return Result.failed(ValueError("filter chain returned no data"), "CHAIN_EMPTY_RESULT")

    chain_data = chain_result.data
    results = chain_data.filter_results

    box_quality_result = results.get("box_quality")
    box_quality_value = (
        float(box_quality_result.features.get("box_quality", float("nan")))
        if box_quality_result is not None
        else float("nan")
    )
    volume_stability_value = (
        float(results.get("volume_platform").features.get("volume_stability", float("nan")))
        if results.get("volume_platform")
        else float("nan")
    )
    passed_filters = [name for name, filter_result in results.items() if getattr(filter_result, "passed", False)]
    _logger.debug(
        "filter_stats symbol=%s box_quality=%.3f volume_stability=%.3f passed_filters=%s",
        symbol,
        box_quality_value,
        volume_stability_value,
        passed_filters,
    )

    price_result = results.get("price_platform")
    if price_result is None or not price_result.passed:
        return Result.success(None, reason_code="PRICE_RULES_NOT_MET")

    volume_result = results.get("volume_platform")
    if volume_result is None or not volume_result.passed:
        return Result.success(None, reason_code="VOLUME_RULES_NOT_MET")

    avg_dollar_volume = float(volume_result.features.get("avg_dollar_volume", 0.0))

    atr_result = results.get("atr_range")
    if atr_result is None or not atr_result.passed:
        return Result.success(None, reason_code="ATR_FILTERED")

    adx_cfg = getattr(config, "adx_entry_filter", None)
    if bool(getattr(adx_cfg, "enabled", False)):
        adx_result = results.get("adx_entry_filter")
        if adx_result is None or not adx_result.passed:
            return Result.success(
                None,
                reason_code=str(getattr(adx_result, "reason", None) or "ADX_FILTERED"),
            )

    if getattr(config, "use_market_cap_filter", False):
        market_cap_result = results.get("market_cap")
        if market_cap_result is None or not market_cap_result.passed:
            return Result.success(
                None,
                reason_code=str(getattr(market_cap_result, "reason", None) or "MARKET_CAP_FILTERED"),
            )

    box_result = results.get("box_quality")
    if box_result is None or not box_result.passed:
        return Result.success(None, reason_code="BOX_QUALITY_FILTERED")

    if getattr(config, "use_low_position_filter", False):
        low_position = results.get("low_position")
        if low_position is None or not low_position.passed:
            return Result.success(None, reason_code="LOW_POSITION_FILTERED")

    if getattr(config, "use_rapid_decline_filter", False):
        rapid_decline = results.get("rapid_decline")
        if rapid_decline is None or not rapid_decline.passed:
            return Result.success(None, reason_code="RAPID_DECLINE_FILTERED")

    if getattr(config, "use_breakthrough_filter", False):
        breakthrough = results.get("breakthrough_potential")
        if breakthrough is None or not breakthrough.passed:
            return Result.success(None, reason_code="BREAKTHROUGH_FILTERED")

    if getattr(config, "use_pullback_confirmation_filter", False):
        pullback_cfg = getattr(config, "pullback_confirmation", None)

        skip_by_score = False
        if pullback_cfg is not None:
            score_threshold = getattr(pullback_cfg, "score_threshold_for_skip", None)
            if score_threshold is not None:
                total = 0.0
                count = 0
                for name, filter_result in results.items():
                    if name == "pullback_confirmation":
                        continue
                    try:
                        score = float(getattr(filter_result, "score", 0.0))
                    except Exception:  # noqa: BLE001
                        score = 0.0
                    if score != score:  # NaN
                        score = 0.0
                    total += max(0.0, min(1.0, score))
                    count += 1

                current_score = float(total / count) if count else 1.0
                if current_score >= float(score_threshold):
                    skip_by_score = True

        if not skip_by_score:
            pullback_confirmation = results.get("pullback_confirmation")
            if pullback_confirmation is None or not pullback_confirmation.passed:
                return Result.success(None, reason_code="PULLBACK_CONFIRMATION_FILTERED")

    support = float(box_result.features.get("support", 0.0))
    resistance = float(box_result.features.get("resistance", 0.0))
    box_quality = float(box_result.features.get("box_quality", 0.0))
    levels = _target_and_invalidation(support, resistance)
    if levels.status is ResultStatus.FAILED:
        return Result.failed(levels.error or ValueError("levels failed"), "LEVELS_FAILED")

    price_features = {
        "box_range": float(price_result.features.get("box_range", 0.0)),
        "box_low": float(price_result.features.get("box_low", 0.0)),
        "box_high": float(price_result.features.get("box_high", 0.0)),
        "ma_diff": float(price_result.features.get("ma_diff", 0.0)),
        "volatility": float(price_result.features.get("volatility", 0.0)),
    }
    volume_features = {
        "volume_change_ratio": float(volume_result.features.get("volume_change_ratio", 0.0)),
        "volume_stability": float(volume_result.features.get("volume_stability", 0.0)),
        "avg_dollar_volume": float(avg_dollar_volume),
    }
    atr_pct = float(atr_result.features.get("atr_pct", 0.0))

    features = _build_platform_features(
        price_features=price_features,
        volume_features=volume_features,
        atr_pct=atr_pct,
        support=support,
        resistance=resistance,
        box_quality=box_quality,
    )

    invalidation_level, target_level = levels.data
    score = float(max(0.0, min(1.0, chain_data.combined_score)))
    reasons = [filter.name for filter in chain.get_enabled_filters()] + list(chain_data.reasons)
    liquidity_metrics = None
    liquidity_result = results.get("liquidity")
    if liquidity_result is not None:
        liquidity_metrics = liquidity_result.features.get("liquidity_metrics")
    candidate = PlatformCandidate(
        symbol=symbol,
        detected_at=detected_at,
        window=window,
        score=score,
        features=features,
        invalidation_level=invalidation_level,
        target_level=target_level,
        reasons=reasons,
        meta={
            **_candidate_meta(config),
            "liquidity_metrics": liquidity_metrics,
            "filter_chain": {
                "logic": chain.logic,
                "passed": chain_data.passed,
                "combined_score": chain_data.combined_score,
                "reasons": list(chain_data.reasons),
                "filter_results": {
                    name: msgspec.structs.asdict(filter_result)
                    for name, filter_result in chain_data.filter_results.items()
                },
            },
        },
    )
    return Result.success(candidate)


_NS_PER_DAY = 86_400 * 1_000_000_000


def _apply_event_guard_filter(
    symbol: str,
    result: Result[PlatformCandidate | None],
    *,
    detected_at: int,
    config: ScannerConfig,
    event_constraints: dict[str, Any] | None,
) -> Result[PlatformCandidate | None]:
    if (
        not getattr(config, "use_event_guard_filter", False)
        or not event_constraints
        or result.status is not ResultStatus.SUCCESS
        or result.data is None
    ):
        return result

    constraints = _resolve_symbol_constraints(event_constraints, symbol)
    if not constraints:
        return result

    earnings_window_days = int(getattr(config, "earnings_window_days", 10) or 10)
    now_ns = int(detected_at)

    if _should_skip_for_earnings(constraints, now_ns=now_ns, window_days=earnings_window_days):
        try:
            result.data.reasons.append("SKIPPED_EARNINGS_WINDOW")
        except Exception:  # noqa: BLE001
            pass
        return Result.success(None, reason_code="SKIPPED_EARNINGS_WINDOW")

    if _should_skip_for_negative_sentiment(constraints):
        try:
            result.data.reasons.append("SKIPPED_NEGATIVE_NEWS")
        except Exception:  # noqa: BLE001
            pass
        return Result.success(None, reason_code="SKIPPED_NEGATIVE_NEWS")

    if _should_skip_for_high_risk(constraints):
        try:
            result.data.reasons.append("SKIPPED_HIGH_RISK_EVENT")
        except Exception:  # noqa: BLE001
            pass
        return Result.success(None, reason_code="SKIPPED_HIGH_RISK_EVENT")

    return result


def _resolve_symbol_constraints(event_constraints: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    if not isinstance(event_constraints, dict):
        return None

    for key in (symbol, str(symbol).upper(), str(symbol).lower()):
        candidate = event_constraints.get(key)
        if isinstance(candidate, dict):
            return candidate

    return None


def _maybe_emit_reversal_candidate(
    symbol: str,
    bars: Sequence[PriceBar],
    window: int,
    config: ScannerConfig,
    detected_at: int,
    regime: str | None,
    *,
    previous_result: Result[PlatformCandidate | None],
    meta: dict[str, Any] | None,
) -> Result[PlatformCandidate | None]:
    """Trigger the reversal detector when no platform candidate was emitted."""

    if (
        previous_result.status is not ResultStatus.SUCCESS
        or previous_result.data is not None
        or not getattr(config, "use_reversal_detector", False)
    ):
        return previous_result

    if not regime or regime.strip().lower() != "bear":
        return previous_result

    reversal_config = getattr(config, "reversal", None)
    if not isinstance(reversal_config, ReversalConfig):
        try:
            reversal_config = msgspec.convert(dict(reversal_config or {}), type=ReversalConfig)
        except Exception:  # noqa: BLE001
            reversal_config = ReversalConfig()

    if not getattr(reversal_config, "enabled", False):
        return previous_result

    detector = ReversalDetector(reversal_config)
    try:
        detection = detector.detect(symbol, bars, regime=regime)
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "reversal_detection_exception symbol=%s error=%s",
            symbol,
            str(exc)[:200],
        )
        return previous_result

    if detection.status is ResultStatus.FAILED:
        _logger.warning(
            "reversal_detection_failed symbol=%s reason=%s error=%s",
            symbol,
            detection.reason_code,
            str(detection.error or "unknown")[:200],
        )
        return previous_result

    if detection.data is None:
        _logger.debug(
            "reversal_detection_no_signal symbol=%s reason=%s",
            symbol,
            detection.reason_code,
        )
        return previous_result

    reversal_candidate = _build_reversal_candidate(
        symbol=symbol,
        bars=bars,
        window=window,
        detected_at=detected_at,
        reversal_signal=detection.data,
        meta=meta,
    )
    _logger.info(
        "reversal_candidate_emitted symbol=%s regime=%s strength=%.4f",
        symbol,
        regime,
        reversal_candidate.reversal_signal_strength or 0.0,
    )
    return Result.success(reversal_candidate, reason_code="REVERSAL_SIGNAL")


def _build_reversal_candidate(
    *,
    symbol: str,
    bars: Sequence[PriceBar],
    window: int,
    detected_at: int,
    reversal_signal: ReversalCandidate,
    meta: dict[str, Any] | None,
) -> PlatformCandidate:
    if not bars:
        raise ValueError("Bars required for reversal candidate.")

    last_bar = bars[-1]
    close_price = _extract_bar_value(last_bar, "close", default=0.0)
    high_price = _extract_bar_value(last_bar, "high", default=close_price)
    low_price = _extract_bar_value(last_bar, "low", default=close_price)
    last_volume = _extract_bar_value(last_bar, "volume", default=0.0)

    lookback = max(2, min(window, len(bars)))
    window_slice = list(bars[-lookback:])
    closes = [_extract_bar_value(bar, "close", default=close_price) for bar in window_slice]
    volumes = [_extract_bar_value(bar, "volume", default=last_volume) for bar in window_slice]

    avg_volume = float(np.mean(volumes)) if volumes else last_volume
    avg_dollar_volume = max(0.0, float(close_price * avg_volume))
    volatility = _estimate_recent_volatility(closes)

    features = PlatformFeatures(
        box_range=_safe_box_range(high_price, low_price),
        box_low=low_price,
        box_high=high_price,
        ma_diff=0.0,
        volatility=volatility,
        atr_pct=None,
        volume_change_ratio=1.0,
        volume_stability=0.0,
        volume_increase_ratio=float(reversal_signal.volume_ratio),
        avg_dollar_volume=avg_dollar_volume,
        box_quality=None,
        support_level=low_price,
        resistance_level=high_price,
    )

    invalidation_level = low_price * INVALIDATION_BUFFER if low_price > 0 else close_price * INVALIDATION_BUFFER
    if not math.isfinite(invalidation_level) or invalidation_level <= 0:
        invalidation_level = close_price * INVALIDATION_BUFFER

    target_level = close_price * TARGET_BUFFER
    if not math.isfinite(target_level) or target_level <= invalidation_level:
        fallback_target = max(close_price * 1.05, invalidation_level * (1.0 / INVALIDATION_BUFFER))
        if fallback_target <= invalidation_level:
            fallback_target = invalidation_level * 1.02 if invalidation_level > 0 else 0.01
        target_level = fallback_target

    score = float(max(0.0, min(1.0, reversal_signal.signal_strength)))
    reasons = ["REVERSAL_SIGNAL"]

    meta_payload: dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
    meta_payload.setdefault(
        "reversal",
        {
            "rsi_value": float(reversal_signal.rsi_value),
            "volume_ratio": float(reversal_signal.volume_ratio),
            "price_stabilized": bool(reversal_signal.price_stabilized),
        },
    )
    meta_payload.setdefault("detector", "reversal")

    return PlatformCandidate(
        symbol=symbol,
        detected_at=detected_at,
        window=window,
        score=score,
        features=features,
        invalidation_level=float(invalidation_level),
        target_level=float(target_level),
        reasons=reasons,
        meta=meta_payload,
        signal_type="reversal",
        reversal_rsi=float(reversal_signal.rsi_value),
        reversal_volume_ratio=float(reversal_signal.volume_ratio),
        reversal_signal_strength=score,
    )


def _estimate_recent_volatility(closes: Sequence[float]) -> float:
    if len(closes) < 2:
        return 0.0
    prices = np.asarray(closes, dtype=float)
    prev = prices[:-1]
    diffs = np.diff(prices)
    valid = prev != 0
    if not np.any(valid):
        return 0.0
    pct_changes = diffs[valid] / prev[valid]
    if pct_changes.size == 0:
        return 0.0
    value = float(np.std(pct_changes))
    return value if math.isfinite(value) else 0.0


def _safe_box_range(high_price: float, low_price: float) -> float:
    if low_price <= 0 or not math.isfinite(low_price) or not math.isfinite(high_price):
        return 0.0
    return max(0.0, (high_price - low_price) / low_price)


def _extract_bar_value(bar: Any, attr: str, *, default: float) -> float:
    value = getattr(bar, attr, None)
    if value is None and hasattr(bar, "get"):
        try:
            value = bar.get(attr, default)
        except Exception:  # noqa: BLE001
            value = default
    if value is None:
        value = default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _should_skip_for_earnings(constraints: dict[str, Any], *, now_ns: int, window_days: int) -> bool:
    if window_days <= 0:
        return False

    reason_codes = constraints.get("reason_codes")
    if not isinstance(reason_codes, list) or not any("EARNINGS" in str(code).upper() for code in reason_codes):
        return False

    windows = constraints.get("no_trade_windows")
    if not isinstance(windows, list) or not windows:
        return False

    end_ns = int(now_ns + int(window_days) * _NS_PER_DAY)
    for window in windows:
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            continue
        try:
            start_ns = int(window[0])
            stop_ns = int(window[1])
        except Exception:  # noqa: BLE001
            continue
        if stop_ns < now_ns or start_ns > end_ns:
            continue
        return True
    return False


def _should_skip_for_high_risk(constraints: dict[str, Any]) -> bool:
    risk_level = constraints.get("risk_level")
    if isinstance(risk_level, str) and risk_level.upper() in ("HIGH", "CRITICAL"):
        return True

    reason_codes = constraints.get("reason_codes")
    if isinstance(reason_codes, list):
        for code in reason_codes:
            text = str(code).upper()
            if "EVENT_RISK_HIGH" in text or "EVENT_RISK_CRITICAL" in text:
                return True
    return False


def _should_skip_for_negative_sentiment(constraints: dict[str, Any]) -> bool:
    for key in ("sentiment", "sentiment_score", "news_sentiment"):
        value = constraints.get(key)
        if isinstance(value, (int, float)):
            return float(value) < -0.5
        if isinstance(value, str) and value.strip().lower() in {"negative", "bearish", "very_negative"}:
            return True
    return False
