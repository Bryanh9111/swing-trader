"""Oversold reversal signal detector module."""

from __future__ import annotations

from typing import Any, Sequence

import msgspec

from common.interface import Result


class ReversalConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Reversal detection configuration."""

    enabled: bool = False
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    volume_lookback: int = 20
    volume_ratio_threshold: float = 2.0
    require_price_stabilization: bool = True
    stabilization_days: int = 2


class ReversalCandidate(msgspec.Struct, frozen=True, kw_only=True):
    """Reversal signal candidate."""

    symbol: str
    rsi_value: float
    rsi_turning_up: bool
    volume_ratio: float
    price_stabilized: bool
    signal_strength: float  # 0-1, composite signal strength


class ReversalDetector:
    """Oversold reversal signal detector."""

    def __init__(self, config: ReversalConfig | None = None) -> None:
        self.config = config or ReversalConfig()

    def detect(
        self,
        symbol: str,
        bars: Sequence[Any],
        regime: str | None = None,
    ) -> Result[ReversalCandidate | None]:
        """Detect reversal signal."""

        if not self.config.enabled:
            return Result.success(None, reason_code="REVERSAL_DISABLED")

        if regime is not None and regime.lower() != "bear":
            return Result.success(None, reason_code="NON_BEAR_REGIME")

        min_bars = max(self.config.rsi_period + 5, self.config.volume_lookback + 1)
        if len(bars) < min_bars:
            return Result.failed(
                ValueError(f"Insufficient bars: {len(bars)} < {min_bars}"),
                "INSUFFICIENT_DATA",
            )

        rsi_values = self._calculate_rsi(bars, self.config.rsi_period)
        if len(rsi_values) < 2:
            return Result.failed(
                ValueError("Failed to calculate RSI"),
                "RSI_CALCULATION_FAILED",
            )

        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]
        rsi_oversold = current_rsi < self.config.rsi_oversold
        rsi_turning_up = current_rsi > prev_rsi

        if not (rsi_oversold and rsi_turning_up):
            return Result.success(None, reason_code="RSI_CONDITION_NOT_MET")

        volume_ratio = self._calculate_volume_ratio(bars, self.config.volume_lookback)
        if volume_ratio < self.config.volume_ratio_threshold:
            return Result.success(None, reason_code="VOLUME_CONDITION_NOT_MET")

        price_stabilized = True
        if self.config.require_price_stabilization:
            price_stabilized = self._check_price_stabilization(bars, self.config.stabilization_days)

        if not price_stabilized:
            return Result.success(None, reason_code="PRICE_NOT_STABILIZED")

        signal_strength = self._calculate_signal_strength(
            current_rsi,
            volume_ratio,
            price_stabilized,
        )

        candidate = ReversalCandidate(
            symbol=symbol,
            rsi_value=current_rsi,
            rsi_turning_up=rsi_turning_up,
            volume_ratio=volume_ratio,
            price_stabilized=price_stabilized,
            signal_strength=signal_strength,
        )
        return Result.success(candidate)

    def _calculate_rsi(self, bars: Sequence[Any], period: int) -> list[float]:
        """Calculate RSI series."""

        closes = [_bar_value(bar, "close", 0.0) for bar in bars]
        if len(closes) < period + 1:
            return []

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains: list[float] = []
        losses: list[float] = []

        for delta in deltas:
            if delta > 0:
                gains.append(delta)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(delta))

        if len(gains) < period:
            return []

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values: list[float] = []
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

        return rsi_values

    def _calculate_volume_ratio(self, bars: Sequence[Any], lookback: int) -> float:
        """Calculate current volume to average volume ratio."""

        volumes = [_bar_value(bar, "volume", 0.0) for bar in bars]
        if len(volumes) < lookback + 1:
            return 0.0

        current_volume = volumes[-1]
        avg_volume = sum(volumes[-(lookback + 1) : -1]) / lookback
        if avg_volume <= 0:
            return 0.0

        return current_volume / avg_volume

    def _check_price_stabilization(self, bars: Sequence[Any], days: int) -> bool:
        """Check price stabilization (no new lows for N consecutive days)."""

        if len(bars) < days + 1:
            return False

        lows = [_bar_value(bar, "low", 0.0) for bar in bars]
        recent_lows = lows[-days:]
        baseline_window = lows[: -days]
        if not baseline_window:
            baseline_window = [lows[-days - 1]]

        reference_low = min(baseline_window)
        tolerance = 0.99

        for low in recent_lows:
            if low < reference_low * tolerance:
                return False
            reference_low = min(reference_low, low)

        return True

    def _calculate_signal_strength(
        self,
        rsi: float,
        volume_ratio: float,
        price_stabilized: bool,
    ) -> float:
        """Calculate composite signal strength."""

        rsi_score = max(0.0, min(1.0, (30.0 - rsi) / 10.0))
        volume_score = max(0.0, min(1.0, (volume_ratio - 1.0) / 3.0))
        stabilization_score = 1.0 if price_stabilized else 0.5

        return rsi_score * 0.4 + volume_score * 0.3 + stabilization_score * 0.3


def _bar_value(bar: Any, field: str, default: float) -> float:
    value = getattr(bar, field, None)
    if value is None and hasattr(bar, "get"):
        try:
            value = bar.get(field, default)
        except Exception:  # noqa: BLE001
            value = default
    if value is None:
        value = default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
