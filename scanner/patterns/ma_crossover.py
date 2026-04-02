"""Simple MA Crossover (golden cross) detector -- demo for the pipeline.

Fires when SMA20 crosses above SMA50 on above-average volume.
Gate-free: sector_bullish / rs_strong are ignored for simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from data.interface import PriceBar
from .interface import TrendPatternConfig, TrendPatternResult


class MACrossoverConfig(TrendPatternConfig, frozen=True, kw_only=True):
    """Configuration for the MA Crossover detector."""

    sma_short_period: int = 20
    sma_long_period: int = 50
    crossover_lookback: int = 3
    volume_mult: float = 1.3
    min_stop_atr_mult: float = 2.0


@dataclass(slots=True)
class MACrossoverDetector:
    """MA Crossover (golden cross) detector -- demo pattern for the pipeline."""

    config: MACrossoverConfig

    def __init__(self, config: MACrossoverConfig | None = None) -> None:
        self.config = config or MACrossoverConfig()

    @property
    def pattern_name(self) -> str:
        return "ma_crossover"

    def _miss(self, reasons: list[str], **kw: Any) -> TrendPatternResult:
        """Shorthand for a non-detection result."""
        return TrendPatternResult(
            detected=False, pattern_type=self.pattern_name, reasons=reasons, **kw)

    def detect(
        self, symbol: str, bars: list[PriceBar], current_date: date,
        *, sector_bullish: bool = False, rs_strong: bool = False,
    ) -> TrendPatternResult:
        """Detect MA crossover.  Gates accepted but ignored (gate-free)."""
        cfg = self.config
        if not cfg.enabled:
            return self._miss(["pattern_disabled"])
        min_bars = cfg.sma_long_period + cfg.crossover_lookback + 1
        if len(bars) < min_bars:
            return self._miss([f"insufficient_bars:{len(bars)}<{min_bars}"])

        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        sma20 = self._sma(closes, cfg.sma_short_period)
        sma50 = self._sma(closes, cfg.sma_long_period)
        n = min(len(sma20), len(sma50))
        sma20, sma50 = sma20[-n:], sma50[-n:]

        # 1) price > SMA50
        if closes[-1] <= sma50[-1]:
            return self._miss(["price_below_sma50"],
                              meta={"close": closes[-1], "sma50": sma50[-1]})
        # 2) golden cross in last N bars
        if not self._has_crossover(sma20, sma50, cfg.crossover_lookback):
            return self._miss(["no_golden_cross"])
        # 3) volume confirmation
        avg_vol = sum(volumes[-20:]) / min(len(volumes), 20)
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 0.0
        if vol_ratio < cfg.volume_mult:
            return self._miss([f"weak_volume:{vol_ratio:.2f}<{cfg.volume_mult}"])

        # Score (0-1)
        pa = (closes[-1] - sma50[-1]) / sma50[-1]
        sp = (sma20[-1] - sma50[-1]) / sma50[-1]
        score = max(0.0, min(1.0,
            0.4 * min(pa / 0.10, 1.0)
            + 0.3 * min(vol_ratio / 3.0, 1.0)
            + 0.3 * min(max(sp, 0.0) / 0.05, 1.0)))
        if score < cfg.min_score:
            return self._miss([f"score_too_low:{score:.2f}<{cfg.min_score}"],
                              score=score)

        # Levels
        entry = closes[-1]
        atr = self._calculate_atr(bars, cfg.atr_period)
        stop = sma50[-1]
        if entry - stop < atr * cfg.min_stop_atr_mult:
            stop = entry - atr * cfg.min_stop_atr_mult
        target = entry + 2.0 * (entry - stop)

        return TrendPatternResult(
            detected=True, pattern_type=self.pattern_name, score=score,
            entry_price=entry, stop_loss=stop, target_price=target,
            reasons=[f"golden_cross:sma20={sma20[-1]:.2f}>sma50={sma50[-1]:.2f}",
                     f"price_above_sma50:{pa * 100:.1f}%",
                     f"volume_confirmed:{vol_ratio:.2f}x_avg"],
            meta={"sma20": sma20[-1], "sma50": sma50[-1],
                  "sma_spread_pct": sp, "volume_ratio": vol_ratio, "atr": atr},
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _sma(values: list[float], period: int) -> list[float]:
        """Simple moving average (rolling window)."""
        if len(values) < period:
            return []
        out: list[float] = []
        s = sum(values[:period])
        out.append(s / period)
        for i in range(period, len(values)):
            s += values[i] - values[i - period]
            out.append(s / period)
        return out

    @staticmethod
    def _has_crossover(short: list[float], long: list[float], lookback: int) -> bool:
        """True if *short* crossed above *long* within the last *lookback* bars."""
        for i in range(len(short) - lookback, len(short)):
            if i < 1:
                continue
            if short[i] >= long[i] and short[i - 1] < long[i - 1]:
                return True
        return False

    @staticmethod
    def _calculate_ema(values: list[float], period: int) -> list[float]:
        """Calculate EMA for a series of values."""
        if len(values) < period:
            return []
        mult = 2 / (period + 1)
        ema = [sum(values[:period]) / period]
        for i in range(period, len(values)):
            ema.append(values[i] * mult + ema[-1] * (1 - mult))
        return ema

    @staticmethod
    def _calculate_atr(bars: list[PriceBar], period: int) -> float:
        """Calculate Average True Range."""
        if len(bars) < 2:
            return 0.0
        trs: list[float] = []
        for i in range(1, len(bars)):
            trs.append(max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            ))
        recent = trs[-period:]
        return sum(recent) / len(recent) if recent else 0.0
