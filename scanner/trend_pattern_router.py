"""Trend Pattern Router — dispatches trend continuation pattern detectors.

The router:
1. Evaluates gate conditions (Sector Regime, Relative Strength)
2. If gates pass, runs all enabled trend pattern detectors
3. Converts TrendPatternResult to PlatformCandidate format
4. Returns the best matching pattern (if any)

The framework ships with a single demo detector (MA Crossover).
Add your own by implementing TrendPatternDetector and registering here.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

import msgspec

from common.interface import Result, ResultStatus
from data.interface import PriceBar

from .interface import PlatformCandidate, PlatformFeatures
from .gates import (
    SectorRegimeGate,
    SectorRegimeConfig,
    RelativeStrengthGate,
    RelativeStrengthConfig,
    GateResult,
)
from .patterns import (
    TrendPatternResult,
    TrendPatternDetector,
)
from .patterns.ma_crossover import MACrossoverDetector, MACrossoverConfig

_logger = logging.getLogger(__name__)

__all__ = [
    "TrendPatternRouterConfig",
    "TrendPatternRouter",
]


class TrendPatternRouterConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for the Trend Pattern Router.

    Attributes:
        enabled: Master switch for trend pattern detection.
        ma_crossover: Configuration for the MA Crossover demo detector.
        sector_regime: Configuration for Sector Regime gate.
        relative_strength: Configuration for Relative Strength gate.
        min_combined_score: Minimum score to emit a candidate.
        prefer_highest_score: If multiple patterns match, use highest score.
    """

    enabled: bool = True

    # Pattern detector configs
    ma_crossover: MACrossoverConfig = msgspec.field(default_factory=MACrossoverConfig)

    # Gate configs
    sector_regime: SectorRegimeConfig = msgspec.field(default_factory=SectorRegimeConfig)
    relative_strength: RelativeStrengthConfig = msgspec.field(default_factory=RelativeStrengthConfig)

    # Routing options
    min_combined_score: float = 0.6
    prefer_highest_score: bool = True


@dataclass(slots=True)
class TrendPatternRouter:
    """Router for trend continuation pattern detection.

    Coordinates gates and pattern detectors to find trend continuation
    setups in growth/momentum stocks.
    """

    config: TrendPatternRouterConfig

    _sector_gate: SectorRegimeGate | None = None
    _rs_gate: RelativeStrengthGate | None = None
    _ma_detector: MACrossoverDetector | None = None

    def __init__(self, config: TrendPatternRouterConfig | None = None) -> None:
        self.config = config or TrendPatternRouterConfig()
        self._sector_gate = None
        self._rs_gate = None
        self._ma_detector = None

    def _get_sector_gate(self) -> SectorRegimeGate:
        if self._sector_gate is None:
            self._sector_gate = SectorRegimeGate(self.config.sector_regime)
        return self._sector_gate

    def _get_rs_gate(self) -> RelativeStrengthGate:
        if self._rs_gate is None:
            self._rs_gate = RelativeStrengthGate(self.config.relative_strength)
        return self._rs_gate

    def _get_ma_detector(self) -> MACrossoverDetector:
        if self._ma_detector is None:
            self._ma_detector = MACrossoverDetector(self.config.ma_crossover)
        return self._ma_detector

    def detect(
        self,
        symbol: str,
        bars: list[PriceBar],
        current_date: date,
        *,
        sector_bars: list[PriceBar] | None = None,
        benchmark_bars: list[PriceBar] | None = None,
        detected_at: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Result[PlatformCandidate | None]:
        """Detect trend continuation patterns.

        Args:
            symbol: Stock symbol.
            bars: Stock price bars (oldest first).
            current_date: Current date for detection.
            sector_bars: Sector ETF bars (XLK/SOXX) for regime gate.
            benchmark_bars: Benchmark bars (QQQ/SPY) for RS gate.
            detected_at: Detection timestamp (ns). Defaults to current time.
            meta: Additional metadata.

        Returns:
            Result[PlatformCandidate | None]: Candidate if pattern detected,
            None if no pattern or gates failed.
        """
        if not self.config.enabled:
            return Result.success(None, reason_code="TREND_PATTERNS_DISABLED")

        if detected_at is None:
            detected_at = int(time.time_ns())

        # Evaluate gates
        gate_result = self._evaluate_gates(
            symbol, bars, current_date, sector_bars, benchmark_bars
        )
        if gate_result.status is ResultStatus.FAILED:
            return Result.failed(
                gate_result.error or ValueError("gate evaluation failed"),
                gate_result.reason_code or "GATE_EVALUATION_FAILED",
            )

        sector_bullish, rs_strong, gate_meta = gate_result.data or (False, False, {})

        # Note: MA Crossover is gate-free, so we still run it even if gates fail.
        # Other pattern detectors may require gates to pass.

        # Run pattern detectors
        pattern_results = self._run_detectors(
            symbol, bars, current_date, sector_bullish, rs_strong
        )

        if not pattern_results:
            return Result.success(None, reason_code="NO_PATTERNS_DETECTED")

        # Select best pattern
        best_result = self._select_best_pattern(pattern_results)
        if best_result is None:
            return Result.success(None, reason_code="NO_QUALIFYING_PATTERNS")

        # Convert to PlatformCandidate
        candidate = self._convert_to_candidate(
            symbol=symbol,
            pattern_result=best_result,
            bars=bars,
            detected_at=detected_at,
            gate_meta=gate_meta,
            meta=meta,
            benchmark_bars=benchmark_bars,
        )

        _logger.info(
            "trend_pattern_detected symbol=%s pattern=%s score=%.3f",
            symbol,
            best_result.pattern_type,
            best_result.score,
        )

        return Result.success(candidate, reason_code="TREND_PATTERN_DETECTED")

    def _evaluate_gates(
        self,
        symbol: str,
        bars: list[PriceBar],
        current_date: date,
        sector_bars: list[PriceBar] | None,
        benchmark_bars: list[PriceBar] | None,
    ) -> Result[tuple[bool, bool, dict[str, Any]]]:
        """Evaluate sector regime and relative strength gates."""
        gate_meta: dict[str, Any] = {}

        sector_gate = self._get_sector_gate()
        sector_result = sector_gate.evaluate(
            symbol=symbol,
            bars=bars,
            current_date=current_date,
            benchmark_bars=sector_bars,
        )
        sector_bullish = sector_result.passed
        gate_meta["sector_regime"] = {
            "passed": sector_result.passed,
            "reason": sector_result.reason,
            "value": sector_result.value,
            "meta": sector_result.meta,
        }

        rs_gate = self._get_rs_gate()
        rs_result = rs_gate.evaluate(
            symbol=symbol,
            bars=bars,
            current_date=current_date,
            benchmark_bars=benchmark_bars,
        )
        rs_strong = rs_result.passed
        gate_meta["relative_strength"] = {
            "passed": rs_result.passed,
            "reason": rs_result.reason,
            "value": rs_result.value,
            "meta": rs_result.meta,
        }

        _logger.debug(
            "gate_evaluation symbol=%s sector_bullish=%s rs_strong=%s",
            symbol,
            sector_bullish,
            rs_strong,
        )

        return Result.success((sector_bullish, rs_strong, gate_meta))

    def _run_detectors(
        self,
        symbol: str,
        bars: list[PriceBar],
        current_date: date,
        sector_bullish: bool,
        rs_strong: bool,
    ) -> list[TrendPatternResult]:
        """Run all enabled pattern detectors and collect results."""
        results: list[TrendPatternResult] = []

        # MA Crossover (gate-free demo pattern)
        if self.config.ma_crossover.enabled:
            ma_detector = self._get_ma_detector()
            ma_result = ma_detector.detect(
                symbol=symbol,
                bars=bars,
                current_date=current_date,
                sector_bullish=sector_bullish,
                rs_strong=rs_strong,
            )
            if ma_result.detected:
                results.append(ma_result)
                _logger.debug(
                    "ma_crossover_detected symbol=%s score=%.3f",
                    symbol,
                    ma_result.score,
                )

        return results

    def _select_best_pattern(
        self, results: list[TrendPatternResult]
    ) -> TrendPatternResult | None:
        """Select the best pattern from multiple detected patterns."""
        if not results:
            return None

        qualifying = [r for r in results if r.score >= self.config.min_combined_score]
        if not qualifying:
            return None

        if self.config.prefer_highest_score:
            return max(qualifying, key=lambda r: r.score)

        return qualifying[0]

    def _convert_to_candidate(
        self,
        symbol: str,
        pattern_result: TrendPatternResult,
        bars: list[PriceBar],
        detected_at: int,
        gate_meta: dict[str, Any],
        meta: dict[str, Any] | None,
        benchmark_bars: list[PriceBar] | None = None,
    ) -> PlatformCandidate:
        """Convert TrendPatternResult to PlatformCandidate format."""
        if not bars:
            raise ValueError("bars required for candidate conversion")

        last_bar = bars[-1]
        close_price = float(last_bar.close)
        high_price = float(last_bar.high)
        low_price = float(last_bar.low)

        entry_price = pattern_result.entry_price or close_price
        stop_loss = pattern_result.stop_loss or (low_price * 0.98)
        target_price = pattern_result.target_price or (entry_price * 1.05)

        pattern_meta = dict(pattern_result.meta) if pattern_result.meta else {}

        box_low = pattern_meta.get("support_level", low_price)
        box_high = pattern_meta.get("resistance_level", high_price)
        if not isinstance(box_low, (int, float)) or not math.isfinite(box_low):
            box_low = low_price
        if not isinstance(box_high, (int, float)) or not math.isfinite(box_high):
            box_high = high_price

        box_range = (box_high - box_low) / box_low if box_low > 0 else 0.0

        atr = pattern_meta.get("atr")
        atr_pct = float(atr / close_price) if atr and close_price > 0 else None

        recent_volumes = [bar.volume for bar in bars[-20:]] if len(bars) >= 20 else [bar.volume for bar in bars]
        avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0.0
        avg_dollar_volume = avg_volume * close_price

        features = PlatformFeatures(
            box_range=float(box_range),
            box_low=float(box_low),
            box_high=float(box_high),
            ma_diff=0.0,
            volatility=pattern_meta.get("volatility", 0.0) or 0.0,
            atr_pct=atr_pct,
            volume_change_ratio=1.0,
            volume_stability=0.0,
            avg_dollar_volume=float(avg_dollar_volume),
            box_quality=None,
            support_level=float(box_low),
            resistance_level=float(box_high),
        )

        candidate_meta: dict[str, Any] = dict(meta) if meta else {}
        candidate_meta["trend_pattern"] = {
            "pattern_type": pattern_result.pattern_type,
            "score": pattern_result.score,
            "reasons": list(pattern_result.reasons),
            "meta": pattern_meta,
        }
        candidate_meta["gates"] = gate_meta
        candidate_meta["detector"] = "trend_pattern_router"

        signal_type = f"trend_{pattern_result.pattern_type}"

        _ss_rs_slope: float | None = None
        if benchmark_bars and bars:
            try:
                from scanner.gates.relative_strength import compute_rs_slope
                _asset_closes = [float(b.close) for b in bars]
                _bench_closes = [float(b.close) for b in benchmark_bars]
                _ss_rs_slope = compute_rs_slope(_asset_closes, _bench_closes)
            except Exception:
                pass

        _ss_dist_to_support: float | None = None
        if close_price > 0 and isinstance(box_low, (int, float)) and float(box_low) > 0:
            _ss_dist_to_support = (close_price - float(box_low)) / close_price

        candidate_meta["shadow_score"] = {
            "ss_version": 1,
            "ss_atr_pct": float(atr_pct) if atr_pct is not None else None,
            "ss_box_quality": None,
            "ss_volatility": float(pattern_meta.get("volatility", 0.0) or 0.0) or None,
            "ss_volume_chg_ratio": None,
            "ss_rs_slope": float(_ss_rs_slope) if _ss_rs_slope is not None else None,
            "ss_rs_lookback": 20,
            "ss_consolidation_days": 30,
            "ss_dist_to_support_pct": float(_ss_dist_to_support) if _ss_dist_to_support is not None else None,
            "ss_pattern_type": str(pattern_result.pattern_type),
        }

        return PlatformCandidate(
            symbol=symbol,
            detected_at=detected_at,
            window=30,
            score=float(pattern_result.score),
            features=features,
            invalidation_level=float(stop_loss),
            target_level=float(target_price),
            reasons=list(pattern_result.reasons) + [f"pattern:{pattern_result.pattern_type}"],
            meta=candidate_meta,
            signal_type=signal_type,
        )


def detect_trend_pattern_candidate(
    symbol: str,
    bars: list[PriceBar],
    current_date: date,
    config: TrendPatternRouterConfig | None = None,
    *,
    sector_bars: list[PriceBar] | None = None,
    benchmark_bars: list[PriceBar] | None = None,
    detected_at: int | None = None,
    meta: dict[str, Any] | None = None,
) -> Result[PlatformCandidate | None]:
    """Convenience function for detecting trend pattern candidates."""
    router = TrendPatternRouter(config)
    return router.detect(
        symbol=symbol,
        bars=bars,
        current_date=current_date,
        sector_bars=sector_bars,
        benchmark_bars=benchmark_bars,
        detected_at=detected_at,
        meta=meta,
    )
