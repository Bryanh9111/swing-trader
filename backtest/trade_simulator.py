"""Trade exit detection utilities for backtests.

This module provides a small, deterministic helper for scanning open positions
against a day's OHLC bar and emitting exit signals when a stop condition is met.
"""

from __future__ import annotations

from datetime import date
import math
from typing import Final

import msgspec
import structlog

from backtest.portfolio_tracker import Position
from data.interface import PriceBar
from indicators.core import compute_atr_last

logger: Final = structlog.get_logger(__name__).bind(component="trade_simulator")


class StagedTakeProfitConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for staged (partial) take profit exits."""

    enabled: bool = False
    first_target_pct: float = 0.10
    second_target_pct: float = 0.20
    exit_fraction: float = 1.0 / 3.0


class TimeStopOptimizationConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for TIME_STOP optimization (V1 - deprecated)."""

    enabled: bool = False
    skip_if_profit_above: float = 0.05
    early_exit_if_loss_below: float = -0.03
    # Early loss exit: exit before time_stop if loss exceeds threshold for N days
    early_loss_exit_enabled: bool = False
    early_loss_exit_days: int = 20
    early_loss_exit_threshold: float = -0.04


class TimeStopV2Config(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for TIME_STOP v2 smart exit logic.

    TIME_STOP v2 treats time stop as a "decision point" rather than a threshold:
    1. If "extend conditions" met → extend holding (up to max_extend_count times)
    2. If "cut conditions" met → exit immediately
    3. Otherwise → normal TIME_STOP exit

    Extend conditions (any one triggers extend):
    - ATR contracting (consolidating, may breakout)
    - Structure intact (price above EMA or not pulled back too far)

    Cut conditions (forces exit even if extended):
    - No new high for N days + momentum weakening
    - Soft trail triggered (price drops from high)
    """

    enabled: bool = False

    # Base parameters
    base_days: int = 30  # Original TIME_STOP trigger point
    extend_step_days: int = 5  # Days to extend each time
    max_extend_count: int = 2  # Maximum extensions allowed

    # Preconditions gate (all must be met to enable v2 logic)
    only_if_bull_regime: bool = True  # Enable only in bull regime
    min_adx_threshold: float = 25.0  # ADX(14) >= threshold
    min_unrealized_r: float = 1.5  # Profit >= N * R (R = stop distance)

    # ATR contraction detection
    atr_period: int = 14
    atr_sma_short: int = 5  # Recent ATR window
    atr_sma_long: int = 20  # Baseline ATR window
    atr_contract_ratio: float = 0.85  # atr_now < atr_base * ratio = contracting

    # Structure intact conditions (satisfy ANY = structure intact)
    ema_period: int = 20
    structure_pullback_atr_mult: float = 1.2  # Close >= High - k*ATR

    # No new high detection
    no_new_high_days: int = 7  # Bull market default
    no_new_high_days_choppy: int = 10  # Choppy market
    momentum_rsi_period: int = 14
    momentum_rsi_threshold: float = 55.0  # RSI < this = weakening

    # Soft trail protection (activated after first extend)
    soft_trail_enabled: bool = True
    soft_trail_atr_mult: float = 1.5  # Exit if Close < HighestClose - k*ATR


class WeakExitConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for early weak-position exits (V27.1).

    WEAK_EXIT (V27.1 deep-drawdown based):
        Targets "slow bleeding" positions that:
        1. Have been held long enough (weak_days)
        2. Experienced significant drawdown (max_drawdown_pct_threshold)
        3. Still haven't recovered (min_current_unrealized_pct)
        4. Never proved themselves (max_runup_r_gate)
        5. Haven't hit partial TP

    FAIL_TO_MOVE (disabled by default in V27.1):
        Previously caused high-frequency false kills on slow-starter positions.

    Safety valves:
        - Skip if ``partial_exits_completed >= 1`` (already hit PT1, let trailing handle).
        - Skip if ``max_runup_r >= max_runup_r_gate`` (position proved itself).
    """

    enabled: bool = False

    # WEAK_EXIT (V27.1): deep-drawdown based — only kill slow-bleeding positions
    weak_days: int = 12  # only target Slow SL territory (>10d)
    max_drawdown_pct_threshold: float = -0.06  # must have dropped at least 6%
    min_current_unrealized_pct: float = -0.02  # must still be losing at least 2%
    max_runup_r_gate: float = 0.25  # never proved itself (tightened from 0.4)

    # FAIL_TO_MOVE (V27.1): disabled — was causing too many false kills
    ftm_enabled: bool = False
    ftm_days: int = 6
    ftm_min_move_pct: float = 0.008
    ftm_max_runup_r_gate: float = 0.3


class ShadowScoreExitConfig(msgspec.Struct, frozen=True, kw_only=True):
    """V27.6 shadow-score tier classification config — INFRASTRUCTURE ONLY.

    STATUS: Currently disabled (enabled: false). The tier classification has
    proven discriminative power (SL_slow: low=24.8% vs high=9.3%) but the
    WEAK_EXIT injection path failed due to R-based runup_r gate math defect
    for low-ATR stocks. Retained for future reuse in position sizing or
    entry filtering. See docs/calibration/shadow_score_exit_v27_5_1.md.

    Composite z-score = sum(z_i * w_i) where z_i = (val - median) / (IQR + eps).
    Higher composite = better quality (less SL_slow risk).
    """

    enabled: bool = False
    # Composite weights (sign encodes direction: positive = higher is better)
    w_atr_pct: float = 1.0           # high ATR% → less SL_slow risk
    w_rs_slope: float = 1.0          # high RS slope → relative strength
    w_dist_to_support: float = -0.5  # close to support → less room to fall
    # Robust z-score constants (from Phase A data calibration)
    median_atr_pct: float = 0.036876
    iqr_atr_pct: float = 0.023208
    median_rs_slope: float = 0.002367
    iqr_rs_slope: float = 0.003612
    median_dist_to_support: float = 0.013769
    iqr_dist_to_support: float = 0.028509
    # Tier boundaries (from Phase A 30/70 percentile on CDEF training set)
    low_threshold: float = -0.5905   # composite P30
    high_threshold: float = 0.6896   # composite P70
    # Low tier: earlier/tighter WEAK_EXIT
    low_weak_days: int = 8
    low_max_dd_pct: float = -0.04
    low_runup_r_gate: float = 0.35
    low_current_unrealized_pct: float = -0.01
    # Low tier winner protection: max_runup_r >= this → never low-tier weak exit
    low_winner_protect_runup_r: float = 1.0
    # High tier: more patience
    high_weak_days: int = 16
    high_max_dd_pct: float = -0.08


class TrailingStopConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for trailing stop exit logic.

    Activates when unrealized profit reaches activate_at_R multiples of R
    (R = entry_price - stop_loss_price). Once active, exits when bar.low
    drops below highest_close * (1 - effective_trail_pct).

    When atr_multiplier > 0, trail width auto-scales to each stock's
    volatility: effective_trail_pct = max(min_trail_pct, atr_multiplier * ATR%).
    Falls back to fixed trail_pct when ATR data is unavailable.

    After partial take-profit (PT1), trailing is forced on residual if
    force_after_partial_tp is True.
    """

    enabled: bool = False
    activate_at_R: float = 1.5
    trail_pct: float = 0.05           # fallback when ATR unavailable
    min_trail_pct: float = 0.05       # floor for ATR-adaptive trail
    atr_multiplier: float = 0.8       # trail = max(min_trail_pct, atr_multiplier * ATR%)
    atr_period: int = 14              # ATR calculation period
    force_after_partial_tp: bool = True


class PositionTrackingData(msgspec.Struct, kw_only=True):
    """Mutable tracking data for TIME_STOP v2 logic.

    This is maintained separately from Position (which is immutable).
    """

    symbol: str
    highest_close_since_entry: float  # Track peak for soft trail
    extend_count: int = 0  # How many times TIME_STOP was extended
    soft_trail_active: bool = False  # Whether soft trail is activated
    last_extend_date: str | None = None  # Date of last extension

    trailing_stop_active: bool = False
    entry_atr_pct: float | None = None  # ATR% locked at trailing activation


class TimeStopV2Decision(msgspec.Struct, frozen=True, kw_only=True):
    """Result of TIME_STOP v2 evaluation."""

    action: str  # "extend", "cut", "stop"
    reason: str  # Human-readable explanation
    soft_trail_price: float | None = None  # If extend, the soft trail exit price


class TimeStopV2Evaluator:
    """Evaluates TIME_STOP v2 conditions for smart exit decisions.

    At TIME_STOP trigger point, this evaluator decides:
    1. "extend" - Continue holding (ATR contracting or structure intact)
    2. "cut" - Exit immediately (no new high + momentum weak, or soft trail hit)
    3. "stop" - Normal TIME_STOP exit (default)
    """

    def __init__(self, config: TimeStopV2Config) -> None:
        self._config = config

    def evaluate(
        self,
        position: "Position",
        bars: list["PriceBar"],
        current_date: date,
        tracking: PositionTrackingData,
        hold_days: int,
        is_bull_regime: bool = True,
    ) -> tuple[TimeStopV2Decision, PositionTrackingData]:
        """Evaluate TIME_STOP v2 conditions.

        Args:
            position: The open position being evaluated.
            bars: Historical price bars for the symbol (most recent last).
            current_date: Current trading date.
            tracking: Mutable tracking data for this position.
            hold_days: Days held so far.
            is_bull_regime: Whether the current market regime is bull.

        Returns:
            Tuple of (decision, updated_tracking_data).
        """
        cfg = self._config
        if not cfg.enabled or len(bars) < max(cfg.atr_sma_long, cfg.ema_period) + cfg.atr_period:
            return TimeStopV2Decision(action="stop", reason="insufficient_data_or_disabled"), tracking

        # Preconditions gate: check whether v2 should be enabled for this position.
        if not self._check_preconditions(position, bars, is_bull_regime):
            # Preconditions not met; let caller fall back to legacy TIME_STOP logic.
            return TimeStopV2Decision(action="none", reason="preconditions_not_met"), tracking

        current_close = float(bars[-1].close)
        entry_price = float(position.entry_price)

        # Update highest close tracking
        new_highest = max(tracking.highest_close_since_entry, current_close)
        updated_tracking = PositionTrackingData(
            symbol=tracking.symbol,
            highest_close_since_entry=new_highest,
            extend_count=tracking.extend_count,
            soft_trail_active=tracking.soft_trail_active,
            last_extend_date=tracking.last_extend_date,
            trailing_stop_active=tracking.trailing_stop_active,
            entry_atr_pct=tracking.entry_atr_pct,
        )

        # Calculate effective TIME_STOP day considering extensions
        effective_stop_day = cfg.base_days + (tracking.extend_count * cfg.extend_step_days)
        if hold_days < effective_stop_day:
            # Not at TIME_STOP yet, but check soft trail if active
            if tracking.soft_trail_active and cfg.soft_trail_enabled:
                soft_trail_price = self._calc_soft_trail_price(bars, new_highest, cfg)
                if soft_trail_price is not None and current_close < soft_trail_price:
                    return TimeStopV2Decision(
                        action="cut",
                        reason=f"soft_trail_triggered (close={current_close:.2f} < trail={soft_trail_price:.2f})",
                    ), updated_tracking
            # Not at TIME_STOP point
            return TimeStopV2Decision(action="none", reason="not_at_time_stop"), updated_tracking

        # At TIME_STOP point - evaluate conditions
        # 1. Check soft trail first (highest priority cut condition)
        if tracking.soft_trail_active and cfg.soft_trail_enabled:
            soft_trail_price = self._calc_soft_trail_price(bars, new_highest, cfg)
            if soft_trail_price is not None and current_close < soft_trail_price:
                return TimeStopV2Decision(
                    action="cut",
                    reason=f"soft_trail_triggered (close={current_close:.2f} < trail={soft_trail_price:.2f})",
                ), updated_tracking

        # Check if max extensions reached (can no longer extend)
        if tracking.extend_count >= cfg.max_extend_count:
            return TimeStopV2Decision(action="stop", reason="max_extend_reached"), updated_tracking

        # 2. Check "no new high + momentum weak" cut condition
        if self._check_no_new_high_weak_momentum(bars, new_highest, cfg):
            return TimeStopV2Decision(
                action="cut",
                reason="no_new_high_and_momentum_weak",
            ), updated_tracking

        # 3. Check extend conditions
        atr_contracting = self._check_atr_contracting(bars, cfg)
        structure_intact = self._check_structure_intact(bars, new_highest, cfg)

        if atr_contracting or structure_intact:
            # Extend - activate soft trail and update tracking
            soft_trail_price = self._calc_soft_trail_price(bars, new_highest, cfg) if cfg.soft_trail_enabled else None
            extend_reason = []
            if atr_contracting:
                extend_reason.append("atr_contracting")
            if structure_intact:
                extend_reason.append("structure_intact")

            extended_tracking = PositionTrackingData(
                symbol=tracking.symbol,
                highest_close_since_entry=new_highest,
                extend_count=tracking.extend_count + 1,
                soft_trail_active=True,
                last_extend_date=str(current_date),
                trailing_stop_active=tracking.trailing_stop_active,
                entry_atr_pct=tracking.entry_atr_pct,
            )

            return TimeStopV2Decision(
                action="extend",
                reason=f"extend_{tracking.extend_count + 1}: {', '.join(extend_reason)}",
                soft_trail_price=soft_trail_price,
            ), extended_tracking

        # Default: normal TIME_STOP
        return TimeStopV2Decision(action="stop", reason="no_extend_conditions_met"), updated_tracking

    def _check_preconditions(self, position: "Position", bars: list["PriceBar"], is_bull_regime: bool) -> bool:
        """Check if v2 preconditions are met.

        v2 is enabled only when ALL conditions are met:
        1) Bull regime (if only_if_bull_regime=True)
        2) ADX(14) >= min_adx_threshold
        3) Profit >= min_unrealized_r * R (R = stop distance)
        """

        cfg = self._config

        if cfg.only_if_bull_regime and not is_bull_regime:
            return False

        if len(bars) < 14:
            return False

        from indicators.core import compute_adx_last

        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
        adx = compute_adx_last(highs=highs, lows=lows, closes=closes, period=14)
        if adx is None or adx < cfg.min_adx_threshold:
            return False

        entry_price = float(position.entry_price)
        stop_loss = float(position.stop_loss_price)
        current_close = float(bars[-1].close)

        r_value = abs(entry_price - stop_loss)
        if r_value <= 0:
            return False

        unrealized_pnl = current_close - entry_price
        unrealized_r = unrealized_pnl / r_value
        if unrealized_r < cfg.min_unrealized_r:
            return False

        return True

    def _check_atr_contracting(self, bars: list["PriceBar"], cfg: TimeStopV2Config) -> bool:
        """Check if ATR is contracting (consolidation/coiling)."""
        from indicators.core import compute_atr_last

        if len(bars) < cfg.atr_sma_long + cfg.atr_period:
            return False

        # Calculate ATR% for recent window
        recent_bars = bars[-(cfg.atr_sma_short + cfg.atr_period) :]
        atr_now = compute_atr_last(recent_bars, period=cfg.atr_period, percentage=True)

        # Calculate ATR% for baseline window
        baseline_bars = bars[-(cfg.atr_sma_long + cfg.atr_period) :]
        atr_base = compute_atr_last(baseline_bars, period=cfg.atr_period, percentage=True)

        if atr_now is None or atr_base is None or atr_base <= 0:
            return False

        # ATR contracting if recent ATR < baseline ATR * ratio
        return atr_now < atr_base * cfg.atr_contract_ratio

    def _check_structure_intact(
        self, bars: list["PriceBar"], highest_close: float, cfg: TimeStopV2Config
    ) -> bool:
        """Check if price structure is intact (any condition = True).

        Conditions (satisfy ANY):
        A. Close > EMA(20)
        B. Close >= HighestClose - k*ATR (not pulled back too far)
        """
        from indicators.core import compute_atr_last, compute_ema_last

        if len(bars) < max(cfg.ema_period, cfg.atr_period + 1):
            return False

        current_close = float(bars[-1].close)

        # Condition A: Close > EMA20
        ema20 = compute_ema_last(bars, period=cfg.ema_period)
        if ema20 is not None and current_close > ema20:
            return True

        # Condition B: Close >= HighestClose - k*ATR
        atr = compute_atr_last(bars, period=cfg.atr_period, percentage=False)
        if atr is not None and atr > 0:
            pullback_threshold = highest_close - (cfg.structure_pullback_atr_mult * atr)
            if current_close >= pullback_threshold:
                return True

        return False

    def _check_no_new_high_weak_momentum(
        self, bars: list["PriceBar"], highest_close: float, cfg: TimeStopV2Config
    ) -> bool:
        """Check if no new high for N days AND momentum is weak."""
        from indicators.core import compute_rsi_last

        if len(bars) < cfg.no_new_high_days:
            return False

        current_close = float(bars[-1].close)

        # Check no new high in recent N days
        recent_closes = [float(b.close) for b in bars[-cfg.no_new_high_days :]]
        recent_max = max(recent_closes)

        # If current close is the high, there IS a new high
        if current_close >= highest_close - 0.001:  # Small tolerance
            return False

        # Also check if recent period has no approach to high
        if recent_max >= highest_close * 0.99:  # Within 1% of high
            return False

        # Check momentum (RSI)
        rsi = compute_rsi_last(bars, period=cfg.momentum_rsi_period)
        if rsi is None:
            return False

        # Weak momentum: RSI below threshold
        return rsi < cfg.momentum_rsi_threshold

    def _calc_soft_trail_price(
        self, bars: list["PriceBar"], highest_close: float, cfg: TimeStopV2Config
    ) -> float | None:
        """Calculate soft trail stop price."""
        from indicators.core import compute_atr_last

        atr = compute_atr_last(bars, period=cfg.atr_period, percentage=False)
        if atr is None or atr <= 0:
            return None

        return highest_close - (cfg.soft_trail_atr_mult * atr)


class ExitSignal(msgspec.Struct, frozen=True, kw_only=True):
    """Immutable signal describing how a position should be exited.

    Attributes:
        symbol: The position symbol to exit.
        exit_price: The price at which the exit should be simulated.
        exit_reason: One of ``STOP_LOSS``, ``TAKE_PROFIT``, ``TIME_STOP``.
    """

    symbol: str
    exit_price: float
    exit_reason: str  # STOP_LOSS | TAKE_PROFIT | TIME_STOP | TRAILING_STOP | PARTIAL_TP_1 | PARTIAL_TP_2 | EARLY_LOSS_EXIT | EARLY_STOP
    exit_quantity: float | None = None  # None => full exit; otherwise partial
    is_partial: bool = False


def classify_shadow_tier(
    shadow_scores: dict[str, object] | None,
    cfg: ShadowScoreExitConfig,
) -> str:
    """Return 'high'/'mid'/'low' based on composite z-score of shadow dimensions.

    Infrastructure function — currently only called from WEAK_EXIT path which is
    disabled. Designed for reuse by future position sizing or entry filtering.
    Falls back to 'mid' when: disabled, missing data, or < 2 valid dimensions.
    """
    if not cfg.enabled or not shadow_scores:
        return "mid"
    _dims = [
        ("ss_atr_pct", cfg.w_atr_pct, cfg.median_atr_pct, cfg.iqr_atr_pct),
        ("ss_rs_slope", cfg.w_rs_slope, cfg.median_rs_slope, cfg.iqr_rs_slope),
        ("ss_dist_to_support_pct", cfg.w_dist_to_support, cfg.median_dist_to_support, cfg.iqr_dist_to_support),
    ]
    components: list[float] = []
    for key, weight, median, iqr in _dims:
        val = shadow_scores.get(key)
        if isinstance(val, (int, float)):
            z = (float(val) - median) / (iqr + 1e-9)
            components.append(z * weight)
    if len(components) < 2:
        return "mid"
    composite = sum(components)
    if composite < cfg.low_threshold:
        return "low"
    if composite > cfg.high_threshold:
        return "high"
    return "mid"


def _resolve_weak_exit_thresholds(
    we_cfg: WeakExitConfig,
    ss_cfg: ShadowScoreExitConfig | None,
    tier: str,
) -> tuple[int, float, float, float]:
    """Return (weak_days, dd_threshold, runup_gate, current_pct) per tier.

    Mid tier or unavailable config → global WeakExitConfig defaults.
    """
    if ss_cfg is None or not ss_cfg.enabled or tier == "mid":
        return (we_cfg.weak_days, we_cfg.max_drawdown_pct_threshold,
                we_cfg.max_runup_r_gate, we_cfg.min_current_unrealized_pct)
    if tier == "low":
        return (ss_cfg.low_weak_days, ss_cfg.low_max_dd_pct,
                ss_cfg.low_runup_r_gate, ss_cfg.low_current_unrealized_pct)
    # high
    return (ss_cfg.high_weak_days, ss_cfg.high_max_dd_pct,
            we_cfg.max_runup_r_gate, we_cfg.min_current_unrealized_pct)


class TradeSimulator:
    """Check positions against daily bar data and emit exit signals."""

    def check_exits(
        self,
        positions: dict[str, Position],
        bar_data: dict[str, PriceBar],
        current_date: date,
        max_hold_days: int = 30,
        *,
        is_bull_regime: bool = True,
        staged_take_profit: StagedTakeProfitConfig | None = None,
        time_stop_optimization: TimeStopOptimizationConfig | None = None,
        time_stop_v2: TimeStopV2Config | None = None,
        trailing_stop: TrailingStopConfig | None = None,
        weak_exit: WeakExitConfig | None = None,
        shadow_score_exit: ShadowScoreExitConfig | None = None,
        runup_tracking: dict[str, float] | None = None,
        drawdown_tracking: dict[str, float] | None = None,
        historical_bars: dict[str, list[PriceBar]] | None = None,
        position_tracking: dict[str, PositionTrackingData] | None = None,
    ) -> tuple[list[ExitSignal], dict[str, PositionTrackingData] | None]:
        """Return exit signals for positions that should be closed on ``current_date``.

        Exit priority when multiple conditions trigger on the same day:
        ``STOP_LOSS`` > ``TAKE_PROFIT``/``TRAILING_STOP`` > ``WEAK_EXIT``/``FAIL_TO_MOVE`` > ``TIME_STOP``.

        Args:
            positions: Mapping of symbol to an open :class:`backtest.portfolio_tracker.Position`.
            bar_data: Mapping of symbol to daily OHLC :class:`data.interface.PriceBar`.
            current_date: Trading date for the provided bar data.
            max_hold_days: Maximum holding period in calendar days. A position is
                considered timed-out when ``(current_date - entry_date).days >= max_hold_days``.

        Returns:
            Tuple of (exit_signals, updated_position_tracking).
            If v2 is disabled, position_tracking is returned unchanged (may be None).
        """

        if max_hold_days <= 0:
            raise ValueError("max_hold_days must be positive")

        signals: list[ExitSignal] = []

        # ZERO-IMPACT SHORT CIRCUIT: determine which features need position tracking
        v2_enabled = time_stop_v2 is not None and time_stop_v2.enabled
        ts_enabled = trailing_stop is not None and trailing_stop.enabled
        needs_tracking = v2_enabled or ts_enabled

        if not needs_tracking:
            v2_evaluator = None
            updated_position_tracking = position_tracking  # Pass through unchanged (may be None)
        else:
            v2_evaluator = TimeStopV2Evaluator(time_stop_v2) if v2_enabled else None
            updated_position_tracking = dict(position_tracking or {})

        for symbol, position in positions.items():
            bar = bar_data.get(symbol)
            if bar is None:
                logger.warning(
                    "missing_bar_data_skip_exit_check",
                    symbol=symbol,
                    current_date=str(current_date),
                )
                continue

            if bar.low <= position.stop_loss_price:
                signals.append(
                    ExitSignal(
                        symbol=symbol,
                        exit_price=position.stop_loss_price,
                        exit_reason="STOP_LOSS",
                    )
                )
                continue

            staged_cfg = staged_take_profit
            if (
                staged_cfg is not None
                and staged_cfg.enabled
                and int(position.partial_exits_completed) < 2
                and float(position.entry_price) > 0
                and bar.high < position.take_profit_price
            ):
                unrealized_pct = (bar.close - position.entry_price) / position.entry_price
                original_qty = position.original_quantity if position.original_quantity is not None else position.quantity
                exit_qty = math.floor(float(original_qty) * float(staged_cfg.exit_fraction))
                if exit_qty > 0 and float(exit_qty) < float(position.quantity):
                    if int(position.partial_exits_completed) == 0 and unrealized_pct >= float(staged_cfg.first_target_pct):
                        signals.append(
                            ExitSignal(
                                symbol=symbol,
                                exit_price=bar.close,
                                exit_reason="PARTIAL_TP_1",
                                exit_quantity=float(exit_qty),
                                is_partial=True,
                            )
                        )
                        continue
                    if int(position.partial_exits_completed) == 1 and unrealized_pct >= float(staged_cfg.second_target_pct):
                        signals.append(
                            ExitSignal(
                                symbol=symbol,
                                exit_price=bar.close,
                                exit_reason="PARTIAL_TP_2",
                                exit_quantity=float(exit_qty),
                                is_partial=True,
                            )
                        )
                        continue

            if bar.high >= position.take_profit_price:
                signals.append(
                    ExitSignal(
                        symbol=symbol,
                        exit_price=position.take_profit_price,
                        exit_reason="TAKE_PROFIT",
                    )
                )
                continue

            hold_days = (current_date - position.entry_date).days

            # Early loss exit: exit before time_stop if loss exceeds threshold
            opt_cfg = time_stop_optimization
            if (
                opt_cfg is not None
                and opt_cfg.enabled
                and opt_cfg.early_loss_exit_enabled
                and hold_days >= opt_cfg.early_loss_exit_days
                and float(position.entry_price) > 0
            ):
                unrealized_pct = (bar.close - position.entry_price) / position.entry_price
                if unrealized_pct < float(opt_cfg.early_loss_exit_threshold):
                    logger.info(
                        "early_loss_exit_triggered",
                        symbol=symbol,
                        unrealized_pct=unrealized_pct,
                        hold_days=hold_days,
                        threshold=opt_cfg.early_loss_exit_threshold,
                    )
                    signals.append(
                        ExitSignal(
                            symbol=symbol,
                            exit_price=bar.close,
                            exit_reason="EARLY_LOSS_EXIT",
                        )
                    )
                    continue

            # --- Shared position tracking + trailing stop + TIME_STOP v2 ---
            trailing_is_active = False
            if needs_tracking:
                tracking = updated_position_tracking.get(symbol)
                if tracking is None:
                    tracking = PositionTrackingData(
                        symbol=symbol,
                        highest_close_since_entry=float(position.entry_price),
                        extend_count=0,
                        soft_trail_active=False,
                        last_extend_date=None,
                        trailing_stop_active=False,
                        entry_atr_pct=None,
                    )

                # (C) Trailing stop exit check — BEFORE peak update to avoid lookahead bias.
                # Trail is based on previous days' peak (known at market open), checked
                # against today's low.  Activation from today's close only takes effect
                # on the NEXT bar.
                if ts_enabled and tracking.trailing_stop_active:
                    # Use ATR% locked at activation time (stable, no daily jitter)
                    effective_trail_pct = trailing_stop.trail_pct  # fallback
                    if trailing_stop.atr_multiplier > 0 and tracking.entry_atr_pct is not None:
                        effective_trail_pct = max(trailing_stop.min_trail_pct, trailing_stop.atr_multiplier * tracking.entry_atr_pct)
                    trail_stop_price = float(tracking.highest_close_since_entry) * (1.0 - effective_trail_pct)
                    if float(bar.low) <= trail_stop_price:
                        # Gap-down handling: if bar opened below trail, realistic
                        # execution is at open (can't get trail_stop on a gap).
                        effective_exit = min(trail_stop_price, float(bar.open))
                        logger.info(
                            "trailing_stop_exit",
                            symbol=symbol,
                            trail_stop_price=trail_stop_price,
                            effective_exit=effective_exit,
                            bar_open=float(bar.open),
                            bar_low=float(bar.low),
                            highest_close=tracking.highest_close_since_entry,
                            effective_trail_pct=effective_trail_pct,
                        )
                        updated_position_tracking[symbol] = tracking
                        signals.append(
                            ExitSignal(
                                symbol=symbol,
                                exit_price=effective_exit,
                                exit_reason="TRAILING_STOP",
                            )
                        )
                        continue

                # (A) Update peak price tracking (shared by trailing stop and v2)
                if float(bar.close) > float(tracking.highest_close_since_entry):
                    tracking = PositionTrackingData(
                        symbol=tracking.symbol,
                        highest_close_since_entry=float(bar.close),
                        extend_count=tracking.extend_count,
                        soft_trail_active=tracking.soft_trail_active,
                        last_extend_date=tracking.last_extend_date,
                        trailing_stop_active=tracking.trailing_stop_active,
                        entry_atr_pct=tracking.entry_atr_pct,
                    )

                # (B) Trailing stop activation check — uses today's close, takes
                # effect starting from the next bar (no same-day activation+exit).
                if ts_enabled and not tracking.trailing_stop_active:
                    entry_price = float(position.entry_price)
                    stop_loss = float(position.stop_loss_price)
                    r_value = abs(entry_price - stop_loss)

                    if r_value > 0:
                        unrealized_r = (float(bar.close) - entry_price) / r_value
                        should_activate = unrealized_r >= trailing_stop.activate_at_R
                        if (
                            not should_activate
                            and trailing_stop.force_after_partial_tp
                            and int(position.partial_exits_completed) >= 1
                        ):
                            should_activate = True

                        if should_activate:
                            # Lock ATR% at activation time for stable trail width
                            locked_atr_pct: float | None = None
                            if trailing_stop.atr_multiplier > 0:
                                bars_for_atr = (historical_bars or {}).get(symbol) or []
                                if bars_for_atr:
                                    locked_atr_pct = compute_atr_last(bars_for_atr, period=trailing_stop.atr_period, percentage=True)
                            tracking = PositionTrackingData(
                                symbol=tracking.symbol,
                                highest_close_since_entry=tracking.highest_close_since_entry,
                                extend_count=tracking.extend_count,
                                soft_trail_active=tracking.soft_trail_active,
                                last_extend_date=tracking.last_extend_date,
                                trailing_stop_active=True,
                                entry_atr_pct=locked_atr_pct,
                            )
                            logger.info(
                                "trailing_stop_activated",
                                symbol=symbol,
                                highest_close=tracking.highest_close_since_entry,
                                partial_exits=int(position.partial_exits_completed),
                                entry_atr_pct=locked_atr_pct,
                            )

                trailing_is_active = tracking.trailing_stop_active
                updated_position_tracking[symbol] = tracking

            # --- WEAK_EXIT (V27.1): deep-drawdown based slow-bleed detection ---
            # Priority: after SL/TP/TRAILING, before TIME_STOP
            # Only kill positions that: held long + dropped deep + still weak + never proved
            we_cfg = weak_exit
            if (
                we_cfg is not None
                and we_cfg.enabled
                and float(position.entry_price) > 0
                and not trailing_is_active  # trailing owns the exit if active
            ):
                _we_entry = float(position.entry_price)
                _we_r = abs(_we_entry - float(position.stop_loss_price))
                _has_partial_tp = int(position.partial_exits_completed) >= 1

                # Compute max_runup_r for safety valve
                _max_runup_r: float | None = None
                if _we_r > 0 and runup_tracking is not None:
                    _max_high = runup_tracking.get(symbol, _we_entry)
                    _max_runup_r = (_max_high - _we_entry) / _we_r

                # Compute max_drawdown_pct from tracking
                _max_dd_pct: float | None = None
                if drawdown_tracking is not None and _we_entry > 0:
                    _min_low = drawdown_tracking.get(symbol, _we_entry)
                    _max_dd_pct = (_min_low - _we_entry) / _we_entry

                # Compute current unrealized pct
                _current_unrealized_pct = (bar.close - _we_entry) / _we_entry if _we_entry > 0 else 0.0

                # V27.6: Shadow score tier-based adaptive WEAK_EXIT thresholds.
                # STATUS: infrastructure pre-wired, currently DISABLED (enabled: false in YAML).
                # When disabled: shadow_score_exit is None → _ss_tier="mid" → all thresholds
                # fall back to global we_cfg defaults → zero behavioral change vs V27.5.1.
                # WHY KEPT: classify_shadow_tier() has proven discriminative power (SL_slow:
                # low=24.8% vs high=9.3%, 15.5pp diff). Retained for future position sizing
                # or entry filtering where the tier signal has better leverage than exit gates.
                # WHY DISABLED: WEAK_EXIT's multi-gate conditions (dd + unrealized + runup_r +
                # partial_tp) intersect to ≈ empty set for low-ATR stocks due to R-denominator
                # scaling defect — 4 iterations all produced zero new triggers.
                _ss_tier = classify_shadow_tier(position.shadow_scores, shadow_score_exit) \
                    if shadow_score_exit is not None else "mid"

                # Winner protection: proven positions (high runup) never get low-tier treatment
                if (
                    _ss_tier == "low"
                    and shadow_score_exit is not None
                    and _max_runup_r is not None
                    and _max_runup_r >= shadow_score_exit.low_winner_protect_runup_r
                ):
                    _ss_tier = "mid"

                _eff_weak_days, _eff_dd_threshold, _eff_runup_gate, _eff_current_pct = \
                    _resolve_weak_exit_thresholds(we_cfg, shadow_score_exit, _ss_tier)

                # WEAK_EXIT: slow-bleed positions that dropped deep and haven't recovered
                if (
                    hold_days >= _eff_weak_days
                    and _max_dd_pct is not None
                    and _max_dd_pct <= _eff_dd_threshold
                    and _current_unrealized_pct <= _eff_current_pct
                    and not _has_partial_tp
                    and _max_runup_r is not None
                    and _max_runup_r < _eff_runup_gate
                ):
                    logger.info(
                        "weak_exit_triggered",
                        symbol=symbol,
                        hold_days=hold_days,
                        max_drawdown_pct=_max_dd_pct,
                        current_unrealized_pct=_current_unrealized_pct,
                        max_runup_r=_max_runup_r,
                        shadow_tier=_ss_tier,
                    )
                    signals.append(
                        ExitSignal(
                            symbol=symbol,
                            exit_price=bar.close,
                            exit_reason="WEAK_EXIT",
                        )
                    )
                    continue

                # FAIL_TO_MOVE (V27.1: disabled by default — high false-kill rate)
                if (
                    we_cfg.ftm_enabled
                    and hold_days >= we_cfg.ftm_days
                    and runup_tracking is not None
                    and not _has_partial_tp
                    and _max_runup_r is not None
                    and _max_runup_r < we_cfg.ftm_max_runup_r_gate
                ):
                    _ftm_max_high = runup_tracking.get(symbol, _we_entry)
                    if _ftm_max_high < _we_entry * (1.0 + we_cfg.ftm_min_move_pct):
                        logger.info(
                            "fail_to_move_exit_triggered",
                            symbol=symbol,
                            hold_days=hold_days,
                            max_high=_ftm_max_high,
                            required=_we_entry * (1.0 + we_cfg.ftm_min_move_pct),
                            max_runup_r=_max_runup_r,
                        )
                        signals.append(
                            ExitSignal(
                                symbol=symbol,
                                exit_price=bar.close,
                                exit_reason="FAIL_TO_MOVE",
                            )
                        )
                        continue

            # TIME_STOP v2 (evaluate before legacy TIME_STOP logic)
            if v2_evaluator is not None:
                tracking = updated_position_tracking[symbol]
                # Peak tracking already done above in shared block

                bars = (historical_bars or {}).get(symbol) or []
                decision, tracking = v2_evaluator.evaluate(
                    position=position,
                    bars=bars,
                    current_date=current_date,
                    tracking=tracking,
                    hold_days=hold_days,
                    is_bull_regime=is_bull_regime,
                )
                updated_position_tracking[symbol] = tracking

                if decision.action == "cut":
                    # Trailing stop active → suppress v2 cut (trailing owns the exit)
                    if trailing_is_active:
                        continue
                    logger.info(
                        "time_stop_v2_cut",
                        symbol=symbol,
                        reason=decision.reason,
                        hold_days=hold_days,
                    )
                    signals.append(
                        ExitSignal(
                            symbol=symbol,
                            exit_price=bar.close,
                            exit_reason="TIME_STOP",
                        )
                    )
                    continue

                if decision.action == "extend":
                    logger.info(
                        "time_stop_v2_extend",
                        symbol=symbol,
                        reason=decision.reason,
                        hold_days=hold_days,
                        extend_count=tracking.extend_count,
                        soft_trail_price=decision.soft_trail_price,
                    )
                    continue

                # When v2 can evaluate, it owns the time-stop cadence; don't apply legacy max_hold_days.
                # If v2 can't run (insufficient data or preconditions not met), fall back to legacy behavior.
                if decision.reason not in {"insufficient_data_or_disabled", "preconditions_not_met"}:
                    if decision.action == "stop":
                        # Trailing stop active → suppress v2 stop
                        if trailing_is_active:
                            continue
                        signals.append(
                            ExitSignal(
                                symbol=symbol,
                                exit_price=bar.close,
                                exit_reason="TIME_STOP",
                            )
                        )
                    continue

            # Trailing stop active → suppress legacy TIME_STOP
            if trailing_is_active:
                continue

            if hold_days >= max_hold_days:
                opt_cfg = time_stop_optimization
                if opt_cfg is not None and opt_cfg.enabled and float(position.entry_price) > 0:
                    unrealized_pct = (bar.close - position.entry_price) / position.entry_price
                    if unrealized_pct > float(opt_cfg.skip_if_profit_above):
                        logger.info(
                            "time_stop_skipped_profitable",
                            symbol=symbol,
                            unrealized_pct=unrealized_pct,
                            hold_days=hold_days,
                        )
                        continue
                    if unrealized_pct < float(opt_cfg.early_exit_if_loss_below):
                        signals.append(
                            ExitSignal(
                                symbol=symbol,
                                exit_price=bar.close,
                                exit_reason="EARLY_STOP",
                            )
                        )
                        continue

                signals.append(
                    ExitSignal(
                        symbol=symbol,
                        exit_price=bar.close,
                        exit_reason="TIME_STOP",
                    )
                )

        return signals, updated_position_tracking
