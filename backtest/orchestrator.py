"""Phase 1 simplified backtest orchestration.

This module wires together:
- The existing EOD scan pipeline (:class:`orchestrator.eod_scan.EODScanOrchestrator`)
- A minimal portfolio + exit simulator (:mod:`backtest.portfolio_tracker`, :mod:`backtest.trade_simulator`)
- Basic reporting (:mod:`backtest.reporter`)

The goal of Phase 1 is to provide an end-to-end backtest loop with pragmatic
error handling and structured logging, without introducing full journal replay
or advanced performance analytics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Final

import msgspec
import structlog

from backtest.portfolio_tracker import PortfolioTracker, Position
from backtest.reporter import BacktestReporter, BacktestStats
from backtest.trade_simulator import (
    PositionTrackingData,
    ShadowScoreExitConfig,
    StagedTakeProfitConfig,
    TimeStopOptimizationConfig,
    TimeStopV2Config,
    TradeSimulator,
    TrailingStopConfig,
    WeakExitConfig,
)
from common.interface import ResultStatus
from data.interface import PriceBar, PriceSeriesSnapshot
from journal.interface import RunType
from orchestrator import EODScanOrchestrator
from portfolio.capital_allocator import CapitalAllocator
from portfolio.interface import AllocationResult, CapitalAllocationConfig
from portfolio.position_rotator import PositionRotator, RotationConfig, RotationDecision
from strategy.interface import IntentType, OrderIntentSet, TradeIntent

logger: Final = structlog.get_logger(__name__).bind(component="backtest_orchestrator")


@dataclass(frozen=True, slots=True)
class _DailyScanArtifacts:
    """Container for per-day extracted artifacts from the EOD scan pipeline."""

    intent_set: OrderIntentSet | None
    price_snapshots: dict[str, PriceSeriesSnapshot]


class BacktestOrchestrator:
    """Main coordinator for a simplified daily backtest loop (Phase 1).

    The orchestrator simulates:
    - Checking exits for open positions using each day's OHLC bar
    - Executing pending entry intents at the open of the day
    - Running the existing EOD scan pipeline to generate next-day entry intents
    - Generating summary statistics and exporting completed trades

    Notes:
        - This implementation focuses on correctness and clarity rather than speed.
        - Missing data or insufficient cash should not abort the backtest; the run continues.
    """

    _VALID_REGIME_MODES: Final[frozenset[str]] = frozenset({"auto", "bull", "choppy", "bear", "none"})

    def __init__(
        self,
        *,
        eod_orchestrator: EODScanOrchestrator,
        initial_capital: float = 2000.0,
        max_hold_days: int = 30,
        output_dir: str = "backtest_results",
        quiet: bool = False,
        regime_mode: str = "none",
        capital_allocation_config: CapitalAllocationConfig | None = None,
        rotation_config: RotationConfig | None = None,
        candidate_filter_config: dict[str, Any] | None = None,
    ) -> None:
        """Create a backtest orchestrator.

        Args:
            eod_orchestrator: Existing EOD scan orchestrator (reused pipeline).
            initial_capital: Starting cash.
            max_hold_days: Maximum holding period in calendar days.
            output_dir: Directory for exported artifacts (e.g., trades CSV).
            quiet: If True, suppress detailed logs; only show one line per day.
            regime_mode: Market regime integration mode (none, auto, bull, choppy, bear).
        """

        if initial_capital < 0:
            raise ValueError("initial_capital must be non-negative")
        if max_hold_days <= 0:
            raise ValueError("max_hold_days must be positive")

        normalised_mode = str(regime_mode).strip().lower()
        if normalised_mode not in self._VALID_REGIME_MODES:
            raise ValueError(f"Unsupported regime_mode {regime_mode!r}. Options: {sorted(self._VALID_REGIME_MODES)}")

        self._eod_orchestrator = eod_orchestrator
        self._initial_capital = float(initial_capital)
        self._max_hold_days = int(max_hold_days)
        self._output_dir = str(output_dir)
        self._quiet = bool(quiet)
        self._regime_mode = normalised_mode
        self._capital_allocator = (
            CapitalAllocator(config=capital_allocation_config) if capital_allocation_config is not None else None
        )
        self._position_rotator = (
            PositionRotator(rotation_config=rotation_config) if rotation_config is not None else None
        )

        # V27.5: ATR% hard filter for trend candidates
        _cf = candidate_filter_config or {}
        self._trend_atr_pct_min: float | None = _cf.get("trend_atr_pct_min")
        self._trend_atr_pct_max: float | None = _cf.get("trend_atr_pct_max")

    def run_backtest(
        self,
        trading_days: list[date],
        config: dict[str, Any],
        trades_filename: str = "trades.csv",
    ) -> BacktestStats:
        """Run a simplified backtest over ``trading_days``.

        Daily loop:
            1) Check exits for open positions using daily OHLC bar
            2) Execute pending entry intents at the open of the day
            3) Run EOD scan; schedule next-day entry intents
            4) Log daily summary (equity/positions/pending)

        Args:
            trading_days: Ordered list of trading dates to simulate.
            config: EOD scan orchestrator configuration mapping.

        Returns:
            Aggregate :class:`backtest.reporter.BacktestStats` for the run.
        """

        portfolio = PortfolioTracker(initial_capital=self._initial_capital)
        simulator = TradeSimulator()
        reporter = BacktestReporter()

        # Fallback configs from initial run_config (used when regime mode is disabled)
        strategy_cfg = self._extract_strategy_config(config)
        fallback_staged_take_profit_cfg: StagedTakeProfitConfig | None = None
        staged_take_profit_raw = strategy_cfg.get("staged_take_profit")
        if isinstance(staged_take_profit_raw, dict) and staged_take_profit_raw.get("enabled", False):
            fallback_staged_take_profit_cfg = StagedTakeProfitConfig(
                enabled=True,
                first_target_pct=float(staged_take_profit_raw.get("first_target_pct", 0.10)),
                second_target_pct=float(staged_take_profit_raw.get("second_target_pct", 0.20)),
                exit_fraction=float(staged_take_profit_raw.get("exit_fraction", 1.0 / 3.0)),
            )

        fallback_time_stop_optimization_cfg: TimeStopOptimizationConfig | None = None
        time_stop_opt_raw = strategy_cfg.get("time_stop_optimization")
        if isinstance(time_stop_opt_raw, dict) and time_stop_opt_raw.get("enabled", False):
            fallback_time_stop_optimization_cfg = TimeStopOptimizationConfig(
                enabled=True,
                skip_if_profit_above=float(time_stop_opt_raw.get("skip_if_profit_above", 0.05)),
                early_exit_if_loss_below=float(time_stop_opt_raw.get("early_exit_if_loss_below", -0.03)),
                early_loss_exit_enabled=bool(time_stop_opt_raw.get("early_loss_exit_enabled", False)),
                early_loss_exit_days=int(time_stop_opt_raw.get("early_loss_exit_days", 20)),
                early_loss_exit_threshold=float(time_stop_opt_raw.get("early_loss_exit_threshold", -0.04)),
            )

        fallback_time_stop_v2_cfg: TimeStopV2Config | None = None
        time_stop_v2_raw = strategy_cfg.get("time_stop_v2")
        if isinstance(time_stop_v2_raw, dict) and time_stop_v2_raw.get("enabled", False):
            fallback_time_stop_v2_cfg = self._build_time_stop_v2_config(time_stop_v2_raw)

        fallback_trailing_stop_cfg: TrailingStopConfig | None = None
        trailing_stop_raw = strategy_cfg.get("trailing_stop")
        if isinstance(trailing_stop_raw, dict) and trailing_stop_raw.get("enabled", False):
            fallback_trailing_stop_cfg = TrailingStopConfig(
                enabled=True,
                activate_at_R=float(trailing_stop_raw.get("activate_at_R", 1.5)),
                trail_pct=float(trailing_stop_raw.get("trail_pct", 0.05)),
                min_trail_pct=float(trailing_stop_raw.get("min_trail_pct", 0.05)),
                atr_multiplier=float(trailing_stop_raw.get("atr_multiplier", 0.8)),
                atr_period=int(trailing_stop_raw.get("atr_period", 14)),
                force_after_partial_tp=bool(trailing_stop_raw.get("force_after_partial_tp", True)),
            )

        fallback_weak_exit_cfg: WeakExitConfig | None = None
        weak_exit_raw = strategy_cfg.get("weak_exit")
        if isinstance(weak_exit_raw, dict) and weak_exit_raw.get("enabled", False):
            fallback_weak_exit_cfg = WeakExitConfig(
                enabled=True,
                weak_days=int(weak_exit_raw.get("weak_days", 12)),
                max_drawdown_pct_threshold=float(weak_exit_raw.get("max_drawdown_pct_threshold", -0.06)),
                min_current_unrealized_pct=float(weak_exit_raw.get("min_current_unrealized_pct", -0.02)),
                max_runup_r_gate=float(weak_exit_raw.get("max_runup_r_gate", 0.25)),
                ftm_enabled=bool(weak_exit_raw.get("ftm_enabled", False)),
                ftm_days=int(weak_exit_raw.get("ftm_days", 6)),
                ftm_min_move_pct=float(weak_exit_raw.get("ftm_min_move_pct", 0.008)),
                ftm_max_runup_r_gate=float(weak_exit_raw.get("ftm_max_runup_r_gate", 0.3)),
            )

        # V27.6: Shadow score exit config
        fallback_shadow_score_exit_cfg: ShadowScoreExitConfig | None = None
        ss_exit_raw = strategy_cfg.get("shadow_score_exit")
        if isinstance(ss_exit_raw, dict) and ss_exit_raw.get("enabled", False):
            fallback_shadow_score_exit_cfg = ShadowScoreExitConfig(
                enabled=True,
                w_atr_pct=float(ss_exit_raw.get("w_atr_pct", 1.0)),
                w_rs_slope=float(ss_exit_raw.get("w_rs_slope", 1.0)),
                w_dist_to_support=float(ss_exit_raw.get("w_dist_to_support", -0.5)),
                median_atr_pct=float(ss_exit_raw.get("median_atr_pct", 0.036876)),
                iqr_atr_pct=float(ss_exit_raw.get("iqr_atr_pct", 0.023208)),
                median_rs_slope=float(ss_exit_raw.get("median_rs_slope", 0.002367)),
                iqr_rs_slope=float(ss_exit_raw.get("iqr_rs_slope", 0.003612)),
                median_dist_to_support=float(ss_exit_raw.get("median_dist_to_support", 0.013769)),
                iqr_dist_to_support=float(ss_exit_raw.get("iqr_dist_to_support", 0.028509)),
                low_threshold=float(ss_exit_raw.get("low_threshold", -0.5905)),
                high_threshold=float(ss_exit_raw.get("high_threshold", 0.6896)),
                low_weak_days=int(ss_exit_raw.get("low_weak_days", 8)),
                low_max_dd_pct=float(ss_exit_raw.get("low_max_dd_pct", -0.04)),
                low_runup_r_gate=float(ss_exit_raw.get("low_runup_r_gate", 0.35)),
                low_current_unrealized_pct=float(ss_exit_raw.get("low_current_unrealized_pct", -0.01)),
                low_winner_protect_runup_r=float(ss_exit_raw.get("low_winner_protect_runup_r", 1.0)),
                high_weak_days=int(ss_exit_raw.get("high_weak_days", 16)),
                high_max_dd_pct=float(ss_exit_raw.get("high_max_dd_pct", -0.08)),
            )

        pending_intents: defaultdict[date, list[TradeIntent]] = defaultdict(list)
        # position_tracking initialized when v2 or trailing stop is actually enabled
        position_tracking: dict[str, PositionTrackingData] | None = None
        # Track min low price per position for max drawdown calculation
        drawdown_tracking: dict[str, float] = {}
        # Track max high price per position for fail-to-move and runup tracking
        runup_tracking: dict[str, float] = {}

        total_days = len(trading_days)

        # Diagnostic accumulators: per-regime candidate funnel
        _regime_diag: dict[str, dict[str, int]] = {}  # regime_label → {days, plat, trend, sched}

        # V27.4.2: Track two consecutive days' bull confirmation for candidate filter.
        # CF only activates when BOTH prev and prev-prev are False (consecutive BULL_U).
        _prev_is_confirmed: bool | None = None
        _prev_prev_is_confirmed: bool | None = None

        for idx, current_date in enumerate(trading_days, start=1):
            day_log = logger.bind(current_date=current_date.isoformat())
            rotation_decision: RotationDecision | None = None

            # V27.5: Always set candidate_filter_fn when ATR config is present.
            # BULL_U consecutive → pattern filter + ATR filter
            # BULL_C / other → ATR filter only (no pattern filter)
            # No ATR config + no regime → None (backward compatible)
            _atr_min = self._trend_atr_pct_min
            _atr_max = self._trend_atr_pct_max
            _has_atr = _atr_min is not None or _atr_max is not None

            if self._regime_mode == "none" and not _has_atr:
                self._eod_orchestrator.candidate_filter_fn = None
            elif (
                _prev_is_confirmed is False
                and _prev_prev_is_confirmed is False
                and self._regime_mode != "none"
            ):
                # BULL_U consecutive: pattern filter + ATR filter
                self._eod_orchestrator.candidate_filter_fn = self._build_candidate_filter(
                    is_confirmed=False, pattern_filter=True,
                    trend_atr_pct_min=_atr_min, trend_atr_pct_max=_atr_max,
                )
            elif _has_atr:
                # BULL_C / CHOP / BEAR / regime=none with ATR: ATR filter only
                self._eod_orchestrator.candidate_filter_fn = self._build_candidate_filter(
                    is_confirmed=True, pattern_filter=False,
                    trend_atr_pct_min=_atr_min, trend_atr_pct_max=_atr_max,
                )
            else:
                self._eod_orchestrator.candidate_filter_fn = None

            # In quiet mode, suppress all logging output during daily scan
            if self._quiet:
                # Suppress structlog and stdlib logging during EOD scan
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    # Temporarily raise logging level to suppress output
                    root_logger = logging.getLogger()
                    old_level = root_logger.level
                    root_logger.setLevel(logging.CRITICAL)
                    try:
                        scan_artifacts = self._run_daily_scan(current_date=current_date, config=config)
                    finally:
                        root_logger.setLevel(old_level)
            else:
                scan_artifacts = self._run_daily_scan(current_date=current_date, config=config)

            bar_data = self._extract_bar_data(scan_artifacts.price_snapshots)

            # Update drawdown and runup tracking for open positions
            for sym, bar in bar_data.items():
                if sym in portfolio.positions:
                    drawdown_tracking[sym] = min(drawdown_tracking.get(sym, bar.low), bar.low)
                    runup_tracking[sym] = max(runup_tracking.get(sym, bar.high), bar.high)

            # Get regime-specific configs (priority) or use fallback
            staged_take_profit_cfg = self._extract_regime_staged_take_profit() or fallback_staged_take_profit_cfg
            time_stop_optimization_cfg = self._extract_regime_time_stop_optimization() or fallback_time_stop_optimization_cfg
            time_stop_v2_cfg = self._extract_regime_time_stop_v2() or fallback_time_stop_v2_cfg
            trailing_stop_cfg = self._extract_regime_trailing_stop() or fallback_trailing_stop_cfg
            weak_exit_cfg = self._extract_regime_weak_exit() or fallback_weak_exit_cfg

            # ZERO-IMPACT SHORT CIRCUIT: Only do v2/trailing work if actually enabled
            v2_actually_enabled = time_stop_v2_cfg is not None and time_stop_v2_cfg.enabled
            ts_actually_enabled = trailing_stop_cfg is not None and trailing_stop_cfg.enabled
            needs_tracking = v2_actually_enabled or ts_actually_enabled

            if needs_tracking:
                if v2_actually_enabled or ts_actually_enabled:
                    historical_bars: dict[str, list[PriceBar]] | None = self._extract_historical_bars(scan_artifacts.price_snapshots)
                    is_bull = self._regime_mode == "bull" or (self._regime_mode == "auto" and self._is_current_regime_bull())
                else:
                    historical_bars = None
                    is_bull = True
                if not v2_actually_enabled:
                    time_stop_v2_cfg = None
                if position_tracking is None:
                    position_tracking = {}
            else:
                historical_bars = None
                is_bull = True
                time_stop_v2_cfg = None

            exits, position_tracking = simulator.check_exits(
                dict(portfolio.positions),
                bar_data,
                current_date,
                max_hold_days=self._max_hold_days,
                is_bull_regime=is_bull,
                staged_take_profit=staged_take_profit_cfg,
                time_stop_optimization=time_stop_optimization_cfg,
                time_stop_v2=time_stop_v2_cfg,
                trailing_stop=trailing_stop_cfg,
                weak_exit=weak_exit_cfg,
                shadow_score_exit=fallback_shadow_score_exit_cfg,
                runup_tracking=runup_tracking,
                drawdown_tracking=drawdown_tracking,
                historical_bars=historical_bars,
                position_tracking=position_tracking,
            )
            for signal in exits:
                try:
                    # Compute max drawdown for this position
                    _dd_entry = portfolio.positions[signal.symbol].entry_price if signal.symbol in portfolio.positions else 0.0
                    _dd_min_low = drawdown_tracking.get(signal.symbol, _dd_entry)
                    _dd_pct = ((_dd_min_low - _dd_entry) / _dd_entry) if _dd_entry > 0 else None

                    if signal.is_partial and signal.exit_quantity is not None:
                        portfolio.reduce_position(
                            signal.symbol,
                            signal.exit_quantity,
                            signal.exit_price,
                            current_date,
                            signal.exit_reason,
                            max_drawdown_pct=_dd_pct,
                        )
                        if signal.symbol not in portfolio.positions and position_tracking is not None:
                            position_tracking.pop(signal.symbol, None)
                        # Clean up tracking on full close
                        if signal.symbol not in portfolio.positions:
                            drawdown_tracking.pop(signal.symbol, None)
                            runup_tracking.pop(signal.symbol, None)
                    else:
                        portfolio.close_position(
                            signal.symbol,
                            exit_price=signal.exit_price,
                            exit_date=current_date,
                            exit_reason=signal.exit_reason,
                            max_drawdown_pct=_dd_pct,
                        )
                        if position_tracking is not None:
                            position_tracking.pop(signal.symbol, None)
                        drawdown_tracking.pop(signal.symbol, None)
                        runup_tracking.pop(signal.symbol, None)
                except KeyError:
                    if not self._quiet:
                        day_log.warning("exit_signal_missing_position", symbol=signal.symbol, exit_reason=signal.exit_reason)
                except ValueError as exc:
                    if not self._quiet:
                        day_log.warning(
                            "exit_signal_rejected",
                            symbol=signal.symbol,
                            exit_reason=signal.exit_reason,
                            error=str(exc)[:200],
                        )

            todays_intents = pending_intents.get(current_date, [])

            # V25 P2: Deterministic sort — ensure identical ordering regardless of
            # insertion order, floating-point ties, or Python sort stability quirks.
            # Primary: score descending; Secondary: symbol ascending (tie-breaker).
            todays_intents = sorted(
                todays_intents,
                key=lambda i: (-self._get_intent_score(i), str(getattr(i, "symbol", ""))),
            )

            allocation_result: AllocationResult | None = None
            bull_conditions_met: int | None = None
            bull_min_required: int | None = None
            is_confirmed: bool | None = None
            if self._capital_allocator is not None:
                current_exposure = sum(position.market_value for position in portfolio.positions.values())
                alloc_res = self._capital_allocator.allocate(
                    total_equity=portfolio.get_total_equity(),
                    current_exposure=current_exposure,
                    current_position_count=len(portfolio.positions),
                )
                if alloc_res.status is ResultStatus.SUCCESS:
                    allocation_result = alloc_res.data

                    # Apply regime-specific overrides (score threshold + rate limits)
                    if self._regime_mode != "none":
                        regime_min_score = self._extract_regime_min_score_threshold()
                        if regime_min_score is not None:
                            allocation_result = msgspec.structs.replace(
                                allocation_result,
                                min_score_threshold=regime_min_score,
                            )
                        regime_rate_limits = self._extract_regime_rate_limits()
                        if regime_rate_limits:
                            replace_kwargs_rl: dict[str, int] = {}
                            if "max_new_positions_per_day" in regime_rate_limits:
                                replace_kwargs_rl["max_new_positions_per_day"] = regime_rate_limits["max_new_positions_per_day"]
                            if "max_positions_to_rotate" in regime_rate_limits:
                                replace_kwargs_rl["max_positions_to_rotate"] = regime_rate_limits["max_positions_to_rotate"]
                            if replace_kwargs_rl:
                                allocation_result = msgspec.structs.replace(
                                    allocation_result,
                                    **replace_kwargs_rl,
                                )

                    # Check BULL confirmation status and apply risk overlay if needed
                    if self._regime_mode != "none" and self._is_current_regime_bull():
                        is_confirmed, confirm_details = self._check_bull_confirmed(scan_artifacts.price_snapshots)
                        bull_conditions_met = int(confirm_details.get("conditions_met", 0))
                        bull_min_required = int(confirm_details.get("min_required", 2))

                        # V21: Apply capital_allocation override for bull_confirmed
                        # Override both target_position_pct (main driver) and max_position_pct (guard rail)
                        if is_confirmed and self._capital_allocator is not None:
                            cap_override = self._extract_cap_alloc_override("bull_confirmed")
                            if cap_override is not None:
                                baseline_config = self._capital_allocator.config
                                replace_kwargs: dict[str, float] = {}

                                override_target = cap_override.get("target_position_pct")
                                if isinstance(override_target, (int, float)):
                                    replace_kwargs["target_position_pct"] = float(override_target)

                                override_max_pct = cap_override.get("max_position_pct")
                                if isinstance(override_max_pct, (int, float)):
                                    replace_kwargs["max_position_pct"] = float(override_max_pct)

                                if replace_kwargs:
                                    effective_config = msgspec.structs.replace(
                                        baseline_config, **replace_kwargs,
                                    )
                                    self._capital_allocator.config = effective_config
                                    # Re-allocate with confirmed overrides
                                    re_alloc = self._capital_allocator.allocate(
                                        total_equity=portfolio.get_total_equity(),
                                        current_exposure=current_exposure,
                                        current_position_count=len(portfolio.positions),
                                    )
                                    if re_alloc.status is ResultStatus.SUCCESS:
                                        allocation_result = re_alloc.data
                                        # Re-apply regime overrides that were clobbered by re-allocation
                                        reapply_kwargs: dict[str, Any] = {}
                                        if regime_min_score is not None:
                                            reapply_kwargs["min_score_threshold"] = regime_min_score
                                        if regime_rate_limits:
                                            if "max_new_positions_per_day" in regime_rate_limits:
                                                reapply_kwargs["max_new_positions_per_day"] = regime_rate_limits["max_new_positions_per_day"]
                                            if "max_positions_to_rotate" in regime_rate_limits:
                                                reapply_kwargs["max_positions_to_rotate"] = regime_rate_limits["max_positions_to_rotate"]
                                        if reapply_kwargs:
                                            allocation_result = msgspec.structs.replace(
                                                allocation_result,
                                                **reapply_kwargs,
                                            )
                                    # Restore baseline config for next day
                                    self._capital_allocator.config = baseline_config

                                    if not self._quiet:
                                        day_log.info(
                                            "bull_confirmed_cap_alloc_override",
                                            conditions_met=bull_conditions_met,
                                            override_target_pct=replace_kwargs.get("target_position_pct"),
                                            override_max_pct=replace_kwargs.get("max_position_pct"),
                                            baseline_target_pct=baseline_config.target_position_pct,
                                            baseline_max_pct=baseline_config.max_position_pct,
                                            new_budget=allocation_result.per_position_budget if allocation_result else None,
                                        )

                        if not is_confirmed:
                            overlay = self._extract_bull_risk_overlay()
                            if overlay is not None:
                                # Apply overlay adjustments (except position_size_multiplier - applied per-intent)
                                new_max_per_day = int(overlay.get("max_new_positions_per_day", allocation_result.max_new_positions_per_day))
                                new_max_rotate = int(overlay.get("max_positions_to_rotate", allocation_result.max_positions_to_rotate))
                                score_add = float(overlay.get("min_score_threshold_add", 0.0))
                                new_min_score = float(allocation_result.min_score_threshold) + score_add

                                allocation_result = msgspec.structs.replace(
                                    allocation_result,
                                    max_new_positions_per_day=new_max_per_day,
                                    max_positions_to_rotate=new_max_rotate,
                                    min_score_threshold=new_min_score,
                                )

                                if not self._quiet:
                                    day_log.info(
                                        "bull_unconfirmed_overlay_applied",
                                        conditions_met=bull_conditions_met,
                                        min_required=bull_min_required,
                                        new_max_per_day=new_max_per_day,
                                        new_max_rotate=new_max_rotate,
                                        new_min_score=new_min_score,
                                    )

                    if not self._quiet:
                        day_log.info(
                            "capital_allocation_computed",
                            available_capital=allocation_result.available_capital,
                            reserved_capital=allocation_result.reserved_capital,
                            max_new_positions=allocation_result.max_new_positions,
                            per_position_budget=allocation_result.per_position_budget,
                            current_exposure=allocation_result.current_exposure,
                            current_position_count=allocation_result.current_position_count,
                            total_equity=allocation_result.total_equity,
                        )
                elif not self._quiet:
                    day_log.warning(
                        "capital_allocation_failed",
                        reason_code=alloc_res.reason_code,
                        error=str(alloc_res.error)[:200] if alloc_res.error else None,
                    )

            # Filter intents by dynamic score threshold
            if allocation_result is not None and todays_intents:
                min_score = float(allocation_result.min_score_threshold)
                filtered_intents: list[TradeIntent] = []
                for intent in todays_intents:
                    score = self._get_intent_score(intent)
                    if score >= min_score:
                        filtered_intents.append(intent)
                    elif not self._quiet:
                        day_log.info(
                            "intent_filtered_by_score",
                            symbol=getattr(intent, "symbol", "?"),
                            score=score,
                            min_score=min_score,
                        )
                todays_intents = filtered_intents

            # V28: Pattern-type filter — give platform and trend independent lanes
            _pf_before = len(todays_intents)
            _pf_kept = _pf_before
            if (
                self._regime_mode != "none"
                and todays_intents
                and is_confirmed is not None  # only after bull_confirmed check ran
            ):
                _effective_regime = "BULL_C" if is_confirmed else "BULL_U"
                _pf_filtered: list[TradeIntent] = []
                for _ti in todays_intents:
                    _pt = self._extract_intent_pattern_type(_ti)
                    if _effective_regime == "BULL_C":
                        # Confirmed bull: keep trend patterns + unknown
                        if _pt in self._TREND_PATTERN_TYPES or _pt == "":
                            _pf_filtered.append(_ti)
                        elif not self._quiet:
                            day_log.info(
                                "intent_filtered_by_pattern_type",
                                symbol=getattr(_ti, "symbol", "?"),
                                pattern_type=_pt,
                                effective_regime=_effective_regime,
                            )
                    else:
                        # Unconfirmed bull: keep platform + unknown
                        if _pt == "platform" or _pt == "":
                            _pf_filtered.append(_ti)
                        elif not self._quiet:
                            day_log.info(
                                "intent_filtered_by_pattern_type",
                                symbol=getattr(_ti, "symbol", "?"),
                                pattern_type=_pt,
                                effective_regime=_effective_regime,
                            )
                todays_intents = _pf_filtered
                _pf_kept = len(todays_intents)

            if allocation_result is not None:
                available_slots = max(int(allocation_result.max_new_positions), 0)
                if available_slots <= 0:
                    # Rotation logic: when positions are full and new opportunities exist, evaluate replacing weak positions
                    if (
                        self._position_rotator is not None
                        and todays_intents
                        and allocation_result is not None
                        and available_slots <= 0
                    ):
                        max_opportunity_score = self._get_max_intent_score(todays_intents)
                        dynamic_max_rotate = int(allocation_result.max_positions_to_rotate)
                        rotation_decision = self._position_rotator.evaluate_rotation(
                            positions=portfolio.positions,
                            market_prices={
                                symbol: bar_data[symbol].close
                                for symbol in portfolio.positions
                                if symbol in bar_data
                            },
                            current_date=current_date,
                            new_opportunity_score=max_opportunity_score,
                            max_positions=allocation_result.max_new_positions
                            + allocation_result.current_position_count,
                            num_new_opportunities=min(len(todays_intents), max(dynamic_max_rotate, 0)),
                        )

                        if rotation_decision.should_rotate:
                            for symbol in rotation_decision.positions_to_close:
                                bar = bar_data.get(symbol)
                                if bar is None:
                                    continue
                                try:
                                    portfolio.close_position(
                                        symbol,
                                        exit_price=bar.close,
                                        exit_date=current_date,
                                        exit_reason="ROTATION",
                                    )
                                    if position_tracking is not None:
                                        position_tracking.pop(symbol, None)
                                    if not self._quiet:
                                        day_log.info(
                                            "position_rotated_out",
                                            symbol=symbol,
                                            health_score=rotation_decision.health_scores.get(symbol),
                                        )
                                except KeyError:
                                    pass

                            available_slots = len(rotation_decision.positions_to_close)
                            # BUG FIX: rotation path must also respect max_new_positions_per_day
                            max_new_today = int(allocation_result.max_new_positions_per_day) if allocation_result else 3
                            available_slots = min(available_slots, max(max_new_today, 0))
                            if not self._quiet:
                                day_log.info(
                                    "rotation_executed",
                                    positions_closed=rotation_decision.positions_to_close,
                                    new_slots=available_slots,
                                    reason=rotation_decision.reason,
                                )
                        elif not self._quiet and rotation_decision.reason != "positions_not_full":
                            day_log.info(
                                "rotation_skipped",
                                reason=rotation_decision.reason,
                            )

                    if available_slots <= 0:
                        if todays_intents and not self._quiet:
                            day_log.info("pending_orders_skipped_no_slots", pending=len(todays_intents))
                        todays_intents = []
                else:
                    original_pending = len(todays_intents)
                    # Apply daily max new positions limit
                    max_new_today = int(allocation_result.max_new_positions_per_day) if allocation_result else 3
                    actual_limit = min(available_slots, max(max_new_today, 0))
                    if original_pending > actual_limit:
                        todays_intents = list(todays_intents)[:actual_limit]
                        if not self._quiet:
                            day_log.info(
                                "pending_orders_limited",
                                pending=original_pending,
                                executed=len(todays_intents),
                                available_slots=available_slots,
                                max_new_per_day=max_new_today,
                            )

            # V25: Two-pass rank sizing — fixed total exposure, proportional weights
            # Pass 1: Symbol set already locked by daily limit (line 524-525)
            # Pass 2: Redistribute FIXED total budget proportionally by rank weights
            # todays_intents is already sorted by score descending (strategy/engine.py)
            _rank_multiplier_map: dict[str, tuple[float, int]] = {}  # intent_id -> (multiplier, rank)
            _rank_budget_map: dict[str, float] = {}  # intent_id -> weighted budget
            if (
                todays_intents
                and bull_conditions_met is not None
                and bull_min_required is not None
                and bull_conditions_met >= bull_min_required
            ):
                rank_cfg = self._extract_rank_sizing_config()
                if rank_cfg is not None and rank_cfg.get("enabled", False):
                    multipliers = rank_cfg.get("multipliers", [1.0])
                    base_budget = (
                        allocation_result.per_position_budget
                        if allocation_result is not None
                        else None
                    )
                    if base_budget is not None and base_budget > 0:
                        n_intents = len(todays_intents)
                        total_pool = base_budget * n_intents  # fixed total exposure
                        weights: list[float] = []
                        for rank_idx, _intent in enumerate(todays_intents):
                            mult_idx = min(rank_idx, len(multipliers) - 1)
                            w = float(multipliers[mult_idx])
                            weights.append(w)
                            _rank_multiplier_map[str(_intent.intent_id)] = (w, rank_idx)
                        weight_sum = sum(weights)
                        if weight_sum > 0:
                            for rank_idx, _intent in enumerate(todays_intents):
                                weighted_budget = total_pool * (weights[rank_idx] / weight_sum)
                                _rank_budget_map[str(_intent.intent_id)] = weighted_budget

            if todays_intents and not self._quiet:
                day_log.info("pending_orders_execute_start", pending=len(todays_intents))
            for intent in list(todays_intents):
                symbol = str(getattr(intent, "symbol", "")).strip().upper()
                bar = bar_data.get(symbol)
                if bar is None:
                    if not self._quiet:
                        day_log.warning("missing_bar_data_skip_open", symbol=symbol, intent_id=intent.intent_id)
                    continue
                if intent.stop_loss_price is None or intent.take_profit_price is None:
                    if not self._quiet:
                        day_log.warning(
                            "missing_bracket_prices_skip_open",
                            symbol=symbol,
                            intent_id=intent.intent_id,
                        )
                    continue

                # Recalculate bracket prices based on actual entry price (bar.open)
                # to handle overnight gaps where actual entry differs from intent's expected entry.
                actual_entry = float(bar.open)
                intent_sl = float(intent.stop_loss_price)
                intent_tp = float(intent.take_profit_price)

                # Get the expected entry price from intent (if available)
                intent_entry = getattr(intent, "entry_price", None)
                if intent_entry is not None and float(intent_entry) > 0:
                    expected_entry = float(intent_entry)
                    # Calculate percentage distances from expected entry
                    sl_pct = (expected_entry - intent_sl) / expected_entry  # For LONG: positive distance below
                    tp_pct = (intent_tp - expected_entry) / expected_entry  # For LONG: positive distance above

                    # Apply same percentage distances to actual entry
                    adjusted_sl = actual_entry * (1.0 - sl_pct)
                    adjusted_tp = actual_entry * (1.0 + tp_pct)

                    # Log if significant adjustment occurred (>1% entry price difference)
                    price_diff_pct = abs(actual_entry - expected_entry) / expected_entry
                    if price_diff_pct > 0.01 and not self._quiet:
                        day_log.info(
                            "bracket_prices_adjusted_for_gap",
                            symbol=symbol,
                            intent_id=intent.intent_id,
                            expected_entry=expected_entry,
                            actual_entry=actual_entry,
                            price_diff_pct=f"{price_diff_pct:.2%}",
                            original_sl=intent_sl,
                            adjusted_sl=adjusted_sl,
                            original_tp=intent_tp,
                            adjusted_tp=adjusted_tp,
                        )
                else:
                    # No expected entry available - use intent prices as-is
                    # but validate they make sense with actual entry
                    adjusted_sl = intent_sl
                    adjusted_tp = intent_tp

                # Final sanity check: ensure bracket prices are valid for a LONG position
                # SL must be below entry, TP must be above entry
                if adjusted_sl >= actual_entry:
                    day_log.warning(
                        "invalid_bracket_sl_above_entry_skip",
                        symbol=symbol,
                        intent_id=intent.intent_id,
                        actual_entry=actual_entry,
                        adjusted_sl=adjusted_sl,
                    )
                    continue
                if adjusted_tp <= actual_entry:
                    day_log.warning(
                        "invalid_bracket_tp_below_entry_skip",
                        symbol=symbol,
                        intent_id=intent.intent_id,
                        actual_entry=actual_entry,
                        adjusted_tp=adjusted_tp,
                    )
                    continue

                # Extract scanner_score and shadow_score from intent metadata
                _intent_meta = getattr(intent, "metadata", None)
                _scanner_score: float | None = None
                _shadow_scores: dict[str, object] | None = None
                if isinstance(_intent_meta, dict):
                    _ss = _intent_meta.get("scanner_score", "")
                    if _ss:
                        try:
                            _scanner_score = float(_ss)
                        except (ValueError, TypeError):
                            pass
                    _shadow_raw = _intent_meta.get("shadow_score", "")
                    if _shadow_raw:
                        try:
                            import json as _json
                            _parsed = _json.loads(_shadow_raw)
                            if isinstance(_parsed, dict):
                                _shadow_scores = _parsed
                        except (ValueError, TypeError):
                            pass

                # Inject market_cap into shadow_scores for CSV diagnostics
                _mcap_for_ss = self._extract_market_cap_map().get(symbol)
                if _mcap_for_ss is not None:
                    if _shadow_scores is None:
                        _shadow_scores = {}
                    _shadow_scores["ss_market_cap"] = _mcap_for_ss

                position = Position(
                    symbol=symbol,
                    entry_price=actual_entry,
                    quantity=float(intent.quantity),
                    entry_date=current_date,
                    stop_loss_price=adjusted_sl,
                    take_profit_price=adjusted_tp,
                    intent_id=str(intent.intent_id),
                    scanner_score=_scanner_score,
                    shadow_scores=_shadow_scores,
                )
                try:
                    budget = allocation_result.per_position_budget if allocation_result is not None else None

                    # V25: Apply two-pass rank sizing (fixed total exposure, proportional weights)
                    _rank_mult = 1.0
                    _rank_idx = -1
                    _intent_id_str = str(intent.intent_id)
                    if _intent_id_str in _rank_budget_map and budget is not None:
                        _rank_mult, _rank_idx = _rank_multiplier_map[_intent_id_str]
                        budget = _rank_budget_map[_intent_id_str]
                        _rs_score = self._get_intent_score(intent)
                        if self._quiet:
                            print(
                                f"  RANK_SIZING: {symbol} rank={_rank_idx} "
                                f"score={_rs_score:.4f} "
                                f"mult={_rank_mult:.2f} budget=${budget:.0f}"
                            )
                        else:
                            day_log.info(
                                "rank_sizing_applied",
                                symbol=symbol,
                                rank=_rank_idx,
                                score=_rs_score,
                                rank_multiplier=_rank_mult,
                                budget=budget,
                            )

                    # V22: Budget multiplier is now controlled by use_budget_multiplier flag
                    # When disabled (default), confirmed/unconfirmed only affects discrete decisions
                    # (max_new_positions_per_day, min_score_threshold) to avoid cascade effects
                    position_multiplier = 1.0
                    multiplier_tier = "DISABLED"

                    # Check if budget multiplier is enabled in risk_overlay config
                    use_budget_multiplier = self._is_budget_multiplier_enabled()

                    if use_budget_multiplier and budget is not None and bull_conditions_met is not None and bull_min_required is not None:
                        intent_score = self._get_intent_score(intent)

                        # Extract atr_pct from intent metadata for PLUS volatility gate
                        intent_atr_pct: float | None = None
                        intent_metadata = getattr(intent, "metadata", None)
                        if isinstance(intent_metadata, dict):
                            atr_pct_str = intent_metadata.get("atr_pct", "")
                            if atr_pct_str:
                                try:
                                    intent_atr_pct = float(atr_pct_str)
                                except (ValueError, TypeError):
                                    pass

                        position_multiplier, multiplier_tier = self._compute_position_multiplier(
                            conditions_met=bull_conditions_met,
                            min_required=bull_min_required,
                            intent_score=intent_score,
                            atr_pct=intent_atr_pct,
                        )
                        budget = budget * position_multiplier
                        if not self._quiet and position_multiplier != 1.0:
                            day_log.info(
                                "position_multiplier_applied",
                                symbol=symbol,
                                tier=multiplier_tier,
                                multiplier=position_multiplier,
                                adjusted_budget=budget,
                                intent_score=intent_score,
                                atr_pct=intent_atr_pct,
                                conditions_met=bull_conditions_met,
                            )

                    # V27.7: ATR-aware position sizing — scale budget by ATR% bucket
                    _atr_sizing_cfg = self._extract_atr_position_sizing_config()
                    _ss_atr_pct: float | None = None
                    if isinstance(_shadow_scores, dict):
                        _raw = _shadow_scores.get("ss_atr_pct")
                        if isinstance(_raw, (int, float)):
                            _ss_atr_pct = float(_raw)
                    _atr_sizing_mult = self._compute_atr_sizing_multiplier(_ss_atr_pct, _atr_sizing_cfg)
                    if _atr_sizing_mult != 1.0 and budget is not None:
                        budget = budget * _atr_sizing_mult
                        if self._quiet:
                            print(
                                f"  ATR_SIZING: {symbol} atr_pct={_ss_atr_pct:.4f} "
                                f"mult={_atr_sizing_mult:.2f} budget=${budget:.0f}"
                            )
                        else:
                            day_log.info(
                                "atr_position_sizing_applied",
                                symbol=symbol,
                                atr_pct=_ss_atr_pct,
                                atr_sizing_multiplier=_atr_sizing_mult,
                                adjusted_budget=budget,
                            )

                    min_position_value = (
                        self._capital_allocator.config.min_position_value
                        if self._capital_allocator is not None
                        else 50.0
                    )

                    # V25 P3: When rank sizing is active and cash is tight,
                    # drop low-rank positions rather than diluting them.
                    # High-rank positions (rank 0-1) are allowed to shrink to fit;
                    # low-rank positions (rank 2+) are skipped entirely if cash
                    # cannot cover at least half their weighted budget, preventing
                    # diluted micro-positions that waste a slot.
                    if (
                        _rank_idx >= 2
                        and _intent_id_str in _rank_budget_map
                        and budget is not None
                        and portfolio.cash < max(budget * 0.5, min_position_value)
                    ):
                        if self._quiet:
                            print(
                                f"  RANK_DROP: {symbol} rank={_rank_idx} "
                                f"cash=${portfolio.cash:.0f} < min=${min_position_value:.0f} — skipped"
                            )
                        else:
                            day_log.info(
                                "rank_sizing_low_rank_dropped",
                                symbol=symbol,
                                rank=_rank_idx,
                                cash=portfolio.cash,
                                min_position_value=min_position_value,
                                budget=budget,
                            )
                        continue

                    portfolio.add_position(
                        position,
                        max_budget=budget,
                        min_position_value=min_position_value,
                    )
                except ValueError as exc:
                    if not self._quiet:
                        day_log.info(
                            "position_open_skipped",
                            symbol=symbol,
                            intent_id=intent.intent_id,
                            error=str(exc)[:200],
                        )
                    continue

            if current_date in pending_intents:
                pending_intents.pop(current_date, None)

            next_trading_day = self._get_next_trading_day(current_date, trading_days)
            scheduled = 0
            if next_trading_day is not None and scan_artifacts.intent_set is not None:
                scheduled = self._schedule_entry_intents(
                    intent_set=scan_artifacts.intent_set,
                    execution_date=next_trading_day,
                    pending_intents=pending_intents,
                    day_log=day_log if not self._quiet else logger,
                )
                if not self._quiet:
                    day_log.info("scheduled_next_day_intents", next_trading_day=next_trading_day.isoformat(), intents=scheduled)
            elif next_trading_day is None:
                if not self._quiet:
                    day_log.info("last_trading_day_no_scheduling")
            else:
                if not self._quiet:
                    day_log.warning("missing_intent_set_skip_scheduling")

            equity = portfolio.get_total_equity()
            if not self._quiet:
                day_log.info(
                    "daily_summary",
                    equity=equity,
                    cash=portfolio.cash,
                    open_positions=len(portfolio.positions),
                    pending_today=len(todays_intents),
                )

            # In quiet mode, print a single progress line per day
            if self._quiet:
                pnl = equity - self._initial_capital
                pnl_pct = (pnl / self._initial_capital) * 100 if self._initial_capital else 0
                # Extract regime from last outputs
                regime_str = ""
                if self._regime_mode != "none":
                    outputs = self._get_last_outputs_by_module()
                    regime_data = outputs.get("market_regime", {})
                    detected_regime = regime_data.get("detected_regime", "?")
                    regime_str = f" | Reg: {detected_regime[:4].upper()}"
                rot_str = ""
                if rotation_decision is not None and rotation_decision.should_rotate:
                    rot_str = f" | Rot: {len(rotation_decision.positions_to_close)}"
                dyn_str = ""
                if self._capital_allocator is not None:
                    max_new_today = allocation_result.max_new_positions_per_day if allocation_result is not None else 3
                    max_rotate_today = allocation_result.max_positions_to_rotate if allocation_result is not None else 2
                    min_score_today = allocation_result.min_score_threshold if allocation_result is not None else 0.90
                    dyn_str = (
                        f" | New/day: {int(max_new_today)}"
                        f" Rot/day: {int(max_rotate_today)}"
                        f" MinS: {float(min_score_today):.2f}"
                    )

                # Diagnostic: scanner candidate breakdown (platform vs trend) + bull confirmation
                _diag_outputs = self._get_last_outputs_by_module()
                _scanner_out = _diag_outputs.get("scanner", {})
                _cands = _scanner_out.get("candidates", []) if isinstance(_scanner_out, dict) else []
                _plat_n = 0
                _trend_n = 0
                for _c in _cands:
                    _m = _c.get("meta", {}) if isinstance(_c, dict) else {}
                    _ss = _m.get("shadow_score", {}) if isinstance(_m, dict) else {}
                    _pt = _ss.get("ss_pattern_type", "") if isinstance(_ss, dict) else ""
                    if _pt == "platform":
                        _plat_n += 1
                    elif _pt:
                        _trend_n += 1
                _bc_str = ""
                if is_confirmed is not None:
                    _bc_str = f" BC:{'Y' if is_confirmed else 'N'}"
                    if bull_conditions_met is not None:
                        _bc_str += f"({bull_conditions_met}/{bull_min_required or '?'})"
                # V28: Effective regime + pattern filter stats
                _ef_str = ""
                if is_confirmed is not None:
                    _ef_str = f" EfReg:{'BULL_C' if is_confirmed else 'BULL_U'}"
                # V27.4: Candidate filter stats
                _cf_fn = self._eod_orchestrator.candidate_filter_fn
                _cf_before = getattr(_cf_fn, "cf_before", 0) if _cf_fn else 0
                _cf_kept = getattr(_cf_fn, "cf_kept", 0) if _cf_fn else 0
                _cf_str = ""
                if _cf_before > 0:
                    _cf_str = f" CF:{_cf_kept}/{_cf_before}"
                _pf_str = ""
                if _pf_before != _pf_kept:
                    _pf_str = f" PF:{_pf_kept}/{_pf_before}"
                diag_str = f" | Cand P:{_plat_n} T:{_trend_n}{_bc_str}{_ef_str}{_cf_str}{_pf_str}"

                print(
                    f"[{idx:3d}/{total_days}] {current_date.isoformat()} | "
                    f"Equity: ${equity:,.0f} ({pnl_pct:+.1f}%) | "
                    f"Pos: {len(portfolio.positions)} | "
                    f"Trades: {len(portfolio.completed_trades)} | "
                    f"Sched: {scheduled}{regime_str}{rot_str}{dyn_str}{diag_str}",
                    flush=True,
                )

                # Accumulate per-regime diagnostics
                _rlabel = detected_regime[:4].upper() if regime_str else "NONE"
                if _rlabel == "BULL" and is_confirmed is not None:
                    _rlabel = "BULL_C" if is_confirmed else "BULL_U"
                _rd = _regime_diag.setdefault(
                    _rlabel, {"days": 0, "plat": 0, "trend": 0, "sched": 0, "pf_in": 0, "pf_out": 0, "cf_in": 0, "cf_out": 0}
                )
                _rd["days"] += 1
                _rd["plat"] += _plat_n
                _rd["trend"] += _trend_n
                _rd["sched"] += scheduled
                _rd["pf_in"] += _pf_before
                _rd["pf_out"] += _pf_kept
                _rd["cf_in"] += _cf_before
                _rd["cf_out"] += _cf_kept

            # V27.4.2: Always save is_confirmed (including None for CHOP/BEAR)
            # to properly break the consecutive-BULL_U chain on regime transitions.
            _prev_prev_is_confirmed = _prev_is_confirmed
            _prev_is_confirmed = is_confirmed

        # Print per-regime diagnostic summary
        if _regime_diag:
            print("\n=== Scanner Candidate Funnel by Regime ===")
            print(
                f"{'Regime':<12} {'Days':>5} {'Platform':>10} {'Trend':>8} {'Scheduled':>10}"
                f" {'P/day':>6} {'T/day':>6} {'CF_In':>6} {'CF_Out':>6} {'PF_In':>6} {'PF_Out':>6}"
            )
            for _rl in sorted(_regime_diag.keys()):
                _rd = _regime_diag[_rl]
                _d = _rd["days"] or 1
                print(
                    f"{_rl:<12} {_rd['days']:>5} {_rd['plat']:>10} {_rd['trend']:>8} {_rd['sched']:>10}"
                    f" {_rd['plat']/_d:>6.1f} {_rd['trend']/_d:>6.1f}"
                    f" {_rd.get('cf_in', 0):>6} {_rd.get('cf_out', 0):>6}"
                    f" {_rd.get('pf_in', 0):>6} {_rd.get('pf_out', 0):>6}"
                )
            print("=" * 80)

        stats = reporter.generate_stats(portfolio.completed_trades, self._initial_capital)
        reporter.save_trades_csv(portfolio.completed_trades, str(Path(self._output_dir) / trades_filename))

        print(
            "Backtest completed:",
            {
                "total_trades": stats.total_trades,
                "win_rate": stats.win_rate,
                "total_pnl": stats.total_pnl,
                "total_pnl_pct": stats.total_pnl_pct,
                "avg_hold_days": stats.avg_hold_days,
            },
        )
        return stats

    def _run_daily_scan(self, *, current_date: date, config: dict[str, Any]) -> _DailyScanArtifacts:
        """Run the EOD scan orchestrator for ``current_date`` and extract artifacts.

        This is a lightweight extraction method that reads module outputs from
        the orchestrator instance directly (no journal replay).
        """

        run_config = deepcopy(config)
        run_config.setdefault("mode", "DRY_RUN")
        self._apply_regime_mode(run_config)

        self._inject_data_end_date(run_config, current_date=current_date)

        # Fix wall-clock bug: use simulated EOD timestamp (21:00 UTC ~ 4 PM ET)
        eod_dt = datetime.combine(current_date, dt_time(21, 0), tzinfo=timezone.utc)
        self._eod_orchestrator.scan_timestamp_ns = int(eod_dt.timestamp() * 1_000_000_000)

        result = self._eod_orchestrator.execute_run(run_config, run_type=RunType.PRE_MARKET_FULL_SCAN)

        self._eod_orchestrator.scan_timestamp_ns = None
        if result.status is ResultStatus.FAILED:
            logger.warning(
                "eod_scan_failed",
                current_date=current_date.isoformat(),
                reason_code=result.reason_code,
                error=str(result.error)[:200] if result.error else None,
            )

        outputs_by_module = self._get_last_outputs_by_module()

        intent_set = self._extract_intent_set(outputs_by_module)
        price_snapshots = self._extract_price_snapshots(outputs_by_module)

        return _DailyScanArtifacts(intent_set=intent_set, price_snapshots=price_snapshots)

    def _get_last_outputs_by_module(self) -> dict[str, Any]:
        """Best-effort extraction of last-run outputs from the EOD orchestrator."""

        outputs = getattr(self._eod_orchestrator, "last_outputs_by_module", None)
        if isinstance(outputs, dict):
            return outputs
        outputs = getattr(self._eod_orchestrator, "_last_outputs_by_module", None)
        if isinstance(outputs, dict):
            return outputs
        return {}

    def _extract_market_cap_map(self) -> dict[str, float]:
        """Extract symbol → market_cap mapping from universe output + static fallback.

        Priority: universe output (real-time) > static JSON file (one-time snapshot).
        The static file at ``<output_dir>/market_cap_static.json`` fills gaps for
        delisted or under-covered symbols in the universe builder output.
        """
        outputs = self._get_last_outputs_by_module()
        universe_data = outputs.get("universe")
        result: dict[str, float] = {}
        if isinstance(universe_data, dict):
            equities = universe_data.get("equities")
            if isinstance(equities, list):
                for eq in equities:
                    if isinstance(eq, dict):
                        sym = eq.get("symbol")
                        mc = eq.get("market_cap")
                        if isinstance(sym, str) and isinstance(mc, (int, float)) and mc > 0:
                            result[sym] = float(mc)

        # Fallback: load static market cap file to fill gaps.
        import json as _json

        static_path = Path(self._output_dir) / "market_cap_static.json"
        if static_path.exists():
            try:
                raw = _json.loads(static_path.read_text())
                static_data = raw.get("data", {})
                for sym, mc in static_data.items():
                    if sym not in result and isinstance(mc, (int, float)) and mc > 0:
                        result[sym] = float(mc)
            except Exception:
                pass

        return result

    def _extract_regime_min_score_threshold(self) -> float | None:
        """Extract min_score_threshold from regime applied_overrides if available.

        Returns:
            The regime-specific min_score_threshold, or None if not available.
        """
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        # First try applied_overrides.strategy.min_score_threshold
        applied = regime_data.get("applied_overrides")
        if isinstance(applied, dict):
            strategy_overrides = applied.get("strategy")
            if isinstance(strategy_overrides, dict):
                min_score = strategy_overrides.get("min_score_threshold")
                if isinstance(min_score, (int, float)):
                    return float(min_score)

        # Fallback: try config.strategy.min_score_threshold
        config = regime_data.get("config")
        if isinstance(config, dict):
            strategy_cfg = config.get("strategy")
            if isinstance(strategy_cfg, dict):
                min_score = strategy_cfg.get("min_score_threshold")
                if isinstance(min_score, (int, float)):
                    return float(min_score)

        return None

    def _extract_regime_rate_limits(self) -> dict[str, int] | None:
        """Extract max_new_positions_per_day and max_positions_to_rotate from regime strategy config.

        Returns:
            Dict with rate limit values, or None if not available.
        """
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        for top_key in ("applied_overrides", "config"):
            top = regime_data.get(top_key)
            if isinstance(top, dict):
                strategy_cfg = top.get("strategy")
                if isinstance(strategy_cfg, dict):
                    result: dict[str, int] = {}
                    for key in ("max_new_positions_per_day", "max_positions_to_rotate"):
                        val = strategy_cfg.get(key)
                        if isinstance(val, (int, float)):
                            result[key] = int(val)
                    if result:
                        return result

        return None

    def _extract_regime_time_stop_optimization(self) -> TimeStopOptimizationConfig | None:
        """Extract time_stop_optimization from regime applied_overrides if available.

        Returns:
            TimeStopOptimizationConfig if enabled in regime config, None otherwise.
        """
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        # Try applied_overrides.strategy.time_stop_optimization
        applied = regime_data.get("applied_overrides")
        if isinstance(applied, dict):
            strategy_overrides = applied.get("strategy")
            if isinstance(strategy_overrides, dict):
                ts_opt = strategy_overrides.get("time_stop_optimization")
                if isinstance(ts_opt, dict) and ts_opt.get("enabled", False):
                    return TimeStopOptimizationConfig(
                        enabled=True,
                        skip_if_profit_above=float(ts_opt.get("skip_if_profit_above", 0.05)),
                        early_exit_if_loss_below=float(ts_opt.get("early_exit_if_loss_below", -0.03)),
                        early_loss_exit_enabled=bool(ts_opt.get("early_loss_exit_enabled", False)),
                        early_loss_exit_days=int(ts_opt.get("early_loss_exit_days", 20)),
                        early_loss_exit_threshold=float(ts_opt.get("early_loss_exit_threshold", -0.04)),
                    )

        # Fallback: try config.strategy.time_stop_optimization
        config = regime_data.get("config")
        if isinstance(config, dict):
            strategy_cfg = config.get("strategy")
            if isinstance(strategy_cfg, dict):
                ts_opt = strategy_cfg.get("time_stop_optimization")
                if isinstance(ts_opt, dict) and ts_opt.get("enabled", False):
                    return TimeStopOptimizationConfig(
                        enabled=True,
                        skip_if_profit_above=float(ts_opt.get("skip_if_profit_above", 0.05)),
                        early_exit_if_loss_below=float(ts_opt.get("early_exit_if_loss_below", -0.03)),
                        early_loss_exit_enabled=bool(ts_opt.get("early_loss_exit_enabled", False)),
                        early_loss_exit_days=int(ts_opt.get("early_loss_exit_days", 20)),
                        early_loss_exit_threshold=float(ts_opt.get("early_loss_exit_threshold", -0.04)),
                    )

        return None

    def _extract_regime_time_stop_v2(self) -> TimeStopV2Config | None:
        """Extract time_stop_v2 from regime applied_overrides if available."""

        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        applied = regime_data.get("applied_overrides")
        if isinstance(applied, dict):
            strategy_overrides = applied.get("strategy")
            if isinstance(strategy_overrides, dict):
                ts_v2 = strategy_overrides.get("time_stop_v2")
                if isinstance(ts_v2, dict) and ts_v2.get("enabled", False):
                    return self._build_time_stop_v2_config(ts_v2)

        config = regime_data.get("config")
        if isinstance(config, dict):
            strategy_cfg = config.get("strategy")
            if isinstance(strategy_cfg, dict):
                ts_v2 = strategy_cfg.get("time_stop_v2")
                if isinstance(ts_v2, dict) and ts_v2.get("enabled", False):
                    return self._build_time_stop_v2_config(ts_v2)

        return None

    @staticmethod
    def _build_time_stop_v2_config(raw: dict[str, Any]) -> TimeStopV2Config:
        return TimeStopV2Config(
            enabled=bool(raw.get("enabled", False)),
            base_days=int(raw.get("base_days", 30)),
            extend_step_days=int(raw.get("extend_step_days", 5)),
            max_extend_count=int(raw.get("max_extend_count", 2)),
            only_if_bull_regime=bool(raw.get("only_if_bull_regime", True)),
            min_adx_threshold=float(raw.get("min_adx_threshold", 25.0)),
            min_unrealized_r=float(raw.get("min_unrealized_r", 1.5)),
            atr_period=int(raw.get("atr_period", 14)),
            atr_sma_short=int(raw.get("atr_sma_short", 5)),
            atr_sma_long=int(raw.get("atr_sma_long", 20)),
            atr_contract_ratio=float(raw.get("atr_contract_ratio", 0.85)),
            ema_period=int(raw.get("ema_period", 20)),
            structure_pullback_atr_mult=float(raw.get("structure_pullback_atr_mult", 1.2)),
            no_new_high_days=int(raw.get("no_new_high_days", 7)),
            no_new_high_days_choppy=int(raw.get("no_new_high_days_choppy", 10)),
            momentum_rsi_period=int(raw.get("momentum_rsi_period", 14)),
            momentum_rsi_threshold=float(raw.get("momentum_rsi_threshold", 55.0)),
            soft_trail_enabled=bool(raw.get("soft_trail_enabled", True)),
            soft_trail_atr_mult=float(raw.get("soft_trail_atr_mult", 1.5)),
        )

    def _extract_regime_trailing_stop(self) -> TrailingStopConfig | None:
        """Extract trailing_stop from regime applied_overrides if available."""

        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        applied = regime_data.get("applied_overrides")
        if isinstance(applied, dict):
            strategy_overrides = applied.get("strategy")
            if isinstance(strategy_overrides, dict):
                ts = strategy_overrides.get("trailing_stop")
                if isinstance(ts, dict) and ts.get("enabled", False):
                    return TrailingStopConfig(
                        enabled=True,
                        activate_at_R=float(ts.get("activate_at_R", 1.5)),
                        trail_pct=float(ts.get("trail_pct", 0.05)),
                        min_trail_pct=float(ts.get("min_trail_pct", 0.05)),
                        atr_multiplier=float(ts.get("atr_multiplier", 0.8)),
                        atr_period=int(ts.get("atr_period", 14)),
                        force_after_partial_tp=bool(ts.get("force_after_partial_tp", True)),
                    )

        config = regime_data.get("config")
        if isinstance(config, dict):
            strategy_cfg = config.get("strategy")
            if isinstance(strategy_cfg, dict):
                ts = strategy_cfg.get("trailing_stop")
                if isinstance(ts, dict) and ts.get("enabled", False):
                    return TrailingStopConfig(
                        enabled=True,
                        activate_at_R=float(ts.get("activate_at_R", 1.5)),
                        trail_pct=float(ts.get("trail_pct", 0.05)),
                        min_trail_pct=float(ts.get("min_trail_pct", 0.05)),
                        atr_multiplier=float(ts.get("atr_multiplier", 0.8)),
                        atr_period=int(ts.get("atr_period", 14)),
                        force_after_partial_tp=bool(ts.get("force_after_partial_tp", True)),
                    )

        return None

    def _extract_regime_weak_exit(self) -> WeakExitConfig | None:
        """Extract weak_exit from regime applied_overrides if available."""

        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        for key in ("applied_overrides", "config"):
            container = regime_data.get(key)
            if isinstance(container, dict):
                strategy = container.get("strategy") if key == "applied_overrides" else container.get("strategy")
                if isinstance(strategy, dict):
                    we = strategy.get("weak_exit")
                    if isinstance(we, dict) and we.get("enabled", False):
                        return WeakExitConfig(
                            enabled=True,
                            weak_days=int(we.get("weak_days", 12)),
                            max_drawdown_pct_threshold=float(we.get("max_drawdown_pct_threshold", -0.06)),
                            min_current_unrealized_pct=float(we.get("min_current_unrealized_pct", -0.02)),
                            max_runup_r_gate=float(we.get("max_runup_r_gate", 0.25)),
                            ftm_enabled=bool(we.get("ftm_enabled", False)),
                            ftm_days=int(we.get("ftm_days", 6)),
                            ftm_min_move_pct=float(we.get("ftm_min_move_pct", 0.008)),
                            ftm_max_runup_r_gate=float(we.get("ftm_max_runup_r_gate", 0.3)),
                        )

        return None

    def _is_current_regime_bull(self) -> bool:
        """Return True if the last detected regime is bull."""

        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return False

        detected = regime_data.get("detected_regime") or regime_data.get("regime_name")
        if not isinstance(detected, str):
            return False

        return detected.strip().lower().startswith("bull")

    def _check_bull_confirmed(self, price_snapshots: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Check if BULL regime is confirmed based on QQQ technical conditions.

        Returns:
            Tuple of (is_confirmed, details_dict)
        """
        details: dict[str, Any] = {"conditions_met": 0, "conditions_checked": []}

        # Get bull_confirmation config from regime
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return True, {"reason": "no_regime_data", "default": True}

        config = regime_data.get("config")
        if not isinstance(config, dict):
            return True, {"reason": "no_config", "default": True}

        bull_conf = config.get("bull_confirmation")
        if not isinstance(bull_conf, dict) or not bull_conf.get("enabled", False):
            return True, {"reason": "bull_confirmation_disabled", "default": True}

        min_required = int(bull_conf.get("min_conditions_required", 2))
        conditions = bull_conf.get("conditions", {})
        if not isinstance(conditions, dict):
            return True, {"reason": "no_conditions", "default": True}

        # Find QQQ data (fallback to SPY)
        qqq_snapshot = price_snapshots.get("QQQ") or price_snapshots.get("SPY")
        if qqq_snapshot is None:
            return True, {"reason": "no_benchmark_data", "default": True}

        try:
            bars = list(qqq_snapshot.bars) if hasattr(qqq_snapshot, "bars") else []
        except Exception:
            return True, {"reason": "bars_extraction_failed", "default": True}

        if len(bars) < 60:
            return True, {"reason": f"insufficient_bars_{len(bars)}", "default": True}

        closes = [float(getattr(b, "close", 0)) for b in bars]
        if not closes or closes[-1] <= 0:
            return True, {"reason": "invalid_closes", "default": True}

        conditions_met = 0

        # Condition 1: QQQ close > EMA50
        if conditions.get("qqq_above_ema50", True):
            ema50 = self._compute_ema(closes, 50)
            if ema50 is not None and closes[-1] > ema50:
                conditions_met += 1
                details["conditions_checked"].append(("qqq_above_ema50", True, closes[-1], ema50))
            else:
                details["conditions_checked"].append(("qqq_above_ema50", False, closes[-1], ema50))

        # Condition 2: QQQ EMA20 > EMA50
        if conditions.get("qqq_ema20_above_ema50", True):
            ema20 = self._compute_ema(closes, 20)
            ema50 = self._compute_ema(closes, 50)
            if ema20 is not None and ema50 is not None and ema20 > ema50:
                conditions_met += 1
                details["conditions_checked"].append(("qqq_ema20_above_ema50", True, ema20, ema50))
            else:
                details["conditions_checked"].append(("qqq_ema20_above_ema50", False, ema20, ema50))

        # Condition 3: QQQ 20d return > threshold
        return_threshold = float(conditions.get("qqq_20d_return_threshold", 0.03))
        if len(closes) >= 21:
            return_20d = (closes[-1] - closes[-21]) / closes[-21]
            if return_20d > return_threshold:
                conditions_met += 1
                details["conditions_checked"].append(("qqq_20d_return", True, return_20d, return_threshold))
            else:
                details["conditions_checked"].append(("qqq_20d_return", False, return_20d, return_threshold))

        # Condition 4: MA50 slope > threshold (from regime detection)
        slope_threshold = float(conditions.get("ma50_slope_threshold", 0.005))
        regime_result = regime_data.get("result") or regime_data
        ma50_slope = regime_result.get("ma50_slope")
        if ma50_slope is not None and float(ma50_slope) > slope_threshold:
            conditions_met += 1
            details["conditions_checked"].append(("ma50_slope", True, ma50_slope, slope_threshold))
        else:
            details["conditions_checked"].append(("ma50_slope", False, ma50_slope, slope_threshold))

        details["conditions_met"] = conditions_met
        details["min_required"] = min_required
        is_confirmed = conditions_met >= min_required

        return is_confirmed, details

    @staticmethod
    def _compute_ema(closes: list[float], period: int) -> float | None:
        """Compute EMA for the given period."""
        if len(closes) < period:
            return None
        multiplier = 2.0 / (period + 1)
        ema = sum(closes[:period]) / period  # SMA for initial
        for close in closes[period:]:
            ema = (close - ema) * multiplier + ema
        return ema

    def _extract_cap_alloc_override(self, state: str) -> dict[str, Any] | None:
        """Extract capital_allocation.overrides.<state> from regime config.

        V21: Regime-aware sizing cap override. Reads from
        regime_config["capital_allocation"]["overrides"][state].

        Args:
            state: The bull confirmation state key (e.g. "bull_confirmed").

        Returns:
            Override dict with fields like max_position_pct, or None.
        """
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        config = regime_data.get("config")
        if not isinstance(config, dict):
            return None

        cap_alloc = config.get("capital_allocation")
        if not isinstance(cap_alloc, dict):
            return None

        overrides = cap_alloc.get("overrides")
        if not isinstance(overrides, dict):
            return None

        state_override = overrides.get(state)
        if not isinstance(state_override, dict):
            return None

        return state_override

    def _extract_bull_risk_overlay(self) -> dict[str, Any] | None:
        """Extract risk_overlay.bull_unconfirmed from regime config."""
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        config = regime_data.get("config")
        if not isinstance(config, dict):
            return None

        risk_overlay = config.get("risk_overlay")
        if not isinstance(risk_overlay, dict):
            return None

        return risk_overlay.get("bull_unconfirmed")

    def _extract_rank_sizing_config(self) -> dict[str, Any] | None:
        """Extract risk_overlay.rank_sizing from regime config.

        V24: Rank-based sizing gives higher-scored intents larger budgets
        within bull_confirmed regime.

        Returns:
            Dict with 'enabled' (bool) and 'multipliers' (list[float]), or None.
        """
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        config = regime_data.get("config")
        if not isinstance(config, dict):
            return None

        risk_overlay = config.get("risk_overlay")
        if not isinstance(risk_overlay, dict):
            return None

        return risk_overlay.get("rank_sizing")

    def _extract_bull_confirmed_plus_config(self) -> dict[str, Any] | None:
        """Extract risk_overlay.bull_confirmed_plus from regime config."""
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        config = regime_data.get("config")
        if not isinstance(config, dict):
            return None

        risk_overlay = config.get("risk_overlay")
        if not isinstance(risk_overlay, dict):
            return None

        return risk_overlay.get("bull_confirmed_plus")

    def _extract_atr_position_sizing_config(self) -> dict[str, Any] | None:
        """Extract risk_overlay.atr_position_sizing from regime config.

        V27.7: ATR-aware position sizing scales budget by ATR% bucket.
        Low ATR% (<3%) gets reduced sizing, high ATR% (5-8%) gets boosted.

        Returns:
            Dict with 'enabled' (bool) and 'buckets' (list of dicts with
            'max_atr_pct' and 'multiplier'), or None.
        """
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        config = regime_data.get("config")
        if not isinstance(config, dict):
            return None

        risk_overlay = config.get("risk_overlay")
        if not isinstance(risk_overlay, dict):
            return None

        return risk_overlay.get("atr_position_sizing")

    @staticmethod
    def _compute_atr_sizing_multiplier(
        atr_pct: float | None,
        atr_sizing_cfg: dict[str, Any] | None,
    ) -> float:
        """Map ATR% to position sizing multiplier using bucket config.

        Buckets are matched in order; first bucket where atr_pct <= max_atr_pct wins.
        Falls back to 1.0 if no bucket matches, ATR% is missing, or config is None/disabled.
        """
        if atr_pct is None or atr_sizing_cfg is None:
            return 1.0
        if not atr_sizing_cfg.get("enabled", False):
            return 1.0
        buckets = atr_sizing_cfg.get("buckets")
        if not isinstance(buckets, list):
            return 1.0
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            max_atr = bucket.get("max_atr_pct")
            mult = bucket.get("multiplier")
            if max_atr is not None and mult is not None and atr_pct <= float(max_atr):
                return float(mult)
        return 1.0

    def _is_budget_multiplier_enabled(self) -> bool:
        """Check if budget multiplier is enabled in risk_overlay config.

        V22 architecture: When disabled (default), confirmed/unconfirmed only affects
        discrete decisions (max_new_positions_per_day, min_score_threshold) to avoid
        the cascade effect where budget changes -> position count changes -> different
        trade selection on subsequent days.

        Returns:
            True if use_budget_multiplier is explicitly set to True, False otherwise.
        """
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return False

        config = regime_data.get("config")
        if not isinstance(config, dict):
            return False

        risk_overlay = config.get("risk_overlay")
        if not isinstance(risk_overlay, dict):
            return False

        # Default to False to avoid cascade effects
        return bool(risk_overlay.get("use_budget_multiplier", False))

    def _compute_position_multiplier(
        self, conditions_met: int, min_required: int, intent_score: float, atr_pct: float | None = None
    ) -> tuple[float, str]:
        """Compute position size multiplier based on 3-tier bull confirmation.

        Tiers:
            - UNCONFIRMED (conditions_met < min_required): Use overlay multiplier (default 0.75)
            - CONFIRMED (conditions_met >= min_required): Baseline 1.0
            - CONFIRMED_PLUS (conditions_met >= plus.min_conditions AND score >= plus.min_score
                             AND atr_pct <= plus.max_atr_pct): 1.2

        Args:
            conditions_met: Number of bull confirmation conditions satisfied
            min_required: Minimum conditions required for CONFIRMED status
            intent_score: Score of the individual trade intent
            atr_pct: ATR as percentage of price (ATR/close), used as volatility gate for PLUS

        Returns:
            Tuple of (multiplier, tier_name) for logging
        """
        # Tier 1: UNCONFIRMED
        if conditions_met < min_required:
            overlay = self._extract_bull_risk_overlay()
            mult = float(overlay.get("position_size_multiplier", 0.75)) if overlay else 0.75
            return mult, "UNCONFIRMED"

        # Check for CONFIRMED_PLUS eligibility (only if explicitly enabled)
        plus_config = self._extract_bull_confirmed_plus_config()
        if plus_config is not None and plus_config.get("enabled", True):
            plus_min_conditions = int(plus_config.get("min_conditions", 3))
            plus_min_score = float(plus_config.get("min_score", 0.85))
            plus_max_atr_pct = float(plus_config.get("max_atr_pct", 0.045))
            plus_multiplier = float(plus_config.get("position_size_multiplier", 1.2))

            # Tier 3: CONFIRMED_PLUS (regime strength + quality + low volatility)
            # ATR gate: only give PLUS to low-volatility stocks to avoid "high-vol end-of-trend" traps
            # Conservative: require known atr_pct <= threshold (None means unknown, don't give PLUS)
            atr_ok = atr_pct is not None and atr_pct <= plus_max_atr_pct
            if conditions_met >= plus_min_conditions and intent_score >= plus_min_score and atr_ok:
                return plus_multiplier, "CONFIRMED_PLUS"

        # Tier 2: CONFIRMED (baseline)
        return 1.0, "CONFIRMED"

    def _extract_regime_staged_take_profit(self) -> StagedTakeProfitConfig | None:
        """Extract staged_take_profit from regime applied_overrides if available.

        Returns:
            StagedTakeProfitConfig if enabled in regime config, None otherwise.
        """
        outputs = self._get_last_outputs_by_module()
        regime_data = outputs.get("market_regime")
        if not isinstance(regime_data, dict):
            return None

        # Try applied_overrides.strategy.staged_take_profit
        applied = regime_data.get("applied_overrides")
        if isinstance(applied, dict):
            strategy_overrides = applied.get("strategy")
            if isinstance(strategy_overrides, dict):
                staged = strategy_overrides.get("staged_take_profit")
                if isinstance(staged, dict) and staged.get("enabled", False):
                    return StagedTakeProfitConfig(
                        enabled=True,
                        first_target_pct=float(staged.get("first_target_pct", 0.10)),
                        second_target_pct=float(staged.get("second_target_pct", 0.20)),
                        exit_fraction=float(staged.get("exit_fraction", 1.0 / 3.0)),
                    )

        # Fallback: try config.strategy.staged_take_profit
        config = regime_data.get("config")
        if isinstance(config, dict):
            strategy_cfg = config.get("strategy")
            if isinstance(strategy_cfg, dict):
                staged = strategy_cfg.get("staged_take_profit")
                if isinstance(staged, dict) and staged.get("enabled", False):
                    return StagedTakeProfitConfig(
                        enabled=True,
                        first_target_pct=float(staged.get("first_target_pct", 0.10)),
                        second_target_pct=float(staged.get("second_target_pct", 0.20)),
                        exit_fraction=float(staged.get("exit_fraction", 1.0 / 3.0)),
                    )

        return None

    @staticmethod
    def _inject_data_end_date(run_config: dict[str, Any], *, current_date: date, lookback_days: int = 180) -> None:
        """Set the Data plugin start/end dates to support ``current_date`` simulation.

        The Data plugin requires BOTH start_date AND end_date to be set for the
        configured dates to be used. If only end_date is set, it falls back to
        using universe.asof_timestamp which defeats the simulation date.
        """
        from datetime import timedelta

        plugins = run_config.get("plugins")
        if not isinstance(plugins, dict):
            return
        data_sources = plugins.get("data_sources")
        if not isinstance(data_sources, list):
            return

        for entry in data_sources:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "")
            if "data" not in name:
                continue
            cfg = entry.get("config")
            if cfg is None:
                cfg = {}
                entry["config"] = cfg
            if isinstance(cfg, dict):
                # Calculate start_date using lookback from current_date
                # This ensures regime detection has enough historical data
                start_date = current_date - timedelta(days=lookback_days)
                cfg["start_date"] = start_date.isoformat()
                cfg["end_date"] = current_date.isoformat()

    def _apply_regime_mode(self, run_config: dict[str, Any]) -> None:
        """Inject market regime mode when configured."""

        if self._regime_mode == "none":
            return
        market_regime = run_config.setdefault("market_regime", {})
        if isinstance(market_regime, dict):
            market_regime["enabled"] = True
            market_regime["mode"] = self._regime_mode

    @staticmethod
    def _extract_strategy_config(run_config: dict[str, Any]) -> dict[str, Any]:
        """Best-effort extraction of the Strategy plugin config."""

        direct = run_config.get("strategy")
        if isinstance(direct, dict):
            return dict(direct)

        plugins = run_config.get("plugins")
        if not isinstance(plugins, dict):
            return {}
        strategies = plugins.get("strategies")
        if not isinstance(strategies, list) or not strategies:
            return {}
        first = strategies[0]
        if not isinstance(first, dict):
            return {}
        cfg = first.get("config")
        if not isinstance(cfg, dict):
            return {}
        return dict(cfg)

    @staticmethod
    def _extract_price_snapshots(outputs_by_module: dict[str, Any]) -> dict[str, PriceSeriesSnapshot]:
        """Extract ``symbol -> PriceSeriesSnapshot`` from the data module output."""

        payload = outputs_by_module.get("data")
        if not isinstance(payload, dict):
            return {}
        series_raw = payload.get("series_by_symbol")
        if not isinstance(series_raw, dict):
            return {}

        snapshots: dict[str, PriceSeriesSnapshot] = {}
        for symbol, raw in series_raw.items():
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            if not isinstance(raw, dict):
                continue
            try:
                snapshots[symbol.strip().upper()] = msgspec.convert(dict(raw), type=PriceSeriesSnapshot)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "price_snapshot_decode_failed",
                    symbol=symbol,
                    error=str(exc)[:200],
                )
                continue
        return snapshots

    @staticmethod
    def _extract_intent_set(outputs_by_module: dict[str, Any]) -> OrderIntentSet | None:
        """Extract :class:`strategy.interface.OrderIntentSet` from strategy module output."""

        payload = outputs_by_module.get("strategy")
        if not isinstance(payload, dict):
            return None
        intents_raw = payload.get("intents")
        if not isinstance(intents_raw, dict):
            return None
        try:
            return msgspec.convert(dict(intents_raw), type=OrderIntentSet)
        except Exception as exc:  # noqa: BLE001
            logger.warning("intent_set_decode_failed", error=str(exc)[:200])
            return None

    @staticmethod
    def _schedule_entry_intents(
        *,
        intent_set: OrderIntentSet,
        execution_date: date,
        pending_intents: defaultdict[date, list[TradeIntent]],
        day_log: Any,
    ) -> int:
        """Schedule next-day entry intents, augmenting them with bracket prices.

        Strategy Engine outputs bracket legs as separate intents. For the
        simplified backtest, we attach stop-loss / take-profit prices directly
        to the entry intent so the execution simulation can build positions.
        """

        scheduled = 0

        for group in list(getattr(intent_set, "intent_groups", []) or []):
            intents = list(getattr(group, "intents", []) or [])
            entry = next((i for i in intents if getattr(i, "intent_type", None) == IntentType.OPEN_LONG), None)
            if entry is None:
                other_entry = next((i for i in intents if getattr(i, "intent_type", None) == IntentType.OPEN_SHORT), None)
                if other_entry is not None:
                    day_log.warning(
                        "unsupported_entry_intent_type_skip",
                        symbol=getattr(other_entry, "symbol", None),
                        intent_type=str(getattr(other_entry, "intent_type", "")),
                    )
                continue

            sl = next((i for i in intents if getattr(i, "intent_type", None) == IntentType.STOP_LOSS), None)
            tp = next((i for i in intents if getattr(i, "intent_type", None) == IntentType.TAKE_PROFIT), None)

            sl_price = getattr(sl, "stop_loss_price", None) if sl is not None else None
            tp_price = getattr(tp, "take_profit_price", None) if tp is not None else None
            if sl_price is None or tp_price is None:
                day_log.warning(
                    "intent_group_missing_bracket_prices_skip",
                    symbol=getattr(entry, "symbol", None),
                    entry_intent_id=getattr(entry, "intent_id", None),
                )
                continue

            augmented = msgspec.structs.replace(
                entry,
                stop_loss_price=float(sl_price),
                take_profit_price=float(tp_price),
            )
            pending_intents[execution_date].append(augmented)
            scheduled += 1

        return scheduled

    @staticmethod
    def _extract_bar_data(price_snapshots: dict[str, PriceSeriesSnapshot]) -> dict[str, PriceBar]:
        """Extract the latest :class:`data.interface.PriceBar` for each symbol.

        Args:
            price_snapshots: Mapping of ``symbol`` to :class:`data.interface.PriceSeriesSnapshot`.

        Returns:
            Mapping of ``symbol`` to the latest available :class:`data.interface.PriceBar`.
        """

        bar_data: dict[str, PriceBar] = {}
        for symbol, snapshot in price_snapshots.items():
            bars = list(getattr(snapshot, "bars", []) or [])
            if not bars:
                continue
            bar_data[str(symbol).strip().upper()] = bars[-1]
        return bar_data

    @staticmethod
    def _extract_historical_bars(price_snapshots: dict[str, PriceSeriesSnapshot]) -> dict[str, list[PriceBar]]:
        """Extract full bar history for each symbol from data snapshots."""

        historical: dict[str, list[PriceBar]] = {}
        for symbol, snapshot in price_snapshots.items():
            bars = list(getattr(snapshot, "bars", []) or [])
            if bars:
                historical[str(symbol).strip().upper()] = bars
        return historical

    @staticmethod
    def _get_next_trading_day(current_date: date, trading_days: list[date]) -> date | None:
        """Return the next trading day after ``current_date`` in ``trading_days``."""

        try:
            idx = trading_days.index(current_date)
        except ValueError:
            return None
        next_idx = idx + 1
        if next_idx >= len(trading_days):
            return None
        return trading_days[next_idx]

    @staticmethod
    def _get_intent_score(intent: TradeIntent) -> float:
        """Extract a numeric intent score (best-effort).

        If the intent has no score metadata, assume it's a high-quality opportunity (0.90).
        """

        score: Any | None = getattr(intent, "score", None)
        if score is None:
            metadata = getattr(intent, "metadata", None)
            if isinstance(metadata, dict):
                score = metadata.get("scanner_score")

        if isinstance(score, (int, float)):
            return float(score)
        if isinstance(score, str):
            try:
                return float(score)
            except ValueError:
                return 0.90

        return 0.90

    @staticmethod
    def _get_max_intent_score(intents: list[TradeIntent]) -> float:
        """Extract the highest score from intents.

        If an intent has no score metadata, assume high-quality opportunity and return 0.90.
        """

        max_score = 0.0
        for intent in intents:
            max_score = max(max_score, BacktestOrchestrator._get_intent_score(intent))

        return max_score

    # Trend pattern types for BULL_C filtering
    _TREND_PATTERN_TYPES: Final[frozenset[str]] = frozenset(
        {"ma_crossover"}
    )

    @staticmethod
    def _extract_intent_pattern_type(intent: TradeIntent) -> str:
        """Extract ss_pattern_type from intent.metadata shadow_score JSON.

        Returns empty string on missing/malformed data (fail-open: unknown
        types are never filtered).
        """
        metadata = getattr(intent, "metadata", None)
        if not isinstance(metadata, dict):
            return ""
        raw = metadata.get("shadow_score", "")
        if not raw:
            return ""
        try:
            import json as _json

            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                pt = parsed.get("ss_pattern_type", "")
                return str(pt) if pt else ""
        except (ValueError, TypeError):
            pass
        return ""

    # -- V27.4: Candidate-level pattern filter (pre-strategy dedup) ----------

    @staticmethod
    def _extract_candidate_pattern_type(candidate: dict) -> str:
        """Extract ss_pattern_type from a candidate dict (pre-serialized format).

        Returns empty string on missing/malformed data (fail-open: unknown
        types are never filtered).
        """
        meta = candidate.get("meta", {}) if isinstance(candidate, dict) else {}
        ss = meta.get("shadow_score", {}) if isinstance(meta, dict) else {}
        pt = ss.get("ss_pattern_type", "") if isinstance(ss, dict) else ""
        return str(pt) if pt else ""

    @staticmethod
    def _extract_candidate_atr_pct(candidate: dict) -> float | None:
        """Extract ss_atr_pct from a candidate dict.

        Returns None on missing/malformed data (fail-open: None passes filter).
        """
        meta = candidate.get("meta", {}) if isinstance(candidate, dict) else {}
        ss = meta.get("shadow_score", {}) if isinstance(meta, dict) else {}
        val = ss.get("ss_atr_pct") if isinstance(ss, dict) else None
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _build_candidate_filter(
        cls,
        is_confirmed: bool,
        *,
        pattern_filter: bool = True,
        trend_atr_pct_min: float | None = None,
        trend_atr_pct_max: float | None = None,
    ) -> Callable[[dict], dict]:
        """Build a candidate filter function for the EOD scan pipeline.

        Args:
            is_confirmed: True for BULL_C (trend lane), False for BULL_U (platform lane).
            pattern_filter: When True, apply V27.4 pattern-type filtering.
            trend_atr_pct_min: V27.5 minimum ATR% for trend candidates (None = no floor).
            trend_atr_pct_max: V27.5 maximum ATR% for trend candidates (None = no ceiling).
        """

        _has_atr_filter = trend_atr_pct_min is not None or trend_atr_pct_max is not None

        def _filter(payload: dict) -> dict:
            result = dict(payload)
            cs = result.get("candidates")
            if not isinstance(cs, dict):
                return result
            cands = cs.get("candidates")
            if not isinstance(cands, list):
                return result

            filtered = []
            for c in cands:
                pt = cls._extract_candidate_pattern_type(c)

                # 1. Pattern filter (V27.4, only when pattern_filter=True)
                if pattern_filter:
                    if is_confirmed:  # BULL_C: trend + unknown
                        if pt not in cls._TREND_PATTERN_TYPES and pt != "":
                            continue
                    else:  # BULL_U: platform + unknown
                        if pt != "platform" and pt != "":
                            continue

                # 2. ATR% filter (V27.5, only for trend candidates)
                if _has_atr_filter and pt in cls._TREND_PATTERN_TYPES:
                    atr = cls._extract_candidate_atr_pct(c)
                    if atr is not None:  # fail-open: None passes
                        if trend_atr_pct_min is not None and atr < trend_atr_pct_min:
                            continue
                        if trend_atr_pct_max is not None and atr > trend_atr_pct_max:
                            continue

                filtered.append(c)

            _filter.cf_before = len(cands)
            _filter.cf_kept = len(filtered)

            result["candidates"] = dict(cs)
            result["candidates"]["candidates"] = filtered
            result["candidates"]["total_detected"] = len(filtered)
            return result

        _filter.cf_before = 0
        _filter.cf_kept = 0
        return _filter
