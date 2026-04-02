from __future__ import annotations

from datetime import date

from backtest.portfolio_tracker import Position
from backtest.trade_simulator import (
    TimeStopOptimizationConfig,
    TrailingStopConfig,
    TradeSimulator,
    WeakExitConfig,
    PositionTrackingData,
)
from data.interface import PriceBar


def test_check_exits_triggers_stop_loss() -> None:
    simulator = TradeSimulator()
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=1.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=95.0,
            take_profit_price=120.0,
            intent_id="I-1",
        )
    }
    bar_data = {
        "AAPL": PriceBar(
            timestamp=1,
            open=100.0,
            high=101.0,
            low=95.0,
            close=99.0,
            volume=100,
        )
    }

    signals, _ = simulator.check_exits(positions, bar_data, current_date=date(2026, 1, 2))

    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
    assert signals[0].exit_reason == "STOP_LOSS"
    assert signals[0].exit_price == 95.0


def test_check_exits_triggers_take_profit() -> None:
    simulator = TradeSimulator()
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=1.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=95.0,
            take_profit_price=110.0,
            intent_id="I-2",
        )
    }
    bar_data = {
        "AAPL": PriceBar(
            timestamp=1,
            open=100.0,
            high=110.0,
            low=99.0,
            close=109.0,
            volume=100,
        )
    }

    signals, _ = simulator.check_exits(positions, bar_data, current_date=date(2026, 1, 2))

    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
    assert signals[0].exit_reason == "TAKE_PROFIT"
    assert signals[0].exit_price == 110.0


def test_check_exits_triggers_time_stop() -> None:
    simulator = TradeSimulator()
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=1.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=95.0,
            take_profit_price=110.0,
            intent_id="I-3",
        )
    }
    bar_data = {
        "AAPL": PriceBar(
            timestamp=1,
            open=100.0,
            high=101.0,
            low=99.0,
            close=98.5,
            volume=100,
        )
    }

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 2, 1),
        max_hold_days=30,
    )

    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
    assert signals[0].exit_reason == "TIME_STOP"
    assert signals[0].exit_price == 98.5


def test_check_exits_time_stop_optimization_skips_profitable_positions() -> None:
    simulator = TradeSimulator()
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=1.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=80.0,
            take_profit_price=150.0,
            intent_id="I-5",
        )
    }
    bar_data = {
        "AAPL": PriceBar(
            timestamp=1,
            open=105.0,
            high=106.0,
            low=104.0,
            close=106.0,  # +6%
            volume=100,
        )
    }

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 2, 1),
        max_hold_days=30,
        time_stop_optimization=TimeStopOptimizationConfig(
            enabled=True,
            skip_if_profit_above=0.05,
            early_exit_if_loss_below=-0.03,
        ),
    )

    assert signals == []


def test_check_exits_time_stop_optimization_exits_early_on_losses() -> None:
    simulator = TradeSimulator()
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=1.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=80.0,
            take_profit_price=150.0,
            intent_id="I-6",
        )
    }
    bar_data = {
        "AAPL": PriceBar(
            timestamp=1,
            open=97.0,
            high=98.0,
            low=96.0,
            close=96.0,  # -4%
            volume=100,
        )
    }

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 2, 1),
        max_hold_days=30,
        time_stop_optimization=TimeStopOptimizationConfig(
            enabled=True,
            skip_if_profit_above=0.05,
            early_exit_if_loss_below=-0.03,
        ),
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "EARLY_STOP"
    assert signals[0].exit_price == 96.0


def test_check_exits_time_stop_optimization_falls_back_to_time_stop() -> None:
    simulator = TradeSimulator()
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=1.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=80.0,
            take_profit_price=150.0,
            intent_id="I-7",
        )
    }
    bar_data = {
        "AAPL": PriceBar(
            timestamp=1,
            open=101.0,
            high=102.0,
            low=100.0,
            close=102.0,  # +2%
            volume=100,
        )
    }

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 2, 1),
        max_hold_days=30,
        time_stop_optimization=TimeStopOptimizationConfig(
            enabled=True,
            skip_if_profit_above=0.05,
            early_exit_if_loss_below=-0.03,
        ),
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "TIME_STOP"
    assert signals[0].exit_price == 102.0


def test_check_exits_priority_stop_loss_over_take_profit_over_time_stop() -> None:
    simulator = TradeSimulator()
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=1.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=95.0,
            take_profit_price=110.0,
            intent_id="I-4",
        )
    }
    bar_data = {
        "AAPL": PriceBar(
            timestamp=1,
            open=100.0,
            high=120.0,  # would trigger take-profit
            low=90.0,  # would trigger stop-loss
            close=115.0,
            volume=100,
        )
    }

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 2, 1),  # would also trigger time stop
        max_hold_days=30,
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "STOP_LOSS"


# ---------------------------------------------------------------------------
# V27.1 Weak Exit: deep-drawdown based, priority, safety valve tests
# ---------------------------------------------------------------------------

_WEAK_CFG = WeakExitConfig(
    enabled=True,
    weak_days=12,
    max_drawdown_pct_threshold=-0.06,  # must have dropped 6%
    min_current_unrealized_pct=-0.02,  # must still be losing 2%
    max_runup_r_gate=0.25,
    ftm_enabled=False,  # disabled in V27.1
)


def test_exit_priority_tp_beats_weak_exit() -> None:
    """Same bar triggers both TP and WEAK_EXIT conditions → TP wins."""
    simulator = TradeSimulator()
    # V27.1: WEAK_EXIT needs: hold>=12, max_dd<=-6%, current<=-2%, runup<0.25R
    # entry=100, SL=90 → R=10
    # bar high=110 → triggers TP
    # Setup WEAK_EXIT conditions: dropped to 93 (max_dd=-7%), current close=97 (-3%)
    positions = {
        "TEST": Position(
            symbol="TEST",
            entry_price=100.0,
            quantity=10.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=90.0,
            take_profit_price=110.0,
            intent_id="I-TP-WEAK",
        )
    }
    bar_data = {
        "TEST": PriceBar(
            timestamp=1,
            open=109.0,
            high=110.0,  # triggers TP
            low=97.0,
            close=97.0,  # current unrealized = -3% (meets -2% threshold)
            volume=100,
        )
    }
    runup = {"TEST": 101.0}  # max_runup_r = 0.1 < 0.25 gate
    drawdown = {"TEST": 93.0}  # max_dd = -7% (meets -6% threshold)

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 13),  # hold_days=12
        weak_exit=_WEAK_CFG,
        runup_tracking=runup,
        drawdown_tracking=drawdown,
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "TAKE_PROFIT"


def test_exit_priority_trailing_beats_weak_exit() -> None:
    """Trailing stop active and triggered on same bar as WEAK_EXIT → TRAILING wins."""
    simulator = TradeSimulator()
    trailing_cfg = TrailingStopConfig(
        enabled=True,
        activate_at_R=1.5,
        trail_pct=0.05,
        min_trail_pct=0.05,
        atr_multiplier=0.0,  # use fixed trail_pct
        force_after_partial_tp=True,
    )
    # entry=100, SL=90, R=10
    # Trailing activated previously: highest_close=115, trail=115*(1-0.05)=109.25
    # Bar low=108 < 109.25 → trailing triggers
    # Also set up WEAK_EXIT conditions: max_dd=-7%, current=-3%, runup<0.25R
    positions = {
        "TEST": Position(
            symbol="TEST",
            entry_price=100.0,
            quantity=10.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=90.0,
            take_profit_price=130.0,
            intent_id="I-TRAIL-WEAK",
        )
    }
    bar_data = {
        "TEST": PriceBar(
            timestamp=1,
            open=110.0,
            high=110.0,
            low=108.0,  # < trail stop 109.25
            close=97.0,  # current unrealized = -3%
            volume=100,
        )
    }
    # Pre-set tracking with trailing active and high peak
    tracking = {
        "TEST": PositionTrackingData(
            symbol="TEST",
            highest_close_since_entry=115.0,
            trailing_stop_active=True,
            entry_atr_pct=None,
        )
    }
    runup = {"TEST": 101.0}  # max_runup_r = 0.1 < 0.25 gate
    drawdown = {"TEST": 93.0}  # max_dd = -7%

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 13),  # hold_days=12
        trailing_stop=trailing_cfg,
        weak_exit=_WEAK_CFG,
        runup_tracking=runup,
        drawdown_tracking=drawdown,
        position_tracking=tracking,
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "TRAILING_STOP"


def test_weak_exit_blocked_by_partial_tp() -> None:
    """Position with partial_exits_completed >= 1 should NOT trigger WEAK_EXIT."""
    simulator = TradeSimulator()
    # V27.1: WEAK_EXIT needs: hold>=12, max_dd<=-6%, current<=-2%, runup<0.25R, no PT
    positions = {
        "TEST": Position(
            symbol="TEST",
            entry_price=100.0,
            quantity=7.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=90.0,
            take_profit_price=130.0,
            intent_id="I-PT-BLOCK",
            partial_exits_completed=1,  # already had PT1 → blocks WEAK_EXIT
            original_quantity=10.0,
        )
    }
    bar_data = {
        "TEST": PriceBar(
            timestamp=1,
            open=96.0,
            high=96.5,
            low=95.5,
            close=97.0,  # current unrealized = -3% (meets -2% threshold)
            volume=100,
        )
    }
    runup = {"TEST": 101.0}  # max_runup_r = 0.1 < 0.25
    drawdown = {"TEST": 93.0}  # max_dd = -7% (meets -6% threshold)

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 13),  # hold_days=12
        weak_exit=_WEAK_CFG,
        runup_tracking=runup,
        drawdown_tracking=drawdown,
    )

    # No WEAK_EXIT (partial_tp blocks it). No SL/TP/TIME_STOP either.
    assert len(signals) == 0


def test_weak_exit_blocked_by_runup_gate() -> None:
    """Position with max_runup_r >= gate should NOT trigger WEAK_EXIT."""
    simulator = TradeSimulator()
    # V27.1: WEAK_EXIT needs: hold>=12, max_dd<=-6%, current<=-2%, runup<0.25R
    positions = {
        "TEST": Position(
            symbol="TEST",
            entry_price=100.0,
            quantity=10.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=90.0,
            take_profit_price=130.0,
            intent_id="I-GATE-BLOCK",
        )
    }
    bar_data = {
        "TEST": PriceBar(
            timestamp=1,
            open=96.0,
            high=96.5,
            low=95.5,
            close=97.0,  # current unrealized = -3% (meets threshold)
            volume=100,
        )
    }
    # max_runup_r = (103-100)/10 = 0.3 >= 0.25 gate → blocked
    runup = {"TEST": 103.0}
    drawdown = {"TEST": 93.0}  # max_dd = -7% (meets threshold)

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 13),  # hold_days=12
        weak_exit=_WEAK_CFG,
        runup_tracking=runup,
        drawdown_tracking=drawdown,
    )

    # Runup gate blocks WEAK_EXIT. No other exit triggered.
    assert len(signals) == 0
