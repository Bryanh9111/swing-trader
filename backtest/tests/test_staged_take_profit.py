from __future__ import annotations

from datetime import date

import pytest

from backtest.portfolio_tracker import PortfolioTracker, Position
from backtest.trade_simulator import StagedTakeProfitConfig, TradeSimulator
from data.interface import PriceBar


def test_staged_take_profit_triggers_first_partial_exit() -> None:
    simulator = TradeSimulator()
    cfg = StagedTakeProfitConfig(enabled=True, first_target_pct=0.10, second_target_pct=0.20, exit_fraction=1.0 / 3.0)
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=9.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=80.0,
            take_profit_price=150.0,
            intent_id="I-1",
        )
    }
    bar_data = {
        "AAPL": PriceBar(
            timestamp=1,
            open=100.0,
            high=109.0,
            low=99.0,
            close=110.0,
            volume=100,
        )
    }

    signals, _ = simulator.check_exits(positions, bar_data, current_date=date(2026, 1, 2), staged_take_profit=cfg)

    assert len(signals) == 1
    assert signals[0].exit_reason == "PARTIAL_TP_1"
    assert signals[0].is_partial is True
    assert signals[0].exit_quantity == pytest.approx(3.0)


def test_staged_take_profit_triggers_second_partial_exit() -> None:
    simulator = TradeSimulator()
    cfg = StagedTakeProfitConfig(enabled=True, first_target_pct=0.10, second_target_pct=0.20, exit_fraction=1.0 / 3.0)
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=6.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=80.0,
            take_profit_price=150.0,
            intent_id="I-2",
            partial_exits_completed=1,
            original_quantity=9.0,
        )
    }
    bar_data = {
        "AAPL": PriceBar(
            timestamp=1,
            open=110.0,
            high=119.0,
            low=109.0,
            close=120.0,
            volume=100,
        )
    }

    signals, _ = simulator.check_exits(positions, bar_data, current_date=date(2026, 1, 3), staged_take_profit=cfg)

    assert len(signals) == 1
    assert signals[0].exit_reason == "PARTIAL_TP_2"
    assert signals[0].is_partial is True
    assert signals[0].exit_quantity == pytest.approx(3.0)


def test_staged_take_profit_updates_remaining_position_correctly() -> None:
    tracker = PortfolioTracker(initial_capital=10_000.0)
    position = tracker.add_position(
        Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=9.0,
            entry_date=date(2026, 1, 1),
            stop_loss_price=80.0,
            take_profit_price=150.0,
            intent_id="I-3",
        )
    )
    assert position.quantity == pytest.approx(9.0)

    tracker.reduce_position("AAPL", exit_quantity=3.0, exit_price=110.0, exit_date=date(2026, 1, 2), exit_reason="PARTIAL_TP_1")
    remaining = tracker.positions["AAPL"]
    assert remaining.quantity == pytest.approx(6.0)
    assert remaining.partial_exits_completed == 1
    assert remaining.original_quantity == pytest.approx(9.0)

    tracker.reduce_position("AAPL", exit_quantity=3.0, exit_price=120.0, exit_date=date(2026, 1, 3), exit_reason="PARTIAL_TP_2")
    remaining = tracker.positions["AAPL"]
    assert remaining.quantity == pytest.approx(3.0)
    assert remaining.partial_exits_completed == 2
    assert remaining.original_quantity == pytest.approx(9.0)
