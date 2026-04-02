from __future__ import annotations

from datetime import date

import pytest

from backtest.portfolio_tracker import CompletedTrade, PortfolioTracker, Position


def test_add_position_rejects_insufficient_cash() -> None:
    tracker = PortfolioTracker(initial_capital=100.0)
    position = Position(
        symbol="AAPL",
        entry_price=50.0,
        quantity=3.0,
        entry_date=date(2026, 1, 1),
        stop_loss_price=45.0,
        take_profit_price=60.0,
        intent_id="I-1",
    )

    with pytest.raises(ValueError, match="insufficient cash"):
        tracker.add_position(position)

    assert tracker.cash == 100.0
    assert tracker.positions == {}


def test_close_position_updates_cash_correctly() -> None:
    tracker = PortfolioTracker(initial_capital=2000.0)
    tracker.add_position(
        Position(
            symbol="AAPL",
            entry_price=100.0,
            quantity=10.0,
            entry_date=date(2026, 1, 2),
            stop_loss_price=90.0,
            take_profit_price=130.0,
            intent_id="I-2",
        )
    )
    assert tracker.cash == 1000.0

    realized = tracker.close_position(
        symbol="AAPL",
        exit_price=120.0,
        exit_date=date(2026, 1, 3),
        exit_reason="TAKE_PROFIT",
    )

    assert realized == pytest.approx(200.0)
    assert tracker.cash == pytest.approx(2200.0)
    assert tracker.positions == {}
    assert len(tracker.completed_trades) == 1
    assert tracker.completed_trades[0].realized_pnl == pytest.approx(200.0)


_SHADOW = {
    "ss_version": 1,
    "ss_atr_pct": 3.0,
    "ss_box_quality": 0.9,
    "ss_volatility": 0.03,
    "ss_volume_chg_ratio": 1.5,
    "ss_rs_slope": 0.01,
    "ss_rs_lookback": 20,
    "ss_consolidation_days": 20,
    "ss_dist_to_support_pct": 0.015,
    "ss_pattern_type": "platform",
}


def test_close_position_passes_shadow_scores() -> None:
    """shadow_scores should propagate from Position to CompletedTrade on close."""
    tracker = PortfolioTracker(initial_capital=5000.0)
    tracker.add_position(
        Position(
            symbol="XYZ",
            entry_price=50.0,
            quantity=10.0,
            entry_date=date(2024, 6, 1),
            stop_loss_price=45.0,
            take_profit_price=60.0,
            intent_id="I-SS-1",
            shadow_scores=_SHADOW,
        )
    )
    tracker.close_position("XYZ", exit_price=55.0, exit_date=date(2024, 6, 10), exit_reason="TAKE_PROFIT")

    trade = tracker.completed_trades[0]
    assert trade.shadow_scores == _SHADOW
    for key in _SHADOW:
        assert key in trade.shadow_scores


def test_reduce_position_passes_shadow_scores() -> None:
    """shadow_scores should propagate to partial trade on reduce_position."""
    tracker = PortfolioTracker(initial_capital=5000.0)
    tracker.add_position(
        Position(
            symbol="ABC",
            entry_price=100.0,
            quantity=20.0,
            entry_date=date(2024, 7, 1),
            stop_loss_price=90.0,
            take_profit_price=120.0,
            intent_id="I-SS-2",
            shadow_scores=_SHADOW,
        )
    )
    tracker.reduce_position("ABC", exit_quantity=10.0, exit_price=110.0, exit_date=date(2024, 7, 5), exit_reason="PARTIAL_TP_1")

    partial = tracker.completed_trades[0]
    assert partial.shadow_scores == _SHADOW

    # Remaining position still has shadow_scores
    assert tracker.positions["ABC"].shadow_scores == _SHADOW


def test_shadow_scores_none_by_default() -> None:
    """shadow_scores defaults to None when not provided."""
    pos = Position(
        symbol="DEF",
        entry_price=10.0,
        quantity=5.0,
        entry_date=date(2024, 1, 1),
        stop_loss_price=9.0,
        take_profit_price=12.0,
        intent_id="I-NO-SS",
    )
    assert pos.shadow_scores is None

    trade = CompletedTrade(
        symbol="DEF",
        entry_date=date(2024, 1, 1),
        entry_price=10.0,
        exit_date=date(2024, 1, 5),
        exit_price=11.0,
        quantity=5.0,
        realized_pnl=5.0,
        exit_reason="TAKE_PROFIT",
    )
    assert trade.shadow_scores is None

