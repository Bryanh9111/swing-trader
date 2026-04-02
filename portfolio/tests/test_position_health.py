from __future__ import annotations

from datetime import date

import pytest

from backtest.portfolio_tracker import Position
from portfolio.position_health import PositionHealthScorer


def _position(
    *,
    symbol: str,
    entry_price: float = 100.0,
    quantity: float = 1.0,
    entry_date: date = date(2026, 1, 1),
    stop_loss_price: float = 90.0,
    take_profit_price: float = 110.0,
) -> Position:
    return Position(
        symbol=symbol,
        entry_price=entry_price,
        quantity=quantity,
        entry_date=entry_date,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        intent_id="I-1",
    )


def test_pnl_score_excellent_when_profitable() -> None:
    scorer = PositionHealthScorer()
    position = _position(symbol="AAA")

    score = scorer.score_position(position=position, current_price=110.0, current_date=date(2026, 1, 2))

    assert score.pnl_score == pytest.approx(1.0)


def test_pnl_score_poor_when_losing() -> None:
    scorer = PositionHealthScorer()
    position = _position(symbol="AAA")

    score = scorer.score_position(position=position, current_price=80.0, current_date=date(2026, 1, 2))

    assert score.pnl_score == pytest.approx(0.0)


def test_time_score_fresh_position() -> None:
    scorer = PositionHealthScorer()
    position = _position(symbol="AAA", entry_date=date(2026, 1, 1))

    score = scorer.score_position(position=position, current_price=100.0, current_date=date(2026, 1, 4))

    assert score.time_score == pytest.approx(1.0)


def test_time_score_mature_position() -> None:
    scorer = PositionHealthScorer()
    position = _position(symbol="AAA", entry_date=date(2026, 1, 1))

    score = scorer.score_position(position=position, current_price=100.0, current_date=date(2026, 1, 20))

    assert score.time_score == pytest.approx(0.4)


def test_trend_score_near_stop_loss() -> None:
    scorer = PositionHealthScorer()
    position = _position(symbol="AAA", stop_loss_price=90.0, take_profit_price=110.0)

    score = scorer.score_position(position=position, current_price=91.0, current_date=date(2026, 1, 2))

    assert score.trend_score == pytest.approx(0.05)


def test_trend_score_near_take_profit() -> None:
    scorer = PositionHealthScorer()
    position = _position(symbol="AAA", stop_loss_price=90.0, take_profit_price=110.0)

    score = scorer.score_position(position=position, current_price=109.0, current_date=date(2026, 1, 2))

    assert score.trend_score == pytest.approx(0.95)


def test_find_replaceable_positions() -> None:
    scorer = PositionHealthScorer()

    poor = _position(symbol="POOR", entry_date=date(2026, 1, 1), stop_loss_price=90.0, take_profit_price=110.0)
    good = _position(symbol="GOOD", entry_date=date(2026, 1, 25), stop_loss_price=90.0, take_profit_price=110.0)

    positions = {"POOR": poor, "GOOD": good}
    prices = {"POOR": 90.0, "GOOD": 110.0}

    replaceable = scorer.find_replaceable_positions(positions, prices, current_date=date(2026, 1, 26))

    assert [s.symbol for s in replaceable] == ["POOR"]
    assert replaceable[0].health_score < scorer.config.min_score_to_keep


def test_score_all_positions_sorted_by_health() -> None:
    scorer = PositionHealthScorer()

    low = _position(symbol="LOW", entry_date=date(2026, 1, 1), stop_loss_price=90.0, take_profit_price=110.0)
    mid = _position(symbol="MID", entry_date=date(2026, 1, 10), stop_loss_price=90.0, take_profit_price=110.0)
    high = _position(symbol="HIGH", entry_date=date(2026, 1, 25), stop_loss_price=90.0, take_profit_price=110.0)

    positions = {"LOW": low, "MID": mid, "HIGH": high}
    prices = {"LOW": 90.0, "MID": 100.0, "HIGH": 110.0}

    scores = scorer.score_all_positions(positions, prices, current_date=date(2026, 1, 26))

    assert [s.symbol for s in scores] == ["LOW", "MID", "HIGH"]

