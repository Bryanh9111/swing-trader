from __future__ import annotations

from datetime import date

from backtest.portfolio_tracker import Position
from portfolio.position_rotator import PositionRotator, RotationConfig


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


def test_rotation_disabled_returns_no_rotate() -> None:
    rotator = PositionRotator(rotation_config=RotationConfig(enabled=False))

    decision = rotator.evaluate_rotation(
        positions={"AAA": _position(symbol="AAA")},
        market_prices={"AAA": 90.0},
        current_date=date(2026, 1, 20),
        new_opportunity_score=1.0,
        max_positions=1,
    )

    assert decision.should_rotate is False
    assert decision.positions_to_close == []
    assert decision.reason == "rotation_disabled"
    assert decision.health_scores == {}


def test_positions_not_full_returns_no_rotate() -> None:
    rotator = PositionRotator()

    decision = rotator.evaluate_rotation(
        positions={"AAA": _position(symbol="AAA"), "BBB": _position(symbol="BBB")},
        market_prices={"AAA": 100.0, "BBB": 100.0},
        current_date=date(2026, 1, 20),
        new_opportunity_score=1.0,
        max_positions=3,
    )

    assert decision.should_rotate is False
    assert decision.positions_to_close == []
    assert decision.reason == "positions_not_full"
    assert decision.health_scores == {}


def test_low_opportunity_score_returns_no_rotate() -> None:
    rotator = PositionRotator(rotation_config=RotationConfig(min_new_opportunity_score=0.95))

    decision = rotator.evaluate_rotation(
        positions={"AAA": _position(symbol="AAA")},
        market_prices={"AAA": 100.0},
        current_date=date(2026, 1, 20),
        new_opportunity_score=0.10,
        max_positions=1,
    )

    assert decision.should_rotate is False
    assert decision.positions_to_close == []
    assert decision.reason.startswith("opportunity_score_too_low")
    assert decision.health_scores == {}


def test_no_replaceable_positions_returns_no_rotate() -> None:
    rotator = PositionRotator()

    positions = {
        "GOOD": _position(symbol="GOOD", entry_date=date(2026, 1, 19)),
        "GREAT": _position(symbol="GREAT", entry_date=date(2026, 1, 19)),
    }
    prices = {"GOOD": 110.0, "GREAT": 110.0}

    decision = rotator.evaluate_rotation(
        positions=positions,
        market_prices=prices,
        current_date=date(2026, 1, 20),
        new_opportunity_score=1.0,
        max_positions=2,
    )

    assert decision.should_rotate is False
    assert decision.positions_to_close == []
    assert decision.reason == "no_replaceable_positions"
    assert set(decision.health_scores) == {"GOOD", "GREAT"}


def test_successful_rotation_selects_weakest() -> None:
    rotator = PositionRotator()

    positions = {
        "WEAKEST": _position(symbol="WEAKEST", entry_date=date(2026, 1, 1)),
        "WEAK": _position(symbol="WEAK", entry_date=date(2026, 1, 1)),
        "GOOD": _position(symbol="GOOD", entry_date=date(2026, 1, 19)),
    }
    prices = {"WEAKEST": 80.0, "WEAK": 93.0, "GOOD": 110.0}

    decision = rotator.evaluate_rotation(
        positions=positions,
        market_prices=prices,
        current_date=date(2026, 1, 20),
        new_opportunity_score=1.0,
        max_positions=3,
        num_new_opportunities=1,
    )

    assert decision.should_rotate is True
    assert decision.positions_to_close == ["WEAKEST"]


def test_rotation_respects_max_positions_to_rotate() -> None:
    rotator = PositionRotator(rotation_config=RotationConfig(max_positions_to_rotate=1))

    positions = {
        "W1": _position(symbol="W1", entry_date=date(2026, 1, 1)),
        "W2": _position(symbol="W2", entry_date=date(2026, 1, 1)),
        "GOOD": _position(symbol="GOOD", entry_date=date(2026, 1, 19)),
    }
    prices = {"W1": 80.0, "W2": 85.0, "GOOD": 110.0}

    decision = rotator.evaluate_rotation(
        positions=positions,
        market_prices=prices,
        current_date=date(2026, 1, 20),
        new_opportunity_score=1.0,
        max_positions=3,
        num_new_opportunities=2,
    )

    assert decision.should_rotate is True
    assert len(decision.positions_to_close) == 1


def test_rotation_requires_pnl_loss_when_configured() -> None:
    rotator = PositionRotator(
        rotation_config=RotationConfig(
            min_health_score_to_keep=0.99,
            require_pnl_loss=True,
            min_loss_pct_to_replace=-0.03,
        )
    )

    positions = {
        "WINNER": _position(symbol="WINNER", entry_date=date(2026, 1, 1)),
    }
    prices = {"WINNER": 110.0}

    decision = rotator.evaluate_rotation(
        positions=positions,
        market_prices=prices,
        current_date=date(2026, 1, 20),
        new_opportunity_score=1.0,
        max_positions=1,
    )

    assert decision.should_rotate is False
    assert decision.positions_to_close == []
    assert decision.reason == "no_replaceable_positions"

