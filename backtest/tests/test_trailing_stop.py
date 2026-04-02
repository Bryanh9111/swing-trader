from __future__ import annotations

from datetime import date

import pytest

from backtest.portfolio_tracker import Position
from backtest.trade_simulator import (
    PositionTrackingData,
    TradeSimulator,
    TrailingStopConfig,
    StagedTakeProfitConfig,
)
from data.interface import PriceBar


ENTRY_PRICE = 100.0
STOP_LOSS_PRICE = 90.0
TAKE_PROFIT_PRICE = 140.0


def _make_position(
    *,
    entry_date: date = date(2026, 1, 1),
    partial_exits_completed: int = 0,
    quantity: float = 1.0,
    original_quantity: float | None = None,
) -> Position:
    return Position(
        symbol="AAPL",
        entry_price=ENTRY_PRICE,
        quantity=quantity,
        entry_date=entry_date,
        stop_loss_price=STOP_LOSS_PRICE,
        take_profit_price=TAKE_PROFIT_PRICE,
        intent_id="I-1",
        partial_exits_completed=partial_exits_completed,
        original_quantity=original_quantity,
    )


def _make_bar(*, close: float, low: float, high: float | None = None, open_: float | None = None) -> PriceBar:
    return PriceBar(
        timestamp=1,
        open=float(close if open_ is None else open_),
        high=float(close if high is None else high),
        low=low,
        close=close,
        volume=100,
    )


def _default_trailing_cfg(
    *,
    enabled: bool = True,
    activate_at_R: float = 1.5,
    trail_pct: float = 0.05,
    min_trail_pct: float = 0.05,
    atr_multiplier: float = 0.0,  # Default 0 = disabled for existing tests
    atr_period: int = 14,
    force_after_partial_tp: bool = True,
) -> TrailingStopConfig:
    return TrailingStopConfig(
        enabled=enabled,
        activate_at_R=activate_at_R,
        trail_pct=trail_pct,
        min_trail_pct=min_trail_pct,
        atr_multiplier=atr_multiplier,
        atr_period=atr_period,
        force_after_partial_tp=force_after_partial_tp,
    )


def _active_tracking(
    *,
    highest_close: float,
    trailing_stop_active: bool = True,
    entry_atr_pct: float | None = None,
) -> PositionTrackingData:
    return PositionTrackingData(
        symbol="AAPL",
        highest_close_since_entry=highest_close,
        extend_count=0,
        soft_trail_active=False,
        last_extend_date=None,
        trailing_stop_active=trailing_stop_active,
        entry_atr_pct=entry_atr_pct,
    )


def test_trailing_stop_not_activated_below_threshold() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=112.0, high=112.0, low=111.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )

    assert signals == []
    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is False


def test_trailing_stop_activates_at_threshold() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=115.0, high=115.0, low=114.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )

    assert signals == []
    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is True


def test_trailing_stop_exit_triggers() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}

    signals_day_1, tracking_day_1 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=120.0, high=120.0, low=116.0)},
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )

    assert signals_day_1 == []
    assert tracking_day_1 is not None
    assert tracking_day_1["AAPL"].trailing_stop_active is True

    signals_day_2, tracking_day_2 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=118.0, high=118.0, low=113.0)},
        current_date=date(2026, 1, 3),
        trailing_stop=cfg,
        position_tracking=tracking_day_1,
    )

    assert len(signals_day_2) == 1
    assert signals_day_2[0].exit_reason == "TRAILING_STOP"
    assert signals_day_2[0].exit_price == pytest.approx(114.0)
    assert tracking_day_2 is not None
    assert tracking_day_2["AAPL"].trailing_stop_active is True


def test_trailing_stop_exit_price_is_trail_price() -> None:
    """Exit price must be trail_stop_price (peak * 0.95), not bar.low or bar.close."""
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}

    # Day 1: activate at close=130, peak=130
    _, tracking_day_1 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=130.0, high=131.0, low=128.0)},
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )
    assert tracking_day_1 is not None
    assert tracking_day_1["AAPL"].trailing_stop_active is True

    # Day 2: low=123 <= 130*0.95=123.5, exit price should be 123.5
    signals, _ = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=124.0, high=125.0, low=123.0)},
        current_date=date(2026, 1, 3),
        trailing_stop=cfg,
        position_tracking=tracking_day_1,
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "TRAILING_STOP"
    assert signals[0].exit_price == pytest.approx(123.5)


def test_no_same_day_activation_and_exit() -> None:
    """Lookahead protection: activation on today's close cannot trigger exit on today's low."""
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}
    # close=120 would activate (2R), low=113 would hit trail (120*0.95=114)
    # But activation uses close (end of day), low already happened → no exit today
    bar_data = {"AAPL": _make_bar(close=120.0, high=121.0, low=113.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )

    # No exit on the activation day (even though low < trail)
    assert signals == []
    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is True


def test_trailing_stop_no_exit_above_trail() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=120.0, high=120.0, low=115.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )

    assert signals == []
    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is True


def test_trailing_stop_peak_updates() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}

    signals_day_1, tracking_day_1 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=115.0, high=115.0, low=114.0)},
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )
    assert signals_day_1 == []
    assert tracking_day_1 is not None
    assert tracking_day_1["AAPL"].trailing_stop_active is True
    assert tracking_day_1["AAPL"].highest_close_since_entry == pytest.approx(115.0)

    signals_day_2, tracking_day_2 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=125.0, high=125.0, low=120.0)},
        current_date=date(2026, 1, 3),
        trailing_stop=cfg,
        position_tracking=tracking_day_1,
    )
    assert signals_day_2 == []
    assert tracking_day_2 is not None
    assert tracking_day_2["AAPL"].trailing_stop_active is True
    assert tracking_day_2["AAPL"].highest_close_since_entry == pytest.approx(125.0)

    signals_day_3, tracking_day_3 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=124.0, high=124.0, low=119.0)},
        current_date=date(2026, 1, 4),
        trailing_stop=cfg,
        position_tracking=tracking_day_2,
    )
    assert signals_day_3 == []
    assert tracking_day_3 is not None
    assert tracking_day_3["AAPL"].trailing_stop_active is True
    assert tracking_day_3["AAPL"].highest_close_since_entry == pytest.approx(125.0)


def test_trailing_stop_forced_after_pt1() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg(force_after_partial_tp=True)

    positions = {"AAPL": _make_position(partial_exits_completed=1)}
    bar_data = {"AAPL": _make_bar(close=105.0, high=105.0, low=104.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )

    assert signals == []
    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is True


def test_trailing_stop_suppresses_legacy_time_stop() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position(entry_date=date(2026, 1, 1))}
    bar_data = {"AAPL": _make_bar(close=120.0, high=120.0, low=115.0)}

    signals, _ = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 2, 5),
        max_hold_days=30,
        trailing_stop=cfg,
        position_tracking={},
    )

    assert signals == []


def test_stop_loss_fires_before_trailing() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=115.0, high=116.0, low=89.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={"AAPL": _active_tracking(highest_close=120.0)},
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "STOP_LOSS"
    assert signals[0].exit_price == pytest.approx(90.0)
    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is True


def test_take_profit_fires_when_trailing_active() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=139.0, high=141.0, low=130.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={"AAPL": _active_tracking(highest_close=120.0)},
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "TAKE_PROFIT"
    assert signals[0].exit_price == pytest.approx(140.0)
    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is True


def test_trailing_stop_disabled_no_effect() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg(enabled=False)

    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=120.0, high=120.0, low=115.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )

    assert signals == []
    assert tracking == {}


def test_trailing_stop_without_v2() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}

    signals_day_1, tracking_day_1 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=115.0, high=115.0, low=114.0)},
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
        time_stop_v2=None,
    )
    assert signals_day_1 == []
    assert tracking_day_1 is not None
    assert tracking_day_1["AAPL"].trailing_stop_active is True

    signals_day_2, _ = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=112.0, high=112.0, low=109.0)},
        current_date=date(2026, 1, 3),
        trailing_stop=cfg,
        position_tracking=tracking_day_1,
        time_stop_v2=None,
    )

    assert len(signals_day_2) == 1
    assert signals_day_2[0].exit_reason == "TRAILING_STOP"
    assert signals_day_2[0].exit_price == pytest.approx(109.25)


def test_trailing_stop_persists_across_days() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}

    signals_day_1, tracking_day_1 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=115.0, high=115.0, low=114.0)},
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )
    assert signals_day_1 == []
    assert tracking_day_1 is not None
    assert tracking_day_1["AAPL"].trailing_stop_active is True

    signals_day_2, tracking_day_2 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=113.0, high=113.0, low=112.0)},
        current_date=date(2026, 1, 3),
        trailing_stop=cfg,
        position_tracking=tracking_day_1,
    )
    assert signals_day_2 == []
    assert tracking_day_2 is not None
    assert tracking_day_2["AAPL"].trailing_stop_active is True

    signals_day_3, _ = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=110.0, high=110.0, low=108.0)},
        current_date=date(2026, 1, 4),
        trailing_stop=cfg,
        position_tracking=tracking_day_2,
    )

    assert len(signals_day_3) == 1
    assert signals_day_3[0].exit_reason == "TRAILING_STOP"
    assert signals_day_3[0].exit_price == pytest.approx(109.25)


def test_staged_tp_fires_when_trailing_active() -> None:
    simulator = TradeSimulator()
    trailing_cfg = _default_trailing_cfg()
    staged_cfg = StagedTakeProfitConfig(enabled=True, first_target_pct=0.10, second_target_pct=0.20, exit_fraction=1.0 / 3.0)

    positions = {"AAPL": _make_position(quantity=9.0)}
    bar_data = {"AAPL": _make_bar(close=111.0, high=112.0, low=110.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        staged_take_profit=staged_cfg,
        trailing_stop=trailing_cfg,
        position_tracking={"AAPL": _active_tracking(highest_close=115.0)},
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "PARTIAL_TP_1"
    assert signals[0].is_partial is True
    assert signals[0].exit_quantity == pytest.approx(3.0)
    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is True


def test_trailing_stop_gap_down_exit_at_open() -> None:
    """When bar gaps below trail_stop, exit_price = bar.open (not trail_stop)."""
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg()

    positions = {"AAPL": _make_position()}

    # Day 1: activate at close=130, peak=130
    _, tracking_day_1 = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=130.0, high=131.0, low=128.0)},
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )
    assert tracking_day_1 is not None
    assert tracking_day_1["AAPL"].trailing_stop_active is True

    # Day 2: trail_stop = 130 * 0.95 = 123.5, but bar opens at 120 (gap down)
    # exit_price should be min(123.5, 120.0) = 120.0
    signals, _ = simulator.check_exits(
        positions,
        {"AAPL": _make_bar(close=119.0, high=121.0, low=118.0, open_=120.0)},
        current_date=date(2026, 1, 3),
        trailing_stop=cfg,
        position_tracking=tracking_day_1,
    )

    assert len(signals) == 1
    assert signals[0].exit_reason == "TRAILING_STOP"
    assert signals[0].exit_price == pytest.approx(120.0)  # bar.open, not 123.5


def test_trailing_stop_force_disabled() -> None:
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg(force_after_partial_tp=False)

    positions = {"AAPL": _make_position(partial_exits_completed=1)}
    bar_data = {"AAPL": _make_bar(close=105.0, high=105.0, low=104.0)}

    signals, tracking = simulator.check_exits(
        positions,
        bar_data,
        current_date=date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
    )

    assert signals == []
    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is False


# ---------------------------------------------------------------------------
# ATR-adaptive trailing stop tests (V22.2)
# ---------------------------------------------------------------------------


def _make_historical_bars(n: int, *, close: float, high_range: float = 5.0) -> list[PriceBar]:
    """Create n historical bars with consistent ATR for testing.

    Returns bars where TR ≈ high_range (each bar: high = close + high_range/2, low = close - high_range/2).
    ATR% ≈ high_range / close.
    """
    bars: list[PriceBar] = []
    for i in range(n):
        bars.append(
            PriceBar(
                timestamp=i,
                open=close,
                high=close + high_range / 2.0,
                low=close - high_range / 2.0,
                close=close,
                volume=100,
            )
        )
    return bars


def test_atr_adaptive_trail_wider_than_fixed() -> None:
    """Stock with 10% ATR locked at activation → effective trail = max(5%, 0.8*10%) = 8%.
    A 6% drop from peak should NOT trigger exit (within 8% trail).
    """
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg(atr_multiplier=0.8, min_trail_pct=0.05)

    peak = 115.0
    # 6% drop from peak: 115 * 0.94 = 108.1
    drop_close = peak * 0.94
    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=drop_close, low=drop_close)}
    # entry_atr_pct=0.10 simulates 10% ATR locked at activation
    tracking = {"AAPL": _active_tracking(highest_close=peak, entry_atr_pct=0.10)}

    exits, updated_tracking = simulator.check_exits(
        positions,
        bar_data,
        date(2026, 2, 15),
        trailing_stop=cfg,
        position_tracking=tracking,
    )

    # 6% drop < 8% trail → no exit
    assert len(exits) == 0


def test_atr_adaptive_trail_floor() -> None:
    """Stock with 3% ATR locked → effective trail = max(5%, 0.8*3%) = max(5%, 2.4%) = 5% (floor holds).
    A 5.5% drop from peak SHOULD trigger exit.
    """
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg(atr_multiplier=0.8, min_trail_pct=0.05)

    peak = 115.0
    # 5.5% drop: 115 * 0.945 = 108.675
    drop_close = peak * 0.945
    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=drop_close, low=drop_close)}
    # entry_atr_pct=0.03 simulates 3% ATR locked at activation
    tracking = {"AAPL": _active_tracking(highest_close=peak, entry_atr_pct=0.03)}

    exits, updated_tracking = simulator.check_exits(
        positions,
        bar_data,
        date(2026, 2, 15),
        trailing_stop=cfg,
        position_tracking=tracking,
    )

    # Floor 5% trail → trail_stop = 115 * 0.95 = 109.25 > 108.675 → exit
    assert len(exits) == 1
    assert exits[0].exit_reason == "TRAILING_STOP"


def test_atr_adaptive_no_atr_at_activation_fallback() -> None:
    """entry_atr_pct=None (no bars at activation) → falls back to fixed trail_pct.
    With trail_pct=0.05, a 6% drop triggers exit.
    """
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg(atr_multiplier=0.8, trail_pct=0.05)

    peak = 115.0
    drop_close = peak * 0.94  # 6% drop
    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=drop_close, low=drop_close)}
    # entry_atr_pct=None simulates no bars available at activation
    tracking = {"AAPL": _active_tracking(highest_close=peak, entry_atr_pct=None)}

    exits, updated_tracking = simulator.check_exits(
        positions,
        bar_data,
        date(2026, 2, 15),
        trailing_stop=cfg,
        position_tracking=tracking,
    )

    # Fallback to 5% trail → trail_stop = 115 * 0.95 = 109.25 > 108.1 → exit
    assert len(exits) == 1
    assert exits[0].exit_reason == "TRAILING_STOP"


def test_atr_adaptive_exit_price() -> None:
    """Verify trail_stop_price = peak * (1 - effective_trail_pct) when ATR-adaptive."""
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg(atr_multiplier=0.8, min_trail_pct=0.05)

    peak = 120.0
    # With ATR=8% locked: effective_trail = max(5%, 0.8*8%) = max(5%, 6.4%) = 6.4%
    # trail_stop = 120 * (1 - 0.064) = 112.32
    # Bar low at 112.0 → exit at 112.0 (gap-down: min(112.32, 112.0))
    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=111.0, low=112.0, open_=112.0)}
    tracking = {"AAPL": _active_tracking(highest_close=peak, entry_atr_pct=0.08)}

    exits, updated_tracking = simulator.check_exits(
        positions,
        bar_data,
        date(2026, 2, 15),
        trailing_stop=cfg,
        position_tracking=tracking,
    )

    assert len(exits) == 1
    assert exits[0].exit_reason == "TRAILING_STOP"
    # exit_price = min(trail_stop, open) = min(112.32, 112.0) = 112.0
    assert abs(exits[0].exit_price - 112.0) < 0.01


def test_atr_multiplier_zero_uses_fixed() -> None:
    """atr_multiplier=0 → always uses fixed trail_pct regardless of entry_atr_pct."""
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg(atr_multiplier=0.0, trail_pct=0.05)

    peak = 115.0
    # 4% drop from peak: 115 * 0.96 = 110.4
    drop_close = peak * 0.96
    positions = {"AAPL": _make_position()}
    bar_data = {"AAPL": _make_bar(close=drop_close, low=drop_close)}
    # Even with high entry_atr_pct, atr_multiplier=0 means fixed trail
    tracking = {"AAPL": _active_tracking(highest_close=peak, entry_atr_pct=0.15)}

    exits, updated_tracking = simulator.check_exits(
        positions,
        bar_data,
        date(2026, 2, 15),
        trailing_stop=cfg,
        position_tracking=tracking,
    )

    # Fixed 5% trail → trail_stop = 115 * 0.95 = 109.25; 110.4 > 109.25 → no exit
    assert len(exits) == 0


def test_atr_locked_at_activation() -> None:
    """ATR% is computed and locked at activation time, not at exit time.
    Verify activation stores entry_atr_pct in tracking.
    """
    simulator = TradeSimulator()
    cfg = _default_trailing_cfg(atr_multiplier=0.8, min_trail_pct=0.05)

    positions = {"AAPL": _make_position()}
    # Day 1: close=115 → unrealized_R = (115-100)/10 = 1.5 → activates
    bar_data = {"AAPL": _make_bar(close=115.0, high=115.0, low=114.0)}
    # Provide historical bars so ATR can be computed at activation
    hist_bars = {"AAPL": _make_historical_bars(20, close=115.0, high_range=115.0 * 0.10)}

    _, tracking = simulator.check_exits(
        positions,
        bar_data,
        date(2026, 1, 2),
        trailing_stop=cfg,
        position_tracking={},
        historical_bars=hist_bars,
    )

    assert tracking is not None
    assert tracking["AAPL"].trailing_stop_active is True
    assert tracking["AAPL"].entry_atr_pct is not None
    assert tracking["AAPL"].entry_atr_pct == pytest.approx(0.10, abs=0.01)
