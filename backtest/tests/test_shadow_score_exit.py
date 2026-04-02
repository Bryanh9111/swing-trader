"""Unit tests for V27.6 Shadow Score Exit Optimization.

Tests cover:
1-6: classify_shadow_tier() classification logic
7-8: _resolve_weak_exit_thresholds() parameter selection
9-10: WEAK_EXIT timing with low/high tiers
11: Winner protection overrides low tier
12: No shadow config → baseline identical
"""
from __future__ import annotations

from datetime import date, timedelta

from backtest.portfolio_tracker import Position
from backtest.trade_simulator import (
    ShadowScoreExitConfig,
    TradeSimulator,
    WeakExitConfig,
    classify_shadow_tier,
    _resolve_weak_exit_thresholds,
)
from data.interface import PriceBar


# --- Shared fixtures ---

def _default_ss_cfg(**overrides) -> ShadowScoreExitConfig:
    defaults = dict(
        enabled=True,
        w_atr_pct=1.0,
        w_rs_slope=1.0,
        w_dist_to_support=-0.5,
        median_atr_pct=0.04,
        iqr_atr_pct=0.02,
        median_rs_slope=0.003,
        iqr_rs_slope=0.003,
        median_dist_to_support=0.02,
        iqr_dist_to_support=0.02,
        low_threshold=-0.5,
        high_threshold=0.5,
        low_weak_days=8,
        low_max_dd_pct=-0.04,
        low_runup_r_gate=0.35,
        low_current_unrealized_pct=-0.01,
        low_winner_protect_runup_r=1.0,
        high_weak_days=16,
        high_max_dd_pct=-0.08,
    )
    defaults.update(overrides)
    return ShadowScoreExitConfig(**defaults)


def _default_we_cfg(**overrides) -> WeakExitConfig:
    defaults = dict(
        enabled=True,
        weak_days=12,
        max_drawdown_pct_threshold=-0.06,
        min_current_unrealized_pct=-0.02,
        max_runup_r_gate=0.25,
    )
    defaults.update(overrides)
    return WeakExitConfig(**defaults)


def _shadow_scores(atr_pct: float = 0.04, rs_slope: float = 0.003,
                   dist_to_support: float = 0.02) -> dict[str, object]:
    return {
        "ss_atr_pct": atr_pct,
        "ss_rs_slope": rs_slope,
        "ss_dist_to_support_pct": dist_to_support,
    }


# ===================================================================
# 1-6: classify_shadow_tier() tests
# ===================================================================

def test_classify_tier_low():
    """Composite z-score well below low_threshold → 'low'."""
    cfg = _default_ss_cfg(low_threshold=-0.5, high_threshold=0.5)
    # All bad: very low ATR (w=+1 → negative z), low RS, far from support
    scores = _shadow_scores(atr_pct=0.01, rs_slope=0.0001, dist_to_support=0.02)
    tier = classify_shadow_tier(scores, cfg)
    assert tier == "low", f"Expected 'low', got '{tier}'"


def test_classify_tier_high():
    """Composite z-score well above high_threshold → 'high'."""
    cfg = _default_ss_cfg(low_threshold=-0.5, high_threshold=0.5)
    # All good: very high ATR, high RS slope, close to support
    scores = _shadow_scores(atr_pct=0.08, rs_slope=0.01, dist_to_support=0.005)
    tier = classify_shadow_tier(scores, cfg)
    assert tier == "high", f"Expected 'high', got '{tier}'"


def test_classify_tier_mid():
    """Composite z-score between thresholds → 'mid'."""
    cfg = _default_ss_cfg(low_threshold=-0.5, high_threshold=0.5)
    # Exactly at median values → composite ≈ 0 → mid
    scores = _shadow_scores(atr_pct=0.04, rs_slope=0.003, dist_to_support=0.02)
    tier = classify_shadow_tier(scores, cfg)
    assert tier == "mid", f"Expected 'mid', got '{tier}'"


def test_classify_tier_none_scores():
    """shadow_scores=None → 'mid' (fail-safe)."""
    cfg = _default_ss_cfg()
    assert classify_shadow_tier(None, cfg) == "mid"


def test_classify_tier_disabled():
    """enabled=False → 'mid' regardless of scores."""
    cfg = _default_ss_cfg(enabled=False)
    scores = _shadow_scores(atr_pct=0.01, rs_slope=0.0001, dist_to_support=0.02)
    assert classify_shadow_tier(scores, cfg) == "mid"


def test_classify_tier_insufficient_dims():
    """Only 1 valid dimension → 'mid' (fail-safe, requires >= 2)."""
    cfg = _default_ss_cfg()
    scores = {"ss_atr_pct": 0.01}  # only 1 dim
    assert classify_shadow_tier(scores, cfg) == "mid"


# ===================================================================
# 7-8: _resolve_weak_exit_thresholds() tests
# ===================================================================

def test_resolve_thresholds_low():
    """Low tier returns low_weak_days and tighter thresholds."""
    we = _default_we_cfg()
    ss = _default_ss_cfg(low_weak_days=8, low_max_dd_pct=-0.04,
                         low_runup_r_gate=0.35, low_current_unrealized_pct=-0.01)
    days, dd, runup, current = _resolve_weak_exit_thresholds(we, ss, "low")
    assert days == 8
    assert dd == -0.04
    assert runup == 0.35
    assert current == -0.01


def test_resolve_thresholds_mid_default():
    """Mid tier or None config → global WeakExitConfig defaults."""
    we = _default_we_cfg(weak_days=12, max_drawdown_pct_threshold=-0.06,
                         max_runup_r_gate=0.25, min_current_unrealized_pct=-0.02)

    # mid tier
    days, dd, runup, current = _resolve_weak_exit_thresholds(we, _default_ss_cfg(), "mid")
    assert days == 12
    assert dd == -0.06
    assert runup == 0.25
    assert current == -0.02

    # None config
    days2, dd2, runup2, current2 = _resolve_weak_exit_thresholds(we, None, "low")
    assert days2 == 12
    assert dd2 == -0.06


# ===================================================================
# 9-10: Full WEAK_EXIT integration (through check_exits)
# ===================================================================

def _make_position(entry_date: date, shadow_scores: dict | None = None) -> Position:
    return Position(
        symbol="TEST",
        entry_price=100.0,
        quantity=10.0,
        entry_date=entry_date,
        stop_loss_price=90.0,
        take_profit_price=120.0,
        intent_id="I-test",
        shadow_scores=shadow_scores,
    )


def _make_bar(close: float = 97.0) -> PriceBar:
    """Bar that doesn't trigger SL (low=91) or TP (high=101)."""
    return PriceBar(timestamp=1, open=98.0, high=101.0, low=91.0,
                    close=close, volume=1000)


def test_weak_exit_low_tier_triggers_earlier():
    """Low tier triggers WEAK_EXIT at day 8, mid tier does not."""
    sim = TradeSimulator()
    we = _default_we_cfg(weak_days=12)
    ss = _default_ss_cfg(low_weak_days=8, low_max_dd_pct=-0.04,
                         low_runup_r_gate=0.35, low_current_unrealized_pct=-0.01)

    # Low-tier shadow scores
    low_scores = _shadow_scores(atr_pct=0.01, rs_slope=0.0001, dist_to_support=0.02)
    entry_date = date(2026, 1, 1)
    current_date = entry_date + timedelta(days=9)  # hold_days=9 (>= 8 for low)

    pos = _make_position(entry_date, shadow_scores=low_scores)
    bar = _make_bar(close=97.0)  # unrealized = -3% (< -0.01)

    signals, _ = sim.check_exits(
        {"TEST": pos}, {"TEST": bar}, current_date,
        max_hold_days=30,
        weak_exit=we,
        shadow_score_exit=ss,
        runup_tracking={"TEST": 100.2},  # max_runup_r ≈ 0.02 < 0.35
        drawdown_tracking={"TEST": 95.0},  # max_dd = -5% < -4%
    )
    assert any(s.exit_reason == "WEAK_EXIT" for s in signals), \
        f"Low tier at day 9 should trigger WEAK_EXIT, got {[s.exit_reason for s in signals]}"

    # Same trade but mid-tier scores → no trigger at day 9 (needs 12)
    mid_scores = _shadow_scores(atr_pct=0.04, rs_slope=0.003, dist_to_support=0.02)
    pos_mid = _make_position(entry_date, shadow_scores=mid_scores)
    signals_mid, _ = sim.check_exits(
        {"TEST": pos_mid}, {"TEST": bar}, current_date,
        max_hold_days=30,
        weak_exit=we,
        shadow_score_exit=ss,
        runup_tracking={"TEST": 100.2},
        drawdown_tracking={"TEST": 95.0},
    )
    assert not any(s.exit_reason == "WEAK_EXIT" for s in signals_mid), \
        "Mid tier at day 9 should NOT trigger WEAK_EXIT (needs 12 days)"


def test_weak_exit_high_tier_survives_longer():
    """High tier does NOT trigger WEAK_EXIT at day 12 (threshold is 16)."""
    sim = TradeSimulator()
    we = _default_we_cfg(weak_days=12)
    ss = _default_ss_cfg(high_weak_days=16, high_max_dd_pct=-0.08)

    high_scores = _shadow_scores(atr_pct=0.08, rs_slope=0.01, dist_to_support=0.005)
    entry_date = date(2026, 1, 1)
    current_date = entry_date + timedelta(days=13)  # hold_days=13

    pos = _make_position(entry_date, shadow_scores=high_scores)
    bar = _make_bar(close=95.0)  # unrealized = -5%

    signals, _ = sim.check_exits(
        {"TEST": pos}, {"TEST": bar}, current_date,
        max_hold_days=30,
        weak_exit=we,
        shadow_score_exit=ss,
        runup_tracking={"TEST": 100.2},
        drawdown_tracking={"TEST": 91.0},  # max_dd = -9% < -8%
    )
    assert not any(s.exit_reason == "WEAK_EXIT" for s in signals), \
        "High tier at day 13 should NOT trigger WEAK_EXIT (needs 16 days)"


# ===================================================================
# 11: Winner protection
# ===================================================================

def test_winner_protect_overrides_low():
    """max_runup_r >= 1.0 forces low tier → mid (winner protection)."""
    sim = TradeSimulator()
    we = _default_we_cfg(weak_days=12)
    ss = _default_ss_cfg(low_weak_days=8, low_winner_protect_runup_r=1.0)

    low_scores = _shadow_scores(atr_pct=0.01, rs_slope=0.0001, dist_to_support=0.02)
    entry_date = date(2026, 1, 1)
    current_date = entry_date + timedelta(days=9)  # >= 8 (low threshold)

    pos = _make_position(entry_date, shadow_scores=low_scores)
    bar = _make_bar(close=97.0)

    # max_runup = 110 → runup_r = (110-100)/10 = 1.0 → winner protection kicks in
    signals, _ = sim.check_exits(
        {"TEST": pos}, {"TEST": bar}, current_date,
        max_hold_days=30,
        weak_exit=we,
        shadow_score_exit=ss,
        runup_tracking={"TEST": 110.0},  # max_runup_r = 1.0 → protect
        drawdown_tracking={"TEST": 95.0},
    )
    assert not any(s.exit_reason == "WEAK_EXIT" for s in signals), \
        "Winner protection should override low tier (downgrade to mid, needs 12 days)"


# ===================================================================
# 12: No shadow config → baseline identical
# ===================================================================

def test_no_shadow_config_unchanged():
    """shadow_score_exit=None → V27.5.1 baseline behavior."""
    sim = TradeSimulator()
    we = _default_we_cfg(weak_days=12)

    scores = _shadow_scores(atr_pct=0.01, rs_slope=0.0001, dist_to_support=0.02)
    entry_date = date(2026, 1, 1)
    current_date = entry_date + timedelta(days=9)  # < 12 days

    pos = _make_position(entry_date, shadow_scores=scores)
    bar = _make_bar(close=97.0)

    # No shadow config → uses global weak_days=12 → day 9 should NOT trigger
    signals, _ = sim.check_exits(
        {"TEST": pos}, {"TEST": bar}, current_date,
        max_hold_days=30,
        weak_exit=we,
        shadow_score_exit=None,  # V27.5.1 baseline: no shadow scoring
        runup_tracking={"TEST": 100.2},
        drawdown_tracking={"TEST": 95.0},
    )
    assert not any(s.exit_reason == "WEAK_EXIT" for s in signals), \
        "Without shadow config, should use global weak_days=12, no trigger at day 9"
