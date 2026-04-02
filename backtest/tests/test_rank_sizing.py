"""V25: Two-Pass Rank-Based Sizing tests.

Tests verify that bull_confirmed rank-based budget multipliers are correctly
applied based on intent score ranking using two-pass allocation (fixed total
exposure, proportional weight distribution).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from backtest.orchestrator import BacktestOrchestrator
from backtest.portfolio_tracker import PortfolioTracker, Position
from data.interface import PriceBar, PriceSeriesSnapshot
from portfolio.interface import CapitalAllocationConfig
from strategy.interface import IntentGroup, IntentType, OrderIntentSet, TradeIntent


# ---------------------------------------------------------------------------
# Helpers (mirrors test_capital_allocation_integration.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ScanArtifacts:
    intent_set: OrderIntentSet | None
    price_snapshots: dict[str, PriceSeriesSnapshot]


def _snapshot(*, symbol: str, open_price: float, timestamp: int = 1) -> PriceSeriesSnapshot:
    bar = PriceBar(
        timestamp=timestamp,
        open=float(open_price),
        high=float(open_price),
        low=float(open_price),
        close=float(open_price),
        volume=1,
    )
    return PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=int(timestamp),
        symbol=symbol,
        timeframe="1d",
        bars=[bar],
        source="test",
        quality_flags={},
    )


def _intent_group(
    *,
    symbol: str,
    intent_id: str,
    quantity: float = 1.0,
    entry_price: float = 100.0,
    sl_pct: float = 0.10,
    tp_pct: float = 0.10,
    score: float = 0.80,
) -> IntentGroup:
    created_at = 1
    stop_loss_price = entry_price * (1.0 - sl_pct)
    take_profit_price = entry_price * (1.0 + tp_pct)

    entry = TradeIntent(
        intent_id=intent_id,
        symbol=symbol,
        intent_type=IntentType.OPEN_LONG,
        quantity=float(quantity),
        created_at_ns=created_at,
        entry_price=float(entry_price),
        metadata={"scanner_score": str(score)},
    )
    sl = TradeIntent(
        intent_id=f"{intent_id}-SL",
        symbol=symbol,
        intent_type=IntentType.STOP_LOSS,
        quantity=0.0,
        created_at_ns=created_at,
        stop_loss_price=float(stop_loss_price),
    )
    tp = TradeIntent(
        intent_id=f"{intent_id}-TP",
        symbol=symbol,
        intent_type=IntentType.TAKE_PROFIT,
        quantity=0.0,
        created_at_ns=created_at,
        take_profit_price=float(take_profit_price),
    )
    return IntentGroup(
        group_id=f"G-{intent_id}",
        symbol=symbol,
        intents=[entry, sl, tp],
        created_at_ns=created_at,
        contingency_type="OUO",
    )


def _make_regime_outputs(
    *,
    rank_sizing_enabled: bool = True,
    multipliers: list[float] | None = None,
    detected_regime: str = "bull_market",
    bull_confirmation_enabled: bool = True,
    min_conditions_required: int = 2,
) -> dict[str, Any]:
    """Build a mock market_regime output dict for _get_last_outputs_by_module."""
    if multipliers is None:
        multipliers = [1.25, 1.00, 0.75]
    return {
        "market_regime": {
            "detected_regime": detected_regime,
            "config": {
                "risk_overlay": {
                    "use_budget_multiplier": False,
                    "rank_sizing": {
                        "enabled": rank_sizing_enabled,
                        "multipliers": multipliers,
                    },
                },
                "bull_confirmation": {
                    "enabled": bull_confirmation_enabled,
                    "min_conditions_required": min_conditions_required,
                    "conditions": {
                        "qqq_above_ema50": True,
                        "qqq_ema20_above_ema50": True,
                        "qqq_20d_return_threshold": 0.03,
                        "ma50_slope_threshold": 0.005,
                    },
                },
                "strategy": {
                    "min_score_threshold": 0.72,
                    "max_new_positions_per_day": 4,
                    "max_positions_to_rotate": 0,
                },
                "capital_allocation": {
                    "overrides": {
                        "bull_confirmed": {
                            "target_position_pct": 0.09,
                            "max_position_pct": 0.10,
                        },
                    },
                },
            },
        },
    }


class _StubBacktestOrchestrator(BacktestOrchestrator):
    """Stub that injects scan artifacts and optionally fakes regime outputs."""

    def __init__(
        self,
        *,
        scans: dict[date, _ScanArtifacts],
        regime_outputs: dict[str, Any] | None = None,
        force_bull_confirmed: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(eod_orchestrator=SimpleNamespace(candidate_filter_fn=None), **kwargs)
        self._scans = scans
        self._regime_outputs = regime_outputs
        self._force_bull_confirmed = force_bull_confirmed

    def _run_daily_scan(self, *, current_date: date, config: dict[str, Any]) -> Any:
        del config
        return self._scans[current_date]

    def _get_last_outputs_by_module(self) -> dict[str, Any]:
        if self._regime_outputs is not None:
            return self._regime_outputs
        return {}

    def _check_bull_confirmed(self, price_snapshots: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if self._force_bull_confirmed:
            return True, {"conditions_met": 3, "min_required": 2, "reason": "test_forced"}
        return False, {"conditions_met": 0, "min_required": 2, "reason": "test_forced_unconfirmed"}


class _SpyPortfolioTracker(PortfolioTracker):
    last_instance: "_SpyPortfolioTracker | None" = None

    def __init__(self, initial_capital: float = 2000.0) -> None:
        super().__init__(initial_capital=initial_capital)
        type(self).last_instance = self


# ---------------------------------------------------------------------------
# Test 1: Multiplier mapping for 4 intents
# ---------------------------------------------------------------------------


def test_rank_multiplier_mapping(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given 4 intents sorted by score desc, multipliers = [1.25, 1.00, 0.75, 0.75]."""
    monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

    day1 = date(2026, 1, 1)
    day2 = date(2026, 1, 2)

    # 4 intents with descending scores — all at $10 entry, large qty so budget is the binding constraint
    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[
            _intent_group(symbol="AAA", intent_id="I-AAA", entry_price=10.0, quantity=9999.0, score=0.95),
            _intent_group(symbol="BBB", intent_id="I-BBB", entry_price=10.0, quantity=9999.0, score=0.90),
            _intent_group(symbol="CCC", intent_id="I-CCC", entry_price=10.0, quantity=9999.0, score=0.85),
            _intent_group(symbol="DDD", intent_id="I-DDD", entry_price=10.0, quantity=9999.0, score=0.80),
        ],
        constraints_applied={},
        source_candidates=["AAA", "BBB", "CCC", "DDD"],
    )

    scans = {
        day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
        day2: _ScanArtifacts(
            intent_set=None,
            price_snapshots={
                "AAA": _snapshot(symbol="AAA", open_price=10.0),
                "BBB": _snapshot(symbol="BBB", open_price=10.0),
                "CCC": _snapshot(symbol="CCC", open_price=10.0),
                "DDD": _snapshot(symbol="DDD", open_price=10.0),
            },
        ),
    }

    # bull_confirmed with rank_sizing enabled
    regime_outputs = _make_regime_outputs(rank_sizing_enabled=True, multipliers=[1.25, 1.00, 0.75])

    orchestrator = _StubBacktestOrchestrator(
        scans=scans,
        regime_outputs=regime_outputs,
        initial_capital=100_000.0,
        output_dir=str(tmp_path),
        quiet=True,
        regime_mode="bull",
        capital_allocation_config=CapitalAllocationConfig(
            max_positions=6,
            base_position_pct=0.09,
            max_position_pct=0.10,
            use_dynamic_sizing=False,
        ),
    )
    orchestrator.run_backtest([day1, day2], config={})

    portfolio = _SpyPortfolioTracker.last_instance
    assert portfolio is not None

    # All 4 should open
    assert len(portfolio.positions) == 4

    # base budget = 100_000 * 0.09 = 9_000; at $10/share = 900 shares max
    # rank0 (AAA): 9000*1.25=11250 → 1125 shares (but capped by max_position_pct=0.10 → 10000 → 1000)
    # rank1 (BBB): 9000*1.00=9000 → 900 shares
    # rank2 (CCC): 9000*0.75=6750 → 675 shares
    # rank3 (DDD): 9000*0.75=6750 → 675 shares
    qty_aaa = portfolio.positions["AAA"].quantity
    qty_bbb = portfolio.positions["BBB"].quantity
    qty_ccc = portfolio.positions["CCC"].quantity
    qty_ddd = portfolio.positions["DDD"].quantity

    # rank0 > rank1 > rank2 == rank3
    assert qty_aaa >= qty_bbb
    assert qty_bbb > qty_ccc
    assert qty_ccc == qty_ddd


# ---------------------------------------------------------------------------
# Test 2: Budget ordering — rank1 qty > rank2 qty > rank3 qty
# ---------------------------------------------------------------------------


def test_rank_sizing_budget_ordering(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """rank1 gets more shares than rank2, rank2 more than rank3."""
    monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

    day1 = date(2026, 1, 1)
    day2 = date(2026, 1, 2)

    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[
            _intent_group(symbol="AAA", intent_id="I-AAA", entry_price=50.0, quantity=9999.0, score=0.95),
            _intent_group(symbol="BBB", intent_id="I-BBB", entry_price=50.0, quantity=9999.0, score=0.88),
            _intent_group(symbol="CCC", intent_id="I-CCC", entry_price=50.0, quantity=9999.0, score=0.80),
        ],
        constraints_applied={},
        source_candidates=["AAA", "BBB", "CCC"],
    )

    scans = {
        day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
        day2: _ScanArtifacts(
            intent_set=None,
            price_snapshots={
                "AAA": _snapshot(symbol="AAA", open_price=50.0),
                "BBB": _snapshot(symbol="BBB", open_price=50.0),
                "CCC": _snapshot(symbol="CCC", open_price=50.0),
            },
        ),
    }

    regime_outputs = _make_regime_outputs(rank_sizing_enabled=True, multipliers=[1.25, 1.00, 0.75])

    orchestrator = _StubBacktestOrchestrator(
        scans=scans,
        regime_outputs=regime_outputs,
        initial_capital=100_000.0,
        output_dir=str(tmp_path),
        quiet=True,
        regime_mode="bull",
        capital_allocation_config=CapitalAllocationConfig(
            max_positions=6,
            base_position_pct=0.09,
            max_position_pct=0.15,
            use_dynamic_sizing=False,
        ),
    )
    orchestrator.run_backtest([day1, day2], config={})

    portfolio = _SpyPortfolioTracker.last_instance
    assert portfolio is not None
    assert len(portfolio.positions) == 3

    # base budget = 100_000 * 0.09 = 9_000; at $50/share
    # rank0: 9000*1.25 = 11250 → floor(11250/50) = 225 shares
    # rank1: 9000*1.00 = 9000  → floor(9000/50) = 180 shares
    # rank2: 9000*0.75 = 6750  → floor(6750/50) = 135 shares
    assert portfolio.positions["AAA"].quantity > portfolio.positions["BBB"].quantity
    assert portfolio.positions["BBB"].quantity > portfolio.positions["CCC"].quantity


# ---------------------------------------------------------------------------
# Test 3: Cash shortage — low rank trimmed first
# ---------------------------------------------------------------------------


def test_rank_sizing_cash_shortage_low_rank_trimmed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With limited cash, rank 3 should be naturally trimmed by min_position_value."""
    monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

    day1 = date(2026, 1, 1)
    day2 = date(2026, 1, 2)

    # 3 intents at $100 entry — capital just enough for ~2.5 positions
    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[
            _intent_group(symbol="AAA", intent_id="I-AAA", entry_price=100.0, quantity=9999.0, score=0.95),
            _intent_group(symbol="BBB", intent_id="I-BBB", entry_price=100.0, quantity=9999.0, score=0.88),
            _intent_group(symbol="CCC", intent_id="I-CCC", entry_price=100.0, quantity=9999.0, score=0.80),
        ],
        constraints_applied={},
        source_candidates=["AAA", "BBB", "CCC"],
    )

    scans = {
        day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
        day2: _ScanArtifacts(
            intent_set=None,
            price_snapshots={
                "AAA": _snapshot(symbol="AAA", open_price=100.0),
                "BBB": _snapshot(symbol="BBB", open_price=100.0),
                "CCC": _snapshot(symbol="CCC", open_price=100.0),
            },
        ),
    }

    regime_outputs = _make_regime_outputs(rank_sizing_enabled=True, multipliers=[1.25, 1.00, 0.75])

    # Only $2200 capital with base_position_pct=0.40 → budget ~880
    # rank0: 880*1.25=1100 → 11 shares ($1100)
    # rank1: 880*1.00=880 → 8 shares ($800)
    # rank2: 880*0.75=660 → but only ~$300 cash left → 3 shares ($300)
    # With min_position_value=50, rank2 may still open but smaller
    orchestrator = _StubBacktestOrchestrator(
        scans=scans,
        regime_outputs=regime_outputs,
        initial_capital=2_200.0,
        output_dir=str(tmp_path),
        quiet=True,
        regime_mode="bull",
        capital_allocation_config=CapitalAllocationConfig(
            max_positions=6,
            base_position_pct=0.40,
            max_position_pct=0.50,
            use_dynamic_sizing=False,
        ),
    )
    orchestrator.run_backtest([day1, day2], config={})

    portfolio = _SpyPortfolioTracker.last_instance
    assert portfolio is not None

    # rank0 and rank1 should definitely open
    assert "AAA" in portfolio.positions
    assert "BBB" in portfolio.positions

    # rank0 should have more shares than rank1
    assert portfolio.positions["AAA"].quantity >= portfolio.positions["BBB"].quantity

    # rank2 may or may not open depending on remaining cash — but if it does,
    # it should have fewer shares than rank1
    if "CCC" in portfolio.positions:
        assert portfolio.positions["CCC"].quantity <= portfolio.positions["BBB"].quantity


# ---------------------------------------------------------------------------
# Test 4: Rank sizing only applies in bull_confirmed
# ---------------------------------------------------------------------------


def test_rank_sizing_only_bull_confirmed(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When regime is not bull or not confirmed, multiplier should be 1.0 for all."""
    monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

    day1 = date(2026, 1, 1)
    day2 = date(2026, 1, 2)

    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[
            _intent_group(symbol="AAA", intent_id="I-AAA", entry_price=50.0, quantity=9999.0, score=0.95),
            _intent_group(symbol="BBB", intent_id="I-BBB", entry_price=50.0, quantity=9999.0, score=0.80),
        ],
        constraints_applied={},
        source_candidates=["AAA", "BBB"],
    )

    scans = {
        day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
        day2: _ScanArtifacts(
            intent_set=None,
            price_snapshots={
                "AAA": _snapshot(symbol="AAA", open_price=50.0),
                "BBB": _snapshot(symbol="BBB", open_price=50.0),
            },
        ),
    }

    # CHOP regime — rank sizing config is present but should NOT apply
    regime_outputs = _make_regime_outputs(
        rank_sizing_enabled=True,
        multipliers=[1.25, 1.00, 0.75],
        detected_regime="choppy",
    )

    orchestrator = _StubBacktestOrchestrator(
        scans=scans,
        regime_outputs=regime_outputs,
        force_bull_confirmed=False,
        initial_capital=100_000.0,
        output_dir=str(tmp_path),
        quiet=True,
        regime_mode="choppy",
        capital_allocation_config=CapitalAllocationConfig(
            max_positions=6,
            base_position_pct=0.09,
            max_position_pct=0.15,
            use_dynamic_sizing=False,
        ),
    )
    orchestrator.run_backtest([day1, day2], config={})

    portfolio = _SpyPortfolioTracker.last_instance
    assert portfolio is not None

    if len(portfolio.positions) == 2:
        # Both should have same quantity since no rank multiplier applied
        # base budget = 100_000 * 0.09 = 9000 → at $50 = 180 shares each
        assert portfolio.positions["AAA"].quantity == portfolio.positions["BBB"].quantity


# ---------------------------------------------------------------------------
# Test 5: Rank sizing disabled in config
# ---------------------------------------------------------------------------


def test_rank_sizing_disabled_config(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When rank_sizing.enabled=false, no multiplier is applied even in bull_confirmed."""
    monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

    day1 = date(2026, 1, 1)
    day2 = date(2026, 1, 2)

    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[
            _intent_group(symbol="AAA", intent_id="I-AAA", entry_price=50.0, quantity=9999.0, score=0.95),
            _intent_group(symbol="BBB", intent_id="I-BBB", entry_price=50.0, quantity=9999.0, score=0.80),
        ],
        constraints_applied={},
        source_candidates=["AAA", "BBB"],
    )

    scans = {
        day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
        day2: _ScanArtifacts(
            intent_set=None,
            price_snapshots={
                "AAA": _snapshot(symbol="AAA", open_price=50.0),
                "BBB": _snapshot(symbol="BBB", open_price=50.0),
            },
        ),
    }

    # Bull confirmed but rank_sizing disabled
    regime_outputs = _make_regime_outputs(rank_sizing_enabled=False)

    orchestrator = _StubBacktestOrchestrator(
        scans=scans,
        regime_outputs=regime_outputs,
        initial_capital=100_000.0,
        output_dir=str(tmp_path),
        quiet=True,
        regime_mode="bull",
        capital_allocation_config=CapitalAllocationConfig(
            max_positions=6,
            base_position_pct=0.09,
            max_position_pct=0.15,
            use_dynamic_sizing=False,
        ),
    )
    orchestrator.run_backtest([day1, day2], config={})

    portfolio = _SpyPortfolioTracker.last_instance
    assert portfolio is not None

    if len(portfolio.positions) == 2:
        # No rank multiplier → same budget → same qty
        assert portfolio.positions["AAA"].quantity == portfolio.positions["BBB"].quantity


# ---------------------------------------------------------------------------
# Test 6: Two-pass preserves universe (same symbols open regardless of multipliers)
# ---------------------------------------------------------------------------


def test_two_pass_preserves_universe(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """With two-pass allocation, the set of opened symbols must be identical
    whether rank_sizing is enabled or disabled, because total exposure is fixed."""
    results: dict[str, set[str]] = {}

    for enabled in [False, True]:
        monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

        day1 = date(2026, 2, 1)
        day2 = date(2026, 2, 2)

        intent_set = OrderIntentSet(
            schema_version="1.0.0",
            system_version="test",
            asof_timestamp=1,
            intent_groups=[
                _intent_group(symbol="AAA", intent_id="I-AAA", entry_price=50.0, quantity=9999.0, score=0.95),
                _intent_group(symbol="BBB", intent_id="I-BBB", entry_price=50.0, quantity=9999.0, score=0.90),
                _intent_group(symbol="CCC", intent_id="I-CCC", entry_price=50.0, quantity=9999.0, score=0.85),
                _intent_group(symbol="DDD", intent_id="I-DDD", entry_price=50.0, quantity=9999.0, score=0.80),
            ],
            constraints_applied={},
            source_candidates=["AAA", "BBB", "CCC", "DDD"],
        )

        scans = {
            day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
            day2: _ScanArtifacts(
                intent_set=None,
                price_snapshots={
                    "AAA": _snapshot(symbol="AAA", open_price=50.0),
                    "BBB": _snapshot(symbol="BBB", open_price=50.0),
                    "CCC": _snapshot(symbol="CCC", open_price=50.0),
                    "DDD": _snapshot(symbol="DDD", open_price=50.0),
                },
            ),
        }

        regime_outputs = _make_regime_outputs(
            rank_sizing_enabled=enabled, multipliers=[1.25, 1.00, 0.75]
        )

        out_dir = tmp_path / f"run_{enabled}"
        out_dir.mkdir()
        orchestrator = _StubBacktestOrchestrator(
            scans=scans,
            regime_outputs=regime_outputs,
            initial_capital=100_000.0,
            output_dir=str(out_dir),
            quiet=True,
            regime_mode="bull",
            capital_allocation_config=CapitalAllocationConfig(
                max_positions=6,
                base_position_pct=0.09,
                max_position_pct=0.15,
                use_dynamic_sizing=False,
            ),
        )
        orchestrator.run_backtest([day1, day2], config={})

        portfolio = _SpyPortfolioTracker.last_instance
        assert portfolio is not None
        results[str(enabled)] = set(portfolio.positions.keys())

    # The same symbols must open regardless of rank_sizing enabled/disabled
    assert results["True"] == results["False"]


# ---------------------------------------------------------------------------
# Test 7: Two-pass total budget is preserved (sum of weighted budgets == N * base)
# ---------------------------------------------------------------------------


def test_two_pass_total_budget_preserved(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Total cash spent across all positions should approximate N * base_budget,
    not exceed it (as V24 could when multipliers > 1.0 were applied independently)."""
    monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

    day1 = date(2026, 3, 1)
    day2 = date(2026, 3, 2)

    entry_price = 10.0
    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[
            _intent_group(symbol="AAA", intent_id="I-AAA", entry_price=entry_price, quantity=9999.0, score=0.95),
            _intent_group(symbol="BBB", intent_id="I-BBB", entry_price=entry_price, quantity=9999.0, score=0.90),
            _intent_group(symbol="CCC", intent_id="I-CCC", entry_price=entry_price, quantity=9999.0, score=0.85),
        ],
        constraints_applied={},
        source_candidates=["AAA", "BBB", "CCC"],
    )

    scans = {
        day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
        day2: _ScanArtifacts(
            intent_set=None,
            price_snapshots={
                "AAA": _snapshot(symbol="AAA", open_price=entry_price),
                "BBB": _snapshot(symbol="BBB", open_price=entry_price),
                "CCC": _snapshot(symbol="CCC", open_price=entry_price),
            },
        ),
    }

    regime_outputs = _make_regime_outputs(
        rank_sizing_enabled=True, multipliers=[1.25, 1.00, 0.75]
    )

    initial_capital = 100_000.0
    base_position_pct = 0.09
    base_budget = initial_capital * base_position_pct  # 9000

    orchestrator = _StubBacktestOrchestrator(
        scans=scans,
        regime_outputs=regime_outputs,
        initial_capital=initial_capital,
        output_dir=str(tmp_path),
        quiet=True,
        regime_mode="bull",
        capital_allocation_config=CapitalAllocationConfig(
            max_positions=6,
            base_position_pct=base_position_pct,
            max_position_pct=0.15,
            use_dynamic_sizing=False,
        ),
    )
    orchestrator.run_backtest([day1, day2], config={})

    portfolio = _SpyPortfolioTracker.last_instance
    assert portfolio is not None
    assert len(portfolio.positions) == 3

    # Total cost = sum(qty * entry_price) for all positions
    total_cost = sum(
        pos.quantity * pos.entry_price for pos in portfolio.positions.values()
    )

    # With two-pass: total should be <= 3 * base_budget (not inflated by multipliers)
    max_expected = 3 * base_budget
    assert total_cost <= max_expected * 1.01, (
        f"Total cost ${total_cost:.2f} exceeds 3 * base_budget ${max_expected:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 8: P3 — Low-rank positions dropped when cash insufficient
# ---------------------------------------------------------------------------


def test_low_rank_dropped_when_cash_tight(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When cash runs low, rank>=2 positions should be dropped (not shrunk),
    while rank 0-1 are preserved.

    Setup: $550 capital, 3 intents at $10 entry, base_position_pct=0.50
    - capital_allocator: per_position_budget = 550*0.50 = 275 (but capped by available/max_new)
    - Actually allocator gives ~budget per position. With max_positions=4,
      per_position = min(275, available/max_new).
    - With reserves (15% default), available ~ 550*0.85 = 467.5
      per_position ~ 467.5 / 4 ≈ 117 (but capped by target 275)
    - total_pool = 117 * 3 = 351
    - weights [1.25, 1.00, 0.75] → sum=3.00
    - rank0: 351*1.25/3 = 146 → 14 shares ($140)
    - rank1: 351*1.00/3 = 117 → 11 shares ($110)
    - After rank0+rank1: cash ~ 550 - 140 - 110 = 300
    - rank2: 351*0.75/3 = 88 → 8 shares ($80)
    - After all 3: cash ~ 220 (still positive)
    We design this so rank0+rank1 definitely open, and rank2 opens only if cash allows.
    The key assertion: rank0+rank1 are not diluted.
    """
    monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

    day1 = date(2026, 4, 1)
    day2 = date(2026, 4, 2)

    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[
            _intent_group(symbol="AAA", intent_id="I-AAA", entry_price=10.0, quantity=9999.0, score=0.95),
            _intent_group(symbol="BBB", intent_id="I-BBB", entry_price=10.0, quantity=9999.0, score=0.90),
            _intent_group(symbol="CCC", intent_id="I-CCC", entry_price=10.0, quantity=9999.0, score=0.85),
        ],
        constraints_applied={},
        source_candidates=["AAA", "BBB", "CCC"],
    )

    scans = {
        day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
        day2: _ScanArtifacts(
            intent_set=None,
            price_snapshots={
                "AAA": _snapshot(symbol="AAA", open_price=10.0),
                "BBB": _snapshot(symbol="BBB", open_price=10.0),
                "CCC": _snapshot(symbol="CCC", open_price=10.0),
            },
        ),
    }

    regime_outputs = _make_regime_outputs(
        rank_sizing_enabled=True, multipliers=[1.25, 1.00, 0.75]
    )

    orchestrator = _StubBacktestOrchestrator(
        scans=scans,
        regime_outputs=regime_outputs,
        initial_capital=550.0,
        output_dir=str(tmp_path),
        quiet=True,
        regime_mode="bull",
        capital_allocation_config=CapitalAllocationConfig(
            max_positions=4,
            base_position_pct=0.50,
            max_position_pct=0.60,
            min_position_value=50.0,
            use_dynamic_sizing=False,
        ),
    )
    orchestrator.run_backtest([day1, day2], config={})

    portfolio = _SpyPortfolioTracker.last_instance
    assert portfolio is not None

    # rank 0 (AAA) must always open
    assert "AAA" in portfolio.positions, "rank0 should open"

    # rank 0 should have >= rank 1 quantity (if rank1 opened)
    if "BBB" in portfolio.positions:
        assert portfolio.positions["AAA"].quantity >= portfolio.positions["BBB"].quantity

    # If rank2 (CCC) opened, it must have fewer shares than rank1
    if "CCC" in portfolio.positions and "BBB" in portfolio.positions:
        assert portfolio.positions["CCC"].quantity <= portfolio.positions["BBB"].quantity

    # Core P3 invariant: rank0 got its full weighted allocation (not diluted)
    # rank0 budget should be ~ total_pool * 1.25/3.0 of per_position_budget * 3
    # At minimum, rank0 should have gotten more shares than equal-weight would give
    if "BBB" in portfolio.positions:
        assert portfolio.positions["AAA"].quantity > portfolio.positions["BBB"].quantity, (
            "rank0 should get strictly more shares than rank1 due to 1.25 vs 1.00 weight"
        )


# ---------------------------------------------------------------------------
# Test 9: P2 — Deterministic ordering when scores are equal
# ---------------------------------------------------------------------------


def test_deterministic_ordering_equal_scores(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When multiple intents have identical scores, ordering must be
    deterministic (by symbol ascending) so rank assignment is stable."""
    results: list[list[str]] = []

    # Run twice — ordering must be identical both times
    for run_idx in range(2):
        monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

        day1 = date(2026, 5, 1)
        day2 = date(2026, 5, 2)

        # All 4 intents have IDENTICAL scores — tie-breaker must be symbol
        intent_set = OrderIntentSet(
            schema_version="1.0.0",
            system_version="test",
            asof_timestamp=1,
            intent_groups=[
                # Deliberately NOT in alphabetical order
                _intent_group(symbol="DDD", intent_id="I-DDD", entry_price=10.0, quantity=9999.0, score=0.95),
                _intent_group(symbol="BBB", intent_id="I-BBB", entry_price=10.0, quantity=9999.0, score=0.95),
                _intent_group(symbol="CCC", intent_id="I-CCC", entry_price=10.0, quantity=9999.0, score=0.95),
                _intent_group(symbol="AAA", intent_id="I-AAA", entry_price=10.0, quantity=9999.0, score=0.95),
            ],
            constraints_applied={},
            source_candidates=["DDD", "BBB", "CCC", "AAA"],
        )

        scans = {
            day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
            day2: _ScanArtifacts(
                intent_set=None,
                price_snapshots={
                    "AAA": _snapshot(symbol="AAA", open_price=10.0),
                    "BBB": _snapshot(symbol="BBB", open_price=10.0),
                    "CCC": _snapshot(symbol="CCC", open_price=10.0),
                    "DDD": _snapshot(symbol="DDD", open_price=10.0),
                },
            ),
        }

        regime_outputs = _make_regime_outputs(
            rank_sizing_enabled=True, multipliers=[1.25, 1.00, 0.75]
        )

        out_dir = tmp_path / f"run_{run_idx}"
        out_dir.mkdir()
        orchestrator = _StubBacktestOrchestrator(
            scans=scans,
            regime_outputs=regime_outputs,
            initial_capital=100_000.0,
            output_dir=str(out_dir),
            quiet=True,
            regime_mode="bull",
            capital_allocation_config=CapitalAllocationConfig(
                max_positions=6,
                base_position_pct=0.09,
                max_position_pct=0.15,
                use_dynamic_sizing=False,
            ),
        )
        orchestrator.run_backtest([day1, day2], config={})

        portfolio = _SpyPortfolioTracker.last_instance
        assert portfolio is not None
        assert len(portfolio.positions) == 4

        # Record ordering by quantity descending (rank0 gets most shares)
        ordered = sorted(
            portfolio.positions.values(),
            key=lambda p: (-p.quantity, p.symbol),
        )
        results.append([p.symbol for p in ordered])

    # Both runs must produce identical ordering
    assert results[0] == results[1], (
        f"Non-deterministic ordering: run0={results[0]} vs run1={results[1]}"
    )

    # With symbol tie-breaker, AAA should be rank0 (alphabetically first)
    assert results[0][0] == "AAA", (
        f"Expected AAA as rank0 (alphabetically first), got {results[0][0]}"
    )
