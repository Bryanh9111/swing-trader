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
    quantity: float,
    entry_price: float = 100.0,
    sl_pct: float = 0.10,
    tp_pct: float = 0.10,
) -> IntentGroup:
    """Create a test intent group with properly scaled bracket prices.

    Args:
        symbol: Stock symbol.
        intent_id: Unique intent identifier.
        quantity: Position size.
        entry_price: Expected entry price (used to calculate SL/TP).
        sl_pct: Stop loss percentage below entry (default 10%).
        tp_pct: Take profit percentage above entry (default 10%).
    """
    created_at = 1
    stop_loss_price = entry_price * (1.0 - sl_pct)
    take_profit_price = entry_price * (1.0 + tp_pct)

    entry = TradeIntent(
        intent_id=intent_id,
        symbol=symbol,
        intent_type=IntentType.OPEN_LONG,
        quantity=float(quantity),
        created_at_ns=created_at,
        entry_price=float(entry_price),  # Include entry_price for bracket recalculation
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


class _StubBacktestOrchestrator(BacktestOrchestrator):
    def __init__(self, *, scans: dict[date, _ScanArtifacts], **kwargs: Any) -> None:
        super().__init__(eod_orchestrator=SimpleNamespace(candidate_filter_fn=None), **kwargs)
        self._scans = scans

    def _run_daily_scan(self, *, current_date: date, config: dict[str, Any]) -> Any:  # type: ignore[override]
        del config
        artifacts = self._scans[current_date]
        return artifacts


class _SpyPortfolioTracker(PortfolioTracker):
    last_instance: "_SpyPortfolioTracker | None" = None

    def __init__(self, initial_capital: float = 2000.0) -> None:
        super().__init__(initial_capital=initial_capital)
        type(self).last_instance = self


def test_capital_allocation_limits_intents(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

    day1 = date(2026, 1, 1)
    day2 = date(2026, 1, 2)

    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[
            _intent_group(symbol="AAPL", intent_id="I-AAPL", quantity=1.0),
            _intent_group(symbol="MSFT", intent_id="I-MSFT", quantity=1.0),
        ],
        constraints_applied={},
        source_candidates=["AAPL", "MSFT"],
    )

    scans = {
        day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
        day2: _ScanArtifacts(
            intent_set=None,
            price_snapshots={
                "AAPL": _snapshot(symbol="AAPL", open_price=100.0),
                "MSFT": _snapshot(symbol="MSFT", open_price=100.0),
            },
        ),
    }

    orchestrator = _StubBacktestOrchestrator(
        scans=scans,
        initial_capital=10_000.0,
        output_dir=str(tmp_path),
        quiet=True,
        capital_allocation_config=CapitalAllocationConfig(max_positions=1, use_dynamic_sizing=False),
    )
    orchestrator.run_backtest([day1, day2], config={})

    portfolio = _SpyPortfolioTracker.last_instance
    assert portfolio is not None
    assert set(portfolio.positions) == {"AAPL"}


def test_capital_allocation_adjusts_quantity(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backtest.orchestrator.PortfolioTracker", _SpyPortfolioTracker)

    day1 = date(2026, 1, 1)
    day2 = date(2026, 1, 2)

    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[_intent_group(symbol="AAPL", intent_id="I-AAPL", quantity=20.0, entry_price=10.0)],
        constraints_applied={},
        source_candidates=["AAPL"],
    )

    scans = {
        day1: _ScanArtifacts(intent_set=intent_set, price_snapshots={}),
        day2: _ScanArtifacts(
            intent_set=None,
            price_snapshots={
                "AAPL": _snapshot(symbol="AAPL", open_price=10.0),
            },
        ),
    }

    orchestrator = _StubBacktestOrchestrator(
        scans=scans,
        initial_capital=1_000.0,
        output_dir=str(tmp_path),
        quiet=True,
        capital_allocation_config=CapitalAllocationConfig(max_positions=1, base_position_pct=0.08, max_position_pct=1.0, use_dynamic_sizing=False),
    )
    orchestrator.run_backtest([day1, day2], config={})

    portfolio = _SpyPortfolioTracker.last_instance
    assert portfolio is not None
    assert portfolio.positions["AAPL"].quantity == pytest.approx(8.0)


def test_portfolio_tracker_budget_respects_min_position_value() -> None:
    tracker = PortfolioTracker(initial_capital=100.0)
    position = Position(
        symbol="AAPL",
        entry_price=10.0,
        quantity=20.0,
        entry_date=date(2026, 1, 1),
        stop_loss_price=9.0,
        take_profit_price=11.0,
        intent_id="I-1",
    )

    with pytest.raises(ValueError, match="insufficient cash"):
        tracker.add_position(position, max_budget=40.0, min_position_value=50.0)

    assert tracker.cash == 100.0
    assert tracker.positions == {}

