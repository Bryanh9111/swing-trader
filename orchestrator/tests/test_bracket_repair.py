"""Unit tests for the shared _repair_missing_stops method on ExecutionPlugin.

Tests cover all 10 planned scenarios from the bracket health-check plan:
1. All entries have active SL → stops_verified > 0, generated_stops = []
2. 1 entry FILLED but no SL → generated_stops has 1 entry with correct prefix
3. Entry has SL but SL already CANCELLED → treated as missing → generates new SL
4. Entry PARTIALLY_FILLED with no SL → generates SL
5. Entry missing stop_loss_price → positions_at_risk with MISSING_STOP_LOSS_PRICE
6. dry_run=True → positions_at_risk with DRY_RUN_STOP_NOT_SUBMITTED
7. SL submit failure → failures + positions_at_risk
8. Symbol-level fallback → parent not matched but symbol matched → protected
9. Terminal entries (CANCELLED/EXPIRED) → skipped
10. Empty mappings → all zeros, no errors
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from common.interface import Result, ResultStatus
from orchestrator.plugins import ExecutionPlugin, ExecutionPluginConfig
from order_state_machine.interface import IntentOrderMapping, OrderState, ReconciliationResult
from strategy.interface import IntentType, TradeIntent


# ---------------------------------------------------------------------------
# Fake order manager
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _FakeOrderManager:
    mappings: list[IntentOrderMapping]
    cancelled: list[str]
    submitted: list[TradeIntent]
    fail_submit: bool = False
    fail_cancel: bool = False

    def reconcile_periodic(self) -> Result[ReconciliationResult]:
        return Result.success(ReconciliationResult(
            updated_mappings=[], orphaned_broker_orders=[], missing_broker_orders=[],
            state_transitions=[], reconciled_at_ns=0,
        ))

    def get_all_orders(self) -> Result[list[IntentOrderMapping]]:
        return Result.success(list(self.mappings))

    def cancel_order(self, client_order_id: str) -> Result[None]:
        self.cancelled.append(client_order_id)
        if self.fail_cancel:
            return Result.failed(RuntimeError("cancel rejected"), "CANCEL_FAILED")
        return Result.success(data=None)

    def submit_order(self, intent: TradeIntent) -> Result[IntentOrderMapping]:
        self.submitted.append(intent)
        if self.fail_submit:
            return Result.failed(RuntimeError("broker error"), "STOP_SUBMIT_FAILED")
        mapping = IntentOrderMapping(
            intent_id=intent.intent_id,
            client_order_id=f"COID-{intent.intent_id}",
            broker_order_id="B1",
            state=OrderState.ACCEPTED,
            created_at_ns=0, updated_at_ns=0,
            intent_snapshot=intent, metadata={},
        )
        self.mappings.append(mapping)
        return Result.success(mapping)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mapping(*, intent: TradeIntent, coid: str, state: OrderState) -> IntentOrderMapping:
    return IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id=coid,
        broker_order_id="B1",
        state=state,
        created_at_ns=0, updated_at_ns=0,
        intent_snapshot=intent, metadata={},
    )


def _entry(*, iid: str, sym: str, sl_price: float | None = None) -> TradeIntent:
    return TradeIntent(
        intent_id=iid, symbol=sym, intent_type=IntentType.OPEN_LONG,
        quantity=10, created_at_ns=0, stop_loss_price=sl_price,
    )


def _stop(*, iid: str, sym: str, parent: str) -> TradeIntent:
    return TradeIntent(
        intent_id=iid, symbol=sym, intent_type=IntentType.STOP_LOSS,
        quantity=10, created_at_ns=0, stop_loss_price=95.0,
        parent_intent_id=parent, reduce_only=True,
    )


def _make_plugin(mappings: list[IntentOrderMapping], *, fail_submit: bool = False) -> tuple[ExecutionPlugin, _FakeOrderManager]:
    mgr = _FakeOrderManager(mappings=list(mappings), cancelled=[], submitted=[], fail_submit=fail_submit)
    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = mgr  # type: ignore[assignment]
    plugin._run_id = "RUN-TEST"
    return plugin, mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_entries_have_active_sl() -> None:
    """Scenario 1: Every entry has a matching active SL → nothing to generate."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    stop = _stop(iid="S1", sym="AAPL", parent="E1")
    mappings = [
        _mapping(intent=entry, coid="C1", state=OrderState.FILLED),
        _mapping(intent=stop, coid="C2", state=OrderState.ACCEPTED),
    ]
    plugin, mgr = _make_plugin(mappings)

    result = plugin._repair_missing_stops(mappings, intent_prefix="T-", generated_by="TEST", dry_run=False)

    assert result["stops_verified"] == 1
    assert result["generated_stops"] == []
    assert result["positions_at_risk"] == []
    assert result["failures"] == []
    assert mgr.submitted == []


def test_filled_entry_no_sl_generates_stop() -> None:
    """Scenario 2: One FILLED entry with no SL → generates 1 stop with correct prefix."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    mappings = [_mapping(intent=entry, coid="C1", state=OrderState.FILLED)]
    plugin, mgr = _make_plugin(mappings)

    result = plugin._repair_missing_stops(mappings, intent_prefix="ID-STOP-", generated_by="INTRADAY_CHECK", dry_run=False)

    assert len(result["generated_stops"]) == 1
    assert result["positions_at_risk"] == []
    assert len(mgr.submitted) == 1
    assert mgr.submitted[0].intent_id == "ID-STOP-E1"
    assert mgr.submitted[0].metadata["generated_by"] == "INTRADAY_CHECK"


def test_cancelled_sl_treated_as_missing() -> None:
    """Scenario 3: Entry has SL but SL is CANCELLED (terminal) → treated as missing."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    stop = _stop(iid="S1", sym="AAPL", parent="E1")
    mappings = [
        _mapping(intent=entry, coid="C1", state=OrderState.FILLED),
        _mapping(intent=stop, coid="C2", state=OrderState.CANCELLED),
    ]
    plugin, mgr = _make_plugin(mappings)

    result = plugin._repair_missing_stops(mappings, intent_prefix="PC-STOP-", generated_by="AFTER_MARKET_RECON", dry_run=False)

    assert len(result["generated_stops"]) == 1
    assert mgr.submitted[0].intent_id == "PC-STOP-E1"


def test_partially_filled_no_sl_generates_stop() -> None:
    """Scenario 4: PARTIALLY_FILLED entry without SL → generates stop."""
    entry = _entry(iid="E1", sym="TSLA", sl_price=190.0)
    mappings = [_mapping(intent=entry, coid="C1", state=OrderState.PARTIALLY_FILLED)]
    plugin, mgr = _make_plugin(mappings)

    result = plugin._repair_missing_stops(mappings, intent_prefix="PM-STOP-", generated_by="PRE_MARKET_CHECK", dry_run=False)

    assert len(result["generated_stops"]) == 1
    assert mgr.submitted[0].intent_id == "PM-STOP-E1"


def test_missing_stop_loss_price_at_risk() -> None:
    """Scenario 5: Entry has no stop_loss_price → positions_at_risk."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=None)
    mappings = [_mapping(intent=entry, coid="C1", state=OrderState.FILLED)]
    plugin, _ = _make_plugin(mappings)

    result = plugin._repair_missing_stops(mappings, intent_prefix="T-", generated_by="TEST", dry_run=False)

    assert result["generated_stops"] == []
    assert len(result["positions_at_risk"]) == 1
    assert result["positions_at_risk"][0]["reason"] == "MISSING_STOP_LOSS_PRICE"


def test_dry_run_does_not_submit() -> None:
    """Scenario 6: dry_run=True → positions_at_risk with DRY_RUN reason."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    mappings = [_mapping(intent=entry, coid="C1", state=OrderState.FILLED)]
    plugin, mgr = _make_plugin(mappings)

    result = plugin._repair_missing_stops(mappings, intent_prefix="T-", generated_by="TEST", dry_run=True)

    assert result["generated_stops"] == []
    assert len(result["positions_at_risk"]) == 1
    assert result["positions_at_risk"][0]["reason"] == "DRY_RUN_STOP_NOT_SUBMITTED"
    assert result["positions_at_risk"][0]["stop_loss_price"] == 95.0
    assert mgr.submitted == []


def test_submit_failure_records_failure_and_at_risk() -> None:
    """Scenario 7: SL submit fails → failures + positions_at_risk."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    mappings = [_mapping(intent=entry, coid="C1", state=OrderState.FILLED)]
    plugin, mgr = _make_plugin(mappings, fail_submit=True)

    result = plugin._repair_missing_stops(mappings, intent_prefix="T-", generated_by="TEST", dry_run=False)

    assert result["generated_stops"] == []
    assert len(result["failures"]) == 1
    assert result["failures"][0]["reason_code"] == "STOP_SUBMIT_FAILED"
    assert len(result["positions_at_risk"]) == 1
    assert result["positions_at_risk"][0]["reason"] == "STOP_SUBMIT_FAILED"


def test_symbol_level_fallback() -> None:
    """Scenario 8: No parent_intent_id match but symbol-level SL exists → protected."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    # SL with a different parent — but same symbol
    stop = _stop(iid="S1", sym="AAPL", parent="E-OTHER")
    mappings = [
        _mapping(intent=entry, coid="C1", state=OrderState.FILLED),
        _mapping(intent=stop, coid="C2", state=OrderState.ACCEPTED),
    ]
    plugin, mgr = _make_plugin(mappings)

    result = plugin._repair_missing_stops(mappings, intent_prefix="T-", generated_by="TEST", dry_run=False)

    assert result["stops_verified"] == 1
    assert "E1" in result["protected_intent_ids"]
    assert result["generated_stops"] == []
    assert mgr.submitted == []


def test_terminal_entries_skipped() -> None:
    """Scenario 9: CANCELLED entries are not checked for missing SL."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    mappings = [_mapping(intent=entry, coid="C1", state=OrderState.CANCELLED)]
    plugin, mgr = _make_plugin(mappings)

    result = plugin._repair_missing_stops(mappings, intent_prefix="T-", generated_by="TEST", dry_run=False)

    assert result["stops_verified"] == 0
    assert result["generated_stops"] == []
    assert result["positions_at_risk"] == []
    assert mgr.submitted == []


def test_empty_mappings() -> None:
    """Scenario 10: Empty mappings → all zeros, no errors."""
    plugin, mgr = _make_plugin([])

    result = plugin._repair_missing_stops([], intent_prefix="T-", generated_by="TEST", dry_run=False)

    assert result["stops_verified"] == 0
    assert result["generated_stops"] == []
    assert result["positions_at_risk"] == []
    assert result["failures"] == []
    assert mgr.submitted == []


# ---------------------------------------------------------------------------
# Integration: intraday_check with repair_brackets=True
# ---------------------------------------------------------------------------

def test_intraday_check_with_repair_brackets() -> None:
    """intraday_check + repair_brackets=True should call _repair_missing_stops."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    mgr = _FakeOrderManager(
        mappings=[_mapping(intent=entry, coid="C1", state=OrderState.FILLED)],
        cancelled=[], submitted=[],
    )
    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = mgr  # type: ignore[assignment]
    plugin._run_id = "RUN-ID"

    result = plugin.execute({"intraday_check": True, "reconcile_first": False, "repair_brackets": True})

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert len(result.data["generated_stops"]) == 1
    assert result.data["positions_at_risk"] == []
    assert mgr.submitted[0].intent_id == "ID-STOP-E1"


def test_intraday_check_without_repair_brackets_is_passive() -> None:
    """intraday_check without repair_brackets → no stop generation (backward compat)."""
    entry = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    mgr = _FakeOrderManager(
        mappings=[_mapping(intent=entry, coid="C1", state=OrderState.FILLED)],
        cancelled=[], submitted=[],
    )
    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = mgr  # type: ignore[assignment]
    plugin._run_id = "RUN-ID"

    result = plugin.execute({"intraday_check": True, "reconcile_first": False})

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert "generated_stops" not in result.data
    assert mgr.submitted == []


# ---------------------------------------------------------------------------
# Helpers for quantity alignment tests
# ---------------------------------------------------------------------------

def _mapping_with_ibkr(
    *, intent: TradeIntent, coid: str, state: OrderState, filled: float = 0.0,
) -> IntentOrderMapping:
    """Create a mapping with ibkr metadata (filled qty)."""
    return IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id=coid,
        broker_order_id="B1",
        state=state,
        created_at_ns=0, updated_at_ns=0,
        intent_snapshot=intent,
        metadata={"ibkr": {"filled": filled}},
    )


def _sl(*, iid: str, sym: str, parent: str, qty: float = 10.0) -> TradeIntent:
    return TradeIntent(
        intent_id=iid, symbol=sym, intent_type=IntentType.STOP_LOSS,
        quantity=qty, created_at_ns=0, stop_loss_price=95.0,
        parent_intent_id=parent, reduce_only=True,
        metadata={"run_id": "run-1", "position_side": "LONG"},
    )


def _tp(*, iid: str, sym: str, parent: str, qty: float = 10.0) -> TradeIntent:
    return TradeIntent(
        intent_id=iid, symbol=sym, intent_type=IntentType.TAKE_PROFIT,
        quantity=qty, created_at_ns=0, take_profit_price=110.0,
        parent_intent_id=parent, reduce_only=True,
        metadata={"run_id": "run-1", "position_side": "LONG"},
    )


def _make_plugin_ex(
    mappings: list[IntentOrderMapping],
    *,
    fail_submit: bool = False,
    fail_cancel: bool = False,
) -> tuple[ExecutionPlugin, _FakeOrderManager]:
    mgr = _FakeOrderManager(
        mappings=list(mappings), cancelled=[], submitted=[],
        fail_submit=fail_submit, fail_cancel=fail_cancel,
    )
    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = mgr  # type: ignore[assignment]
    plugin._run_id = "RUN-TEST"
    return plugin, mgr


# ---------------------------------------------------------------------------
# Quantity alignment tests
# ---------------------------------------------------------------------------

def test_qty_mismatch_cancel_and_resubmit() -> None:
    """Partial fill 30/51 → cancel old SL + submit SL with qty=30."""
    entry_intent = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    sl_intent = _sl(iid="SL1", sym="AAPL", parent="E1", qty=51.0)
    mappings = [
        _mapping_with_ibkr(intent=entry_intent, coid="CE1", state=OrderState.PARTIALLY_FILLED, filled=30.0),
        _mapping(intent=sl_intent, coid="CSL1", state=OrderState.SUBMITTED),
    ]
    plugin, mgr = _make_plugin_ex(mappings)

    result = plugin._repair_quantity_mismatch(mappings, dry_run=False)

    assert result["aligned"] == 0
    assert len(result["repaired"]) == 1
    assert result["repaired"][0]["old_qty"] == 51.0
    assert result["repaired"][0]["new_qty"] == 30.0
    assert "CSL1" in mgr.cancelled
    assert len(mgr.submitted) == 1
    assert mgr.submitted[0].quantity == 30.0


def test_qty_aligned_no_action() -> None:
    """SL qty matches filled qty → no action."""
    entry_intent = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    sl_intent = _sl(iid="SL1", sym="AAPL", parent="E1", qty=51.0)
    mappings = [
        _mapping_with_ibkr(intent=entry_intent, coid="CE1", state=OrderState.FILLED, filled=51.0),
        _mapping(intent=sl_intent, coid="CSL1", state=OrderState.SUBMITTED),
    ]
    plugin, mgr = _make_plugin_ex(mappings)

    result = plugin._repair_quantity_mismatch(mappings, dry_run=False)

    assert result["aligned"] == 1
    assert result["repaired"] == []
    assert mgr.cancelled == []
    assert mgr.submitted == []


def test_qty_mismatch_dry_run() -> None:
    """dry_run=True → at_risk, no cancel/submit."""
    entry_intent = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    sl_intent = _sl(iid="SL1", sym="AAPL", parent="E1", qty=51.0)
    mappings = [
        _mapping_with_ibkr(intent=entry_intent, coid="CE1", state=OrderState.PARTIALLY_FILLED, filled=30.0),
        _mapping(intent=sl_intent, coid="CSL1", state=OrderState.SUBMITTED),
    ]
    plugin, mgr = _make_plugin_ex(mappings)

    result = plugin._repair_quantity_mismatch(mappings, dry_run=True)

    assert result["repaired"] == []
    assert len(result["positions_at_risk"]) == 1
    assert result["positions_at_risk"][0]["reason"] == "DRY_RUN_QTY_MISMATCH"
    assert mgr.cancelled == []
    assert mgr.submitted == []


def test_qty_mismatch_cancel_fails_no_resubmit() -> None:
    """Guardrail C: cancel fails → no resubmit (avoid double-active)."""
    entry_intent = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    sl_intent = _sl(iid="SL1", sym="AAPL", parent="E1", qty=51.0)
    mappings = [
        _mapping_with_ibkr(intent=entry_intent, coid="CE1", state=OrderState.PARTIALLY_FILLED, filled=30.0),
        _mapping(intent=sl_intent, coid="CSL1", state=OrderState.SUBMITTED),
    ]
    plugin, mgr = _make_plugin_ex(mappings, fail_cancel=True)

    result = plugin._repair_quantity_mismatch(mappings, dry_run=False)

    assert result["repaired"] == []
    assert len(result["failures"]) == 1
    assert result["failures"][0]["category"] == "qty_align_cancel"
    assert len(result["positions_at_risk"]) == 1
    assert result["positions_at_risk"][0]["reason"] == "CANCEL_FAILED_NO_RESUBMIT"
    assert mgr.submitted == []  # No resubmit!


def test_qty_mismatch_zero_filled_skip() -> None:
    """Guardrail A: filled=0 → skip (no position, nothing to align)."""
    entry_intent = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    sl_intent = _sl(iid="SL1", sym="AAPL", parent="E1", qty=51.0)
    mappings = [
        _mapping_with_ibkr(intent=entry_intent, coid="CE1", state=OrderState.PARTIALLY_FILLED, filled=0.0),
        _mapping(intent=sl_intent, coid="CSL1", state=OrderState.SUBMITTED),
    ]
    plugin, mgr = _make_plugin_ex(mappings)

    result = plugin._repair_quantity_mismatch(mappings, dry_run=False)

    assert result["aligned"] == 0
    assert result["repaired"] == []
    assert mgr.cancelled == []
    assert mgr.submitted == []


def test_qty_mismatch_both_sl_and_tp() -> None:
    """Both SL and TP get aligned when qty mismatches."""
    entry_intent = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    sl_intent = _sl(iid="SL1", sym="AAPL", parent="E1", qty=51.0)
    tp_intent = _tp(iid="TP1", sym="AAPL", parent="E1", qty=51.0)
    mappings = [
        _mapping_with_ibkr(intent=entry_intent, coid="CE1", state=OrderState.PARTIALLY_FILLED, filled=30.0),
        _mapping(intent=sl_intent, coid="CSL1", state=OrderState.SUBMITTED),
        _mapping(intent=tp_intent, coid="CTP1", state=OrderState.SUBMITTED),
    ]
    plugin, mgr = _make_plugin_ex(mappings)

    result = plugin._repair_quantity_mismatch(mappings, dry_run=False)

    assert result["aligned"] == 0
    assert len(result["repaired"]) == 2
    assert "CSL1" in mgr.cancelled
    assert "CTP1" in mgr.cancelled
    assert len(mgr.submitted) == 2
    for submitted in mgr.submitted:
        assert submitted.quantity == 30.0


def test_oca_cancelled_sibling_cleaned_up() -> None:
    """When TP is FILLED and SL becomes CANCELLED (via OCA), SL is terminal and skipped."""
    entry_intent = _entry(iid="E1", sym="AAPL", sl_price=95.0)
    sl_intent = _sl(iid="SL1", sym="AAPL", parent="E1", qty=51.0)
    tp_intent = _tp(iid="TP1", sym="AAPL", parent="E1", qty=51.0)
    mappings = [
        _mapping_with_ibkr(intent=entry_intent, coid="CE1", state=OrderState.FILLED, filled=51.0),
        # OCA: TP filled → SL auto-cancelled
        _mapping(intent=tp_intent, coid="CTP1", state=OrderState.FILLED),
        _mapping(intent=sl_intent, coid="CSL1", state=OrderState.CANCELLED),
    ]
    plugin, mgr = _make_plugin_ex(mappings)

    result = plugin._repair_quantity_mismatch(mappings, dry_run=False)

    # Cancelled SL is filtered out (terminal), no siblings to align
    assert result["repaired"] == []
    assert result["failures"] == []
    assert mgr.cancelled == []
    assert mgr.submitted == []


# ---------------------------------------------------------------------------
# Dict-based intent_snapshot tests (deserialization from disk)
# ---------------------------------------------------------------------------

def _dict_mapping(*, itype: str, symbol: str, qty: float = 51.0,
                  state: OrderState = OrderState.SUBMITTED,
                  parent: str | None = None,
                  sl_price: float = 0.0, tp_price: float = 0.0,
                  entry_price: float = 0.0,
                  filled: float = 0.0, avg_fill: float = 0.0) -> IntentOrderMapping:
    """Create a mapping whose intent_snapshot is a plain dict (as loaded from disk)."""
    intent_dict: dict[str, Any] = {
        "intent_id": f"I-{symbol}-{itype}",
        "intent_type": itype,
        "symbol": symbol,
        "quantity": qty,
        "stop_loss_price": sl_price,
        "take_profit_price": tp_price,
        "entry_price": entry_price,
    }
    if parent:
        intent_dict["parent_intent_id"] = parent
    meta: dict[str, Any] = {}
    if filled or avg_fill:
        meta["ibkr"] = {"filled": filled, "remaining": qty - filled, "avgFillPrice": avg_fill}
    return IntentOrderMapping(
        intent_id=intent_dict["intent_id"],
        client_order_id=f"C-{symbol}-{itype}",
        state=state,
        created_at_ns=0,
        updated_at_ns=0,
        intent_snapshot=intent_dict,  # type: ignore[arg-type]
        metadata=meta,
    )


def test_derive_order_stats_dict_intent() -> None:
    """_derive_order_stats correctly reads fields from dict intent_snapshot."""
    mappings = [
        _dict_mapping(itype="OPEN_LONG", symbol="AAPL", qty=100, entry_price=150.0,
                      state=OrderState.FILLED, filled=100, avg_fill=150.50),
        _dict_mapping(itype="STOP_LOSS", symbol="AAPL", qty=100, sl_price=145.0,
                      state=OrderState.SUBMITTED, parent="I-AAPL-OPEN_LONG"),
        _dict_mapping(itype="TAKE_PROFIT", symbol="AAPL", qty=100, tp_price=160.0,
                      state=OrderState.SUBMITTED, parent="I-AAPL-OPEN_LONG"),
    ]
    stats = ExecutionPlugin._derive_order_stats(mappings)

    assert stats["intents_generated"] == 1  # 1 unique entry symbol
    assert stats["orders_submitted"] == 3
    assert stats["fills_entry_count"] == 1
    assert stats["fills_exit_count"] == 0
    assert stats["orders_cancelled"] == 0

    details = stats["order_details"]
    assert len(details) == 3
    # Entry order detail
    entry_d = [d for d in details if d["intent_type"] == "OPEN_LONG"][0]
    assert entry_d["symbol"] == "AAPL"
    assert entry_d["ordered_qty"] == 100.0
    assert entry_d["filled_qty"] == 100.0
    assert entry_d["price"] == 150.0
    assert entry_d["avg_fill_price"] == 150.50
    # SL order detail
    sl_d = [d for d in details if d["intent_type"] == "STOP_LOSS"][0]
    assert sl_d["symbol"] == "AAPL"
    assert sl_d["price"] == 145.0


def test_build_bracket_summary_dict_intent() -> None:
    """_build_bracket_summary correctly reads fields from dict intent_snapshot."""
    mappings = [
        _dict_mapping(itype="OPEN_LONG", symbol="AAPL", qty=100,
                      state=OrderState.FILLED),
        _dict_mapping(itype="STOP_LOSS", symbol="AAPL", qty=100, sl_price=145.0,
                      state=OrderState.SUBMITTED, parent="I-AAPL-OPEN_LONG"),
        _dict_mapping(itype="TAKE_PROFIT", symbol="AAPL", qty=100, tp_price=160.0,
                      state=OrderState.SUBMITTED, parent="I-AAPL-OPEN_LONG"),
    ]
    summary = ExecutionPlugin._build_bracket_summary(mappings)

    assert "AAPL" in summary
    assert summary["AAPL"]["sl"]["price"] == 145.0
    assert summary["AAPL"]["sl"]["qty"] == 100.0
    assert summary["AAPL"]["tp"]["price"] == 160.0
    assert summary["AAPL"]["tp"]["qty"] == 100.0


def test_get_helpers_dict_intent() -> None:
    """_get_intent_type, _get_symbol, _get_parent handle dict intent_snapshot."""
    m = _dict_mapping(itype="STOP_LOSS", symbol="TSLA", parent="E1")
    assert ExecutionPlugin._get_intent_type(m) == "STOP_LOSS"
    assert ExecutionPlugin._get_symbol(m) == "TSLA"
    assert ExecutionPlugin._get_parent(m) == "E1"


# ---------------------------------------------------------------------------
# Broker bracket fallback tests
# ---------------------------------------------------------------------------

def _make_plugin_with_broker_brackets(
    mappings: list[IntentOrderMapping],
    broker_brackets: dict[str, dict[str, Any]],
) -> ExecutionPlugin:
    """Build an ExecutionPlugin with a fake adapter that returns broker brackets."""

    class _FakeBrokerAdapter:
        def get_open_bracket_orders(self) -> Result[dict[str, dict[str, Any]]]:
            return Result.success(broker_brackets)

    mgr = _FakeOrderManager(
        mappings=mappings, cancelled=[], submitted=[],
    )
    plugin = ExecutionPlugin(config={"enabled": True})
    plugin._manager = mgr
    plugin._adapter = _FakeBrokerAdapter()
    return plugin


def test_bracket_fallback_uses_broker_when_mappings_empty() -> None:
    """When AST mappings have no brackets, fall back to IBKR open orders."""
    # Entry is FILLED but SL/TP are EXPIRED (not tracked in mappings anymore)
    entry = _dict_mapping(itype="OPEN_LONG", symbol="AAPL", qty=100, state=OrderState.FILLED)
    sl = _dict_mapping(itype="STOP_LOSS", symbol="AAPL", qty=100, sl_price=145.0,
                       state=OrderState.EXPIRED, parent="E1")
    mappings = [entry, sl]

    broker_brackets = {
        "AAPL": {
            "sl": {"price": 145.0, "qty": 100.0, "status": "PreSubmitted"},
            "tp": {"price": 160.0, "qty": 100.0, "status": "PreSubmitted"},
        },
    }
    plugin = _make_plugin_with_broker_brackets(mappings, broker_brackets)

    result = plugin._build_bracket_summary_with_broker_fallback(
        mappings, symbols=["AAPL"],
    )

    assert "AAPL" in result
    assert result["AAPL"]["sl"]["price"] == 145.0
    assert result["AAPL"]["sl"]["status"] == "PreSubmitted"
    assert result["AAPL"]["tp"]["price"] == 160.0


def test_bracket_fallback_prefers_mappings_over_broker() -> None:
    """AST mapping brackets take priority over broker open orders."""
    sl = _dict_mapping(itype="STOP_LOSS", symbol="AAPL", qty=100, sl_price=145.0,
                       state=OrderState.SUBMITTED, parent="E1")
    mappings = [sl]

    broker_brackets = {
        "AAPL": {
            "sl": {"price": 140.0, "qty": 100.0, "status": "PreSubmitted"},
            "tp": {"price": 160.0, "qty": 100.0, "status": "PreSubmitted"},
        },
    }
    plugin = _make_plugin_with_broker_brackets(mappings, broker_brackets)

    result = plugin._build_bracket_summary_with_broker_fallback(
        mappings, symbols=["AAPL"],
    )

    # SL from mappings (145.0), TP from broker (160.0)
    assert result["AAPL"]["sl"]["price"] == 145.0
    assert result["AAPL"]["tp"]["price"] == 160.0


def test_bracket_fallback_no_adapter() -> None:
    """Without adapter, returns mapping-only summary (no crash)."""
    mappings: list[IntentOrderMapping] = []
    plugin = ExecutionPlugin(config={"enabled": True})
    plugin._adapter = None

    result = plugin._build_bracket_summary_with_broker_fallback(
        mappings, symbols=["AAPL"],
    )
    assert result == {}


# ---------------------------------------------------------------------------
# Shared adapter injection tests
# ---------------------------------------------------------------------------


class _FakeSharedAdapter:
    """Minimal adapter stub for shared-adapter injection tests."""

    def __init__(self):
        self.disconnect_called = False

    def fetch_broker_order(self, mapping):
        return Result.success(None)

    def is_connected(self):
        return True

    def disconnect(self):
        self.disconnect_called = True


def test_shared_adapter_injection() -> None:
    """validate_config with _shared_adapter skips creating a new adapter."""

    fake_adapter = _FakeSharedAdapter()
    plugin = ExecutionPlugin()

    config = {
        "enabled": True,
        "adapter": "ibkr",
        "broker": {"host": "127.0.0.1", "port": 4002},
        "_shared_adapter": fake_adapter,
    }
    result = plugin.validate_config(config)
    assert result.status is ResultStatus.SUCCESS

    # _build_manager should have used the shared adapter.
    assert plugin._adapter is fake_adapter
    assert plugin._manager is not None


def test_shared_adapter_cleanup_skips_disconnect() -> None:
    """cleanup() does NOT disconnect a shared adapter."""

    fake_adapter = _FakeSharedAdapter()
    plugin = ExecutionPlugin()

    config = {
        "enabled": True,
        "adapter": "ibkr",
        "broker": {"host": "127.0.0.1", "port": 4002},
        "_shared_adapter": fake_adapter,
    }
    plugin.validate_config(config)

    result = plugin.cleanup()
    assert result.status in (ResultStatus.SUCCESS, ResultStatus.DEGRADED)
    assert fake_adapter.disconnect_called is False
