from __future__ import annotations

from typing import Any

import pytest

from common.exceptions import PartialDataError
from common.interface import Result, ResultStatus
from order_state_machine.interface import IntentOrderMapping, OrderState
from order_state_machine.reconciliation import Reconciler
from order_state_machine.state_machine import StateMachine
from strategy.interface import TradeIntent


def _make_mapping(
    *,
    intent: TradeIntent,
    intent_id: str,
    client_order_id: str,
    state: OrderState,
    broker_order_id: str | None = None,
    updated_at_ns: int = 0,
    metadata: dict[str, Any] | None = None,
) -> IntentOrderMapping:
    return IntentOrderMapping(
        intent_id=intent_id,
        client_order_id=client_order_id,
        broker_order_id=broker_order_id,
        state=state,
        created_at_ns=0,
        updated_at_ns=updated_at_ns,
        intent_snapshot=intent,
        metadata=dict(metadata or {}),
    )


def test_match_broker_order_prefers_broker_id_then_client_id(mock_trade_intent: TradeIntent) -> None:
    sm = StateMachine()
    r = Reconciler(sm)
    mapping_broker = _make_mapping(
        intent=mock_trade_intent,
        intent_id="I-1",
        client_order_id="O-1",
        broker_order_id="B-1",
        state=OrderState.SUBMITTED,
    )
    mapping_client = _make_mapping(
        intent=mock_trade_intent,
        intent_id="I-2",
        client_order_id="O-2",
        broker_order_id=None,
        state=OrderState.SUBMITTED,
    )

    broker_order = {"id": "B-1", "clientOrderId": "O-2", "status": "OPEN"}
    match = r._match_broker_order(
        broker_order,
        local_by_broker_id={"B-1": mapping_broker},
        local_by_client_id={"O-2": mapping_client},
    )
    assert match is not None
    mapping, info = match
    assert mapping.intent_id == "I-1"
    assert info["match"] == "broker_order_id"


def test_normalize_broker_state_synonyms_and_unknown() -> None:
    r = Reconciler(StateMachine())
    assert r._normalize_broker_state({"status": "new"}) is OrderState.SUBMITTED
    assert r._normalize_broker_state({"status": "CANCELED"}) is OrderState.CANCELLED
    assert r._normalize_broker_state({"status": "weird"}) is None


def test_find_transition_path_bfs_and_none() -> None:
    r = Reconciler(StateMachine())
    # BFS finds shortest path; SUBMITTED → FILLED is now a direct transition.
    assert r._find_transition_path(OrderState.PENDING, OrderState.FILLED) == [
        OrderState.SUBMITTED,
        OrderState.FILLED,
    ]
    assert r._find_transition_path(OrderState.FILLED, OrderState.ACCEPTED) is None


def test_reconcile_startup_matching_orphan_missing_and_corrections(
    reconciler_fixture: Reconciler,
    mock_broker_orders: list[dict[str, Any]],
    sample_mappings: list[IntentOrderMapping],
    mock_trade_intent: TradeIntent,
) -> None:
    local = [
        *sample_mappings,
        _make_mapping(
            intent=mock_trade_intent,
            intent_id="I-4",
            client_order_id="O-4",
            broker_order_id=None,
            state=OrderState.ACCEPTED,
            updated_at_ns=10,
        ),
    ]

    result = reconciler_fixture.reconcile_startup(local, mock_broker_orders, timestamp_ns=1_000)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None

    data = result.data
    assert any(o.get("id") == "B-ORPHAN" for o in data.orphaned_broker_orders)
    assert "I-4" in data.missing_broker_orders
    updated_by_id = {m.intent_id: m for m in data.updated_mappings}

    # I-1: SUBMITTED -> ACCEPTED (broker says ACCEPTED)
    assert updated_by_id["I-1"].state is OrderState.ACCEPTED

    # I-2: broker_id backfill + fills + ACCEPTED -> FILLED
    assert updated_by_id["I-2"].broker_order_id == "B-2"
    assert updated_by_id["I-2"].state is OrderState.FILLED
    processed = updated_by_id["I-2"].metadata.get("processed_trade_ids")
    assert processed == ["T-1", "T-2"]

    # I-4: missing from broker => EXPIRED
    assert updated_by_id["I-4"].state is OrderState.EXPIRED


def test_reconcile_startup_idempotent_no_duplicate_fills(
    reconciler_fixture: Reconciler,
    mock_broker_orders: list[dict[str, Any]],
    sample_mappings: list[IntentOrderMapping],
) -> None:
    first = reconciler_fixture.reconcile_startup(sample_mappings, mock_broker_orders, timestamp_ns=1_000)
    assert first.status is ResultStatus.SUCCESS
    assert first.data is not None

    updated = {m.intent_id: m for m in (first.data.updated_mappings or [])}
    local2 = [updated.get(m.intent_id, m) for m in sample_mappings]

    second = reconciler_fixture.reconcile_startup(local2, mock_broker_orders, timestamp_ns=2_000)
    assert second.status is ResultStatus.SUCCESS
    assert second.data is not None
    updated2 = {m.intent_id: m for m in (second.data.updated_mappings or [])}

    if "I-2" in updated2:
        assert updated2["I-2"].metadata.get("processed_trade_ids") == ["T-1", "T-2"]


def test_reconcile_fills_trade_id_dedup(mock_trade_intent: TradeIntent) -> None:
    r = Reconciler(StateMachine())
    mapping = _make_mapping(
        intent=mock_trade_intent,
        intent_id="I-1",
        client_order_id="O-1",
        state=OrderState.ACCEPTED,
        metadata={"processed_trade_ids": ["T-1"]},
    )
    broker_order = {"status": "FILLED", "fills": [{"id": "T-1"}, {"id": "T-2"}, {"id": "T-2"}]}
    out = r._reconcile_fills(mapping, broker_order, now_ns=123, reason="test")
    assert out.changed is True
    assert out.mapping.metadata["processed_trade_ids"] == ["T-1", "T-2"]

    out2 = r._reconcile_fills(out.mapping, broker_order, now_ns=456, reason="test")
    assert out2.changed is False


def test_transition_to_target_state_multi_step_adds_event_history(mock_trade_intent: TradeIntent) -> None:
    r = Reconciler(StateMachine())
    mapping = _make_mapping(intent=mock_trade_intent, intent_id="I-1", client_order_id="O-1", state=OrderState.PENDING)
    out = r._transition_to_target_state(mapping, OrderState.FILLED, now_ns=999, reason="startup")
    assert out.changed is True
    assert out.mapping.state is OrderState.FILLED
    # SUBMITTED → FILLED is now direct, so PENDING → SUBMITTED → FILLED = 2 steps.
    assert len(out.transitions) == 2
    history = out.mapping.metadata.get("event_history")
    assert isinstance(history, list)
    assert len(history) == 2


def test_reconcile_periodic_timeout_and_missing_from_broker(
    broker_fetch_spy: Any,
    mock_trade_intent: TradeIntent,
) -> None:
    spy = broker_fetch_spy
    sm = StateMachine()
    r = Reconciler(sm, fetch_broker_order=spy.fetch)

    in_flight = _make_mapping(
        intent=mock_trade_intent,
        intent_id="I-1",
        client_order_id="O-1",
        state=OrderState.SUBMITTED,
        updated_at_ns=0,
    )
    spy.by_intent_id["I-1"] = {"id": "B-1", "clientOrderId": "O-1", "status": "ACCEPTED"}
    out = r.reconcile_periodic([in_flight], timeout_threshold_ns=10, timestamp_ns=100)
    assert out.status is ResultStatus.SUCCESS
    assert spy.calls == ["I-1"]
    assert out.data is not None
    assert out.data.updated_mappings[0].state is OrderState.ACCEPTED

    missing = _make_mapping(
        intent=mock_trade_intent,
        intent_id="I-2",
        client_order_id="O-2",
        state=OrderState.ACCEPTED,
        updated_at_ns=0,
    )
    spy.by_intent_id["I-2"] = None
    out2 = r.reconcile_periodic([missing], timeout_threshold_ns=10, timestamp_ns=100)
    assert out2.status is ResultStatus.SUCCESS
    assert out2.data is not None
    assert out2.data.missing_broker_orders == ["I-2"]
    assert out2.data.updated_mappings[0].state is OrderState.EXPIRED


def test_reconcile_periodic_degraded_when_broker_query_not_configured(mock_trade_intent: TradeIntent) -> None:
    r = Reconciler(StateMachine(), fetch_broker_order=None)
    mapping = _make_mapping(
        intent=mock_trade_intent,
        intent_id="I-1",
        client_order_id="O-1",
        state=OrderState.SUBMITTED,
        updated_at_ns=0,
    )
    out = r.reconcile_periodic([mapping], timeout_threshold_ns=1, timestamp_ns=10)
    assert out.status is ResultStatus.DEGRADED
    assert isinstance(out.error, PartialDataError)
    assert out.reason_code == "PERIODIC_RECONCILIATION_PARTIAL"


def test_reconcile_startup_partial_errors_return_degraded(mock_trade_intent: TradeIntent) -> None:
    r = Reconciler(StateMachine())
    mapping = _make_mapping(
        intent=mock_trade_intent,
        intent_id="I-1",
        client_order_id="O-1",
        broker_order_id="B-1",
        state=OrderState.ACCEPTED,
    )
    # Broker claims PENDING (no legal transition path from ACCEPTED -> PENDING).
    broker_orders = [{"id": "B-1", "clientOrderId": "O-1", "status": "PENDING"}]
    out = r.reconcile_startup([mapping], broker_orders, timestamp_ns=1)
    assert out.status is ResultStatus.DEGRADED
    assert isinstance(out.error, PartialDataError)
    assert out.reason_code == "STARTUP_RECONCILIATION_PARTIAL"

