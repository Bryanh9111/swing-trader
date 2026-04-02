from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from common.interface import Result, ResultStatus
from order_state_machine.id_generator import IDGenerator
from order_state_machine.interface import IntentOrderMapping, OrderState
from order_state_machine.persistence import JSONLPersistence
from order_state_machine.reconciliation import Reconciler
from order_state_machine.state_machine import StateMachine
from strategy.interface import IntentType, TradeIntent


def _dt_to_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


def _make_intent(*, symbol: str, created_at_ns: int, intent_type: IntentType = IntentType.OPEN_LONG, quantity: float = 10.0) -> TradeIntent:
    return TradeIntent(
        intent_id="ignored",
        symbol=symbol,
        intent_type=intent_type,
        quantity=quantity,
        created_at_ns=created_at_ns,
        entry_price=100.0,
    )


def _make_mapping(*, intent_id: str, order_id: str, intent: TradeIntent, state: OrderState, ts: int) -> IntentOrderMapping:
    return IntentOrderMapping(
        intent_id=intent_id,
        client_order_id=order_id,
        broker_order_id=None,
        state=state,
        created_at_ns=ts,
        updated_at_ns=ts,
        intent_snapshot=intent,
        metadata={},
    )


def test_end_to_end_workflow_filled_and_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JSONLPersistence(storage_dir=tmp_path / "store")
    sm = StateMachine()
    gen = IDGenerator()

    fixed_now = _dt_to_ns(datetime(2025, 1, 2, tzinfo=UTC))
    monkeypatch.setattr("order_state_machine.id_generator.time.time_ns", lambda: fixed_now)

    intent1 = _make_intent(symbol="AAPL", created_at_ns=fixed_now)
    intent_id1 = gen.generate_intent_id(intent1).data
    order_id1 = gen.generate_order_id(intent_id1 or "I-?", "2025-01-02-run-abcdef").data
    assert intent_id1 and order_id1

    m = _make_mapping(intent_id=intent_id1, order_id=order_id1, intent=intent1, state=OrderState.PENDING, ts=1)
    assert store.save(m).status is ResultStatus.SUCCESS

    for state in (OrderState.SUBMITTED, OrderState.ACCEPTED, OrderState.FILLED):
        out = sm.transition(m, state, reason=f"to-{state.value}", timestamp_ns=2)
        assert out.status is ResultStatus.SUCCESS and out.data is not None
        m = out.data
        assert store.save(m).status is ResultStatus.SUCCESS

    assert store.load(intent_id1).data == m
    assert m.state is OrderState.FILLED

    intent2 = _make_intent(symbol="MSFT", created_at_ns=fixed_now + 1, quantity=5.0)
    intent_id2 = gen.generate_intent_id(intent2).data
    order_id2 = gen.generate_order_id(intent_id2 or "I-?", "2025-01-02-run-abcdef").data
    assert intent_id2 and order_id2

    m2 = _make_mapping(intent_id=intent_id2, order_id=order_id2, intent=intent2, state=OrderState.PENDING, ts=1)
    m2 = sm.transition(m2, OrderState.SUBMITTED, reason="submit", timestamp_ns=2).data or m2
    m2 = sm.transition(m2, OrderState.REJECTED, reason="reject", timestamp_ns=3).data or m2
    assert m2.state is OrderState.REJECTED


def test_integration_startup_and_periodic_reconciliation_and_restart_safe_seeding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JSONLPersistence(storage_dir=tmp_path / "store")
    sm = StateMachine()
    gen = IDGenerator()

    fixed_now = _dt_to_ns(datetime(2025, 1, 2, tzinfo=UTC))
    monkeypatch.setattr("order_state_machine.id_generator.time.time_ns", lambda: fixed_now)

    intent = _make_intent(symbol="AAPL", created_at_ns=fixed_now)
    intent_id = gen.generate_intent_id(intent).data or "I-missing"
    order_id = gen.generate_order_id(intent_id, "run-1").data or "O-missing"

    local = _make_mapping(intent_id=intent_id, order_id=order_id, intent=intent, state=OrderState.ACCEPTED, ts=10)
    assert store.save(local).status is ResultStatus.SUCCESS

    startup = Reconciler(sm).reconcile_startup(
        local_mappings=[local],
        broker_orders=[{"id": "B-1", "clientOrderId": order_id, "status": "FILLED", "fills": [{"id": "T-1"}]}],
        timestamp_ns=20,
    )
    assert startup.status is ResultStatus.SUCCESS
    updated = {m.intent_id: m for m in (startup.data.updated_mappings if startup.data else [])}
    assert updated[intent_id].state is OrderState.FILLED
    assert updated[intent_id].broker_order_id == "B-1"

    assert store.save(updated[intent_id]).status is ResultStatus.SUCCESS
    reloaded = store.load_all()
    assert reloaded.status is ResultStatus.SUCCESS
    assert len(reloaded.data or []) >= 1

    calls: list[str] = []

    # Periodic reconciliation: timed out SUBMITTED => query broker => ACCEPTED.
    def fetch_order(m: IntentOrderMapping) -> Result[dict[str, object] | None]:
        calls.append(m.intent_id)
        return Result.success({"id": "B-2", "clientOrderId": m.client_order_id, "status": "ACCEPTED"})

    periodic = Reconciler(sm, fetch_broker_order=fetch_order).reconcile_periodic(
        mappings_in_flight=[
            _make_mapping(intent_id="I-2", order_id="O-2", intent=intent, state=OrderState.SUBMITTED, ts=0),
        ],
        timeout_threshold_ns=1,
        timestamp_ns=100,
    )
    assert periodic.status is ResultStatus.SUCCESS
    assert calls == ["I-2"]

    # Duplicate prevention: deterministic intent id + persistence.exists.
    assert gen.generate_intent_id(intent).data == gen.generate_intent_id(intent).data
    assert store.exists(intent_id) is True

    # Restart-safe: seed counter from persisted mapping count.
    count = len(reloaded.data or [])
    gen2 = IDGenerator(initial_counter=count)
    new_order = gen2.generate_order_id("I-new", "run-1").data
    assert new_order and new_order.endswith(f"-{count + 1}")
