from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from common.interface import Result
from order_state_machine.id_generator import IDGenerator
from order_state_machine.interface import IntentOrderMapping, OrderState
from order_state_machine.persistence import JSONLPersistence
from order_state_machine.reconciliation import Reconciler
from order_state_machine.state_machine import StateMachine
from strategy.interface import IntentType, TradeIntent


@dataclass(frozen=True, slots=True)
class BrokerFetchSpy:
    fetch: Callable[[IntentOrderMapping], Result[dict[str, Any] | None]]
    calls: list[str]
    by_intent_id: dict[str, dict[str, Any] | None]


def _make_intent(*, intent_id: str = "intent-1", created_at_ns: int = 1_700_000_000_000_000_000) -> TradeIntent:
    return TradeIntent(
        intent_id=intent_id,
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=created_at_ns,
        entry_price=101.25,
        stop_loss_price=None,
        take_profit_price=None,
        parent_intent_id=None,
        linked_intent_ids=[],
        reduce_only=False,
        contingency_type=None,
        ladder_level=None,
        reason_codes=["TEST"],
        metadata={"source": "pytest"},
    )


def _make_mapping(
    *,
    intent: TradeIntent,
    intent_id: str = "I-20250101-deadbeefdead",
    client_order_id: str = "O-20250101-run-1",
    broker_order_id: str | None = None,
    state: OrderState = OrderState.PENDING,
    created_at_ns: int = 1_700_000_000_000_000_000,
    updated_at_ns: int = 1_700_000_000_000_000_000,
    metadata: dict[str, Any] | None = None,
) -> IntentOrderMapping:
    return IntentOrderMapping(
        intent_id=intent_id,
        client_order_id=client_order_id,
        broker_order_id=broker_order_id,
        state=state,
        created_at_ns=created_at_ns,
        updated_at_ns=updated_at_ns,
        intent_snapshot=intent,
        metadata=dict(metadata or {}),
    )


@pytest.fixture()
def mock_trade_intent() -> TradeIntent:
    return _make_intent()


@pytest.fixture()
def id_generator_fixture() -> IDGenerator:
    return IDGenerator(initial_counter=0)


@pytest.fixture()
def state_machine_fixture() -> StateMachine:
    return StateMachine()


@pytest.fixture()
def temp_persistence_fixture(tmp_path: Path) -> JSONLPersistence:
    return JSONLPersistence(storage_dir=tmp_path)


@pytest.fixture()
def broker_fetch_spy() -> BrokerFetchSpy:
    calls: list[str] = []
    by_intent_id: dict[str, dict[str, Any] | None] = {}

    def fetch(mapping: IntentOrderMapping) -> Result[dict[str, Any] | None]:
        calls.append(mapping.intent_id)
        return Result.success(by_intent_id.get(mapping.intent_id))

    return BrokerFetchSpy(fetch=fetch, calls=calls, by_intent_id=by_intent_id)


@pytest.fixture()
def reconciler_fixture(state_machine_fixture: StateMachine, broker_fetch_spy: BrokerFetchSpy) -> Reconciler:
    return Reconciler(state_machine_fixture, fetch_broker_order=broker_fetch_spy.fetch)


@pytest.fixture()
def mock_broker_orders() -> list[dict[str, Any]]:
    return [
        {"id": "B-1", "clientOrderId": "O-1", "status": "ACCEPTED"},
        {"id": "B-2", "clientOrderId": "O-2", "status": "FILLED", "fills": [{"id": "T-1"}, {"id": "T-2"}]},
        {"id": "B-ORPHAN", "clientOrderId": "O-ORPHAN", "status": "OPEN"},
    ]


@pytest.fixture()
def sample_mappings(mock_trade_intent: TradeIntent) -> list[IntentOrderMapping]:
    return [
        _make_mapping(
            intent=mock_trade_intent,
            intent_id="I-1",
            client_order_id="O-1",
            broker_order_id="B-1",
            state=OrderState.SUBMITTED,
            updated_at_ns=100,
        ),
        _make_mapping(
            intent=mock_trade_intent,
            intent_id="I-2",
            client_order_id="O-2",
            broker_order_id=None,
            state=OrderState.ACCEPTED,
            updated_at_ns=200,
            metadata={"processed_trade_ids": ["T-1"]},
        ),
        _make_mapping(
            intent=mock_trade_intent,
            intent_id="I-3",
            client_order_id="O-3",
            broker_order_id="B-3",
            state=OrderState.PENDING,
            updated_at_ns=300,
        ),
    ]

