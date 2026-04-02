from __future__ import annotations

from typing import Any

import msgspec
import msgspec.json
import msgspec.structs

from common.interface import Result
from journal.interface import SnapshotBase
from order_state_machine.interface import (
    IDGeneratorProtocol,
    IntentOrderMapping,
    IntentOrderMappingSet,
    OrderState,
    PersistenceProtocol,
    ReconciliationProtocol,
    ReconciliationResult,
    StateMachineProtocol,
    StateTransition,
)
from strategy.interface import TradeIntent

if not hasattr(msgspec.json, "DecodeError"):
    setattr(msgspec.json, "DecodeError", msgspec.DecodeError)
if not hasattr(msgspec.json, "EncodeError"):
    setattr(msgspec.json, "EncodeError", msgspec.EncodeError)


def test_order_state_terminal_states_and_is_terminal() -> None:
    terminal = OrderState.terminal_states()
    assert OrderState.FILLED in terminal
    assert OrderState.CANCELLED.is_terminal() is True
    assert OrderState.SUBMITTED.is_terminal() is False


def test_intent_order_mapping_schema_and_msgspec_roundtrip(mock_trade_intent: TradeIntent) -> None:
    mapping = IntentOrderMapping(
        intent_id="I-1",
        client_order_id="O-1",
        broker_order_id=None,
        state=OrderState.PENDING,
        created_at_ns=1,
        updated_at_ns=2,
        # TradeIntent is intentionally type-erased (Any) at runtime, so decode returns a dict.
        intent_snapshot=msgspec.structs.asdict(mock_trade_intent),
        metadata={"a": 1},
    )
    encoded = msgspec.json.encode(mapping)
    decoded = msgspec.json.decode(encoded, type=IntentOrderMapping)
    assert decoded == mapping


def test_intent_order_mapping_set_inherits_snapshot_base(mock_trade_intent: TradeIntent) -> None:
    snapshot = IntentOrderMappingSet(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=123,
        mappings=[
            IntentOrderMapping(
                intent_id="I-1",
                client_order_id="O-1",
                state=OrderState.PENDING,
                created_at_ns=1,
                updated_at_ns=1,
                intent_snapshot=mock_trade_intent,
            )
        ],
        run_id="run-1",
    )
    assert isinstance(snapshot, SnapshotBase)


def test_state_transition_and_reconciliation_result_all_fields() -> None:
    transition = StateTransition(
        intent_id="I-1",
        from_state=OrderState.PENDING,
        to_state=OrderState.SUBMITTED,
        timestamp_ns=123,
        reason="submit",
        metadata={"x": 1},
    )
    result = ReconciliationResult(
        updated_mappings=[],
        orphaned_broker_orders=[{"id": "B-1"}],
        missing_broker_orders=["I-2"],
        state_transitions=[transition],
        reconciled_at_ns=456,
    )
    assert result.state_transitions[0].reason == "submit"


def test_protocols_are_runtime_checkable() -> None:
    class Dummy(IDGeneratorProtocol, StateMachineProtocol, PersistenceProtocol, ReconciliationProtocol):  # type: ignore[misc]
        def generate_intent_id(self, intent: TradeIntent) -> Result[str]:  # type: ignore[override]
            return Result.success("I")

        def generate_order_id(self, intent_id: str, run_id: str) -> Result[str]:  # type: ignore[override]
            return Result.success("O")

        def transition(self, mapping: IntentOrderMapping, new_state: OrderState, *, reason: str, timestamp_ns: int | None = None, metadata: dict[str, Any] | None = None) -> Result[IntentOrderMapping]:  # type: ignore[override]
            return Result.success(mapping)

        def is_valid_transition(self, from_state: OrderState, to_state: OrderState) -> bool:  # type: ignore[override]
            return True

        def save(self, mapping: IntentOrderMapping) -> Result[None]:  # type: ignore[override]
            return Result.success(None)

        def load(self, intent_id: str) -> Result[IntentOrderMapping]:  # type: ignore[override]
            raise NotImplementedError

        def load_all(self) -> Result[list[IntentOrderMapping]]:  # type: ignore[override]
            return Result.success([])

        def exists(self, intent_id: str) -> bool:  # type: ignore[override]
            return False

        def reconcile_startup(self, local_mappings: list[IntentOrderMapping], broker_orders: list[dict[str, Any]], *, timestamp_ns: int | None = None) -> Result[ReconciliationResult]:  # type: ignore[override]
            return Result.success(ReconciliationResult(reconciled_at_ns=0))

        def reconcile_periodic(self, mappings_in_flight: list[IntentOrderMapping], timeout_threshold_ns: int, *, timestamp_ns: int | None = None) -> Result[ReconciliationResult]:  # type: ignore[override]
            return Result.success(ReconciliationResult(reconciled_at_ns=0))

    dummy = Dummy()
    assert isinstance(dummy, IDGeneratorProtocol)
    assert isinstance(dummy, StateMachineProtocol)
    assert isinstance(dummy, PersistenceProtocol)
    assert isinstance(dummy, ReconciliationProtocol)
