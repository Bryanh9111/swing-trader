"""Unit tests for OrderManager orchestration (Phase 3.5)."""

from __future__ import annotations

import time
from unittest.mock import DEFAULT, Mock

import pytest

from common.interface import Result, ResultStatus
from execution import OrderManager
from execution.interface import FillDetail
from order_state_machine import IDGenerator, OrderState, Persistence, Reconciler, StateMachine
from order_state_machine.interface import IntentOrderMapping, ReconciliationResult, StateTransition
from strategy.interface import IntentType, TradeIntent


def _make_intent(*, intent_id: str = "I-1", quantity: float = 10.0, run_id: str = "run-1") -> TradeIntent:
    return TradeIntent(
        intent_id=intent_id,
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=float(quantity),
        created_at_ns=1_700_000_000_000_000_000,
        entry_price=None,
        stop_loss_price=None,
        take_profit_price=None,
        parent_intent_id=None,
        linked_intent_ids=[],
        reduce_only=False,
        contingency_type=None,
        ladder_level=None,
        reason_codes=["TEST"],
        metadata={"run_id": run_id},
    )


def _make_mapping(
    *,
    intent: TradeIntent,
    client_order_id: str = "O-1",
    broker_order_id: str | None = "B-1",
    state: OrderState = OrderState.SUBMITTED,
    created_at_ns: int = 1,
    updated_at_ns: int = 1,
    metadata: dict[str, object] | None = None,
) -> IntentOrderMapping:
    return IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id=client_order_id,
        broker_order_id=broker_order_id,
        state=state,
        created_at_ns=int(created_at_ns),
        updated_at_ns=int(updated_at_ns),
        intent_snapshot=intent,
        metadata=dict(metadata or {}),
    )


def _make_manager(
    *,
    adapter: Mock | None = None,
    persistence: Mock | None = None,
    reconciler: Mock | None = None,
    state_machine: StateMachine | None = None,
    initial_mappings: list[IntentOrderMapping] | None = None,
) -> tuple[OrderManager, Mock, Mock, Mock, StateMachine]:
    adapter_owned = adapter is None
    persistence_owned = persistence is None
    reconciler_owned = reconciler is None

    adapter_mock = adapter or Mock()
    persistence_mock = persistence or Mock(spec=Persistence)
    reconciler_mock = reconciler or Mock(spec=Reconciler)
    sm = state_machine or StateMachine()

    if persistence_owned:
        persistence_mock.load_all.return_value = Result.success(list(initial_mappings or []))
        persistence_mock.save.return_value = Result.success(None)

    if adapter_owned:
        adapter_mock.is_connected.return_value = False
        adapter_mock.get_all_orders.return_value = Result.success([])
    else:
        # Ensure startup reconciliation can't crash tests when the adapter mock
        # wasn't configured for get_all_orders/is_connected.
        if not isinstance(getattr(getattr(adapter_mock, "is_connected", None), "return_value", None), bool):
            adapter_mock.is_connected.return_value = False
        if not isinstance(getattr(getattr(adapter_mock, "get_all_orders", None), "return_value", None), Result):
            adapter_mock.get_all_orders.return_value = Result.success([])

    if reconciler_owned:
        reconciler_mock.reconcile_startup.return_value = Result.success(ReconciliationResult(reconciled_at_ns=0))
        reconciler_mock.reconcile_periodic.return_value = Result.success(ReconciliationResult(reconciled_at_ns=0))

    manager = OrderManager(
        adapter=adapter_mock,
        id_generator=IDGenerator(),
        state_machine=sm,
        persistence=persistence_mock,
        reconciler=reconciler_mock,
    )
    return manager, adapter_mock, persistence_mock, reconciler_mock, sm


def test_initialization_loads_mappings() -> None:
    """Initialization loads persisted mappings into the in-memory cache."""

    # Arrange
    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)

    # Act
    manager, adapter, persistence, _reconciler, _sm = _make_manager(initial_mappings=[mapping])

    # Assert
    assert manager.get_all_orders().data == [mapping]
    persistence.load_all.assert_called_once()
    adapter.is_connected.assert_called_once()


def test_initialization_performs_startup_reconciliation() -> None:
    """Initialization invokes startup reconciliation (best-effort)."""

    # Arrange
    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    updated = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.ACCEPTED, updated_at_ns=99)

    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.get_all_orders.return_value = Result.success([updated])

    reconciler = Mock(spec=Reconciler)
    reconciler.reconcile_startup.return_value = Result.success(
        ReconciliationResult(updated_mappings=[updated], reconciled_at_ns=123)
    )

    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.success([mapping])
    persistence.save.return_value = Result.success(None)

    # Act
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, persistence=persistence, reconciler=reconciler, initial_mappings=[mapping]
    )

    # Assert
    status = manager.get_order_status("O-1")
    assert status.status is ResultStatus.SUCCESS
    assert status.data is not None
    assert status.data.state is OrderState.ACCEPTED
    persistence.save.assert_called()


def test_initialization_handles_persistence_failure() -> None:
    """Initialization swallows persistence exceptions and does not crash."""

    # Arrange
    adapter = Mock()
    persistence = Mock(spec=Persistence)
    persistence.load_all.side_effect = RuntimeError("boom")

    # Act
    _manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, persistence=persistence
    )

    # Assert
    adapter.is_connected.assert_not_called()


def test_submit_order_success() -> None:
    """submit_order() caches and persists a successful adapter mapping."""

    # Arrange
    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)

    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.submit_order.return_value = Result.success(mapping)

    manager, _adapter, persistence, _reconciler, _sm = _make_manager(adapter=adapter)

    # Act
    result = manager.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert result.data == mapping
    persistence.save.assert_called_once_with(mapping)


def test_submit_order_not_connected() -> None:
    """submit_order() fails fast when adapter is not connected."""

    # Arrange
    intent = _make_intent()
    adapter = Mock()
    adapter.is_connected.return_value = False
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter)

    # Act
    result = manager.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "BROKER_NOT_CONNECTED"
    adapter.submit_order.assert_not_called()


def test_submit_order_adapter_fails() -> None:
    """submit_order() returns adapter failure results unchanged."""

    # Arrange
    intent = _make_intent()
    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.submit_order.return_value = Result.failed(RuntimeError("nope"), "SUBMIT_FAILED")
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter)

    # Act
    result = manager.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "SUBMIT_FAILED"


def test_submit_order_persistence_degraded() -> None:
    """submit_order() degrades when persistence.save fails after submission."""

    # Arrange
    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1")
    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.submit_order.return_value = Result.success(mapping)

    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.success([])
    persistence.save.return_value = Result.failed(RuntimeError("disk"), "PERSIST_FAILED")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, persistence=persistence
    )

    # Act
    result = manager.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.DEGRADED
    assert result.data == mapping


def test_cancel_order_success() -> None:
    """cancel_order() transitions to CANCELLED, persists, and returns SUCCESS."""

    # Arrange
    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.cancel_order.return_value = Result.success(None)

    manager, _adapter, persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[mapping])

    # Act
    result = manager.cancel_order("O-1")

    # Assert
    assert result.status is ResultStatus.SUCCESS
    updated = manager.get_order_status("O-1").data
    assert updated is not None
    assert updated.state is OrderState.CANCELLED
    persistence.save.assert_called()


def test_cancel_order_not_found() -> None:
    """cancel_order() fails when the mapping does not exist."""

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager()
    result = manager.cancel_order("O-404")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ORDER_NOT_FOUND"


def test_cancel_order_invalid_transition() -> None:
    """cancel_order() fails when state machine rejects the cancellation transition."""

    # Arrange
    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.FILLED)
    adapter = Mock()
    adapter.cancel_order.return_value = Result.success(None)
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[mapping])

    # Act
    result = manager.cancel_order("O-1")

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "SYSTEM_STATE_ERROR"
    adapter.cancel_order.assert_not_called()


def test_cancel_order_adapter_fails() -> None:
    """cancel_order() returns adapter failures unchanged."""

    # Arrange
    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    adapter = Mock()
    adapter.cancel_order.return_value = Result.failed(RuntimeError("nope"), "CANCEL_FAILED")
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[mapping])

    # Act
    result = manager.cancel_order("O-1")

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CANCEL_FAILED"


def test_get_order_status_cached() -> None:
    """get_order_status() returns the cached mapping without broker refresh."""

    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(initial_mappings=[mapping])
    result = manager.get_order_status("O-1")
    assert result.status is ResultStatus.SUCCESS
    assert result.data == mapping


def test_get_order_status_not_found() -> None:
    """get_order_status() fails with ORDER_NOT_FOUND for unknown ids."""

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager()
    result = manager.get_order_status("O-404")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ORDER_NOT_FOUND"


def test_refresh_order_status_updates_from_broker() -> None:
    """refresh_order_status() applies a valid transition from broker view and persists."""

    # Arrange
    intent = _make_intent()
    current = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED, metadata={"a": 1})
    broker_view = _make_mapping(
        intent=intent, client_order_id="O-1", state=OrderState.ACCEPTED, metadata={"b": 2}
    )
    adapter = Mock()
    adapter.get_order_status.return_value = Result.success(broker_view)
    manager, _adapter, persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[current])

    # Act
    result = manager.refresh_order_status("O-1")

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.state is OrderState.ACCEPTED
    assert result.data.metadata["a"] == 1
    assert result.data.metadata["b"] == 2
    persistence.save.assert_called()


def test_refresh_order_status_not_found() -> None:
    """refresh_order_status() fails with ORDER_NOT_FOUND for unknown ids."""

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager()
    result = manager.refresh_order_status("O-404")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ORDER_NOT_FOUND"


def test_refresh_order_status_adapter_exception() -> None:
    """refresh_order_status() converts adapter exceptions into a FAILED Result."""

    intent = _make_intent()
    current = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    adapter = Mock()
    adapter.get_order_status.side_effect = RuntimeError("boom")
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[current])
    result = manager.refresh_order_status("O-1")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ADAPTER_STATUS_EXCEPTION"


def test_refresh_order_status_broker_failed_passthrough() -> None:
    """refresh_order_status() returns FAILED broker results unchanged."""

    intent = _make_intent()
    current = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    adapter = Mock()
    adapter.get_order_status.return_value = Result.failed(RuntimeError("nope"), "STATUS_FAILED")
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[current])
    result = manager.refresh_order_status("O-1")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "STATUS_FAILED"


def test_refresh_order_status_transition_failure_returns_failed() -> None:
    """refresh_order_status() fails when state_machine.transition returns FAILED."""

    intent = _make_intent()
    current = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    broker_view = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.ACCEPTED)
    adapter = Mock()
    adapter.get_order_status.return_value = Result.success(broker_view)

    sm = Mock()
    sm.is_valid_transition.return_value = True
    sm.transition.return_value = Result.failed(RuntimeError("nope"), "STATE_TRANSITION_FAILED")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, state_machine=sm, initial_mappings=[current]
    )
    result = manager.refresh_order_status("O-1")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "STATE_TRANSITION_FAILED"


def test_refresh_order_status_persistence_failure_degrades_after_transition() -> None:
    """refresh_order_status() degrades when persistence fails after applying a transition."""

    intent = _make_intent()
    current = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    broker_view = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.ACCEPTED)
    adapter = Mock()
    adapter.get_order_status.return_value = Result.success(broker_view)

    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.success([current])
    persistence.save.return_value = Result.failed(RuntimeError("disk"), "PERSIST_FAILED")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, persistence=persistence, initial_mappings=[current]
    )
    result = manager.refresh_order_status("O-1")
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None


def test_refresh_order_status_invalid_transition() -> None:
    """refresh_order_status() degrades when broker reports an illegal state rewind."""

    # Arrange
    intent = _make_intent()
    current = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    broker_view = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.PENDING)
    adapter = Mock()
    adapter.get_order_status.return_value = Result.success(broker_view)
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[current])

    # Act
    result = manager.refresh_order_status("O-1")

    # Assert
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "INVALID_BROKER_TRANSITION"
    assert result.data == current


def test_get_all_orders() -> None:
    """get_all_orders() returns the cached mapping set."""

    intent1 = _make_intent(intent_id="I-1")
    intent2 = _make_intent(intent_id="I-2")
    m1 = _make_mapping(intent=intent1, client_order_id="O-1")
    m2 = _make_mapping(intent=intent2, client_order_id="O-2")
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(initial_mappings=[m1, m2])
    result = manager.get_all_orders()
    assert result.status is ResultStatus.SUCCESS
    orders = result.data or []
    assert {m.client_order_id for m in orders} == {"O-1", "O-2"}
    by_id = {m.client_order_id: m for m in orders}
    assert by_id["O-1"] == m1
    assert by_id["O-2"] == m2


def test_reconcile_startup_success() -> None:
    """reconcile_startup() returns reconciler result and persists updated mappings."""

    # Arrange
    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    updated = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.ACCEPTED, updated_at_ns=99)

    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.get_all_orders.return_value = Result.success([updated])

    reconciler = Mock(spec=Reconciler)
    reconciler.reconcile_startup.return_value = Result.success(
        ReconciliationResult(updated_mappings=[updated], reconciled_at_ns=123)
    )

    manager, _adapter, persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, reconciler=reconciler, initial_mappings=[mapping]
    )

    # Act
    result = manager.reconcile_startup()

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert manager.get_order_status("O-1").data is not None
    persistence.save.assert_called()


def test_reconcile_startup_not_connected() -> None:
    """reconcile_startup() degrades when broker is not connected."""

    manager, adapter, _persistence, _reconciler, _sm = _make_manager()
    adapter.is_connected.return_value = False
    result = manager.reconcile_startup()
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "STARTUP_RECONCILIATION_SKIPPED"


def test_reconcile_startup_empty_broker_snapshot() -> None:
    """reconcile_startup() degrades when broker snapshot is empty with in-flight mappings."""

    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    manager, adapter, _persistence, _reconciler, _sm = _make_manager(initial_mappings=[mapping])
    adapter.is_connected.return_value = True
    adapter.get_all_orders.return_value = Result.success([])

    result = manager.reconcile_startup()
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "STARTUP_RECONCILIATION_SKIPPED"


def test_reconcile_startup_broker_fetch_exception_degrades() -> None:
    """reconcile_startup() degrades when adapter.get_all_orders raises."""

    manager, adapter, _persistence, _reconciler, _sm = _make_manager()
    adapter.is_connected.return_value = True
    adapter.get_all_orders.side_effect = RuntimeError("boom")
    result = manager.reconcile_startup()
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "STARTUP_BROKER_FETCH_EXCEPTION"


def test_reconcile_startup_reconciler_exception_degrades() -> None:
    """reconcile_startup() degrades when reconciler.reconcile_startup raises."""

    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)

    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.get_all_orders.return_value = Result.success([mapping])

    reconciler = Mock(spec=Reconciler)
    reconciler.reconcile_startup.side_effect = RuntimeError("boom")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, reconciler=reconciler, initial_mappings=[mapping]
    )
    result = manager.reconcile_startup()
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "STARTUP_RECONCILIATION_EXCEPTION"


def test_reconcile_startup_apply_degraded_returns_degraded() -> None:
    """reconcile_startup() degrades when applying reconciliation updates cannot be fully persisted."""

    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    updated = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.ACCEPTED, updated_at_ns=99)

    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.get_all_orders.return_value = Result.success([updated])

    reconciler = Mock(spec=Reconciler)
    reconciler.reconcile_startup.return_value = Result.success(
        ReconciliationResult(updated_mappings=[updated], reconciled_at_ns=123)
    )

    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.success([mapping])
    persistence.save.return_value = Result.failed(RuntimeError("disk"), "PERSIST_FAILED")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, persistence=persistence, reconciler=reconciler, initial_mappings=[mapping]
    )
    result = manager.reconcile_startup()
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None


def test_reconcile_periodic_success() -> None:
    """reconcile_periodic() returns reconciler result when successful."""

    manager, _adapter, _persistence, reconciler, _sm = _make_manager()
    reconciler.reconcile_periodic.return_value = Result.success(ReconciliationResult(reconciled_at_ns=123))
    result = manager.reconcile_periodic()
    assert result.status is ResultStatus.SUCCESS


def test_reconcile_periodic_applies_transitions() -> None:
    """reconcile_periodic() applies transition audit entries to cached mappings."""

    # Arrange
    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    transition = StateTransition(
        intent_id="I-1",
        from_state=OrderState.SUBMITTED,
        to_state=OrderState.ACCEPTED,
        timestamp_ns=int(time.time_ns()),
        reason="TEST",
        metadata={"x": 1},
    )

    reconciler = Mock(spec=Reconciler)
    reconciler.reconcile_periodic.return_value = Result.success(
        ReconciliationResult(state_transitions=[transition], reconciled_at_ns=int(time.time_ns()))
    )
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        reconciler=reconciler, initial_mappings=[mapping]
    )

    # Act
    result = manager.reconcile_periodic()

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert manager.get_order_status("O-1").data is not None
    assert manager.get_order_status("O-1").data.state is OrderState.ACCEPTED


def test_reconcile_periodic_failed_degrades() -> None:
    """reconcile_periodic() degrades when reconciler returns FAILED."""

    manager, _adapter, _persistence, reconciler, _sm = _make_manager()
    reconciler.reconcile_periodic.return_value = Result.failed(RuntimeError("boom"), "PERIODIC_FAILED")
    result = manager.reconcile_periodic()
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "PERIODIC_FAILED"


def test_reconcile_periodic_apply_degraded_returns_degraded() -> None:
    """reconcile_periodic() degrades when persistence cannot save updated mappings."""

    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    updated = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.ACCEPTED, updated_at_ns=99)

    reconciler = Mock(spec=Reconciler)
    reconciler.reconcile_periodic.return_value = Result.success(
        ReconciliationResult(updated_mappings=[updated], reconciled_at_ns=123)
    )

    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.success([mapping])
    persistence.save.return_value = Result.failed(RuntimeError("disk"), "PERSIST_FAILED")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        persistence=persistence, reconciler=reconciler, initial_mappings=[mapping]
    )
    result = manager.reconcile_periodic()
    assert result.status is ResultStatus.DEGRADED


def test_apply_transitions_partial_returns_degraded() -> None:
    """_apply_transitions returns DEGRADED when some transitions are invalid or cannot be persisted."""

    intent1 = _make_intent(intent_id="I-1")
    intent2 = _make_intent(intent_id="I-2")
    m1 = _make_mapping(intent=intent1, client_order_id="O-1", state=OrderState.SUBMITTED, metadata={"a": 1})
    m2 = _make_mapping(intent=intent2, client_order_id="O-2", state=OrderState.FILLED)

    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.success([m1, m2])
    persistence.save.return_value = Result.failed(RuntimeError("disk"), "PERSIST_FAILED")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(persistence=persistence, initial_mappings=[m1, m2])

    transitions = [
        StateTransition(
            intent_id="I-UNKNOWN",
            from_state=OrderState.SUBMITTED,
            to_state=OrderState.ACCEPTED,
            timestamp_ns=1,
            reason="SKIP",
        ),
        StateTransition(
            intent_id="I-1",
            from_state=OrderState.SUBMITTED,
            to_state=OrderState.ACCEPTED,
            timestamp_ns=2,
            reason="VALID",
            metadata={"b": 2},
        ),
        StateTransition(
            intent_id="I-2",
            from_state=OrderState.FILLED,
            to_state=OrderState.ACCEPTED,
            timestamp_ns=3,
            reason="INVALID",
        ),
    ]

    result = manager._apply_transitions(transitions)  # noqa: SLF001 - intentional internal coverage
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "RECONCILIATION_APPLY_PARTIAL"


def test_generate_execution_report_success() -> None:
    """generate_execution_report() succeeds for terminal orders (no fills)."""

    # Arrange
    intent = _make_intent(quantity=10.0)
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.CANCELLED)
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(initial_mappings=[mapping])

    # Act
    result = manager.generate_execution_report("O-1")

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.final_state is OrderState.CANCELLED
    assert result.data.filled_quantity == 0.0
    assert result.data.remaining_quantity == 10.0


def test_generate_execution_report_not_found() -> None:
    """generate_execution_report() fails when the mapping is missing."""

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager()
    result = manager.generate_execution_report("O-404")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ORDER_NOT_FOUND"


def test_generate_execution_report_not_terminal() -> None:
    """generate_execution_report() fails when the order state is not terminal."""

    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(initial_mappings=[mapping])
    result = manager.generate_execution_report("O-1")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ORDER_NOT_TERMINAL"


def test_generate_execution_report_with_fills() -> None:
    """generate_execution_report() includes adapter-provided fills."""

    intent = _make_intent(quantity=10.0)
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.FILLED)

    fills_list = [
        FillDetail(execution_id="E-1", fill_price=100.0, fill_quantity=10.0, fill_time_ns=1, commission=0.5)
    ]
    adapter = Mock()
    adapter.get_fills.return_value = fills_list
    adapter._fills = {"O-1": fills_list}  # noqa: SLF001 - kept for backwards compat
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[mapping])

    result = manager.generate_execution_report("O-1")
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert len(result.data.fills) == 1
    assert result.data.commissions == 0.5


def test_generate_execution_report_calculates_avg_price() -> None:
    """generate_execution_report() calculates a volume-weighted average fill price."""

    intent = _make_intent(quantity=10.0)
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.FILLED)

    fills_list = [
        FillDetail(execution_id="E-1", fill_price=100.0, fill_quantity=2.0, fill_time_ns=1),
        FillDetail(execution_id="E-2", fill_price=110.0, fill_quantity=8.0, fill_time_ns=2),
    ]
    adapter = Mock()
    adapter.get_fills.return_value = fills_list
    adapter._fills = {"O-1": fills_list}  # noqa: SLF001
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[mapping])

    result = manager.generate_execution_report("O-1")
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.avg_fill_price == pytest.approx((2.0 * 100.0 + 8.0 * 110.0) / 10.0)


def test_initialization_load_degraded_does_not_crash() -> None:
    """Initialization tolerates degraded persistence loads and continues."""

    # Arrange
    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1")

    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.degraded([mapping], RuntimeError("warn"), "LOAD_DEGRADED")
    persistence.save.return_value = Result.success(None)

    adapter = Mock()
    adapter.is_connected.return_value = False
    adapter.get_all_orders.return_value = Result.success([])

    # Act
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, persistence=persistence)

    # Assert
    assert manager.get_order_status("O-1").status is ResultStatus.SUCCESS


def test_initialization_load_failed_returns_without_reconcile() -> None:
    """Initialization stops after a FAILED load_all result (no reconciliation)."""

    # Arrange
    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.failed(RuntimeError("boom"), "LOAD_FAILED")
    adapter = Mock()

    # Act
    _manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, persistence=persistence)

    # Assert
    adapter.is_connected.assert_not_called()


def test_submit_order_adapter_raises_exception() -> None:
    """submit_order() converts adapter exceptions into a FAILED Result."""

    # Arrange
    intent = _make_intent()
    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.submit_order.side_effect = RuntimeError("boom")
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter)

    # Act
    result = manager.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ADAPTER_SUBMIT_EXCEPTION"


def test_submit_order_persistence_exception_degrades() -> None:
    """submit_order() degrades when persistence.save raises unexpectedly."""

    # Arrange
    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1")
    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.submit_order.return_value = Result.success(mapping)

    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.success([])
    persistence.save.side_effect = RuntimeError("boom")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, persistence=persistence
    )

    # Act
    result = manager.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "PERSISTENCE_EXCEPTION"


def test_cancel_order_adapter_raises_exception() -> None:
    """cancel_order() converts adapter exceptions into a FAILED Result."""

    # Arrange
    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    adapter = Mock()
    adapter.cancel_order.side_effect = RuntimeError("boom")
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[mapping])

    # Act
    result = manager.cancel_order("O-1")

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ADAPTER_CANCEL_EXCEPTION"


def test_cancel_order_transition_failure_returns_failed() -> None:
    """cancel_order() fails when state_machine.transition returns FAILED."""

    # Arrange
    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)

    adapter = Mock()
    adapter.cancel_order.return_value = Result.success(None)

    sm = Mock()
    sm.is_valid_transition.return_value = True
    sm.transition.return_value = Result.failed(RuntimeError("nope"), "STATE_TRANSITION_FAILED")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, state_machine=sm, initial_mappings=[mapping]
    )

    # Act
    result = manager.cancel_order("O-1")

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "STATE_TRANSITION_FAILED"


def test_cancel_order_degraded_when_adapter_degraded() -> None:
    """cancel_order() returns DEGRADED when adapter returns DEGRADED and persistence succeeds."""

    # Arrange
    intent = _make_intent()
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    adapter = Mock()
    adapter.cancel_order.return_value = Result.degraded(None, RuntimeError("slow"), "CANCEL_DEGRADED")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[mapping])

    # Act
    result = manager.cancel_order("O-1")

    # Assert
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "CANCEL_DEGRADED"


def test_refresh_order_status_same_state_persists_metadata_update() -> None:
    """refresh_order_status() persists metadata-only changes when state is unchanged."""

    # Arrange
    intent = _make_intent()
    current = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED, metadata={"a": 1})
    broker_view = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED, metadata={"b": 2})
    adapter = Mock()
    adapter.get_order_status.return_value = Result.success(broker_view)

    manager, _adapter, persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[current])

    # Act
    result = manager.refresh_order_status("O-1")

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.metadata["a"] == 1
    assert result.data.metadata["b"] == 2
    persistence.save.assert_called()


def test_refresh_order_status_broker_degraded_propagates() -> None:
    """refresh_order_status() returns DEGRADED when the broker result is DEGRADED."""

    # Arrange
    intent = _make_intent()
    current = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)
    broker_view = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.ACCEPTED)
    adapter = Mock()
    adapter.get_order_status.return_value = Result.degraded(broker_view, RuntimeError("stale"), "BROKER_DEGRADED")
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(adapter=adapter, initial_mappings=[current])

    # Act
    result = manager.refresh_order_status("O-1")

    # Assert
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "BROKER_DEGRADED"


def test_reconcile_startup_broker_fetch_failed_degrades() -> None:
    """reconcile_startup() degrades when adapter.get_all_orders returns FAILED."""

    # Arrange
    manager, adapter, _persistence, _reconciler, _sm = _make_manager()
    adapter.is_connected.return_value = True
    adapter.get_all_orders.return_value = Result.failed(RuntimeError("boom"), "BROKER_SNAPSHOT_FAILED")

    # Act
    result = manager.reconcile_startup()

    # Assert
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "BROKER_SNAPSHOT_FAILED"


def test_reconcile_startup_reconciler_returns_no_data_fails() -> None:
    """reconcile_startup() fails when reconciler returns a Result with data=None."""

    # Arrange
    intent = _make_intent(intent_id="I-1")
    mapping = _make_mapping(intent=intent, client_order_id="O-1", state=OrderState.SUBMITTED)

    adapter = Mock()
    adapter.is_connected.return_value = True
    adapter.get_all_orders.return_value = Result.success([mapping])

    reconciler = Mock(spec=Reconciler)
    reconciler.reconcile_startup.return_value = Result.failed(RuntimeError("boom"), "RECON_FAIL")

    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(
        adapter=adapter, reconciler=reconciler, initial_mappings=[mapping]
    )

    # Act
    result = manager.reconcile_startup()

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "RECON_FAIL"


def test_reconcile_periodic_reconciler_exception_degrades() -> None:
    """reconcile_periodic() degrades when reconciler.reconcile_periodic raises."""

    manager, _adapter, _persistence, reconciler, _sm = _make_manager()
    reconciler.reconcile_periodic.side_effect = RuntimeError("boom")
    result = manager.reconcile_periodic()
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "PERIODIC_RECONCILIATION_EXCEPTION"


def test_generate_execution_report_fallback_fields_for_missing_data() -> None:
    """generate_execution_report() falls back for missing symbol/run_id/broker_order_id."""

    # Arrange
    intent = TradeIntent(
        intent_id="I-1",
        symbol="  ",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=1,
        entry_price=None,
        stop_loss_price=None,
        take_profit_price=None,
        parent_intent_id=None,
        linked_intent_ids=[],
        reduce_only=False,
        contingency_type=None,
        ladder_level=None,
        reason_codes=["TEST"],
        metadata={},
    )
    mapping = _make_mapping(intent=intent, client_order_id="O-1", broker_order_id=None, state=OrderState.CANCELLED)
    manager, _adapter, _persistence, _reconciler, _sm = _make_manager(initial_mappings=[mapping])

    # Act
    report = manager.generate_execution_report("O-1")

    # Assert
    assert report.status is ResultStatus.SUCCESS
    assert report.data is not None
    assert report.data.run_id == "unknown"
    assert report.data.broker_order_id == "unknown"
    assert report.data.symbol == "UNKNOWN"
