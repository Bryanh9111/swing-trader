"""End-to-end integration tests for the Execution layer (Phase 3.5).

These tests wire together:
- ``OrderManager`` (execution)
- ``StateMachine`` + ``Reconciler`` + ``JSONLPersistence`` (order_state_machine)
- A lightweight in-memory adapter implementing ``BrokerAdapterProtocol``
"""

from __future__ import annotations

import time
from pathlib import Path

from msgspec.structs import replace

from common.interface import Result, ResultStatus
from execution import OrderManager
from execution.interface import BrokerAdapterProtocol, FillDetail
from order_state_machine import IDGenerator, OrderState, Persistence, Reconciler, StateMachine
from order_state_machine.interface import IntentOrderMapping
from strategy.interface import IntentType, TradeIntent


class InMemoryAdapter(BrokerAdapterProtocol):
    """Simple in-memory adapter used for integration tests."""

    def __init__(self, *, readonly: bool = False) -> None:
        self._connected = True
        self._readonly = readonly
        self._counter = 0
        self._mappings: dict[str, IntentOrderMapping] = {}
        self._fills: dict[str, list[FillDetail]] = {}

    def connect(self) -> Result[None]:  # type: ignore[override]
        self._connected = True
        return Result.success(None)

    def disconnect(self) -> Result[None]:  # type: ignore[override]
        self._connected = False
        return Result.success(None)

    def is_connected(self) -> bool:  # type: ignore[override]
        return bool(self._connected)

    def submit_order(self, intent: TradeIntent) -> Result[IntentOrderMapping]:  # type: ignore[override]
        if self._readonly:
            return Result.failed(PermissionError("readonly"), "READONLY_MODE")
        if not self._connected:
            return Result.failed(ConnectionError("not connected"), "CONNECTION_FAILED")

        self._counter += 1
        now_ns = int(time.time_ns())
        client_order_id = f"O-{self._counter}"
        broker_order_id = f"B-{self._counter}"
        mapping = IntentOrderMapping(
            intent_id=intent.intent_id,
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
            state=OrderState.SUBMITTED,
            created_at_ns=now_ns,
            updated_at_ns=now_ns,
            intent_snapshot=intent,
            metadata={"broker": "in_memory"},
        )
        self._mappings[client_order_id] = mapping
        self._fills.setdefault(client_order_id, [])
        return Result.success(mapping)

    def cancel_order(self, client_order_id: str) -> Result[None]:  # type: ignore[override]
        if self._readonly:
            return Result.failed(PermissionError("readonly"), "READONLY_MODE")
        if not self._connected:
            return Result.failed(ConnectionError("not connected"), "CONNECTION_FAILED")
        if str(client_order_id) not in self._mappings:
            return Result.failed(KeyError("not found"), "ORDER_NOT_FOUND")
        return Result.success(None)

    def get_order_status(self, client_order_id: str) -> Result[IntentOrderMapping]:  # type: ignore[override]
        mapping = self._mappings.get(str(client_order_id))
        if mapping is None:
            return Result.failed(KeyError("not found"), "ORDER_NOT_FOUND")
        return Result.success(mapping)

    def get_all_orders(self) -> Result[list[IntentOrderMapping]]:  # type: ignore[override]
        return Result.success(list(self._mappings.values()))


def _make_intent(*, intent_id: str, qty: float = 10.0, run_id: str = "run-1") -> TradeIntent:
    return TradeIntent(
        intent_id=intent_id,
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=float(qty),
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


def _make_manager(*, adapter: InMemoryAdapter, storage_dir: Path, reconciler: Reconciler | None = None) -> OrderManager:
    sm = StateMachine()
    store = Persistence(storage_dir=storage_dir)
    rec = reconciler or Reconciler(sm, fetch_broker_order=None)
    return OrderManager(
        adapter=adapter,
        id_generator=IDGenerator(),
        state_machine=sm,
        persistence=store,
        reconciler=rec,
    )


def test_e2e_submit_and_track_order(tmp_path: Path) -> None:
    """Submit an order and verify it is tracked in the manager cache."""

    # Arrange
    adapter = InMemoryAdapter()
    manager = _make_manager(adapter=adapter, storage_dir=tmp_path)
    intent = _make_intent(intent_id="I-1")

    # Act
    submit = manager.submit_order(intent)

    # Assert
    assert submit.status is ResultStatus.SUCCESS
    assert submit.data is not None
    cached = manager.get_order_status(submit.data.client_order_id)
    assert cached.status is ResultStatus.SUCCESS


def test_e2e_submit_cancel_order(tmp_path: Path) -> None:
    """Submit an order and then cancel it via OrderManager."""

    # Arrange
    adapter = InMemoryAdapter()
    manager = _make_manager(adapter=adapter, storage_dir=tmp_path)
    intent = _make_intent(intent_id="I-1")
    mapping = manager.submit_order(intent).data
    assert mapping is not None

    # Act
    cancel = manager.cancel_order(mapping.client_order_id)

    # Assert
    assert cancel.status is ResultStatus.SUCCESS
    assert manager.get_order_status(mapping.client_order_id).data is not None
    assert manager.get_order_status(mapping.client_order_id).data.state is OrderState.CANCELLED


def test_e2e_startup_reconciliation_corrects_state(tmp_path: Path) -> None:
    """Startup reconciliation updates local in-flight state to match broker snapshot."""

    # Arrange
    adapter = InMemoryAdapter()
    sm = StateMachine()
    store = Persistence(storage_dir=tmp_path)
    reconciler = Reconciler(sm, fetch_broker_order=None)

    intent = _make_intent(intent_id="I-1")
    now_ns = int(time.time_ns())
    local = IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id="O-1",
        broker_order_id="B-1",
        state=OrderState.SUBMITTED,
        created_at_ns=now_ns,
        updated_at_ns=now_ns,
        intent_snapshot=intent,
        metadata={},
    )
    store.save(local)

    # Broker view: already FILLED.
    broker_view = IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id="O-1",
        broker_order_id="B-1",
        state=OrderState.FILLED,
        created_at_ns=now_ns,
        updated_at_ns=now_ns,
        intent_snapshot=intent,
        metadata={},
    )
    adapter._mappings["O-1"] = broker_view

    # Act
    manager = OrderManager(
        adapter=adapter,
        id_generator=IDGenerator(),
        state_machine=sm,
        persistence=store,
        reconciler=reconciler,
    )

    # Assert
    status = manager.get_order_status("O-1")
    assert status.status is ResultStatus.SUCCESS
    assert status.data is not None
    assert status.data.state is OrderState.FILLED


def test_e2e_periodic_reconciliation_detects_timeout(tmp_path: Path) -> None:
    """Periodic reconciliation expires timed-out in-flight mappings when broker cannot find them."""

    # Arrange
    adapter = InMemoryAdapter()
    sm = StateMachine()
    store = Persistence(storage_dir=tmp_path)

    intent = _make_intent(intent_id="I-1")
    now_ns = int(time.time_ns())
    timed_out_ns = now_ns - (OrderManager._DEFAULT_TIMEOUT_THRESHOLD_NS + 1_000)  # noqa: SLF001
    local = IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id="O-1",
        broker_order_id="B-1",
        state=OrderState.SUBMITTED,
        created_at_ns=timed_out_ns,
        updated_at_ns=timed_out_ns,
        intent_snapshot=intent,
        metadata={},
    )
    store.save(local)

    def fetch_broker_order(_mapping: IntentOrderMapping) -> Result[dict | None]:
        return Result.success(None)

    reconciler = Reconciler(sm, fetch_broker_order=fetch_broker_order)
    manager = OrderManager(
        adapter=adapter,
        id_generator=IDGenerator(),
        state_machine=sm,
        persistence=store,
        reconciler=reconciler,
    )

    # Act
    reconcile = manager.reconcile_periodic()

    # Assert
    assert reconcile.status in {ResultStatus.SUCCESS, ResultStatus.DEGRADED}
    assert manager.get_order_status("O-1").data is not None
    assert manager.get_order_status("O-1").data.state is OrderState.EXPIRED


def test_e2e_execution_report_after_fill(tmp_path: Path) -> None:
    """Execution report includes fills after a terminal fill state is observed."""

    # Arrange
    adapter = InMemoryAdapter()
    manager = _make_manager(adapter=adapter, storage_dir=tmp_path)
    intent = _make_intent(intent_id="I-1", qty=10.0)
    mapping = manager.submit_order(intent).data
    assert mapping is not None

    adapter._fills[mapping.client_order_id] = [
        FillDetail(execution_id="E-1", fill_price=100.0, fill_quantity=10.0, fill_time_ns=1, commission=0.5)
    ]
    adapter._mappings[mapping.client_order_id] = replace(mapping, state=OrderState.ACCEPTED)
    assert manager.refresh_order_status(mapping.client_order_id).status is ResultStatus.SUCCESS
    adapter._mappings[mapping.client_order_id] = replace(
        adapter._mappings[mapping.client_order_id], state=OrderState.FILLED
    )
    assert manager.refresh_order_status(mapping.client_order_id).status is ResultStatus.SUCCESS

    # Act
    report = manager.generate_execution_report(mapping.client_order_id)

    # Assert
    assert report.status is ResultStatus.SUCCESS
    assert report.data is not None
    assert report.data.final_state is OrderState.FILLED
    assert report.data.filled_quantity == 10.0
    assert len(report.data.fills) == 1


def test_e2e_readonly_mode_blocks_submission(tmp_path: Path) -> None:
    """Readonly adapter blocks submission and manager surfaces the failure."""

    # Arrange
    adapter = InMemoryAdapter(readonly=True)
    manager = _make_manager(adapter=adapter, storage_dir=tmp_path)
    intent = _make_intent(intent_id="I-1")

    # Act
    result = manager.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "READONLY_MODE"
    assert manager.get_all_orders().data == []


def test_e2e_adapter_disconnect_degrades_gracefully(tmp_path: Path) -> None:
    """Disconnected adapter returns a clean FAILED Result for submission."""

    # Arrange
    adapter = InMemoryAdapter()
    adapter.disconnect()
    manager = _make_manager(adapter=adapter, storage_dir=tmp_path)
    intent = _make_intent(intent_id="I-1")

    # Act
    result = manager.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "BROKER_NOT_CONNECTED"


def test_e2e_multiple_orders_tracked_independently(tmp_path: Path) -> None:
    """Submitting multiple orders results in independently tracked mappings."""

    # Arrange
    adapter = InMemoryAdapter()
    manager = _make_manager(adapter=adapter, storage_dir=tmp_path)

    # Act
    m1 = manager.submit_order(_make_intent(intent_id="I-1")).data
    m2 = manager.submit_order(_make_intent(intent_id="I-2")).data
    assert m1 is not None and m2 is not None

    manager.cancel_order(m1.client_order_id)

    # Assert
    orders = manager.get_all_orders().data or []
    by_id = {m.client_order_id: m for m in orders}
    assert by_id[m1.client_order_id].state is OrderState.CANCELLED
    assert by_id[m2.client_order_id].state is OrderState.SUBMITTED
