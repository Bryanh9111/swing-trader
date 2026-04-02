"""High-level order orchestration for AST Execution layer.

This module implements Phase 3.5-style orchestration logic that coordinates:

- Broker adapter operations (submit/cancel/query)
- Local intent->order mapping tracking
- Persistence of mappings across restarts
- Startup + periodic reconciliation (via ``order_state_machine.Reconciler``)
- ExecutionReport generation for terminal orders

All public methods return ``common.interface.Result[T]`` and avoid raising as
control flow across the boundary.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from msgspec.structs import replace

from common.exceptions import BrokerConnectionError, OperationalError, SystemStateError
from common.interface import Result, ResultStatus
from execution.interface import BrokerAdapterProtocol, ExecutionReport, FillDetail
from order_state_machine import IDGenerator, Persistence, Reconciler, StateMachine
from order_state_machine.interface import (
    IntentOrderMapping,
    OrderState,
    ReconciliationResult,
    StateTransition,
)
from strategy.interface import TradeIntent

__all__ = ["OrderManager"]

_MODULE = "execution.order_manager"
_LOGGER = logging.getLogger(_MODULE)


class OrderManager:
    """High-level order orchestration for AST Execution layer.

    Responsibilities:
        - Submit/cancel orders through a broker adapter.
        - Track intent->order mappings and persist them.
        - Perform startup and periodic reconciliation.
        - Generate ``ExecutionReport`` for terminal orders.

    Thread safety:
        All access and mutation of ``_mappings`` is guarded by ``_lock``.

    Notes:
        The OrderManager treats reconciliation failures as non-fatal and
        degrades gracefully (logs and continues operating with cached state).
    """

    _DEFAULT_TIMEOUT_THRESHOLD_NS = 10 * 60 * 1_000_000_000  # 10 minutes

    def __init__(
        self,
        adapter: BrokerAdapterProtocol,
        id_generator: IDGenerator,
        state_machine: StateMachine,
        persistence: Persistence,
        reconciler: Reconciler,
    ) -> None:
        """Create an OrderManager and perform startup initialization.

        Args:
            adapter: Broker adapter implementation (e.g., IBKR adapter).
            id_generator: ID generator used by upstream/downstream components.
            state_machine: Order state machine used for transition validation.
            persistence: Persistence implementation for mappings.
            reconciler: Reconciler implementation for startup/periodic corrections.
        """

        self._adapter = adapter
        self._id_gen = id_generator
        self._sm = state_machine
        self._persistence = persistence
        self._reconciler = reconciler

        self._lock = threading.RLock()
        self._mappings: dict[str, IntentOrderMapping] = {}

        self._initialize()

    def _initialize(self) -> None:
        """Load mappings and perform startup reconciliation (best-effort)."""

        try:
            load_result = self._persistence.load_all()
        except Exception as exc:  # noqa: BLE001 - boundary should not raise
            _LOGGER.exception("Failed to load persisted mappings: %s", repr(exc))
            return

        mappings = list(load_result.data or [])
        with self._lock:
            self._mappings = {m.client_order_id: m for m in mappings}

        if load_result.status is ResultStatus.DEGRADED:
            _LOGGER.warning(
                "Loaded persisted mappings in degraded mode (reason_code=%s, error=%s)",
                load_result.reason_code,
                repr(load_result.error),
            )
        elif load_result.status is ResultStatus.FAILED:
            _LOGGER.error(
                "Failed to load persisted mappings (reason_code=%s, error=%s)",
                load_result.reason_code,
                repr(load_result.error),
            )
            return

        startup = self.reconcile_startup()
        if startup.status is ResultStatus.FAILED:
            _LOGGER.warning(
                "Startup reconciliation failed; continuing with cached mappings (reason_code=%s, error=%s)",
                startup.reason_code,
                repr(startup.error),
            )
        elif startup.status is ResultStatus.DEGRADED:
            _LOGGER.warning(
                "Startup reconciliation degraded; continuing (reason_code=%s, error=%s)",
                startup.reason_code,
                repr(startup.error),
            )

    def submit_order(self, intent: TradeIntent) -> Result[IntentOrderMapping]:
        """Submit an order to the broker and persist the resulting mapping.

        Flow:
            1) Verify adapter connectivity.
            2) Submit the order via the adapter.
            3) Persist the returned mapping.

        Args:
            intent: Strategy trade intent to submit.

        Returns:
            Result[IntentOrderMapping]: Mapping for the submitted order.
        """

        try:
            if not self._adapter.is_connected():
                err = BrokerConnectionError(
                    "Broker adapter is not connected",
                    module=_MODULE,
                    reason_code="BROKER_NOT_CONNECTED",
                )
                return Result.failed(err, err.reason_code)

            submit_result = self._adapter.submit_order(intent)
        except Exception as exc:  # noqa: BLE001 - adapter boundary protection
            return Result.failed(exc, "ADAPTER_SUBMIT_EXCEPTION")

        if submit_result.status is ResultStatus.FAILED or submit_result.data is None:
            return submit_result

        mapping = submit_result.data

        with self._lock:
            self._mappings[mapping.client_order_id] = mapping

        persist_result = self._safe_persist(mapping)
        if persist_result.status is ResultStatus.SUCCESS:
            return Result.success(mapping)

        return Result.degraded(
            data=mapping,
            error=persist_result.error or RuntimeError("persistence failed"),
            reason_code=persist_result.reason_code or "PERSISTENCE_FAILED",
        )

    def cancel_order(self, client_order_id: str) -> Result[None]:
        """Cancel an order through the broker adapter and persist the transition.

        Flow:
            1) Locate the local mapping.
            2) Validate cancellation transition legality.
            3) Call adapter.cancel_order(...).
            4) Apply state transition and persist on success.

        Args:
            client_order_id: Client order id used to reference the order.

        Returns:
            Result[None]: Success if the cancellation request was accepted and the
                mapping was updated; failed/degraded otherwise.
        """

        with self._lock:
            mapping = self._mappings.get(str(client_order_id))

        if mapping is None:
            err = OperationalError(
                f"Order not found: {client_order_id}",
                module=_MODULE,
                reason_code="ORDER_NOT_FOUND",
                details={"client_order_id": str(client_order_id)},
            )
            return Result.failed(err, err.reason_code)

        if mapping.state is OrderState.PENDING:
            ts_ns = int(time.time_ns())
            transition_result = self._sm.transition(
                mapping,
                OrderState.CANCELLED,
                reason="CANCELLED_LOCAL",
                timestamp_ns=ts_ns,
                metadata={"order_manager": {"cancelled_at_ns": ts_ns, "local_only": True}},
            )
            if transition_result.status is ResultStatus.FAILED or transition_result.data is None:
                err = transition_result.error or RuntimeError("state transition failed")
                return Result.failed(err, transition_result.reason_code or "STATE_TRANSITION_FAILED")

            updated = transition_result.data
            with self._lock:
                self._mappings[updated.client_order_id] = updated

            persist_result = self._safe_persist(updated)
            if persist_result.status is ResultStatus.SUCCESS:
                return Result.success(data=None)

            return Result.degraded(
                data=None,
                error=persist_result.error or RuntimeError("persistence failed"),
                reason_code=persist_result.reason_code or "PERSISTENCE_FAILED",
            )

        if not self._sm.is_valid_transition(mapping.state, OrderState.CANCELLED):
            err = SystemStateError(
                "Invalid cancellation transition",
                module=_MODULE,
                details={
                    "client_order_id": str(client_order_id),
                    "intent_id": mapping.intent_id,
                    "from_state": mapping.state.value,
                    "to_state": OrderState.CANCELLED.value,
                },
            )
            return Result.failed(err, err.reason_code)

        try:
            cancel_result = self._adapter.cancel_order(str(client_order_id))
        except Exception as exc:  # noqa: BLE001 - adapter boundary protection
            return Result.failed(exc, "ADAPTER_CANCEL_EXCEPTION")

        if cancel_result.status is ResultStatus.FAILED:
            return cancel_result

        ts_ns = int(time.time_ns())
        transition_result = self._sm.transition(
            mapping,
            OrderState.CANCELLED,
            reason="CANCEL_REQUESTED",
            timestamp_ns=ts_ns,
            metadata={"order_manager": {"cancelled_at_ns": ts_ns}},
        )
        if transition_result.status is ResultStatus.FAILED or transition_result.data is None:
            err = transition_result.error or RuntimeError("state transition failed")
            return Result.failed(err, transition_result.reason_code or "STATE_TRANSITION_FAILED")

        updated = transition_result.data
        with self._lock:
            self._mappings[updated.client_order_id] = updated

        persist_result = self._safe_persist(updated)
        if persist_result.status is ResultStatus.SUCCESS:
            if cancel_result.status is ResultStatus.DEGRADED:
                return Result.degraded(None, cancel_result.error or RuntimeError("degraded cancel"), cancel_result.reason_code or "CANCEL_DEGRADED")
            return Result.success(data=None)

        return Result.degraded(
            data=None,
            error=persist_result.error or RuntimeError("persistence failed"),
            reason_code=persist_result.reason_code or "PERSISTENCE_FAILED",
        )

    def get_order_status(self, client_order_id: str) -> Result[IntentOrderMapping]:
        """Return the cached order status (no broker refresh).

        Args:
            client_order_id: Client order id used to reference the order.

        Returns:
            Result[IntentOrderMapping]: Cached mapping for this order.
        """

        with self._lock:
            mapping = self._mappings.get(str(client_order_id))

        if mapping is None:
            err = OperationalError(
                f"Order not found: {client_order_id}",
                module=_MODULE,
                reason_code="ORDER_NOT_FOUND",
                details={"client_order_id": str(client_order_id)},
            )
            return Result.failed(err, err.reason_code)

        return Result.success(mapping)

    def refresh_order_status(self, client_order_id: str) -> Result[IntentOrderMapping]:
        """Refresh the order status from the broker and persist updates.

        Flow:
            1) Fetch broker's latest mapping view.
            2) Apply a validated state transition (if any).
            3) Persist the updated mapping.

        Args:
            client_order_id: Client order id to refresh.

        Returns:
            Result[IntentOrderMapping]: Updated mapping after refresh.
        """

        with self._lock:
            current = self._mappings.get(str(client_order_id))

        if current is None:
            err = OperationalError(
                f"Order not found: {client_order_id}",
                module=_MODULE,
                reason_code="ORDER_NOT_FOUND",
                details={"client_order_id": str(client_order_id)},
            )
            return Result.failed(err, err.reason_code)

        try:
            broker_result = self._adapter.get_order_status(str(client_order_id))
        except Exception as exc:  # noqa: BLE001 - adapter boundary protection
            return Result.failed(exc, "ADAPTER_STATUS_EXCEPTION")

        if broker_result.status is ResultStatus.FAILED or broker_result.data is None:
            return broker_result

        broker_view = broker_result.data
        merged_metadata = dict(current.metadata)
        merged_metadata.update(dict(broker_view.metadata or {}))

        base = replace(
            current,
            broker_order_id=broker_view.broker_order_id or current.broker_order_id,
            metadata=merged_metadata,
        )

        if broker_view.state == current.state:
            if base == current:
                return Result.success(current)

            with self._lock:
                self._mappings[base.client_order_id] = base
            persist = self._safe_persist(base)
            if persist.status is ResultStatus.SUCCESS:
                return Result.success(base)
            return Result.degraded(
                data=base,
                error=persist.error or RuntimeError("persistence failed"),
                reason_code=persist.reason_code or "PERSISTENCE_FAILED",
            )

        if not self._sm.is_valid_transition(current.state, broker_view.state):
            err = SystemStateError(
                "Invalid broker refresh transition",
                module=_MODULE,
                details={
                    "client_order_id": str(client_order_id),
                    "intent_id": current.intent_id,
                    "from_state": current.state.value,
                    "to_state": broker_view.state.value,
                },
            )
            return Result.degraded(
                data=current,
                error=err,
                reason_code="INVALID_BROKER_TRANSITION",
            )

        ts_ns = int(time.time_ns())
        transition = self._sm.transition(
            base,
            broker_view.state,
            reason="BROKER_REFRESH",
            timestamp_ns=ts_ns,
            metadata={"broker_refresh": {"timestamp_ns": ts_ns}},
        )
        if transition.status is ResultStatus.FAILED or transition.data is None:
            err = transition.error or RuntimeError("state transition failed")
            return Result.failed(err, transition.reason_code or "STATE_TRANSITION_FAILED")

        updated = transition.data
        with self._lock:
            self._mappings[updated.client_order_id] = updated

        persist = self._safe_persist(updated)
        if persist.status is ResultStatus.SUCCESS:
            if broker_result.status is ResultStatus.DEGRADED:
                return Result.degraded(updated, broker_result.error or RuntimeError("degraded status"), broker_result.reason_code or "BROKER_REFRESH_DEGRADED")
            return Result.success(updated)

        return Result.degraded(
            data=updated,
            error=persist.error or RuntimeError("persistence failed"),
            reason_code=persist.reason_code or "PERSISTENCE_FAILED",
        )

    def cancel_all_open_orders(
        self,
        *,
        intent_types: list[str] | None = None,
    ) -> Result[dict[str, Any]]:
        """Cancel all open (non-terminal) orders, optionally filtered by intent type.

        Args:
            intent_types: If provided, only cancel orders whose intent_type
                matches one of these values (e.g. ``["OPEN_LONG", "OPEN_SHORT"]``).

        Returns:
            Result with a summary dict: ``{"cancelled": N, "failed": N, "skipped": N}``.
        """
        _TERMINAL = {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED}

        with self._lock:
            candidates = list(self._mappings.values())

        intent_types_set: set[str] | None = None
        if intent_types is not None:
            intent_types_set = {t.upper() for t in intent_types}

        cancelled = 0
        failed = 0
        skipped = 0

        for mapping in candidates:
            if mapping.state in _TERMINAL:
                skipped += 1
                continue
            if intent_types_set is not None:
                mapping_type = str(getattr(mapping, "intent_type", "") or "").upper()
                if mapping_type not in intent_types_set:
                    skipped += 1
                    continue

            result = self.cancel_order(mapping.client_order_id)
            if result.status is ResultStatus.SUCCESS or result.status is ResultStatus.DEGRADED:
                cancelled += 1
            else:
                failed += 1

        summary: dict[str, Any] = {
            "cancelled": cancelled,
            "failed": failed,
            "skipped": skipped,
        }

        if failed > 0:
            return Result.degraded(
                summary,
                RuntimeError(f"{failed} order(s) failed to cancel"),
                "PARTIAL_CANCEL",
            )
        return Result.success(summary)

    def get_all_orders(self) -> Result[list[IntentOrderMapping]]:
        """Return all tracked orders (cached)."""

        with self._lock:
            return Result.success(list(self._mappings.values()))

    def reconcile_startup(self) -> Result[ReconciliationResult]:
        """Perform startup reconciliation (best-effort, non-fatal on errors).

        Returns:
            Result[ReconciliationResult]: Reconciliation outcome, degraded when the
                broker snapshot cannot be obtained or persistence updates fail.
        """

        with self._lock:
            local_mappings = list(self._mappings.values())

        now_ns = int(time.time_ns())
        empty = ReconciliationResult(
            updated_mappings=[],
            orphaned_broker_orders=[],
            missing_broker_orders=[],
            state_transitions=[],
            reconciled_at_ns=now_ns,
        )

        if not self._adapter.is_connected():
            err = BrokerConnectionError(
                "Startup reconciliation skipped: broker not connected",
                module=_MODULE,
                reason_code="BROKER_NOT_CONNECTED",
            )
            return Result.degraded(empty, err, "STARTUP_RECONCILIATION_SKIPPED")

        try:
            broker_all = self._adapter.get_all_orders()
        except Exception as exc:  # noqa: BLE001 - adapter boundary protection
            return Result.degraded(empty, exc, "STARTUP_BROKER_FETCH_EXCEPTION")

        if broker_all.status is ResultStatus.FAILED:
            return Result.degraded(
                empty,
                broker_all.error or RuntimeError("broker snapshot failed"),
                broker_all.reason_code or "STARTUP_BROKER_FETCH_FAILED",
            )

        broker_orders = self._to_broker_orders_snapshot(broker_all.data or [])
        if not broker_orders and any(
            (not m.state.is_terminal() and m.state is not OrderState.PENDING) for m in local_mappings
        ):
            err = OperationalError(
                "Startup reconciliation skipped: broker snapshot is empty with in-flight local mappings",
                module=_MODULE,
                reason_code="EMPTY_BROKER_SNAPSHOT",
                details={"local_in_flight": True},
            )
            return Result.degraded(empty, err, "STARTUP_RECONCILIATION_SKIPPED")

        try:
            reconcile_result = self._reconciler.reconcile_startup(local_mappings, broker_orders)
        except Exception as exc:  # noqa: BLE001 - reconciliation boundary
            return Result.degraded(empty, exc, "STARTUP_RECONCILIATION_EXCEPTION")

        if reconcile_result.data is None:
            return Result.failed(
                reconcile_result.error or RuntimeError("startup reconciliation failed"),
                reconcile_result.reason_code or "STARTUP_RECONCILIATION_FAILED",
            )

        apply_result = self._apply_reconciliation_result(reconcile_result.data)
        if apply_result.status is ResultStatus.SUCCESS:
            return reconcile_result

        return Result.degraded(
            reconcile_result.data,
            apply_result.error or RuntimeError("persistence failed"),
            apply_result.reason_code or "PERSISTENCE_FAILED",
        )

    def reconcile_periodic(self) -> Result[ReconciliationResult]:
        """Perform periodic reconciliation with the broker.

        This method:
            - Fetches broker orders (best-effort, mainly for observability).
            - Runs reconciler.periodic reconciliation for in-flight mappings.
            - Applies transitions and persists updated mappings.

        Returns:
            Result[ReconciliationResult]: Reconciliation outcome; degraded on
                partial failures.
        """

        with self._lock:
            in_flight = [m for m in self._mappings.values() if not m.state.is_terminal()]

        now_ns = int(time.time_ns())
        empty = ReconciliationResult(
            updated_mappings=[],
            orphaned_broker_orders=[],
            missing_broker_orders=[],
            state_transitions=[],
            reconciled_at_ns=now_ns,
        )

        # Requirement: fetch all broker orders as part of periodic reconciliation.
        try:
            broker_all = self._adapter.get_all_orders()
            if broker_all.status is ResultStatus.FAILED:
                _LOGGER.warning(
                    "Periodic broker snapshot fetch failed (reason_code=%s, error=%s)",
                    broker_all.reason_code,
                    repr(broker_all.error),
                )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Periodic broker snapshot fetch raised: %s", repr(exc))

        try:
            reconcile_result = self._reconciler.reconcile_periodic(
                in_flight,
                timeout_threshold_ns=self._DEFAULT_TIMEOUT_THRESHOLD_NS,
            )
        except Exception as exc:  # noqa: BLE001 - reconciliation boundary
            _LOGGER.exception("Periodic reconciliation crashed: %s", repr(exc))
            return Result.degraded(empty, exc, "PERIODIC_RECONCILIATION_EXCEPTION")

        if reconcile_result.status is ResultStatus.FAILED or reconcile_result.data is None:
            err = reconcile_result.error or RuntimeError("periodic reconciliation failed")
            _LOGGER.warning(
                "Periodic reconciliation failed (reason_code=%s, error=%s)",
                reconcile_result.reason_code,
                repr(err),
            )
            return Result.degraded(empty, err, reconcile_result.reason_code or "PERIODIC_RECONCILIATION_FAILED")

        apply_result = self._apply_reconciliation_result(reconcile_result.data)
        if apply_result.status is ResultStatus.SUCCESS:
            return reconcile_result

        return Result.degraded(
            reconcile_result.data,
            apply_result.error or RuntimeError("persistence failed"),
            apply_result.reason_code or "PERSISTENCE_FAILED",
        )

    def generate_execution_report(self, client_order_id: str) -> Result[ExecutionReport]:
        """Generate an ExecutionReport for a terminal order.

        Conditions:
            The order must be in a terminal state (FILLED/CANCELLED/REJECTED/EXPIRED).

        Flow:
            1) Load mapping and verify terminal state.
            2) Collect fills (best-effort from ``adapter._fills``).
            3) Compute filled/remaining quantities and average fill price.
            4) Compute total commissions (when available).
            5) Collect state transitions for this intent (best-effort).

        Args:
            client_order_id: Client order id for which to generate a report.

        Returns:
            Result[ExecutionReport]: Execution report payload.
        """

        with self._lock:
            mapping = self._mappings.get(str(client_order_id))

        if mapping is None:
            err = OperationalError(
                f"Order not found: {client_order_id}",
                module=_MODULE,
                reason_code="ORDER_NOT_FOUND",
                details={"client_order_id": str(client_order_id)},
            )
            return Result.failed(err, err.reason_code)

        if not mapping.state.is_terminal():
            err = OperationalError(
                "Execution report requires a terminal order state",
                module=_MODULE,
                reason_code="ORDER_NOT_TERMINAL",
                details={"client_order_id": mapping.client_order_id, "state": mapping.state.value},
            )
            return Result.failed(err, err.reason_code)

        fills: list[FillDetail] = []
        try:
            if hasattr(self._adapter, "get_fills") and callable(getattr(self._adapter, "get_fills", None)):
                raw_fills = self._adapter.get_fills(mapping.client_order_id)
                if isinstance(raw_fills, list):
                    fills = [f for f in raw_fills if isinstance(f, FillDetail)]
            else:
                adapter_fills = getattr(self._adapter, "_fills", None)
                if isinstance(adapter_fills, dict):
                    raw = adapter_fills.get(mapping.client_order_id, [])
                    if isinstance(raw, list):
                        fills = [f for f in raw if isinstance(f, FillDetail)]
        except Exception as exc:  # noqa: BLE001 - best-effort
            _LOGGER.warning("Failed to read adapter fills: %s", repr(exc))

        filled_qty = float(sum(float(f.fill_quantity) for f in fills))
        intended_qty = float(getattr(mapping.intent_snapshot, "quantity", 0.0) or 0.0)
        remaining_qty = max(0.0, intended_qty - filled_qty)

        avg_price: float | None
        if filled_qty > 0:
            notional = sum(float(f.fill_quantity) * float(f.fill_price) for f in fills)
            avg_price = float(notional / filled_qty)
        else:
            avg_price = None

        commissions: float | None
        if any(f.commission is not None for f in fills):
            commissions = float(sum(float(f.commission or 0.0) for f in fills))
        else:
            commissions = None

        transitions: list[StateTransition] = []
        try:
            log = getattr(self._sm, "transition_log", ())
            if isinstance(log, (tuple, list)):
                transitions = [t for t in log if isinstance(t, StateTransition) and t.intent_id == mapping.intent_id]
                transitions.sort(key=lambda t: int(t.timestamp_ns))
        except Exception as exc:  # noqa: BLE001 - best-effort
            _LOGGER.warning("Failed to collect state transitions: %s", repr(exc))

        run_id = self._extract_run_id(mapping.intent_snapshot)
        broker_order_id = str(mapping.broker_order_id or "unknown")
        symbol = str(getattr(mapping.intent_snapshot, "symbol", "")).strip().upper() or "UNKNOWN"
        executed_at_ns = int(time.time_ns())

        report = ExecutionReport(
            run_id=run_id,
            intent_id=mapping.intent_id,
            client_order_id=mapping.client_order_id,
            broker_order_id=broker_order_id,
            symbol=symbol,
            final_state=mapping.state,
            filled_quantity=filled_qty,
            remaining_quantity=remaining_qty,
            avg_fill_price=avg_price,
            commissions=commissions,
            fills=list(fills),
            state_transitions=transitions,
            executed_at_ns=executed_at_ns,
            metadata={"mapping_metadata": dict(mapping.metadata)},
        )
        return Result.success(report)

    def _apply_transitions(self, transitions: list[StateTransition]) -> Result[None]:
        """Apply provided transitions to cached mappings and persist (best-effort).

        This method is designed for reconciliation integration where transitions
        are provided as an audit trail. To avoid double-recording transitions in
        a shared ``StateMachine``, it updates mappings directly (immutable replace)
        after validating transition legality.

        Args:
            transitions: Ordered state transitions to apply.

        Returns:
            Result[None]: Success when all transitions were applied/persisted;
                degraded when some transitions could not be applied/persisted.
        """

        updated: list[IntentOrderMapping] = []
        invalid_count = 0
        first_error: BaseException | None = None

        with self._lock:
            by_intent: dict[str, str] = {m.intent_id: cid for cid, m in self._mappings.items()}

            for transition in transitions:
                client_id = by_intent.get(transition.intent_id)
                if client_id is None:
                    continue
                current = self._mappings.get(client_id)
                if current is None:
                    continue

                if current.state == transition.to_state:
                    continue

                if not self._sm.is_valid_transition(current.state, transition.to_state):
                    invalid_count += 1
                    if first_error is None:
                        first_error = SystemStateError(
                            "Invalid transition during reconciliation apply",
                            module=_MODULE,
                            details={
                                "intent_id": transition.intent_id,
                                "from_state": current.state.value,
                                "to_state": transition.to_state.value,
                            },
                        )
                    continue

                merged_metadata = dict(current.metadata)
                merged_metadata.update(dict(transition.metadata or {}))

                new_mapping = replace(
                    current,
                    state=transition.to_state,
                    updated_at_ns=int(transition.timestamp_ns),
                    metadata=merged_metadata,
                )
                self._mappings[client_id] = new_mapping
                updated.append(new_mapping)

        persist_errors = 0
        for mapping in updated:
            persist_result = self._safe_persist(mapping)
            if persist_result.status is ResultStatus.FAILED:
                persist_errors += 1
                if first_error is None:
                    first_error = persist_result.error

        if invalid_count or persist_errors:
            err = first_error or RuntimeError("reconciliation apply had failures")
            return Result.degraded(
                data=None,
                error=err,
                reason_code="RECONCILIATION_APPLY_PARTIAL",
            )

        return Result.success(data=None)

    def _apply_reconciliation_result(self, result: ReconciliationResult) -> Result[None]:
        """Apply reconciliation updates to mappings and persist (best-effort)."""

        to_persist: list[IntentOrderMapping] = []
        updated_intents: set[str] = set()
        with self._lock:
            for mapping in result.updated_mappings:
                self._mappings[mapping.client_order_id] = mapping
                to_persist.append(mapping)
                updated_intents.add(mapping.intent_id)

        # Persist updated mappings.
        first_error: BaseException | None = None
        error_count = 0
        for mapping in to_persist:
            persist_result = self._safe_persist(mapping)
            if persist_result.status is ResultStatus.FAILED:
                error_count += 1
                if first_error is None:
                    first_error = persist_result.error

        # Apply the audit transitions to keep cached mappings consistent if the
        # updated_mappings set is incomplete (should be rare).
        transitions = [t for t in list(result.state_transitions or []) if t.intent_id not in updated_intents]
        transitions_apply = self._apply_transitions(transitions)
        if transitions_apply.status is ResultStatus.DEGRADED:
            error_count += 1
            if first_error is None:
                first_error = transitions_apply.error

        if error_count:
            return Result.degraded(
                data=None,
                error=first_error or RuntimeError("reconciliation apply degraded"),
                reason_code="RECONCILIATION_APPLY_DEGRADED",
            )

        return Result.success(data=None)

    def _safe_persist(self, mapping: IntentOrderMapping) -> Result[None]:
        """Persist a mapping, converting unexpected exceptions into Result."""

        try:
            return self._persistence.save(mapping)
        except Exception as exc:  # noqa: BLE001 - persistence boundary protection
            return Result.failed(exc, "PERSISTENCE_EXCEPTION")

    def _to_broker_orders_snapshot(self, broker_mappings: list[IntentOrderMapping]) -> list[dict[str, Any]]:
        """Convert adapter mapping views into reconciler broker-order dict schema."""

        snapshot: list[dict[str, Any]] = []
        for mapping in broker_mappings:
            snapshot.append(
                {
                    "intent_id": mapping.intent_id,
                    "client_order_id": mapping.client_order_id,
                    "broker_order_id": mapping.broker_order_id,
                    "status": mapping.state.value,
                    "metadata": dict(mapping.metadata),
                }
            )
        return snapshot

    def _extract_run_id(self, intent: TradeIntent) -> str:
        """Best-effort extraction of run_id from intent metadata."""

        meta = getattr(intent, "metadata", None)
        if isinstance(meta, dict):
            run_id = meta.get("run_id")
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
        return "unknown"
