"""Reconciliation logic for the Order State Machine (Pattern 6, Phase 3.4).

Implements:
- Startup reconciliation: local mappings vs broker orders snapshot.
- Periodic reconciliation: in-flight timeout checks + targeted broker queries.
- Idempotency: trade_id dedup + event history + replay-safe state corrections.

Broker schema is intentionally flexible (dict-based). This module attempts to
extract the following fields from broker order dicts:
- broker_order_id (aka order_id / id / venue_order_id)
- client_order_id (aka clientOrderId / cl_ord_id)
- status/state (aka status / state / order_status)
- fills/trades/executions with trade_id (aka id / execution_id)

The reconciler never persists; it returns updated mappings + transitions for
callers to persist/publish.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from msgspec.structs import replace

from common.exceptions import OperationalError, PartialDataError, SystemStateError
from common.interface import Result, ResultStatus
from order_state_machine.interface import (
    IntentOrderMapping,
    OrderState,
    ReconciliationProtocol,
    ReconciliationResult,
    StateMachineProtocol,
    StateTransition,
)

__all__ = ["Reconciler"]

_MODULE = "order_state_machine.reconciliation"

_BROKER_ID_KEYS: tuple[str, ...] = ("broker_order_id", "order_id", "id", "venue_order_id", "brokerId")
_CLIENT_ID_KEYS: tuple[str, ...] = (
    "client_order_id",
    "clientOrderId",
    "client_orderid",
    "client_id",
    "cl_ord_id",
    "clOrdId",
)
_STATUS_KEYS: tuple[str, ...] = ("status", "state", "order_status", "orderState", "order_state")

_FILLS_KEYS: tuple[str, ...] = ("fills", "trades", "executions")
_TRADE_ID_KEYS: tuple[str, ...] = ("trade_id", "tradeId", "execution_id", "executionId", "id")

_EVENT_HISTORY_KEY = "event_history"
_PROCESSED_TRADE_IDS_KEY = "processed_trade_ids"


BrokerFetchOrder: type = Callable[[IntentOrderMapping], Result[dict[str, Any] | None]]


@dataclass(frozen=True, slots=True)
class _ReconcileOutcome:
    mapping: IntentOrderMapping
    transitions: list[StateTransition]
    changed: bool


class Reconciler(ReconciliationProtocol):
    """Default reconciler implementing Pattern 6.

    Args:
        state_machine: State machine used to apply validated transitions.
        fetch_broker_order: Optional callback used by periodic reconciliation to
            query broker truth for a single mapping.
    """

    def __init__(
        self,
        state_machine: StateMachineProtocol,
        *,
        fetch_broker_order: BrokerFetchOrder | None = None,
    ) -> None:
        self._sm = state_machine
        self._fetch_broker_order = fetch_broker_order

    def reconcile_startup(
        self,
        local_mappings: list[IntentOrderMapping],
        broker_orders: list[dict[str, Any]],
        *,
        timestamp_ns: int | None = None,
    ) -> Result[ReconciliationResult]:
        now_ns = self._now_ns(timestamp_ns)

        updated_by_intent: dict[str, IntentOrderMapping] = {}
        transitions: list[StateTransition] = []
        orphaned_broker_orders: list[dict[str, Any]] = []
        missing_broker_orders: list[str] = []

        local_by_broker_id: dict[str, IntentOrderMapping] = {}
        local_by_client_id: dict[str, IntentOrderMapping] = {}
        for mapping in local_mappings:
            if mapping.broker_order_id:
                local_by_broker_id[str(mapping.broker_order_id)] = mapping
            local_by_client_id[str(mapping.client_order_id)] = mapping

        matched_intent_ids: set[str] = set()

        first_error: BaseException | None = None
        error_count = 0

        for broker_order in broker_orders:
            try:
                match = self._match_broker_order(
                    broker_order,
                    local_by_broker_id=local_by_broker_id,
                    local_by_client_id=local_by_client_id,
                )
                if match is None:
                    orphaned_broker_orders.append(dict(broker_order))
                    continue

                mapping, match_info = match
                matched_intent_ids.add(mapping.intent_id)

                outcome = self._reconcile_mapping_against_broker(
                    mapping,
                    broker_order,
                    now_ns=now_ns,
                    reason="STARTUP_RECONCILIATION",
                    match_info=match_info,
                )
                if outcome.changed:
                    updated_by_intent[outcome.mapping.intent_id] = outcome.mapping
                transitions.extend(outcome.transitions)
            except Exception as exc:  # noqa: BLE001 - reconciliation is a boundary
                error_count += 1
                if first_error is None:
                    first_error = exc

        # Detect local mappings missing from broker snapshot (non-terminal only).
        broker_key_set = self._broker_presence_keys(broker_orders)
        for mapping in local_mappings:
            if mapping.state.is_terminal() or mapping.state is OrderState.PENDING:
                continue
            if mapping.intent_id in matched_intent_ids:
                continue
            if not self._mapping_present_in_broker_keys(mapping, broker_key_set):
                missing_broker_orders.append(mapping.intent_id)
                try:
                    outcome = self._force_terminal_when_missing(
                        mapping,
                        now_ns=now_ns,
                        reason="STARTUP_MISSING_FROM_BROKER",
                    )
                    if outcome.changed:
                        updated_by_intent[outcome.mapping.intent_id] = outcome.mapping
                    transitions.extend(outcome.transitions)
                except Exception as exc:  # noqa: BLE001
                    error_count += 1
                    if first_error is None:
                        first_error = exc

        result = ReconciliationResult(
            updated_mappings=list(updated_by_intent.values()),
            orphaned_broker_orders=orphaned_broker_orders,
            missing_broker_orders=missing_broker_orders,
            state_transitions=transitions,
            reconciled_at_ns=now_ns,
        )

        if error_count:
            err = PartialDataError(
                "Startup reconciliation completed with errors",
                module=_MODULE,
                reason_code="STARTUP_RECONCILIATION_PARTIAL",
                details={"error_count": error_count, "first_error": repr(first_error)},
                original_error=first_error,
            )
            return Result.degraded(result, err, "STARTUP_RECONCILIATION_PARTIAL")

        return Result.success(result)

    def reconcile_periodic(
        self,
        mappings_in_flight: list[IntentOrderMapping],
        timeout_threshold_ns: int,
        *,
        timestamp_ns: int | None = None,
    ) -> Result[ReconciliationResult]:
        now_ns = self._now_ns(timestamp_ns)

        updated_by_intent: dict[str, IntentOrderMapping] = {}
        transitions: list[StateTransition] = []
        orphaned_broker_orders: list[dict[str, Any]] = []
        missing_broker_orders: list[str] = []

        first_error: BaseException | None = None
        error_count = 0

        for mapping in mappings_in_flight:
            if mapping.state.is_terminal():
                continue

            should_query = False
            if mapping.state in (OrderState.SUBMITTED, OrderState.ACCEPTED):
                should_query = self._is_timed_out(mapping, timeout_threshold_ns, now_ns=now_ns)
            elif mapping.state is OrderState.PARTIALLY_FILLED:
                should_query = True

            if not should_query:
                continue

            if self._fetch_broker_order is None:
                error_count += 1
                if first_error is None:
                    first_error = OperationalError(
                        "Periodic reconciliation requires fetch_broker_order callback",
                        module=_MODULE,
                        reason_code="BROKER_QUERY_NOT_CONFIGURED",
                        details={"intent_id": mapping.intent_id},
                    )
                continue

            broker_result = self._fetch_broker_order(mapping)
            if broker_result.status is ResultStatus.FAILED:
                error_count += 1
                if first_error is None:
                    first_error = broker_result.error or RuntimeError("broker query failed")
                continue

            broker_order = broker_result.data
            if broker_order is None:
                missing_broker_orders.append(mapping.intent_id)
                try:
                    outcome = self._force_terminal_when_missing(
                        mapping,
                        now_ns=now_ns,
                        reason="PERIODIC_MISSING_FROM_BROKER",
                    )
                    if outcome.changed:
                        updated_by_intent[outcome.mapping.intent_id] = outcome.mapping
                    transitions.extend(outcome.transitions)
                except Exception as exc:  # noqa: BLE001
                    error_count += 1
                    if first_error is None:
                        first_error = exc
                continue

            try:
                outcome = self._reconcile_mapping_against_broker(
                    mapping,
                    broker_order,
                    now_ns=now_ns,
                    reason="PERIODIC_RECONCILIATION",
                    match_info={"match": "targeted_query"},
                )
                if outcome.changed:
                    updated_by_intent[outcome.mapping.intent_id] = outcome.mapping
                transitions.extend(outcome.transitions)
            except Exception as exc:  # noqa: BLE001
                error_count += 1
                if first_error is None:
                    first_error = exc

        result = ReconciliationResult(
            updated_mappings=list(updated_by_intent.values()),
            orphaned_broker_orders=orphaned_broker_orders,
            missing_broker_orders=missing_broker_orders,
            state_transitions=transitions,
            reconciled_at_ns=now_ns,
        )

        if error_count:
            err = PartialDataError(
                "Periodic reconciliation completed with errors",
                module=_MODULE,
                reason_code="PERIODIC_RECONCILIATION_PARTIAL",
                details={"error_count": error_count, "first_error": repr(first_error)},
                original_error=first_error,
            )
            return Result.degraded(result, err, "PERIODIC_RECONCILIATION_PARTIAL")

        return Result.success(result)

    def _match_broker_order(
        self,
        broker_order: Mapping[str, Any],
        *,
        local_by_broker_id: Mapping[str, IntentOrderMapping],
        local_by_client_id: Mapping[str, IntentOrderMapping],
    ) -> tuple[IntentOrderMapping, dict[str, Any]] | None:
        broker_order_id = self._get_first_str(broker_order, _BROKER_ID_KEYS)
        if broker_order_id:
            mapping = local_by_broker_id.get(broker_order_id)
            if mapping is not None:
                return mapping, {"match": "broker_order_id", "broker_order_id": broker_order_id}

        client_order_id = self._get_first_str(broker_order, _CLIENT_ID_KEYS)
        if client_order_id:
            mapping = local_by_client_id.get(client_order_id)
            if mapping is not None:
                return mapping, {"match": "client_order_id", "client_order_id": client_order_id}

        return None

    def _reconcile_mapping_against_broker(
        self,
        mapping: IntentOrderMapping,
        broker_order: Mapping[str, Any],
        *,
        now_ns: int,
        reason: str,
        match_info: Mapping[str, Any],
    ) -> _ReconcileOutcome:
        transitions: list[StateTransition] = []
        changed = False

        normalized_state = self._normalize_broker_state(broker_order)
        if normalized_state is None:
            return _ReconcileOutcome(mapping=mapping, transitions=[], changed=False)

        mapping_with_ids = self._ensure_broker_ids(mapping, broker_order, now_ns=now_ns)
        if mapping_with_ids != mapping:
            mapping = mapping_with_ids
            changed = True

        fill_outcome = self._reconcile_fills(mapping, broker_order, now_ns=now_ns, reason=reason)
        mapping = fill_outcome.mapping
        if fill_outcome.changed:
            changed = True

        if mapping.state != normalized_state:
            reconcile_meta = {"reconciliation": {"reason": reason, "broker_state": normalized_state.value, **match_info}}
            transition_outcome = self._transition_to_target_state(
                mapping,
                normalized_state,
                now_ns=now_ns,
                reason=reason,
                base_metadata=reconcile_meta,
            )
            mapping = transition_outcome.mapping
            if transition_outcome.changed:
                changed = True
            transitions.extend(transition_outcome.transitions)

        return _ReconcileOutcome(mapping=mapping, transitions=transitions, changed=changed)

    def _reconcile_fills(
        self,
        mapping: IntentOrderMapping,
        broker_order: Mapping[str, Any],
        *,
        now_ns: int,
        reason: str,
    ) -> _ReconcileOutcome:
        fills = self._extract_fills(broker_order)
        if not fills:
            return _ReconcileOutcome(mapping=mapping, transitions=[], changed=False)

        processed = self._processed_trade_ids(mapping)
        newly_processed: list[str] = []
        new_fill_payloads: list[dict[str, Any]] = []

        for fill in fills:
            trade_id = self._get_first_str(fill, _TRADE_ID_KEYS)
            if not trade_id:
                continue
            if trade_id in processed:
                continue
            processed.add(trade_id)
            newly_processed.append(trade_id)
            new_fill_payloads.append(dict(fill))

        if not newly_processed:
            return _ReconcileOutcome(mapping=mapping, transitions=[], changed=False)

        meta = {
            _PROCESSED_TRADE_IDS_KEY: sorted(processed),
            "reconciliation_fills": {
                "reason": reason,
                "new_trade_ids": newly_processed,
                "new_fills": new_fill_payloads,
            },
        }
        updated = self._with_metadata(mapping, meta, now_ns=now_ns)
        return _ReconcileOutcome(mapping=updated, transitions=[], changed=(updated != mapping))

    def _transition_to_target_state(
        self,
        mapping: IntentOrderMapping,
        target_state: OrderState,
        *,
        now_ns: int,
        reason: str,
        base_metadata: Mapping[str, Any] | None = None,
    ) -> _ReconcileOutcome:
        if mapping.state == target_state:
            return _ReconcileOutcome(mapping=mapping, transitions=[], changed=False)

        path = self._find_transition_path(mapping.state, target_state)
        if path is None:
            err = SystemStateError(
                "No valid transition path for reconciliation",
                module=_MODULE,
                reason_code="RECONCILIATION_NO_PATH",
                details={
                    "intent_id": mapping.intent_id,
                    "from_state": mapping.state.value,
                    "to_state": target_state.value,
                },
            )
            raise err

        transitions: list[StateTransition] = []
        changed = False
        current = mapping
        for next_state in path:
            step_meta = dict(base_metadata or {})
            step_out = self._apply_transition(
                current,
                next_state,
                now_ns=now_ns,
                reason=f"{reason}:{next_state.value}",
                metadata=step_meta,
            )
            current = step_out.mapping
            if step_out.changed:
                changed = True
            transitions.extend(step_out.transitions)

        return _ReconcileOutcome(mapping=current, transitions=transitions, changed=changed)

    def _apply_transition(
        self,
        mapping: IntentOrderMapping,
        new_state: OrderState,
        *,
        now_ns: int,
        reason: str,
        metadata: Mapping[str, Any] | None,
    ) -> _ReconcileOutcome:
        if mapping.state == new_state:
            return _ReconcileOutcome(mapping=mapping, transitions=[], changed=False)

        transition_record = self._generate_state_transition(
            mapping,
            new_state,
            timestamp_ns=now_ns,
            reason=reason,
            metadata=dict(metadata or {}),
        )

        merged_metadata = dict(metadata or {})
        merged_metadata[_EVENT_HISTORY_KEY] = self._append_event_history(mapping, transition_record)

        result = self._sm.transition(
            mapping,
            new_state,
            reason=reason,
            timestamp_ns=now_ns,
            metadata=merged_metadata,
        )
        if result.status is ResultStatus.FAILED or result.data is None:
            error = result.error or RuntimeError("state transition failed")
            raise OperationalError(
                "Failed to apply reconciliation transition",
                module=_MODULE,
                reason_code=result.reason_code or "RECONCILIATION_TRANSITION_FAILED",
                details={
                    "intent_id": mapping.intent_id,
                    "from_state": mapping.state.value,
                    "to_state": new_state.value,
                    "error": repr(error),
                },
                original_error=error,
            )

        updated = result.data
        return _ReconcileOutcome(mapping=updated, transitions=[transition_record], changed=True)

    def _force_terminal_when_missing(
        self,
        mapping: IntentOrderMapping,
        *,
        now_ns: int,
        reason: str,
    ) -> _ReconcileOutcome:
        if mapping.state.is_terminal() or mapping.state is OrderState.PENDING:
            return _ReconcileOutcome(mapping=mapping, transitions=[], changed=False)

        meta = {"reconciliation": {"reason": reason, "note": "not_found_in_broker"}}
        return self._transition_to_target_state(
            mapping,
            OrderState.EXPIRED,
            now_ns=now_ns,
            reason=reason,
            base_metadata=meta,
        )

    def _ensure_broker_ids(
        self,
        mapping: IntentOrderMapping,
        broker_order: Mapping[str, Any],
        *,
        now_ns: int,
    ) -> IntentOrderMapping:
        if mapping.broker_order_id:
            return mapping

        broker_id = self._get_first_str(broker_order, _BROKER_ID_KEYS)
        if not broker_id:
            return mapping

        meta = {"reconciliation": {"broker_order_id": broker_id}}
        return self._with_metadata(
            replace(mapping, broker_order_id=broker_id),
            meta,
            now_ns=now_ns,
        )

    def _with_metadata(self, mapping: IntentOrderMapping, delta: Mapping[str, Any], *, now_ns: int) -> IntentOrderMapping:
        if not delta:
            return mapping
        merged = dict(mapping.metadata)
        changed = False
        for k, v in delta.items():
            if merged.get(k) != v:
                merged[k] = v
                changed = True
        if not changed:
            return mapping
        return replace(mapping, metadata=merged, updated_at_ns=now_ns)

    def _processed_trade_ids(self, mapping: IntentOrderMapping) -> set[str]:
        raw = mapping.metadata.get(_PROCESSED_TRADE_IDS_KEY, [])
        if isinstance(raw, (set, frozenset)):
            return set(str(x) for x in raw)
        if isinstance(raw, list | tuple):
            return set(str(x) for x in raw)
        return set()

    def _append_event_history(self, mapping: IntentOrderMapping, transition: StateTransition) -> list[dict[str, Any]]:
        raw = mapping.metadata.get(_EVENT_HISTORY_KEY, [])
        history: list[dict[str, Any]]
        if isinstance(raw, list):
            history = [dict(x) if isinstance(x, dict) else {"event": str(x)} for x in raw]
        else:
            history = []

        entry = {
            "intent_id": transition.intent_id,
            "from_state": transition.from_state.value,
            "to_state": transition.to_state.value,
            "timestamp_ns": transition.timestamp_ns,
            "reason": transition.reason,
            "metadata": dict(transition.metadata),
        }
        history.append(entry)
        return history

    def _generate_state_transition(
        self,
        mapping: IntentOrderMapping,
        new_state: OrderState,
        *,
        timestamp_ns: int,
        reason: str,
        metadata: dict[str, Any],
    ) -> StateTransition:
        return StateTransition(
            intent_id=mapping.intent_id,
            from_state=mapping.state,
            to_state=new_state,
            timestamp_ns=int(timestamp_ns),
            reason=str(reason),
            metadata=dict(metadata),
        )

    def _is_timed_out(self, mapping: IntentOrderMapping, timeout_threshold_ns: int, *, now_ns: int) -> bool:
        try:
            elapsed = int(now_ns) - int(mapping.updated_at_ns)
            return elapsed >= int(timeout_threshold_ns)
        except Exception:  # noqa: BLE001
            return True

    def _normalize_broker_state(self, broker_order: Mapping[str, Any]) -> OrderState | None:
        raw = self._get_first_any(broker_order, _STATUS_KEYS)
        if raw is None:
            return None

        value = getattr(raw, "value", raw)
        s = str(value).strip().upper()
        if not s:
            return None

        synonyms: dict[str, OrderState] = {
            "PENDING": OrderState.PENDING,
            "NEW": OrderState.SUBMITTED,
            "SUBMITTED": OrderState.SUBMITTED,
            "SENT": OrderState.SUBMITTED,
            "ACK": OrderState.ACCEPTED,
            "ACKNOWLEDGED": OrderState.ACCEPTED,
            "ACCEPTED": OrderState.ACCEPTED,
            "OPEN": OrderState.ACCEPTED,
            "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
            "PARTIAL": OrderState.PARTIALLY_FILLED,
            "FILLED": OrderState.FILLED,
            "DONE": OrderState.FILLED,
            "CANCELLED": OrderState.CANCELLED,
            "CANCELED": OrderState.CANCELLED,
            "CANCELLED_BY_USER": OrderState.CANCELLED,
            "REJECTED": OrderState.REJECTED,
            "EXPIRED": OrderState.EXPIRED,
        }
        if s in synonyms:
            return synonyms[s]

        try:
            return OrderState[s]
        except Exception:  # noqa: BLE001
            return None

    def _extract_fills(self, broker_order: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        for key in _FILLS_KEYS:
            value = broker_order.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, Mapping)]
        return []

    def _find_transition_path(self, from_state: OrderState, to_state: OrderState) -> list[OrderState] | None:
        if from_state == to_state:
            return []

        # Build adjacency by probing the state machine for allowed transitions.
        states = list(OrderState)
        adjacency: dict[OrderState, list[OrderState]] = {s: [] for s in states}
        for a in states:
            for b in states:
                if a == b:
                    continue
                if self._sm.is_valid_transition(a, b):
                    adjacency[a].append(b)

        visited: set[OrderState] = {from_state}
        q: deque[tuple[OrderState, list[OrderState]]] = deque([(from_state, [])])

        while q:
            state, path = q.popleft()
            for nxt in adjacency.get(state, []):
                if nxt in visited:
                    continue
                new_path = [*path, nxt]
                if nxt == to_state:
                    return new_path
                visited.add(nxt)
                q.append((nxt, new_path))

        return None

    def _broker_presence_keys(self, broker_orders: Sequence[Mapping[str, Any]]) -> set[str]:
        keys: set[str] = set()
        for order in broker_orders:
            b = self._get_first_str(order, _BROKER_ID_KEYS)
            c = self._get_first_str(order, _CLIENT_ID_KEYS)
            if b:
                keys.add(f"broker:{b}")
            if c:
                keys.add(f"client:{c}")
        return keys

    def _mapping_present_in_broker_keys(self, mapping: IntentOrderMapping, broker_keys: set[str]) -> bool:
        if mapping.broker_order_id and f"broker:{mapping.broker_order_id}" in broker_keys:
            return True
        if f"client:{mapping.client_order_id}" in broker_keys:
            return True
        return False

    def _get_first_str(self, obj: Mapping[str, Any], keys: Iterable[str]) -> str | None:
        value = self._get_first_any(obj, keys)
        if value is None:
            return None
        s = str(getattr(value, "value", value)).strip()
        return s or None

    def _get_first_any(self, obj: Mapping[str, Any], keys: Iterable[str]) -> Any | None:
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return None

    def _now_ns(self, timestamp_ns: int | None) -> int:
        return int(time.time_ns() if timestamp_ns is None else timestamp_ns)
