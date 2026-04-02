"""Order State Machine interface contracts for Phase 3.4.

This module defines the public schemas and protocols for AST's Order State Machine.
The Order State Machine manages the intent-to-order lifecycle, including:

- Intent ID generation (deterministic)
- Intent-to-order mapping persistence
- Order state tracking (PENDING → SUBMITTED → FILLED/CANCELLED/REJECTED)
- Reconciliation logic (idempotent)
- Duplicate prevention

All schemas follow repo-wide conventions:

- Immutable value objects built with ``msgspec.Struct(frozen=True, kw_only=True)``.
- Stable, machine-readable state codes.
- Journal-friendly snapshot schemas inherit ``journal.interface.SnapshotBase``.
- Interfaces return ``common.interface.Result[T]`` to capture success/degraded/fail outcomes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

import msgspec

from common.interface import Result
from journal.interface import SnapshotBase

if TYPE_CHECKING:
    from strategy.interface import TradeIntent
else:  # pragma: no cover - type-only imports may not exist in early phases.
    TradeIntent = Any  # type: ignore[assignment]

__all__ = [
    "OrderState",
    "IntentOrderMapping",
    "IntentOrderMappingSet",
    "ReconciliationResult",
    "StateTransition",
    "IDGeneratorProtocol",
    "StateMachineProtocol",
    "PersistenceProtocol",
    "ReconciliationProtocol",
]


class OrderState(str, Enum):
    """Order state in the intent-to-order lifecycle.

    Values:
        PENDING: Intent approved by Risk Gate, not yet submitted to broker.
        SUBMITTED: Submitted to broker, awaiting acknowledgment.
        ACCEPTED: Broker acknowledged, order is active.
        PARTIALLY_FILLED: Partial execution.
        FILLED: Fully executed (terminal state).
        CANCELLED: Cancelled by user/system (terminal state).
        REJECTED: Rejected by broker (terminal state).
        EXPIRED: Expired by time/condition (terminal state).
    """

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    @classmethod
    def terminal_states(cls) -> set[OrderState]:
        """Return the set of terminal states."""
        return {cls.FILLED, cls.CANCELLED, cls.REJECTED, cls.EXPIRED}

    def is_terminal(self) -> bool:
        """Return True if this state is terminal."""
        return self in self.terminal_states()


class IntentOrderMapping(msgspec.Struct, frozen=True, kw_only=True):
    """Intent-to-order mapping with state tracking.

    This structure represents the mapping between a strategy intent and its
    corresponding broker order, tracking the lifecycle state and metadata.

    Attributes:
        intent_id: Deterministic intent ID (hash-based).
        client_order_id: Generated order ID for broker submission.
        broker_order_id: Venue/broker order ID (filled after submission).
        state: Current order state.
        created_at_ns: Nanosecond Unix epoch timestamp when mapping was created.
        updated_at_ns: Nanosecond Unix epoch timestamp of last state transition.
        intent_snapshot: Original intent for replay/audit.
        metadata: Broker-specific fields (e.g., commission, fees, fill details).
    """

    intent_id: str
    client_order_id: str
    broker_order_id: str | None = msgspec.field(default=None)
    state: OrderState
    created_at_ns: int
    updated_at_ns: int
    intent_snapshot: TradeIntent
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


class IntentOrderMappingSet(SnapshotBase, frozen=True, kw_only=True):
    """Journal snapshot containing all intent-order mappings.

    Attributes:
        schema_version: See ``journal.interface.SnapshotBase``.
        system_version: See ``journal.interface.SnapshotBase``.
        asof_timestamp: See ``journal.interface.SnapshotBase``.

        mappings: List of all intent-order mappings.
        run_id: Run ID associated with this snapshot.
    """

    mappings: list[IntentOrderMapping] = msgspec.field(default_factory=list)
    run_id: str


class StateTransition(msgspec.Struct, frozen=True, kw_only=True):
    """Record of a state transition for audit/replay.

    Attributes:
        intent_id: Intent ID this transition applies to.
        from_state: Previous state.
        to_state: New state.
        timestamp_ns: Nanosecond Unix epoch timestamp of transition.
        reason: Human-readable reason for transition.
        metadata: Additional transition details.
    """

    intent_id: str
    from_state: OrderState
    to_state: OrderState
    timestamp_ns: int
    reason: str
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


class ReconciliationResult(msgspec.Struct, frozen=True, kw_only=True):
    """Result of reconciliation operation.

    Attributes:
        updated_mappings: Mappings that were updated during reconciliation.
        orphaned_broker_orders: Broker orders not found in local mappings.
        missing_broker_orders: Local mappings not found in broker.
        state_transitions: List of state transitions applied.
        reconciled_at_ns: Nanosecond Unix epoch timestamp of reconciliation.
    """

    updated_mappings: list[IntentOrderMapping] = msgspec.field(default_factory=list)
    orphaned_broker_orders: list[dict[str, Any]] = msgspec.field(default_factory=list)
    missing_broker_orders: list[str] = msgspec.field(default_factory=list)
    state_transitions: list[StateTransition] = msgspec.field(default_factory=list)
    reconciled_at_ns: int


@runtime_checkable
class IDGeneratorProtocol(Protocol):
    """Protocol for generating intent IDs and order IDs.

    Implementations should ensure:
    - Intent IDs are deterministic (same intent → same ID).
    - Order IDs are unique and restart-safe.
    """

    def generate_intent_id(self, intent: TradeIntent) -> Result[str]:
        """Generate deterministic intent ID from intent payload.

        Args:
            intent: Strategy engine trade intent.

        Returns:
            Result[str]: Generated intent ID.
        """

    def generate_order_id(self, intent_id: str, run_id: str) -> Result[str]:
        """Generate unique order ID for broker submission.

        Args:
            intent_id: Intent ID this order ID is for.
            run_id: Current run ID for namespacing.

        Returns:
            Result[str]: Generated order ID.
        """


@runtime_checkable
class StateMachineProtocol(Protocol):
    """Protocol for order state machine operations.

    Implementations should ensure:
    - State transitions follow defined rules.
    - Invalid transitions are rejected.
    - Idempotent state updates (repeated transitions have no effect).
    """

    def transition(
        self,
        mapping: IntentOrderMapping,
        new_state: OrderState,
        *,
        reason: str,
        timestamp_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Result[IntentOrderMapping]:
        """Apply state transition to mapping.

        Args:
            mapping: Current intent-order mapping.
            new_state: Target state.
            reason: Human-readable reason for transition.
            timestamp_ns: Optional timestamp override for tests.
            metadata: Optional additional metadata.

        Returns:
            Result[IntentOrderMapping]: Updated mapping or error.
        """

    def is_valid_transition(self, from_state: OrderState, to_state: OrderState) -> bool:
        """Check if state transition is valid.

        Args:
            from_state: Current state.
            to_state: Target state.

        Returns:
            bool: True if transition is valid.
        """


@runtime_checkable
class PersistenceProtocol(Protocol):
    """Protocol for intent-order mapping persistence.

    Implementations should ensure:
    - Mappings are durably persisted (survive crashes).
    - Intent ID uniqueness constraint is enforced.
    - Restart-safe (can reload all mappings).
    """

    def save(self, mapping: IntentOrderMapping) -> Result[None]:
        """Save or update intent-order mapping.

        Args:
            mapping: Mapping to persist.

        Returns:
            Result[None]: Success or error.
        """

    def load(self, intent_id: str) -> Result[IntentOrderMapping]:
        """Load mapping by intent ID.

        Args:
            intent_id: Intent ID to load.

        Returns:
            Result[IntentOrderMapping]: Loaded mapping or error.
        """

    def load_all(self) -> Result[list[IntentOrderMapping]]:
        """Load all persisted mappings.

        Returns:
            Result[list[IntentOrderMapping]]: All mappings or error.
        """

    def exists(self, intent_id: str) -> bool:
        """Check if mapping exists for intent ID.

        Args:
            intent_id: Intent ID to check.

        Returns:
            bool: True if mapping exists.
        """


@runtime_checkable
class ReconciliationProtocol(Protocol):
    """Protocol for reconciliation operations.

    Implementations should ensure:
    - Idempotent reconciliation (repeated runs have no adverse effects).
    - State corrections based on broker truth.
    - Orphaned order detection.
    """

    def reconcile_startup(
        self,
        local_mappings: list[IntentOrderMapping],
        broker_orders: list[dict[str, Any]],
        *,
        timestamp_ns: int | None = None,
    ) -> Result[ReconciliationResult]:
        """Reconcile local state with broker state on startup.

        Args:
            local_mappings: All local intent-order mappings.
            broker_orders: All active/recent orders from broker.
            timestamp_ns: Optional timestamp override for tests.

        Returns:
            Result[ReconciliationResult]: Reconciliation result.
        """

    def reconcile_periodic(
        self,
        mappings_in_flight: list[IntentOrderMapping],
        timeout_threshold_ns: int,
        *,
        timestamp_ns: int | None = None,
    ) -> Result[ReconciliationResult]:
        """Reconcile in-flight orders periodically.

        Args:
            mappings_in_flight: Mappings in non-terminal states.
            timeout_threshold_ns: Timeout threshold for in-flight orders.
            timestamp_ns: Optional timestamp override for tests.

        Returns:
            Result[ReconciliationResult]: Reconciliation result.
        """
