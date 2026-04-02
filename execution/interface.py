"""Execution interface contracts for Phase 3.5 (broker adapter boundary).

This module defines the public schemas and protocols for AST's Phase 3.5
Execution layer. The Execution layer sits between downstream order lifecycle
tracking (``order_state_machine``) and concrete broker integrations (e.g.,
IBKR, simulated paper broker, backtest adapters).

Core responsibilities expressed by this interface:

- Broker adapter boundary: a small, consistent set of operations required by
  orchestrators/state-machines to connect, submit, cancel, and query orders.
- Execution reporting: a stable, broker-agnostic report schema that summarizes
  final execution outcomes (fills, costs, and audited state transitions).

Public API overview:

- ``BrokerAdapterProtocol``: runtime-checkable protocol implemented by broker
  adapters (IBKR, paper, mock). All operations return ``Result[T]``.
- ``BrokerConnectionConfig``: immutable connection configuration for adapters.
- ``FillDetail``: canonical fill detail sub-schema used by execution reports.
- ``ExecutionReport``: terminal execution summary for a single order/intent.

Design conventions (aligned with Phases 1–3.4):

- Immutable value objects built with ``msgspec.Struct(frozen=True, kw_only=True)``.
- Timestamps are nanosecond Unix epoch integers (UTC) with ``*_ns`` suffix.
- Interfaces return ``common.interface.Result[T]`` to capture success/degraded/fail
  outcomes without raising as control flow across boundaries.
- Broker-specific or adapter-specific fields are carried in ``metadata`` to keep
  the public contract stable as integrations evolve.
"""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

import msgspec

from common.interface import Result
from journal.interface import SnapshotBase
from order_state_machine.interface import (
    IntentOrderMapping,
    OrderState,
    StateTransition,
)

if TYPE_CHECKING:
    from strategy.interface import TradeIntent
else:  # pragma: no cover - type-only imports may not exist in early phases.
    TradeIntent = Any  # type: ignore[assignment]

__all__ = [
    "FillDetail",
    "ExecutionReport",
    "ExecutionOutput",
    "BrokerAdapterProtocol",
    "BrokerConnectionConfig",
]


class FillDetail(msgspec.Struct, frozen=True, kw_only=True):
    """Normalized representation of a single execution fill.

    This schema is intentionally broker-agnostic. Adapter implementations should
    map venue-specific execution records into this structure so downstream
    components (journaling, analytics, reconciliation) can reason about fills
    without depending on broker SDK objects.

    Attributes:
        execution_id: Unique broker execution identifier for this fill (or an
            adapter-generated stable id if the broker does not provide one).
        fill_price: Execution price for this fill.
        fill_quantity: Quantity executed by this fill (shares/contracts).
        fill_time_ns: Nanosecond Unix epoch timestamp of the fill (UTC).
        commission: Optional commission attributable to this fill; ``None`` when
            the broker reports only aggregate commissions.
        metadata: Optional broker-specific fields (e.g., liquidity flag, venue,
            currency, FX rate, clearing, or execution conditions).
    """

    execution_id: str
    fill_price: float
    fill_quantity: float
    fill_time_ns: int
    commission: float | None = msgspec.field(default=None)
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


class ExecutionReport(msgspec.Struct, frozen=True, kw_only=True):
    """Terminal execution summary for a submitted order associated with an intent.

    ``ExecutionReport`` captures the final, reconciled outcome after the order is
    considered complete (i.e., it has entered a terminal state such as FILLED,
    CANCELLED, REJECTED, or EXPIRED). It is designed to be emitted by the
    Execution layer and stored by journaling/replay pipelines.

    The report intentionally includes both:

    - High-level aggregates (filled/remaining quantities, average fill price,
      total commissions), and
    - Fine-grained audit details (individual fills and state transitions).

    Attributes:
        run_id: Run ID that originated the intent/order submission.
        intent_id: Deterministic intent identifier (from Strategy Engine).
        client_order_id: Client order id used to address the order in the broker
            adapter and within AST.
        broker_order_id: Broker/venue order identifier once assigned.
        symbol: Instrument identifier (e.g., ``"AAPL"``).
        final_state: Final order state (terminal) as tracked by the Order State
            Machine.

        filled_quantity: Total executed quantity across all fills.
        remaining_quantity: Quantity remaining at termination (0 for FILLED).
        avg_fill_price: Volume-weighted average execution price; ``None`` when
            there were no fills.
        commissions: Total commissions/fees; ``None`` when not provided by the
            broker or not yet known.

        fills: Ordered list of individual fill records (may be empty).
        state_transitions: Ordered list of state transitions applied by the Order
            State Machine for this intent/order.

        executed_at_ns: Nanosecond Unix epoch timestamp when the report was
            produced (UTC). This is typically the time the order reached its
            terminal state after reconciliation.
        metadata: Broker-specific fields (e.g., order type, tif, account,
            exchange/route, currency, realized pnl, warnings).
    """

    run_id: str
    intent_id: str
    client_order_id: str
    broker_order_id: str
    symbol: str
    final_state: OrderState
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: float | None = msgspec.field(default=None)
    commissions: float | None = msgspec.field(default=None)
    fills: list[FillDetail] = msgspec.field(default_factory=list)
    state_transitions: list[StateTransition] = msgspec.field(default_factory=list)
    executed_at_ns: int
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


class ExecutionOutput(SnapshotBase, frozen=True, kw_only=True):
    """Execution plugin output snapshot for journaling and replay.

    This snapshot wraps the complete Execution output including generated execution
    reports and pass-through data from upstream modules. It embeds the standard
    snapshot metadata envelope from ``journal.interface.SnapshotBase``.

    Attributes:
        schema_version: See ``journal.interface.SnapshotBase``.
        system_version: See ``journal.interface.SnapshotBase``.
        asof_timestamp: See ``journal.interface.SnapshotBase``.
        reports: List of execution reports (each ExecutionReport serialized).
        intents: Pass-through OrderIntentSet from Strategy Engine (serialized).
        risk_decisions: Pass-through RiskDecisionSet from Risk Gate (serialized).
    """

    SCHEMA_VERSION = "1.4.0"

    reports: list[dict[str, Any]] = msgspec.field(default_factory=list)
    intents: dict[str, Any] = msgspec.field(default_factory=dict)
    risk_decisions: dict[str, Any] = msgspec.field(default_factory=dict)


class BrokerConnectionConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for establishing and maintaining broker connectivity.

    This config is consumed by broker adapters implementing
    :class:`~execution.interface.BrokerAdapterProtocol`.

    Notes:
        - ``account=None`` indicates the adapter should auto-detect the active
          trading account when the broker API supports it.
        - ``timeout`` is expressed in seconds and should apply to connection and
          request/response waits (adapter-specific).
        - ``max_reconnect_attempts`` controls how many reconnect attempts an
          adapter may perform after unexpected disconnects.

    Attributes:
        host: Broker gateway host (defaults to local gateway).
        port: Broker gateway port (e.g., 7497 for paper TWS, 4002 for IBG).
        client_id: Client identifier used by certain brokers to track sessions.
        readonly: When True, adapter must not submit/cancel orders.
        account: Optional account identifier; ``None`` means auto-detect.
        timeout: Operation timeout in seconds (best-effort across adapters).
        max_reconnect_attempts: Maximum reconnect attempts before failing.
        enable_dynamic_client_id: Enable dynamic client_id allocation on
            connection conflict (Error 326).
        client_id_range: Range of client_id values to try when dynamic allocation
            is enabled. Default (1, 32) covers standard IBKR client_id range.
    """

    host: str = msgspec.field(default="127.0.0.1")
    port: int
    client_id: int = msgspec.field(default=1)
    readonly: bool = msgspec.field(default=False)
    account: str | None = msgspec.field(default=None)
    timeout: int = msgspec.field(default=20)
    max_reconnect_attempts: int = msgspec.field(default=5)
    enable_dynamic_client_id: bool = msgspec.field(default=False)
    client_id_range: tuple[int, int] = msgspec.field(default=(1, 32))


@runtime_checkable
class BrokerAdapterProtocol(Protocol):
    """Protocol for a broker adapter used by the Execution layer.

    Implementations provide a minimal surface area so upstream components can
    operate against a uniform boundary:

    - Connection lifecycle (connect/disconnect/is_connected)
    - Order submission/cancellation
    - Order status queries for reconciliation

    All operations return ``Result[T]`` and should avoid raising as part of the
    normal control flow across the boundary. Broker SDK exceptions should be
    captured in ``Result.failed`` or ``Result.degraded`` with an appropriate
    ``reason_code``.
    """

    def connect(self) -> Result[None]:
        """Establish a broker connection and prepare the adapter for use.

        Returns:
            Result[None]: Success when connected and ready; failed/degraded
                otherwise.
        """

    def disconnect(self) -> Result[None]:
        """Disconnect from the broker and release resources.

        Returns:
            Result[None]: Success when disconnected; failed/degraded otherwise.
        """

    def is_connected(self) -> bool:
        """Return True when the adapter believes it is connected and usable."""

    def submit_order(self, intent: TradeIntent) -> Result[IntentOrderMapping]:
        """Submit an order to the broker for the supplied trade intent.

        Implementations should:
            - Enforce ``BrokerConnectionConfig.readonly`` if applicable.
            - Provide an idempotent mapping where possible (replays do not create
              duplicate orders).
            - Populate ``IntentOrderMapping.broker_order_id`` once acknowledged.

        Args:
            intent: Trade intent to translate into a broker order.

        Returns:
            Result[IntentOrderMapping]: Mapping containing ids, state, and
                broker-specific metadata.
        """

    def cancel_order(self, client_order_id: str) -> Result[None]:
        """Request cancellation of an existing broker order.

        Args:
            client_order_id: The client order id used to reference the order.

        Returns:
            Result[None]: Success when a cancel request is accepted/queued.
        """

    def get_order_status(self, client_order_id: str) -> Result[IntentOrderMapping]:
        """Fetch the latest broker view of the supplied order.

        Implementations should translate broker status fields into
        :class:`~order_state_machine.interface.OrderState` and return an updated
        :class:`~order_state_machine.interface.IntentOrderMapping`.

        Args:
            client_order_id: The client order id used to reference the order.

        Returns:
            Result[IntentOrderMapping]: Latest mapping view for this order.
        """

    def get_all_orders(self) -> Result[list[IntentOrderMapping]]:
        """Fetch all known active/recent orders from the broker session.

        This method is primarily used for reconciliation (detecting orphaned
        broker orders and updating local mappings).

        Returns:
            Result[list[IntentOrderMapping]]: List of broker orders translated
                into mapping objects.
        """
