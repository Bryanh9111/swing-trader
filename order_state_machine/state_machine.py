"""Order State Machine implementation (Phase 3.4).

This module implements the Pattern 3 order state machine described in
``docs/sessions/session-2025-12-19-order-state-machine.md``.

Design goals:
    - Validate all state transitions against an explicit transition matrix.
    - Enforce terminal state rules (terminal -> anything but itself is invalid).
    - Provide idempotency: transitioning to the current state succeeds and does
      not mutate the mapping.
    - Record transition audit entries via :class:`~order_state_machine.interface.StateTransition`.
"""

from __future__ import annotations

import time
from typing import Any, Final, TypeAlias

from msgspec.structs import replace

from common.exceptions import SystemStateError
from common.interface import Result
from order_state_machine.interface import (
    IntentOrderMapping,
    OrderState,
    StateMachineProtocol,
    StateTransition,
)

__all__ = [
    "StateMachine",
]

_MODULE: Final[str] = "order_state_machine.state_machine"

TransitionRules: TypeAlias = dict[OrderState, set[OrderState]]


class StateMachine(StateMachineProtocol):
    """Default implementation of :class:`~order_state_machine.interface.StateMachineProtocol`."""

    def __init__(self) -> None:
        self._terminal_states: set[OrderState] = OrderState.terminal_states()
        self._rules: TransitionRules = self._build_transition_rules()
        self._validate_transition_rules(self._rules)
        self._transitions: list[StateTransition] = []

    @property
    def terminal_states(self) -> set[OrderState]:
        """Return terminal states (copy) for convenience/inspection."""

        return set(self._terminal_states)

    @property
    def transition_log(self) -> tuple[StateTransition, ...]:
        """Return an immutable snapshot of recorded transitions."""

        return tuple(self._transitions)

    def drain_transition_log(self) -> list[StateTransition]:
        """Drain and return the transition log.

        Not part of the protocol, but useful for audit pipelines and tests.
        """

        drained = list(self._transitions)
        self._transitions.clear()
        return drained

    def transition(
        self,
        mapping: IntentOrderMapping,
        new_state: OrderState,
        *,
        reason: str,
        timestamp_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Result[IntentOrderMapping]:
        """Apply a validated state transition to an intent-order mapping."""

        try:
            ts_ns = int(self._current_time_ns() if timestamp_ns is None else timestamp_ns)
            from_state = mapping.state
            to_state = new_state

            # Idempotency: repeated transitions must succeed but not mutate state.
            if from_state == to_state:
                return Result.success(mapping, "IDEMPOTENT")

            if not self.is_valid_transition(from_state, to_state):
                err = SystemStateError(
                    "Invalid order state transition",
                    module=_MODULE,
                    details={
                        "intent_id": mapping.intent_id,
                        "from_state": from_state.value,
                        "to_state": to_state.value,
                        "terminal_from_state": from_state in self._terminal_states,
                    },
                )
                return Result.failed(err, err.reason_code)

            updated_mapping = self._create_updated_mapping(
                mapping=mapping,
                new_state=to_state,
                timestamp_ns=ts_ns,
                metadata=metadata,
            )

            self._transitions.append(
                StateTransition(
                    intent_id=mapping.intent_id,
                    from_state=from_state,
                    to_state=to_state,
                    timestamp_ns=ts_ns,
                    reason=reason,
                    metadata=dict(metadata or {}),
                )
            )
            return Result.success(updated_mapping)
        except Exception as exc:  # noqa: BLE001 - state machine must not crash caller
            return Result.failed(exc, "STATE_MACHINE_ERROR")

    def is_valid_transition(self, from_state: OrderState, to_state: OrderState) -> bool:
        """Return True when the given state change is allowed."""

        if from_state == to_state:
            return True

        if from_state in self._terminal_states:
            return False

        allowed = self._rules.get(from_state)
        if not allowed:
            return False
        return to_state in allowed

    def _build_transition_rules(self) -> dict[OrderState, set[OrderState]]:
        """Construct the transition matrix.

        Notes:
            - Self-transitions are handled by the idempotency rule.
            - Terminal states are handled by the terminal state rule.
            - Rules should only describe forward progress; callers must not
              "rewind" state (e.g., FILLED -> ACCEPTED).
        """

        return {
            OrderState.PENDING: {
                OrderState.SUBMITTED,
                OrderState.CANCELLED,
            },
            OrderState.SUBMITTED: {
                OrderState.ACCEPTED,
                OrderState.PARTIALLY_FILLED,
                OrderState.FILLED,
                OrderState.CANCELLED,
                OrderState.REJECTED,
                OrderState.EXPIRED,
            },
            OrderState.ACCEPTED: {
                OrderState.PARTIALLY_FILLED,
                OrderState.FILLED,
                OrderState.CANCELLED,
                OrderState.EXPIRED,
            },
            OrderState.PARTIALLY_FILLED: {
                OrderState.FILLED,
                OrderState.CANCELLED,
                OrderState.EXPIRED,
            },
        }

    def _validate_transition_rules(self, rules: TransitionRules) -> None:
        """Validate that the transition matrix is internally consistent."""

        for from_state, allowed in rules.items():
            if from_state in self._terminal_states:
                msg = f"Transition rules must not include terminal from_state={from_state.value}"
                raise ValueError(msg)
            if not allowed:
                msg = f"Transition rules must not contain empty allowed set for {from_state.value}"
                raise ValueError(msg)
            if from_state in allowed:
                msg = (
                    f"Transition rules should not include self-transition for {from_state.value}; "
                    "idempotency handles it separately"
                )
                raise ValueError(msg)

            for to_state in allowed:
                if to_state is OrderState.PENDING and from_state is not OrderState.PENDING:
                    msg = f"Illegal rewind transition {from_state.value} -> {to_state.value}"
                    raise ValueError(msg)

    def _create_updated_mapping(
        self,
        *,
        mapping: IntentOrderMapping,
        new_state: OrderState,
        timestamp_ns: int,
        metadata: dict[str, Any] | None,
    ) -> IntentOrderMapping:
        """Return a new mapping instance reflecting the applied state transition."""

        merged_metadata: dict[str, Any]
        if metadata:
            merged_metadata = dict(mapping.metadata)
            merged_metadata.update(metadata)
        else:
            merged_metadata = mapping.metadata

        return replace(
            mapping,
            state=new_state,
            updated_at_ns=timestamp_ns,
            metadata=merged_metadata,
        )

    def _current_time_ns(self) -> int:
        """Return the current Unix epoch time in nanoseconds."""

        return int(time.time_ns())
