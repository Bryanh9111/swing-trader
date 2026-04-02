"""Safe Mode state management for Risk Gate (Phase 3.3).

This module owns the minimal Safe Mode circuit breaker described in
``docs/ROADMAP.md`` Phase 3.3.

Responsibilities
----------------
1. Evaluate Safe Mode trigger conditions:
   - Data source unavailable
   - Broker disconnect detected
   - Risk budget exhausted (e.g., daily loss limit)
2. Manage Safe Mode lifecycle:
   - Persistent state (survives process restarts)
   - State transitions (ACTIVE ↔ SAFE_REDUCING ↔ HALTED)
   - Query interface for current state/snapshot
3. Integrate with monitoring/eventing:
   - Emit CRITICAL alerts when Safe Mode is triggered/escalated
   - Publish domain events on state transitions

All public APIs use ``common.interface.Result[T]`` for error propagation and
support deterministic serialization via ``msgspec``.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any, Final

import msgspec
import msgspec.json

from common.interface import DomainEvent, EventBus, Result, ResultStatus
from monitoring.interface import Alert, AlertEmitter, AlertLevel
from risk_gate.interface import SafeModeState

__all__ = [
    "SafeModeTriggerEvaluation",
    "SafeModePersistedState",
    "SafeModeTransition",
    "SafeModeManager",
    "evaluate_safe_mode_triggers",
]


_MODULE_NAME: Final[str] = "risk_gate.safe_mode"
_SCHEMA_VERSION: Final[str] = "3.3.0"


class SafeModeTriggerEvaluation(msgspec.Struct, frozen=True, kw_only=True):
    """Evaluation output for Safe Mode triggers.

    Attributes:
        data_source_unavailable: True when the market data source is unavailable
            (or when availability is unknown and fail-closed behaviour is enabled).
        broker_disconnected: True when the broker connection is unavailable
            (or unknown under fail-closed behaviour).
        risk_budget_exhausted: True when the daily loss/risk budget is exhausted.
        recommended_state: Recommended Safe Mode state given the trigger set.
        reason_codes: Stable reason codes describing the active triggers.
        details: Diagnostic payload for auditing/observability.
    """

    data_source_unavailable: bool
    broker_disconnected: bool
    risk_budget_exhausted: bool
    recommended_state: SafeModeState
    reason_codes: list[str] = msgspec.field(default_factory=list)
    details: dict[str, Any] = msgspec.field(default_factory=dict)


class SafeModePersistedState(msgspec.Struct, frozen=True, kw_only=True):
    """Persisted Safe Mode snapshot stored on disk via msgspec JSON."""

    schema_version: str = msgspec.field(default=_SCHEMA_VERSION)
    state: SafeModeState = msgspec.field(default=SafeModeState.ACTIVE)
    updated_at_ns: int = msgspec.field(default=0)
    reason_codes: list[str] = msgspec.field(default_factory=list)
    details: dict[str, Any] = msgspec.field(default_factory=dict)


class SafeModeTransition(msgspec.Struct, frozen=True, kw_only=True):
    """State transition record produced by :class:`SafeModeManager`."""

    from_state: SafeModeState
    to_state: SafeModeState
    timestamp_ns: int
    reason_codes: list[str] = msgspec.field(default_factory=list)
    details: dict[str, Any] = msgspec.field(default_factory=dict)


def evaluate_safe_mode_triggers(
    *,
    data_source_available: bool | None = None,
    broker_connected: bool | None = None,
    risk_budget_exhausted: bool | None = None,
    risk_budget_remaining: float | None = None,
    fail_closed_on_unknown: bool = True,
) -> Result[SafeModeTriggerEvaluation]:
    """Evaluate Safe Mode trigger conditions and recommend a target state.

    The function is designed to be called from multiple subsystems:
    - Data layer: ``data_source_available``
    - Execution layer: ``broker_connected``
    - Portfolio/risk layer: ``risk_budget_exhausted`` or ``risk_budget_remaining``

    Fail-closed policy:
        When ``fail_closed_on_unknown=True``, ``None`` inputs for availability
        fields are treated as unavailable/disconnected.
    """

    try:
        if risk_budget_remaining is not None:
            try:
                risk_budget_remaining = float(risk_budget_remaining)
            except (TypeError, ValueError) as exc:
                return Result.failed(exc, "SAFE_MODE_INVALID_RISK_BUDGET_REMAINING")

        if risk_budget_exhausted is None and risk_budget_remaining is not None:
            risk_budget_exhausted = bool(risk_budget_remaining <= 0.0)

        data_source_unavailable = (
            (data_source_available is False)
            if data_source_available is not None
            else bool(fail_closed_on_unknown)
        )
        broker_disconnected = (
            (broker_connected is False) if broker_connected is not None else bool(fail_closed_on_unknown)
        )
        risk_budget_exhausted = bool(risk_budget_exhausted) if risk_budget_exhausted is not None else False

        reason_codes: list[str] = []
        if data_source_unavailable:
            reason_codes.append("SAFE_MODE.DATA_SOURCE_UNAVAILABLE")
        if broker_disconnected:
            reason_codes.append("SAFE_MODE.BROKER_DISCONNECTED")
        if risk_budget_exhausted:
            reason_codes.append("SAFE_MODE.RISK_BUDGET_EXHAUSTED")

        if broker_disconnected:
            recommended_state = SafeModeState.HALTED
        elif data_source_unavailable or risk_budget_exhausted:
            recommended_state = SafeModeState.SAFE_REDUCING
        else:
            recommended_state = SafeModeState.ACTIVE

        details = {
            "inputs": {
                "data_source_available": data_source_available,
                "broker_connected": broker_connected,
                "risk_budget_exhausted": risk_budget_exhausted,
                "risk_budget_remaining": risk_budget_remaining,
                "fail_closed_on_unknown": fail_closed_on_unknown,
            }
        }

        return Result.success(
            SafeModeTriggerEvaluation(
                data_source_unavailable=data_source_unavailable,
                broker_disconnected=broker_disconnected,
                risk_budget_exhausted=risk_budget_exhausted,
                recommended_state=recommended_state,
                reason_codes=reason_codes,
                details=details,
            )
        )
    except Exception as exc:  # noqa: BLE001 - caller expects Result propagation
        return Result.failed(exc, "SAFE_MODE_TRIGGER_EVALUATION_ERROR")


class SafeModeManager:
    """Manage Safe Mode state lifecycle with persistence + alert/event emission."""

    _STATE_TOPIC = "events.risk_gate.safe_mode_state_changed"
    _DEFAULT_RUN_ID = "unknown"

    def __init__(
        self,
        *,
        state_path: str | Path = "artifacts/system_state/safe_mode.json",
        alert_emitter: AlertEmitter | None = None,
        event_bus: EventBus | None = None,
        run_id: str | None = None,
        module_name: str = _MODULE_NAME,
        allow_auto_exit_from_safe_reducing: bool = True,
        allow_auto_exit_from_halted: bool = False,
    ) -> None:
        self._lock = RLock()
        self._state_path = Path(state_path)
        self._alert_emitter = alert_emitter
        self._event_bus = event_bus
        self._run_id = str(run_id) if run_id else self._DEFAULT_RUN_ID
        self._module_name = str(module_name or _MODULE_NAME)
        self._allow_auto_exit_from_safe_reducing = bool(allow_auto_exit_from_safe_reducing)
        self._allow_auto_exit_from_halted = bool(allow_auto_exit_from_halted)

        self._snapshot = SafeModePersistedState(
            schema_version=_SCHEMA_VERSION,
            state=SafeModeState.ACTIVE,
            updated_at_ns=0,
            reason_codes=[],
            details={},
        )

    def load(self) -> Result[SafeModePersistedState]:
        """Load persisted Safe Mode state from disk.

        Behaviour on missing file:
            Returns SUCCESS with default ``ACTIVE`` state.

        Behaviour on unreadable/corrupt file:
            Returns DEGRADED and falls back to ``SAFE_REDUCING`` fail-closed.
        """

        with self._lock:
            try:
                if not self._state_path.exists():
                    self._snapshot = SafeModePersistedState(
                        schema_version=_SCHEMA_VERSION,
                        state=SafeModeState.ACTIVE,
                        updated_at_ns=0,
                        reason_codes=[],
                        details={"note": "no persisted state found"},
                    )
                    return Result.success(self._snapshot, "SAFE_MODE_STATE_DEFAULTED")

                raw = self._state_path.read_bytes()
                decoded = msgspec.json.decode(raw, type=SafeModePersistedState)

                self._snapshot = SafeModePersistedState(
                    schema_version=str(decoded.schema_version or _SCHEMA_VERSION),
                    state=decoded.state,
                    updated_at_ns=int(decoded.updated_at_ns),
                    reason_codes=list(decoded.reason_codes or []),
                    details=dict(decoded.details or {}),
                )
                return Result.success(self._snapshot, "SAFE_MODE_STATE_LOADED")
            except Exception as exc:  # noqa: BLE001 - fail-closed on state corruption
                self._snapshot = SafeModePersistedState(
                    schema_version=_SCHEMA_VERSION,
                    state=SafeModeState.SAFE_REDUCING,
                    updated_at_ns=int(time.time_ns()),
                    reason_codes=["SAFE_MODE.STATE_LOAD_FAILED"],
                    details={
                        "path": str(self._state_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                return Result.degraded(self._snapshot, exc, "SAFE_MODE_STATE_LOAD_FAILED")

    def get_state(self) -> SafeModeState:
        """Return the current in-memory Safe Mode state."""

        with self._lock:
            return self._snapshot.state

    def get_snapshot(self) -> SafeModePersistedState:
        """Return a copy of the current persisted snapshot."""

        with self._lock:
            snap = self._snapshot
            return SafeModePersistedState(
                schema_version=str(snap.schema_version),
                state=snap.state,
                updated_at_ns=int(snap.updated_at_ns),
                reason_codes=list(snap.reason_codes or []),
                details=dict(snap.details or {}),
            )

    def evaluate_and_update(
        self,
        evaluation: SafeModeTriggerEvaluation,
        *,
        current_time_ns: int | None = None,
    ) -> Result[SafeModeTransition | None]:
        """Apply automatic transitions based on a trigger evaluation.

        Notes
        -----
        - Escalation is always allowed (ACTIVE->SAFE_REDUCING->HALTED).
        - Auto exit from SAFE_REDUCING to ACTIVE is controlled by
          ``allow_auto_exit_from_safe_reducing``.
        - Auto exit from HALTED is disabled by default; use :meth:`set_state`.
        """

        timestamp_ns = int(time.time_ns() if current_time_ns is None else current_time_ns)
        with self._lock:
            current_state = self._snapshot.state
            recommended = evaluation.recommended_state
            target = self._resolve_auto_target(current_state, recommended)
            if target == current_state:
                return Result.success(None, "SAFE_MODE_NO_STATE_CHANGE")

            return self._apply_transition(
                from_state=current_state,
                to_state=target,
                timestamp_ns=timestamp_ns,
                reason_codes=list(evaluation.reason_codes or []),
                details={"trigger_evaluation": msgspec.to_builtins(evaluation)},
            )

    def set_state(
        self,
        state: SafeModeState,
        *,
        reason_codes: list[str] | None = None,
        details: dict[str, Any] | None = None,
        current_time_ns: int | None = None,
    ) -> Result[SafeModeTransition]:
        """Manually set Safe Mode state (explicit operator/system action)."""

        timestamp_ns = int(time.time_ns() if current_time_ns is None else current_time_ns)
        with self._lock:
            from_state = self._snapshot.state
            to_state = SafeModeState(state)
            if from_state == to_state:
                transition = SafeModeTransition(
                    from_state=from_state,
                    to_state=to_state,
                    timestamp_ns=timestamp_ns,
                    reason_codes=list(reason_codes or []),
                    details=dict(details or {}),
                )
                return Result.success(transition, "SAFE_MODE_ALREADY_IN_STATE")

            if not self._is_transition_allowed(from_state, to_state):
                return Result.failed(
                    ValueError(f"Illegal Safe Mode transition {from_state.value}->{to_state.value}"),
                    "SAFE_MODE_ILLEGAL_TRANSITION",
                )

            return self._apply_transition(
                from_state=from_state,
                to_state=to_state,
                timestamp_ns=timestamp_ns,
                reason_codes=list(reason_codes or []),
                details=dict(details or {}),
            )

    def persist(self) -> Result[Path]:
        """Persist the current state snapshot to disk."""

        with self._lock:
            return self._persist_snapshot(self._snapshot)

    def _resolve_auto_target(self, current: SafeModeState, recommended: SafeModeState) -> SafeModeState:
        if current == SafeModeState.HALTED and not self._allow_auto_exit_from_halted:
            return SafeModeState.HALTED
        if current == SafeModeState.SAFE_REDUCING and recommended == SafeModeState.ACTIVE:
            return SafeModeState.ACTIVE if self._allow_auto_exit_from_safe_reducing else SafeModeState.SAFE_REDUCING
        return recommended

    def _is_transition_allowed(self, from_state: SafeModeState, to_state: SafeModeState) -> bool:
        if from_state == to_state:
            return True
        if from_state == SafeModeState.ACTIVE:
            return to_state in (SafeModeState.SAFE_REDUCING, SafeModeState.HALTED)
        if from_state == SafeModeState.SAFE_REDUCING:
            return to_state in (SafeModeState.ACTIVE, SafeModeState.HALTED)
        if from_state == SafeModeState.HALTED:
            return to_state in (SafeModeState.SAFE_REDUCING, SafeModeState.ACTIVE)
        return False

    def _apply_transition(
        self,
        *,
        from_state: SafeModeState,
        to_state: SafeModeState,
        timestamp_ns: int,
        reason_codes: list[str],
        details: dict[str, Any],
    ) -> Result[SafeModeTransition]:
        transition = SafeModeTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp_ns=int(timestamp_ns),
            reason_codes=list(reason_codes or []),
            details=dict(details or {}),
        )

        severity_increased = self._severity(to_state) > self._severity(from_state)

        self._snapshot = SafeModePersistedState(
            schema_version=_SCHEMA_VERSION,
            state=to_state,
            updated_at_ns=int(timestamp_ns),
            reason_codes=list(reason_codes or []),
            details=dict(details or {}),
        )

        persist_res = self._persist_snapshot(self._snapshot)

        alert_error: BaseException | None = None
        event_error: BaseException | None = None
        if severity_increased:
            alert_error = self._emit_critical_alert(transition)
        event_error = self._publish_state_change_event(transition)

        if persist_res.status == ResultStatus.FAILED:
            return Result.failed(persist_res.error or RuntimeError("persist failed"), "SAFE_MODE_PERSIST_FAILED")
        if persist_res.status == ResultStatus.DEGRADED or alert_error is not None or event_error is not None:
            first_error = persist_res.error or alert_error or event_error or RuntimeError("degraded")
            return Result.degraded(transition, first_error, "SAFE_MODE_TRANSITION_DEGRADED")

        return Result.success(transition, "SAFE_MODE_TRANSITION_APPLIED")

    def _persist_snapshot(self, snapshot: SafeModePersistedState) -> Result[Path]:
        path = self._state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = msgspec.json.encode(snapshot)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=str(path.parent),
                prefix=f".{path.name}.",
            ) as tmp:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, path)
            return Result.success(path, "SAFE_MODE_STATE_PERSISTED")
        except Exception as exc:  # noqa: BLE001 - surface persistence failure via Result
            try:
                if "tmp_path" in locals() and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
            return Result.degraded(path, exc, "SAFE_MODE_STATE_PERSIST_FAILED")

    def _emit_critical_alert(self, transition: SafeModeTransition) -> BaseException | None:
        if self._alert_emitter is None:
            return None
        try:
            alert = Alert(
                alert_id=uuid.uuid4().hex,
                level=AlertLevel.CRITICAL,
                title=f"Safe Mode triggered: {transition.to_state.value}",
                message=(
                    "Safe Mode state escalated.\n"
                    f"from={transition.from_state.value} to={transition.to_state.value}\n"
                    f"reason_codes={list(transition.reason_codes or [])}"
                ),
                timestamp=int(transition.timestamp_ns),
                run_id=self._run_id,
                module_name=self._module_name,
                metadata={
                    "from_state": transition.from_state.value,
                    "to_state": transition.to_state.value,
                    "reason_codes": list(transition.reason_codes or []),
                    "details": dict(transition.details or {}),
                },
            )
            self._alert_emitter.emit_alert(alert)
            return None
        except Exception as exc:  # noqa: BLE001 - monitoring must not crash trading loop
            return exc

    def _publish_state_change_event(self, transition: SafeModeTransition) -> BaseException | None:
        if self._event_bus is None:
            return None
        try:
            event = DomainEvent(
                event_id=uuid.uuid4().hex,
                event_type="safe_mode_state_changed",
                run_id=self._run_id,
                module=self._module_name,
                timestamp_ns=int(transition.timestamp_ns),
                data={
                    "from_state": transition.from_state.value,
                    "to_state": transition.to_state.value,
                    "reason_codes": list(transition.reason_codes or []),
                    "details": dict(transition.details or {}),
                },
            )
            self._event_bus.publish(self._STATE_TOPIC, event)
            return None
        except Exception as exc:  # noqa: BLE001 - event bus is best-effort
            return exc

    @staticmethod
    def _severity(state: SafeModeState) -> int:
        return {
            SafeModeState.ACTIVE: 0,
            SafeModeState.SAFE_REDUCING: 1,
            SafeModeState.HALTED: 2,
        }.get(state, 99)
