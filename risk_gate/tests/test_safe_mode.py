from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from common.interface import DomainEvent, ResultStatus
from monitoring.interface import Alert, AlertLevel
from risk_gate.safe_mode import SafeModeManager, evaluate_safe_mode_triggers
from risk_gate.interface import SafeModeState


@dataclass(slots=True)
class _RecordingAlertEmitter:
    alerts: list[Alert] = field(default_factory=list)
    raise_on_emit: bool = False

    def emit_alert(self, alert: Alert) -> None:
        if self.raise_on_emit:
            raise RuntimeError("alert failed")
        self.alerts.append(alert)

    def get_alerts(self, run_id: str) -> list[Alert]:
        return [a for a in self.alerts if a.run_id == run_id]


@dataclass(slots=True)
class _RecordingEventBus:
    published: list[tuple[str, DomainEvent]] = field(default_factory=list)
    raise_on_publish: bool = False

    def publish(self, topic: str, event: DomainEvent) -> None:
        if self.raise_on_publish:
            raise RuntimeError("publish failed")
        self.published.append((topic, event))

    def subscribe(self, pattern: str, handler: object) -> None:  # pragma: no cover - not needed here
        raise NotImplementedError


def test_evaluate_safe_mode_triggers_fail_closed_unknown() -> None:
    res = evaluate_safe_mode_triggers(data_source_available=None, broker_connected=None, fail_closed_on_unknown=True)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.data_source_unavailable is True
    assert res.data.broker_disconnected is True
    assert res.data.recommended_state is SafeModeState.HALTED
    assert "SAFE_MODE.DATA_SOURCE_UNAVAILABLE" in res.data.reason_codes
    assert "SAFE_MODE.BROKER_DISCONNECTED" in res.data.reason_codes


def test_evaluate_safe_mode_triggers_recommendations() -> None:
    res = evaluate_safe_mode_triggers(data_source_available=True, broker_connected=True, risk_budget_exhausted=False)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.recommended_state is SafeModeState.ACTIVE
    assert res.data.reason_codes == []

    res = evaluate_safe_mode_triggers(data_source_available=False, broker_connected=True)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.recommended_state is SafeModeState.SAFE_REDUCING
    assert "SAFE_MODE.DATA_SOURCE_UNAVAILABLE" in res.data.reason_codes

    res = evaluate_safe_mode_triggers(broker_connected=False, data_source_available=True)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.recommended_state is SafeModeState.HALTED
    assert "SAFE_MODE.BROKER_DISCONNECTED" in res.data.reason_codes


def test_evaluate_safe_mode_triggers_risk_budget_remaining() -> None:
    res = evaluate_safe_mode_triggers(risk_budget_remaining=10.0, data_source_available=True, broker_connected=True)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.risk_budget_exhausted is False
    assert res.data.recommended_state is SafeModeState.ACTIVE

    res = evaluate_safe_mode_triggers(risk_budget_remaining=0.0, data_source_available=True, broker_connected=True)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.risk_budget_exhausted is True
    assert res.data.recommended_state is SafeModeState.SAFE_REDUCING
    assert "SAFE_MODE.RISK_BUDGET_EXHAUSTED" in res.data.reason_codes


def test_evaluate_safe_mode_triggers_invalid_risk_budget_remaining() -> None:
    res = evaluate_safe_mode_triggers(risk_budget_remaining="nope")  # type: ignore[arg-type]
    assert res.status is ResultStatus.FAILED
    assert res.reason_code == "SAFE_MODE_INVALID_RISK_BUDGET_REMAINING"


def test_safe_mode_manager_load_default_and_persistence(tmp_path: Path) -> None:
    state_path = tmp_path / "safe_mode.json"
    manager = SafeModeManager(state_path=state_path)
    load_res = manager.load()
    assert load_res.status is ResultStatus.SUCCESS
    assert manager.get_state() is SafeModeState.ACTIVE

    transition_res = manager.set_state(SafeModeState.SAFE_REDUCING, reason_codes=["X"], current_time_ns=123)
    assert transition_res.status is ResultStatus.SUCCESS
    assert manager.get_state() is SafeModeState.SAFE_REDUCING

    persist_res = manager.persist()
    assert persist_res.status is ResultStatus.SUCCESS
    assert state_path.exists()

    manager2 = SafeModeManager(state_path=state_path)
    load_res2 = manager2.load()
    assert load_res2.status is ResultStatus.SUCCESS
    assert manager2.get_state() is SafeModeState.SAFE_REDUCING
    assert manager2.get_snapshot().reason_codes == ["X"]


def test_safe_mode_manager_state_transitions_and_auto_exit(tmp_path: Path) -> None:
    state_path = tmp_path / "safe_mode.json"
    bus = _RecordingEventBus()
    alerts = _RecordingAlertEmitter()
    manager = SafeModeManager(state_path=state_path, event_bus=bus, alert_emitter=alerts, run_id="run-1")
    manager.load()

    eval_res = evaluate_safe_mode_triggers(data_source_available=False, broker_connected=True)
    transition = manager.evaluate_and_update(eval_res.data, current_time_ns=1)
    assert transition.status is ResultStatus.SUCCESS
    assert transition.data is not None
    assert manager.get_state() is SafeModeState.SAFE_REDUCING
    assert len(alerts.alerts) == 1
    assert alerts.alerts[0].level is AlertLevel.CRITICAL
    assert len(bus.published) == 1

    eval_res = evaluate_safe_mode_triggers(data_source_available=True, broker_connected=True, risk_budget_exhausted=False)
    transition = manager.evaluate_and_update(eval_res.data, current_time_ns=2)
    assert transition.status is ResultStatus.SUCCESS
    assert transition.data is not None
    assert manager.get_state() is SafeModeState.ACTIVE
    assert len(alerts.alerts) == 1
    assert len(bus.published) == 2

    manager.set_state(SafeModeState.HALTED, current_time_ns=3)
    assert manager.get_state() is SafeModeState.HALTED
    eval_res = evaluate_safe_mode_triggers(data_source_available=True, broker_connected=True)
    transition = manager.evaluate_and_update(eval_res.data, current_time_ns=4)
    assert transition.status is ResultStatus.SUCCESS
    assert transition.data is None  # auto-exit from HALTED disabled by default
    assert manager.get_state() is SafeModeState.HALTED


def test_safe_mode_manager_manual_state_switch_noop(tmp_path: Path) -> None:
    manager = SafeModeManager(state_path=tmp_path / "safe_mode.json")
    manager.load()
    manager.set_state(SafeModeState.SAFE_REDUCING, current_time_ns=1)

    res = manager.set_state(SafeModeState.SAFE_REDUCING, reason_codes=["NOOP"], current_time_ns=2)
    assert res.status is ResultStatus.SUCCESS
    assert res.reason_code == "SAFE_MODE_ALREADY_IN_STATE"
    assert res.data is not None
    assert res.data.from_state is SafeModeState.SAFE_REDUCING
    assert res.data.to_state is SafeModeState.SAFE_REDUCING


def test_safe_mode_manager_load_corrupt_state_is_fail_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "safe_mode.json"
    state_path.write_bytes(b"{not json")
    manager = SafeModeManager(state_path=state_path)
    res = manager.load()
    assert res.status is ResultStatus.DEGRADED
    assert res.data is not None
    assert manager.get_state() is SafeModeState.SAFE_REDUCING
    assert "SAFE_MODE.STATE_LOAD_FAILED" in res.data.reason_codes


def test_safe_mode_manager_degraded_when_alert_or_persist_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alerts = _RecordingAlertEmitter(raise_on_emit=True)
    manager = SafeModeManager(state_path=tmp_path / "safe_mode.json", alert_emitter=alerts)
    manager.load()
    res = manager.set_state(SafeModeState.SAFE_REDUCING, current_time_ns=1)
    assert res.status is ResultStatus.DEGRADED
    assert res.reason_code == "SAFE_MODE_TRANSITION_DEGRADED"

    def _replace(*args: object, **kwargs: object) -> None:
        raise OSError("nope")

    monkeypatch.setattr("risk_gate.safe_mode.os.replace", _replace)
    manager2 = SafeModeManager(state_path=tmp_path / "subdir" / "safe_mode.json")
    manager2.load()
    res = manager2.set_state(SafeModeState.SAFE_REDUCING, current_time_ns=1)
    assert res.status is ResultStatus.DEGRADED
    assert res.reason_code == "SAFE_MODE_TRANSITION_DEGRADED"
