from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from common.events import InMemoryEventBus
from common.interface import Result, ResultStatus
from common.logging import StructlogLoggerFactory
from journal import ArtifactManager, JournalWriter
from journal.interface import RunType
from orchestrator import EODScanOrchestrator, register_stub_plugins
from plugins.interface import PluginCategory, PluginMetadata
from plugins.registry import PluginRegistry


class _BaseTestPlugin:
    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        return Result.success(dict(config) if isinstance(config, Mapping) else {})

    def cleanup(self) -> Result[None]:
        return Result.success(data=None)


class EventGuardTestPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="event_guard_test",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.RISK_POLICY,
        enabled=True,
        description="Test event guard plugin.",
    )

    execute_called: bool = False

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        EventGuardTestPlugin.execute_called = True
        return Result.success({"schema_version": "1.0.0", "constraints": {}})


class StrategyTestPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="strategy_test",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.STRATEGY,
        enabled=True,
        description="Test strategy plugin.",
    )

    execute_called: bool = False

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        StrategyTestPlugin.execute_called = True
        return Result.success({"schema_version": "1.0.0", "intents": {"schema_version": "1.0.0", "intent_groups": []}})


class RiskGateTestPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="risk_gate_test",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.RISK_POLICY,
        enabled=True,
        description="Test risk gate plugin.",
    )

    execute_called: bool = False
    last_validated_config: Mapping[str, Any] | None = None

    def validate_config(self, config: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        RiskGateTestPlugin.last_validated_config = dict(config) if isinstance(config, Mapping) else {}
        return Result.success(dict(config) if isinstance(config, Mapping) else {})

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        RiskGateTestPlugin.execute_called = True
        return Result.success(
            {
                "schema_version": "1.0.0",
                "intents": payload.get("intents", {}),
                "risk_decisions": {"schema_version": "1.0.0", "decisions": []},
            }
        )


class ExecutionTestPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="execution_test",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.SIGNAL,
        enabled=True,
        description="Test execution plugin.",
    )

    last_payload: Mapping[str, Any] | None = None

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        ExecutionTestPlugin.last_payload = dict(payload)
        return Result.success({"schema_version": "1.0.0", "reports": []})


@pytest.fixture()
def orchestrator_env(tmp_path: Path) -> tuple[EODScanOrchestrator, ArtifactManager]:
    artifact_base = tmp_path / "artifacts"
    artifact_manager = ArtifactManager(base_path=artifact_base)
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_stub_plugins(registry)

    for plugin in (EventGuardTestPlugin, StrategyTestPlugin, RiskGateTestPlugin, ExecutionTestPlugin):
        registry.register_plugin(plugin.metadata.name, plugin)

    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )

    return orchestrator, artifact_manager


def _intraday_config(run_type: RunType) -> dict[str, Any]:
    return {
        "mode": "DRY_RUN",
        "run_type": run_type.value,
        "event_guard_plugin": "event_guard_test",
        "strategy_plugin": "strategy_test",
        "risk_gate_plugin": "risk_gate_test",
        "execution_plugin": "execution_test",
        "plugins": {
            "data_sources": [
                {"name": "universe_stub"},
                {"name": "data_stub"},
            ],
            "scanners": [
                {"name": "scanner_stub"},
            ],
            "risk_policies": [
                {"name": "event_guard_test", "config": {}},
                {"name": "risk_gate_test", "config": {"risk_gate": {"safe_mode_state": "ACTIVE"}}},
            ],
            "strategies": [{"name": "strategy_test", "config": {}}],
            "signals": [{"name": "execution_test", "config": {}}],
        },
    }


def _new_run_id(artifact_manager: ArtifactManager) -> str:
    directories = [path for path in artifact_manager.base_path.iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0].name


def _reset_plugins() -> None:
    EventGuardTestPlugin.execute_called = False
    StrategyTestPlugin.execute_called = False
    RiskGateTestPlugin.execute_called = False
    RiskGateTestPlugin.last_validated_config = None
    ExecutionTestPlugin.last_payload = None


def test_intraday_1030_shortcircuits_to_execution_only(
    orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager],
) -> None:
    """1030 intraday run should short-circuit: only execution plugin called."""
    orchestrator, artifact_manager = orchestrator_env
    _reset_plugins()

    result = orchestrator.execute_run(_intraday_config(RunType.INTRADAY_CHECK_1030))
    assert result.status is ResultStatus.SUCCESS

    # Execution plugin IS called with intraday_check payload
    assert ExecutionTestPlugin.last_payload is not None
    assert ExecutionTestPlugin.last_payload.get("intraday_check") is True
    assert ExecutionTestPlugin.last_payload.get("reconcile_first") is True
    assert ExecutionTestPlugin.last_payload.get("check_stops") is True

    # Upstream modules are NOT called (short-circuit)
    assert EventGuardTestPlugin.execute_called is False
    assert StrategyTestPlugin.execute_called is False
    assert RiskGateTestPlugin.execute_called is False

    run_id = _new_run_id(artifact_manager)
    metadata_result = artifact_manager.read_metadata(run_id)
    assert metadata_result.status is ResultStatus.SUCCESS
    assert metadata_result.data is not None
    assert metadata_result.data.run_type is RunType.INTRADAY_CHECK_1030


def test_intraday_1230_shortcircuits_to_execution_only(
    orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager],
) -> None:
    """1230 intraday run should short-circuit identical to 1030 (no safe_mode)."""
    orchestrator, artifact_manager = orchestrator_env
    _reset_plugins()

    result = orchestrator.execute_run(_intraday_config(RunType.INTRADAY_CHECK_1230))
    assert result.status is ResultStatus.SUCCESS

    # Execution plugin IS called with intraday_check payload
    assert ExecutionTestPlugin.last_payload is not None
    assert ExecutionTestPlugin.last_payload.get("intraday_check") is True
    assert ExecutionTestPlugin.last_payload.get("reconcile_first") is True
    assert ExecutionTestPlugin.last_payload.get("check_stops") is True

    # No safe_mode (same as 1030)
    assert "safe_mode" not in ExecutionTestPlugin.last_payload

    # Upstream modules are NOT called (short-circuit)
    assert EventGuardTestPlugin.execute_called is False
    assert StrategyTestPlugin.execute_called is False
    assert RiskGateTestPlugin.execute_called is False

    run_id = _new_run_id(artifact_manager)
    metadata_result = artifact_manager.read_metadata(run_id)
    assert metadata_result.status is ResultStatus.SUCCESS
    assert metadata_result.data is not None
    assert metadata_result.data.run_type is RunType.INTRADAY_CHECK_1230


def test_intraday_1430_shortcircuits_with_safe_mode(
    orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager],
) -> None:
    """1430 intraday run should short-circuit with safe_mode=SAFE_REDUCING."""
    orchestrator, artifact_manager = orchestrator_env
    _reset_plugins()

    result = orchestrator.execute_run(_intraday_config(RunType.INTRADAY_CHECK_1430))
    assert result.status is ResultStatus.SUCCESS

    # Execution plugin IS called with intraday_check + safe_mode
    assert ExecutionTestPlugin.last_payload is not None
    assert ExecutionTestPlugin.last_payload.get("intraday_check") is True
    assert ExecutionTestPlugin.last_payload.get("reconcile_first") is True
    assert ExecutionTestPlugin.last_payload.get("check_stops") is True
    assert ExecutionTestPlugin.last_payload.get("safe_mode") == "SAFE_REDUCING"

    # Upstream modules are NOT called (short-circuit)
    assert EventGuardTestPlugin.execute_called is False
    assert StrategyTestPlugin.execute_called is False
    assert RiskGateTestPlugin.execute_called is False

    run_id = _new_run_id(artifact_manager)
    metadata_result = artifact_manager.read_metadata(run_id)
    assert metadata_result.status is ResultStatus.SUCCESS
    assert metadata_result.data is not None
    assert metadata_result.data.run_type is RunType.INTRADAY_CHECK_1430
