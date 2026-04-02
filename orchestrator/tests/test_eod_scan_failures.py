from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from common.events import InMemoryEventBus
from common.exceptions import ConfigurationError, OperationalError, RecoverableError
from common.interface import DomainEvent, EventBus, Result, ResultStatus
from common.logging import StructlogLoggerFactory
from orchestrator import EODScanOrchestrator, register_stub_plugins
from orchestrator.stubs import DataStubPlugin, ScannerStubPlugin, UniverseStubPlugin
from plugins.interface import PluginCategory, PluginMetadata
from plugins.registry import PluginRegistry


def _stub_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "mode": "DRY_RUN",
        "plugins": {
            "data_sources": [
                {"name": "universe_stub"},
                {"name": "data_stub"},
            ],
            "scanners": [
                {"name": "scanner_stub"},
            ],
        },
    }
    config.update(overrides)
    return config


@dataclass
class FakeJournalWriter:
    results: list[Result[Any]] | None = None

    def __post_init__(self) -> None:
        self.results = list(self.results or [])
        self.calls: list[tuple[str, Any, Mapping[str, Any], Mapping[str, Any], list[DomainEvent]]] = []

    def persist_complete_run(
        self,
        run_id: str,
        metadata: Any,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        events: list[DomainEvent],
    ) -> Result[Any]:
        self.calls.append((run_id, metadata, inputs, outputs, list(events)))
        if self.results:
            return self.results.pop(0)
        return Result.success(data=None)


def _build_orchestrator(
    _tmp_path: Path,
    *,
    registry: PluginRegistry | None = None,
    event_bus: EventBus | None = None,
    journal_writer: FakeJournalWriter | None = None,
    lifecycle_manager: Any | None = None,
    register_plugins: bool = True,
) -> tuple[EODScanOrchestrator, FakeJournalWriter]:
    registry = registry or PluginRegistry()
    if register_plugins:
        register_stub_plugins(registry)
    journal_writer = journal_writer or FakeJournalWriter()
    event_bus = event_bus or InMemoryEventBus()
    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=StructlogLoggerFactory(),
        plugin_registry=registry,
        lifecycle_manager=lifecycle_manager,
        system_version_getter=lambda: "test-version",
    )
    return orchestrator, journal_writer


class RaisingEventBus:
    def __init__(self, factory: Callable[[], Exception]) -> None:
        self._factory = factory

    def publish(self, topic: str, event: DomainEvent) -> None:
        raise self._factory()

    def subscribe(self, pattern: str, handler: Callable[[DomainEvent], None]) -> None:
        return None


def test_execute_run_rejects_non_mapping_config(tmp_path: Path) -> None:
    orchestrator, _ = _build_orchestrator(tmp_path)

    result = orchestrator.execute_run(["not", "a", "mapping"])  # type: ignore[arg-type]

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "EOD_INVALID_CONFIG"


def test_execute_run_rejects_invalid_mode(tmp_path: Path) -> None:
    orchestrator, _ = _build_orchestrator(tmp_path)

    result = orchestrator.execute_run(_stub_config(mode="invalid-mode"))

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "EOD_INVALID_CONFIG"


def test_execute_run_plugin_registry_failure(tmp_path: Path) -> None:
    class FailingRegistry(PluginRegistry):
        def load_from_config(self, _: Mapping[str, Any]) -> Result[Any]:
            error = ConfigurationError(
                "registry load failed",
                module="tests.registry",
                reason_code="TEST_PLUGIN_LOAD_FAILED",
            )
            return Result.failed(error, error.reason_code)

    registry = FailingRegistry()
    orchestrator, _ = _build_orchestrator(tmp_path, registry=registry, register_plugins=False)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ConfigurationError)
    assert result.reason_code == "TEST_PLUGIN_LOAD_FAILED"


def test_execute_run_plugin_registry_degraded(tmp_path: Path) -> None:
    class DegradedRegistry(PluginRegistry):
        def __init__(self) -> None:
            super().__init__()
            register_stub_plugins(self)

        def load_from_config(self, _: Mapping[str, Any]) -> Result[Any]:
            instances = {
                "universe_stub": UniverseStubPlugin(),
                "data_stub": DataStubPlugin(),
                "scanner_stub": ScannerStubPlugin(),
            }
            error = RuntimeError("partial load")
            return Result.degraded(instances, error, "TEST_PLUGIN_DEGRADED")

    registry = DegradedRegistry()
    orchestrator, writer = _build_orchestrator(tmp_path, registry=registry, register_plugins=False)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == "TEST_PLUGIN_DEGRADED"
    assert len(writer.calls) == 1


def test_execute_run_universe_plugin_name_validation(tmp_path: Path) -> None:
    orchestrator, _ = _build_orchestrator(tmp_path)

    bad_config = _stub_config(universe_plugin=" ")
    result = orchestrator.execute_run(bad_config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ConfigurationError)


def test_execute_run_data_plugin_name_validation(tmp_path: Path) -> None:
    orchestrator, _ = _build_orchestrator(tmp_path)

    bad_config = _stub_config(data_plugin=123)
    result = orchestrator.execute_run(bad_config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ConfigurationError)


def test_execute_run_data_plugin_missing(tmp_path: Path) -> None:
    class MissingDataRegistry(PluginRegistry):
        def __init__(self) -> None:
            super().__init__()
            register_stub_plugins(self)

        def load_from_config(self, _: Mapping[str, Any]) -> Result[Any]:
            return Result.success(
                {
                    "universe_stub": UniverseStubPlugin(),
                }
            )

        def get_plugin(self, name: str) -> type[Any]:
            if name == "data_stub":
                raise ConfigurationError(
                    "data plugin missing",
                    module="tests.registry",
                    reason_code="TEST_DATA_PLUGIN_MISSING",
                )
            return super().get_plugin(name)

    registry = MissingDataRegistry()
    orchestrator, _ = _build_orchestrator(tmp_path, registry=registry, register_plugins=False)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "TEST_DATA_PLUGIN_MISSING"


def test_execute_run_scanner_plugin_missing(tmp_path: Path) -> None:
    class MissingScannerRegistry(PluginRegistry):
        def __init__(self) -> None:
            super().__init__()
            register_stub_plugins(self)

        def load_from_config(self, _: Mapping[str, Any]) -> Result[Any]:
            return Result.success(
                {
                    "universe_stub": UniverseStubPlugin(),
                    "data_stub": DataStubPlugin(),
                }
            )

        def get_plugin(self, name: str) -> type[Any]:
            if name == "scanner_stub":
                raise ConfigurationError(
                    "scanner plugin missing",
                    module="tests.registry",
                    reason_code="TEST_SCANNER_PLUGIN_MISSING",
                )
            return super().get_plugin(name)

    registry = MissingScannerRegistry()
    orchestrator, _ = _build_orchestrator(tmp_path, registry=registry, register_plugins=False)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "TEST_SCANNER_PLUGIN_MISSING"


def test_execute_run_plugin_instantiation_permission_error(tmp_path: Path) -> None:
    class PermissionDataPlugin:
        metadata = PluginMetadata(
            name="permission_data",
            version="1.0.0",
            category=PluginCategory.DATA_SOURCE,
            enabled=True,
            description="permission guarded",
        )

        def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
            raise PermissionError("permission denied")

        def validate_config(self, config: Mapping[str, Any]) -> Result[Any]:
            return Result.success(config)

        def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
            return Result.success(None)

        def execute(self, payload: Mapping[str, Any]) -> Result[Any]:
            return Result.success(payload)

        def cleanup(self) -> Result[None]:
            return Result.success(None)

    class PermissionRegistry(PluginRegistry):
        def __init__(self) -> None:
            super().__init__()
            register_stub_plugins(self)
            self.register_plugin("permission_data", PermissionDataPlugin)

        def load_from_config(self, _: Mapping[str, Any]) -> Result[Any]:
            return Result.success(
                {
                    "universe_stub": UniverseStubPlugin(),
                    "scanner_stub": ScannerStubPlugin(),
                }
            )

    registry = PermissionRegistry()
    orchestrator, _ = _build_orchestrator(tmp_path, registry=registry, register_plugins=False)
    config = _stub_config(data_plugin="permission_data")

    result = orchestrator.execute_run(config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ConfigurationError)


def test_execute_run_plugin_instantiation_generic_error(tmp_path: Path) -> None:
    class BrokenDataPlugin:
        metadata = PluginMetadata(
            name="broken_data",
            version="1.0.0",
            category=PluginCategory.DATA_SOURCE,
            enabled=True,
            description="broken init",
        )

        def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
            raise RuntimeError("broken init")

        def validate_config(self, config: Mapping[str, Any]) -> Result[Any]:
            return Result.success(config)

        def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
            return Result.success(None)

        def execute(self, payload: Mapping[str, Any]) -> Result[Any]:
            return Result.success(payload)

        def cleanup(self) -> Result[None]:
            return Result.success(None)

    class BrokenRegistry(PluginRegistry):
        def __init__(self) -> None:
            super().__init__()
            register_stub_plugins(self)
            self.register_plugin("broken_data", BrokenDataPlugin)

        def load_from_config(self, _: Mapping[str, Any]) -> Result[Any]:
            return Result.success(
                {
                    "universe_stub": UniverseStubPlugin(),
                    "scanner_stub": ScannerStubPlugin(),
                }
            )

    registry = BrokenRegistry()
    orchestrator, _ = _build_orchestrator(tmp_path, registry=registry, register_plugins=False)
    config = _stub_config(data_plugin="broken_data")

    result = orchestrator.execute_run(config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ConfigurationError)


def test_module_init_failure_normalises_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator, writer = _build_orchestrator(tmp_path)

    def failing_init(self: UniverseStubPlugin, _: Any) -> Result[Any]:
        error = RuntimeError("init failed")
        return Result.failed(error, "TEST_INIT_FAIL")

    monkeypatch.setattr(UniverseStubPlugin, "init", failing_init)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "EOD_RUN_FAILED"
    assert isinstance(result.error, OperationalError)
    assert len(writer.calls) == 1
    assert "init failed" in str(result.error)


def test_module_init_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator, _ = _build_orchestrator(tmp_path)

    def degraded_init(self: UniverseStubPlugin, _: Any) -> Result[Any]:
        error = RecoverableError(
            "init degraded",
            module="tests.universe",
            reason_code="TEST_INIT_DEGRADED",
        )
        return Result.degraded(data=None, error=error, reason_code=error.reason_code)

    monkeypatch.setattr(UniverseStubPlugin, "init", degraded_init)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "TEST_INIT_DEGRADED"


def test_module_execute_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator, writer = _build_orchestrator(tmp_path)

    def permission_execute(self: ScannerStubPlugin, _: Any) -> None:
        raise PermissionError("execute denied")

    monkeypatch.setattr(ScannerStubPlugin, "execute", permission_execute)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, OperationalError)
    assert len(writer.calls) == 1


def test_module_cleanup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator, writer = _build_orchestrator(tmp_path)

    def failing_cleanup(self: ScannerStubPlugin) -> Result[Any]:
        error = RuntimeError("cleanup failed")
        return Result.failed(error, "TEST_CLEANUP_FAILED")

    monkeypatch.setattr(ScannerStubPlugin, "cleanup", failing_cleanup)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "EOD_RUN_FAILED"
    assert len(writer.calls) == 1
    assert "cleanup failed" in str(result.error)


def test_module_cleanup_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator, _ = _build_orchestrator(tmp_path)

    def degraded_cleanup(self: ScannerStubPlugin) -> Result[Any]:
        error = RecoverableError(
            "cleanup degraded",
            module="tests.scanner",
            reason_code="TEST_CLEANUP_DEGRADED",
        )
        return Result.degraded(data=None, error=error, reason_code=error.reason_code)

    monkeypatch.setattr(ScannerStubPlugin, "cleanup", degraded_cleanup)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "TEST_CLEANUP_DEGRADED"


def test_event_bus_permission_error(tmp_path: Path) -> None:
    bus = RaisingEventBus(lambda: PermissionError("publish denied"))
    orchestrator, writer = _build_orchestrator(tmp_path, event_bus=bus)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "EOD_EVENT_FAILED"
    assert isinstance(result.error, OperationalError)
    assert len(writer.calls) == 1


def test_event_bus_os_error(tmp_path: Path) -> None:
    bus = RaisingEventBus(lambda: OSError("transient"))
    orchestrator, writer = _build_orchestrator(tmp_path, event_bus=bus)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "EOD_EVENT_FAILED"
    assert isinstance(result.error, RecoverableError)
    assert len(writer.calls) == 1


def test_journal_writer_failure(tmp_path: Path) -> None:
    error = OperationalError(
        "persist failed",
        module="tests.journal",
        reason_code="TEST_PERSIST_FAILED",
    )
    writer = FakeJournalWriter(results=[Result.failed(error, error.reason_code)])
    orchestrator, _ = _build_orchestrator(tmp_path, journal_writer=writer)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "TEST_PERSIST_FAILED"
    assert isinstance(result.error, OperationalError)
    assert len(writer.calls) == 1


def test_journal_writer_degraded(tmp_path: Path) -> None:
    error = RecoverableError(
        "persist degraded",
        module="tests.journal",
        reason_code="TEST_PERSIST_DEGRADED",
    )
    writer = FakeJournalWriter(results=[Result.degraded(None, error, error.reason_code)])
    orchestrator, _ = _build_orchestrator(tmp_path, journal_writer=writer)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "TEST_PERSIST_DEGRADED"
    assert isinstance(result.error, RecoverableError)
