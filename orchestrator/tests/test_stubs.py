from __future__ import annotations

from typing import Any, Mapping

from common.interface import ResultStatus
from orchestrator.stubs import (
    DataStubPlugin,
    ScannerStubPlugin,
    UniverseStubPlugin,
    register_stub_plugins,
)
from plugins.registry import PluginRegistry


def _build_context(module_name: str, events: list[tuple[str, str, Mapping[str, Any] | None]]) -> Mapping[str, Any]:
    def emitter(event_type: str, mod_name: str, data: Mapping[str, Any] | None) -> None:
        events.append((event_type, mod_name, data))

    return {"module_name": module_name, "emit_event": emitter}


def test_universe_stub_execute_emits_events() -> None:
    plugin = UniverseStubPlugin()
    events: list[tuple[str, str, Mapping[str, Any] | None]] = []

    init_result = plugin.init(_build_context("universe", events))
    assert init_result.status is ResultStatus.SUCCESS

    execute_result = plugin.execute({})
    assert execute_result.status is ResultStatus.SUCCESS
    payload = execute_result.data
    assert isinstance(payload, Mapping)
    assert payload["symbols"] == []
    assert "asof" in payload

    cleanup_result = plugin.cleanup()
    assert cleanup_result.status is ResultStatus.SUCCESS

    event_types = [event[0] for event in events]
    assert "module.initialised" in event_types
    assert "module.executed" in event_types
    assert "module.cleaned_up" in event_types


def test_stub_validate_config_defaults() -> None:
    plugin = DataStubPlugin()
    result = plugin.validate_config({})

    assert result.status is ResultStatus.SUCCESS
    assert getattr(result.data, "enabled", None) is True
    assert getattr(result.data, "notes", None) is None


def test_register_stub_plugins_idempotent() -> None:
    registry = PluginRegistry()
    register_stub_plugins(registry)
    register_stub_plugins(registry)

    plugin_class = registry.get_plugin("scanner_stub")
    plugin = plugin_class()
    init_result = plugin.init({"module_name": "scanner"})
    assert init_result.status is ResultStatus.SUCCESS
