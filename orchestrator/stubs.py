"""Stub plugin implementations for the PRE_MARKET_FULL_SCAN orchestrator.

These lightweight plugins satisfy the plugin contracts defined in
``plugins.interface`` while returning empty payloads. They enable the Phase 1.5
orchestrator to exercise the full pipeline without depending on production
data sources or scanners.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import msgspec

from common.interface import Result
from plugins.interface import PluginBase, PluginCategory, PluginContext, PluginMetadata

from risk_gate.interface import DecisionType, RiskDecision, RiskDecisionSet, RiskGateOutput, SafeModeState
from strategy.interface import IntentSnapshot, OrderIntentSet

__all__ = [
    "UniverseStubPlugin",
    "DataStubPlugin",
    "ScannerStubPlugin",
    "StrategyStubPlugin",
    "RiskGateStubPlugin",
    "register_stub_plugins",
]

EventEmitter = Callable[[str, str, Mapping[str, Any] | None], None]


class StubPluginConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Configuration schema shared by the stub plugins."""

    enabled: bool = True
    notes: str | None = None


class _BaseStubPlugin:
    """Common behaviour shared by all stub plugins."""

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._module_name: str = self.__class__.__name__
        self._emit_event: EventEmitter | None = None

    def init(self, context: PluginContext | None = None) -> Result[None]:
        """Initialise the plugin and capture the event emitter callback."""

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str):
                self._module_name = module_name

            emitter = context.get("emit_event")
            if callable(emitter):
                self._emit_event = emitter  # type: ignore[assignment]

        self._emit("module.initialised", {"timestamp_ns": time.time_ns()})
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[StubPluginConfig]:
        """Return a validated configuration payload."""

        if not isinstance(config, Mapping):
            return Result.success(StubPluginConfig())

        payload = {
            "enabled": bool(config.get("enabled", True)),
            "notes": str(config["notes"]) if "notes" in config and config["notes"] is not None else None,
        }
        return Result.success(StubPluginConfig(**payload))

    def cleanup(self) -> Result[None]:
        """Emit a cleanup event for observability."""

        self._emit("module.cleaned_up", {"timestamp_ns": time.time_ns()})
        return Result.success(data=None)

    def _emit(self, event_type: str, data: Mapping[str, Any] | None = None) -> None:
        if self._emit_event is None:
            return
        self._emit_event(event_type, self._module_name, data)


class UniverseStubPlugin(_BaseStubPlugin, PluginBase[StubPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Universe plugin returning an empty universe payload."""

    metadata = PluginMetadata(
        name="universe_stub",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
        description="Stub universe provider returning an empty symbol list.",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        universe_payload = {"schema_version": "1.0.0", "symbols": [], "asof": 0}
        self._emit("module.executed", {"symbols_count": 0})
        return Result.success(universe_payload)


class DataStubPlugin(_BaseStubPlugin, PluginBase[StubPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Data plugin returning an empty data snapshot."""

    metadata = PluginMetadata(
        name="data_stub",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
        description="Stub data source returning empty OHLCV payloads.",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        data_payload = {"schema_version": "1.0.0", "bars": {}, "asof": 0}
        self._emit("module.executed", {"bars_count": 0})
        return Result.success(data_payload)


class ScannerStubPlugin(_BaseStubPlugin, PluginBase[StubPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Scanner plugin returning an empty candidate list."""

    metadata = PluginMetadata(
        name="scanner_stub",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.SCANNER,
        enabled=True,
        description="Stub scanner producing zero trade candidates.",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        candidates: Mapping[str, Any] = {"schema_version": "1.0.0", "candidates": []}
        self._emit("module.executed", {"candidates_count": 0})
        return Result.success(candidates)


class StrategyStubPlugin(_BaseStubPlugin, PluginBase[StubPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Strategy plugin returning an empty intent set payload."""

    metadata = PluginMetadata(
        name="strategy_stub",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.STRATEGY,
        enabled=True,
        description="Stub strategy plugin producing zero trade intents.",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        self._emit("module.executed", {"intent_groups": 0})
        # Keep stub outputs journal-silent (no schema_version) for deterministic
        # end-to-end orchestrator + replay tests.
        return Result.success({"intent_sets": []})


class RiskGateStubPlugin(_BaseStubPlugin, PluginBase[StubPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Risk Gate plugin returning an empty decision payload."""

    metadata = PluginMetadata(
        name="risk_gate_stub",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.RISK_POLICY,
        enabled=True,
        description="Stub risk gate plugin allowing all intents.",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        self._emit("module.executed", {"decisions": 0})
        # Keep stub outputs journal-silent (no schema_version) for deterministic replay tests.
        return Result.success({"decisions": []})


def register_stub_plugins(registry: "PluginRegistry") -> None:
    """Register all stub plugins with ``registry`` for orchestrator integration."""

    from plugins.registry import PluginRegistry  # Local import to avoid cycles.
    from common.exceptions import ConfigurationError

    if not isinstance(registry, PluginRegistry):
        raise TypeError("registry must be an instance of PluginRegistry.")

    for plugin in (
        UniverseStubPlugin,
        DataStubPlugin,
        ScannerStubPlugin,
        StrategyStubPlugin,
        RiskGateStubPlugin,
    ):
        try:
            registry.register_plugin(plugin.metadata.name, plugin)
        except ConfigurationError:
            # Ignore duplicates to keep the helper idempotent for repeated tests.
            continue
