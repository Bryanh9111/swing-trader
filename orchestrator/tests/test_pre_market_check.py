from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pytest

from common.events import InMemoryEventBus
from common.interface import Result, ResultStatus
from common.logging import StructlogLoggerFactory
from journal import ArtifactManager, JournalWriter
from journal.interface import RunType
from orchestrator import EODScanOrchestrator, register_stub_plugins
from orchestrator.plugins import ExecutionPlugin, ExecutionPluginConfig
from order_state_machine.interface import IntentOrderMapping, OrderState, ReconciliationResult
from plugins.interface import PluginCategory, PluginMetadata
from plugins.registry import PluginRegistry
from strategy.interface import IntentType, TradeIntent


class _BaseTestPlugin:
    init_called = False
    execute_called = False

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        type(self).init_called = True
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        return Result.success(dict(config) if isinstance(config, Mapping) else {})

    def cleanup(self) -> Result[None]:
        return Result.success(data=None)


class UniverseFailPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="universe_fail",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
        description="Fails if executed (universe).",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        type(self).execute_called = True
        raise AssertionError("Universe plugin should not execute during PRE_MARKET_CHECK.")


class DataFailPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="data_fail",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
        description="Fails if executed (data).",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        type(self).execute_called = True
        raise AssertionError("Data plugin should not execute during PRE_MARKET_CHECK.")


class ScannerFailPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="scanner_fail",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.SCANNER,
        enabled=True,
        description="Fails if executed (scanner).",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        type(self).execute_called = True
        raise AssertionError("Scanner plugin should not execute during PRE_MARKET_CHECK.")


class EventGuardFailPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="event_guard_fail",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.RISK_POLICY,
        enabled=True,
        description="Fails if executed (event guard).",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        type(self).execute_called = True
        raise AssertionError("EventGuard plugin should not execute during PRE_MARKET_CHECK.")


class StrategyFailPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="strategy_fail",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.STRATEGY,
        enabled=True,
        description="Fails if executed (strategy).",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        type(self).execute_called = True
        raise AssertionError("Strategy plugin should not execute during PRE_MARKET_CHECK.")


class RiskGateFailPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="risk_gate_fail",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.RISK_POLICY,
        enabled=True,
        description="Fails if executed (risk gate).",
    )

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        type(self).execute_called = True
        raise AssertionError("RiskGate plugin should not execute during PRE_MARKET_CHECK.")


class ExecutionSpyPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="execution_spy",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.SIGNAL,
        enabled=True,
        description="Spy execution plugin for PRE_MARKET_CHECK orchestrator tests.",
    )

    last_payload: Mapping[str, Any] | None = None

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        type(self).execute_called = True
        ExecutionSpyPlugin.last_payload = dict(payload)
        return Result.success(
            {
                "schema_version": "1.0.0",
                "pre_market_check": True,
                "connection": {"connected": True, "funds_available": True},
                "stops_verified": 0,
                "generated_stops": [],
                "positions_at_risk": [],
                "overnight_events": [],
                "orders_validated": 0,
                "conflicting_orders_cancelled": 0,
            }
        )


@pytest.fixture()
def orchestrator_env(tmp_path: Path) -> tuple[EODScanOrchestrator, ArtifactManager]:
    artifact_base = tmp_path / "artifacts"
    artifact_manager = ArtifactManager(base_path=artifact_base)
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_stub_plugins(registry)

    for plugin in (
        UniverseFailPlugin,
        DataFailPlugin,
        ScannerFailPlugin,
        EventGuardFailPlugin,
        StrategyFailPlugin,
        RiskGateFailPlugin,
        ExecutionSpyPlugin,
    ):
        registry.register_plugin(plugin.metadata.name, plugin)

    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )

    return orchestrator, artifact_manager


def _new_run_id(artifact_manager: ArtifactManager) -> str:
    directories = [path for path in artifact_manager.base_path.iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0].name


def test_pre_market_check_run_type_only_executes_execution_plugin(
    orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager],
) -> None:
    orchestrator, artifact_manager = orchestrator_env

    for plugin in (
        UniverseFailPlugin,
        DataFailPlugin,
        ScannerFailPlugin,
        EventGuardFailPlugin,
        StrategyFailPlugin,
        RiskGateFailPlugin,
        ExecutionSpyPlugin,
    ):
        plugin.init_called = False
        plugin.execute_called = False
    ExecutionSpyPlugin.last_payload = None

    config = {
        "mode": "PAPER",
        "run_type": RunType.PRE_MARKET_CHECK.value,
        "universe_plugin": "universe_fail",
        "data_plugin": "data_fail",
        "scanner_plugin": "scanner_fail",
        "event_guard_plugin": "event_guard_fail",
        "strategy_plugin": "strategy_fail",
        "risk_gate_plugin": "risk_gate_fail",
        "execution_plugin": "execution_spy",
        "plugins": {
            "data_sources": [{"name": "universe_fail", "config": {}}, {"name": "data_fail", "config": {}}],
            "scanners": [{"name": "scanner_fail", "config": {}}],
            "risk_policies": [{"name": "event_guard_fail", "config": {}}, {"name": "risk_gate_fail", "config": {}}],
            "strategies": [{"name": "strategy_fail", "config": {}}],
            "signals": [{"name": "execution_spy", "config": {}}],
        },
    }

    result = orchestrator.execute_run(config)
    assert result.status is ResultStatus.SUCCESS

    assert ExecutionSpyPlugin.execute_called is True
    assert ExecutionSpyPlugin.last_payload is not None
    assert ExecutionSpyPlugin.last_payload.get("pre_market_check") is True
    assert ExecutionSpyPlugin.last_payload.get("reconcile_first") is True
    assert ExecutionSpyPlugin.last_payload.get("check_stops") is True
    assert ExecutionSpyPlugin.last_payload.get("scan_overnight_events") is True
    assert ExecutionSpyPlugin.last_payload.get("validate_orders") is True

    assert UniverseFailPlugin.init_called is False
    assert DataFailPlugin.init_called is False
    assert ScannerFailPlugin.init_called is False
    assert EventGuardFailPlugin.init_called is False
    assert StrategyFailPlugin.init_called is False
    assert RiskGateFailPlugin.init_called is False

    run_id = _new_run_id(artifact_manager)
    metadata_result = artifact_manager.read_metadata(run_id)
    assert metadata_result.status is ResultStatus.SUCCESS
    assert metadata_result.data is not None
    assert metadata_result.data.run_type is RunType.PRE_MARKET_CHECK


@dataclass(slots=True)
class _FakeAdapter:
    connected: bool = True
    account_summary: dict[str, Any] | None = None
    connect_calls: int = 0

    def is_connected(self) -> bool:
        return bool(self.connected)

    def connect(self) -> Result[None]:
        self.connect_calls += 1
        self.connected = True
        return Result.success(data=None)

    def get_account_summary(self) -> Result[dict[str, Any]]:
        return Result.success(data=dict(self.account_summary or {}))


@dataclass(slots=True)
class _FakeOrderManager:
    mappings: list[IntentOrderMapping]
    submitted: list[TradeIntent]
    cancelled: list[str] | None = None

    def reconcile_periodic(self) -> Result[ReconciliationResult]:
        return Result.success(
            ReconciliationResult(
                updated_mappings=[],
                orphaned_broker_orders=[],
                missing_broker_orders=[],
                state_transitions=[],
                reconciled_at_ns=0,
            )
        )

    def get_all_orders(self) -> Result[list[IntentOrderMapping]]:
        return Result.success(list(self.mappings))

    def submit_order(self, intent: TradeIntent) -> Result[IntentOrderMapping]:
        self.submitted.append(intent)
        mapping = IntentOrderMapping(
            intent_id=intent.intent_id,
            client_order_id=f"COID-{intent.intent_id}",
            broker_order_id="B1",
            state=OrderState.ACCEPTED,
            created_at_ns=0,
            updated_at_ns=0,
            intent_snapshot=intent,
            metadata={},
        )
        self.mappings.append(mapping)
        return Result.success(mapping)

    def cancel_order(self, client_order_id: str) -> Result[None]:
        if self.cancelled is None:
            self.cancelled = []
        self.cancelled.append(str(client_order_id))
        return Result.success(data=None)


def _mapping_for_intent(*, intent: TradeIntent, client_order_id: str, state: OrderState) -> IntentOrderMapping:
    return IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id=client_order_id,
        broker_order_id="B1",
        state=state,
        created_at_ns=0,
        updated_at_ns=0,
        intent_snapshot=intent,
        metadata={},
    )


def _entry_intent(*, intent_id: str, symbol: str, stop_loss_price: float | None = None) -> TradeIntent:
    return TradeIntent(
        intent_id=intent_id,
        symbol=symbol,
        intent_type=IntentType.OPEN_LONG,
        quantity=10,
        created_at_ns=0,
        stop_loss_price=stop_loss_price,
    )


def _stop_intent(*, intent_id: str, symbol: str, parent_intent_id: str) -> TradeIntent:
    return TradeIntent(
        intent_id=intent_id,
        symbol=symbol,
        intent_type=IntentType.STOP_LOSS,
        quantity=10,
        created_at_ns=0,
        stop_loss_price=95.0,
        parent_intent_id=parent_intent_id,
        reduce_only=True,
    )


def test_execution_pre_market_reports_connection_health_and_funds() -> None:
    adapter = _FakeAdapter(
        connected=True,
        account_summary={
            "buying_power": 1000.0,
            "total_cash_value": 100.0,
        },
    )
    manager = _FakeOrderManager(mappings=[], submitted=[])

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._adapter = adapter  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute({"pre_market_check": True, "reconcile_first": False, "check_stops": False})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["connection"]["connected"] is True
    assert result.data["connection"]["funds_available"] is True


def test_execution_pre_market_verifies_existing_protective_stops() -> None:
    adapter = _FakeAdapter(connected=True, account_summary={})
    entry = _entry_intent(intent_id="E1", symbol="AAPL", stop_loss_price=95.0)
    stop = _stop_intent(intent_id="S1", symbol="AAPL", parent_intent_id="E1")

    manager = _FakeOrderManager(
        mappings=[
            _mapping_for_intent(intent=entry, client_order_id="C1", state=OrderState.FILLED),
            _mapping_for_intent(intent=stop, client_order_id="C2", state=OrderState.ACCEPTED),
        ],
        submitted=[],
    )

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._adapter = adapter  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute({"pre_market_check": True, "reconcile_first": False, "check_stops": True})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["stops_verified"] == 1
    assert result.data["positions_at_risk"] == []


def test_execution_pre_market_generates_missing_protective_stops() -> None:
    adapter = _FakeAdapter(connected=True, account_summary={})
    entry = _entry_intent(intent_id="E1", symbol="AAPL", stop_loss_price=95.0)

    manager = _FakeOrderManager(
        mappings=[_mapping_for_intent(intent=entry, client_order_id="C1", state=OrderState.FILLED)],
        submitted=[],
    )

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._adapter = adapter  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute({"pre_market_check": True, "reconcile_first": False, "check_stops": True})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["generated_stops"]
    assert manager.submitted
    submitted_intent = manager.submitted[0]
    assert submitted_intent.intent_type is IntentType.STOP_LOSS
    assert submitted_intent.metadata is not None
    assert submitted_intent.metadata.get("order_type") == "STOP_MARKET"
    assert submitted_intent.metadata.get("generated_by") == "PRE_MARKET_CHECK"
    assert result.data["positions_at_risk"] == []


def test_execution_pre_market_flags_missing_stop_loss_price() -> None:
    adapter = _FakeAdapter(connected=True, account_summary={})
    entry = _entry_intent(intent_id="E1", symbol="AAPL", stop_loss_price=None)

    manager = _FakeOrderManager(
        mappings=[_mapping_for_intent(intent=entry, client_order_id="C1", state=OrderState.FILLED)],
        submitted=[],
    )

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._adapter = adapter  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute({"pre_market_check": True, "reconcile_first": False, "check_stops": True})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert manager.submitted == []
    assert result.data["positions_at_risk"]
    assert result.data["positions_at_risk"][0]["reason"] == "MISSING_STOP_LOSS_PRICE"


def test_execution_pre_market_includes_overnight_sections_when_enabled() -> None:
    adapter = _FakeAdapter(connected=True, account_summary={})
    manager = _FakeOrderManager(mappings=[], submitted=[])

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._adapter = adapter  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute(
        {
            "pre_market_check": True,
            "reconcile_first": False,
            "check_stops": False,
            "scan_overnight_events": True,
            "validate_orders": True,
        }
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["overnight_events"] == []
    assert result.data["orders_validated"] == 0
    assert result.data["conflicting_orders_cancelled"] == 0


def test_execution_pre_market_scans_overnight_events_and_finds_critical() -> None:
    adapter = _FakeAdapter(connected=True, account_summary={})
    entry = _entry_intent(intent_id="E1", symbol="AAPL", stop_loss_price=95.0)
    manager = _FakeOrderManager(
        mappings=[_mapping_for_intent(intent=entry, client_order_id="C1", state=OrderState.FILLED)],
        submitted=[],
    )

    now_ns = int(datetime.now(UTC).timestamp() * 1e9)
    event_guard_snapshot = {
        "events": [
            {
                "symbol": "AAPL",
                "event_type": "EARNINGS",
                "risk_level": "HIGH",
                "event_date": now_ns,
                "source": "polygon_rest",
            }
        ]
    }

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._adapter = adapter  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute(
        {
            "pre_market_check": True,
            "reconcile_first": False,
            "check_stops": False,
            "scan_overnight_events": True,
            "validate_orders": False,
            "event_guard_snapshot": event_guard_snapshot,
        }
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["overnight_events"]
    assert result.data["overnight_events"][0]["symbol"] == "AAPL"
    assert result.data["overnight_events"][0]["risk_level"] == "HIGH"


def test_execution_pre_market_validates_orders_and_cancels_conflicting() -> None:
    adapter = _FakeAdapter(connected=True, account_summary={})
    entry = _entry_intent(intent_id="E1", symbol="AAPL", stop_loss_price=95.0)
    manager = _FakeOrderManager(
        mappings=[_mapping_for_intent(intent=entry, client_order_id="C1", state=OrderState.SUBMITTED)],
        submitted=[],
        cancelled=[],
    )

    now_ns = int(datetime.now(UTC).timestamp() * 1e9)
    event_guard_snapshot = {
        "events": [
            {
                "symbol": "AAPL",
                "event_type": "EARNINGS",
                "risk_level": "CRITICAL",
                "event_date": now_ns,
                "source": "polygon_rest",
            }
        ]
    }

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._adapter = adapter  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute(
        {
            "pre_market_check": True,
            "reconcile_first": False,
            "check_stops": False,
            "scan_overnight_events": True,
            "validate_orders": True,
            "event_guard_snapshot": event_guard_snapshot,
        }
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["orders_validated"] == 1
    assert result.data["conflicting_orders_cancelled"] == 1
    assert manager.cancelled == ["C1"]


def test_execution_pre_market_validates_orders_skips_non_critical_events() -> None:
    adapter = _FakeAdapter(connected=True, account_summary={})
    entry = _entry_intent(intent_id="E1", symbol="MSFT", stop_loss_price=95.0)
    manager = _FakeOrderManager(
        mappings=[_mapping_for_intent(intent=entry, client_order_id="C1", state=OrderState.SUBMITTED)],
        submitted=[],
        cancelled=[],
    )

    now_ns = int(datetime.now(UTC).timestamp() * 1e9)
    event_guard_snapshot = {
        "events": [
            {
                "symbol": "MSFT",
                "event_type": "DIVIDEND",
                "risk_level": "LOW",
                "event_date": now_ns,
                "source": "polygon_rest",
            }
        ]
    }

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._adapter = adapter  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute(
        {
            "pre_market_check": True,
            "reconcile_first": False,
            "check_stops": False,
            "scan_overnight_events": True,
            "validate_orders": True,
            "event_guard_snapshot": event_guard_snapshot,
        }
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["orders_validated"] == 1
    assert result.data["conflicting_orders_cancelled"] == 0
    assert manager.cancelled == []
