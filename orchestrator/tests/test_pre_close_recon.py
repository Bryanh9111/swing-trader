from __future__ import annotations

from dataclasses import dataclass
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
        raise AssertionError("Universe plugin should not execute during AFTER_MARKET_RECON.")


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
        raise AssertionError("Data plugin should not execute during AFTER_MARKET_RECON.")


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
        raise AssertionError("Scanner plugin should not execute during AFTER_MARKET_RECON.")


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
        raise AssertionError("EventGuard plugin should not execute during AFTER_MARKET_RECON.")


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
        raise AssertionError("Strategy plugin should not execute during AFTER_MARKET_RECON.")


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
        raise AssertionError("RiskGate plugin should not execute during AFTER_MARKET_RECON.")


class ExecutionSpyPlugin(_BaseTestPlugin):
    metadata = PluginMetadata(
        name="execution_spy",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.SIGNAL,
        enabled=True,
        description="Spy execution plugin for AFTER_MARKET_RECON orchestrator tests.",
    )

    last_payload: Mapping[str, Any] | None = None

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        type(self).execute_called = True
        ExecutionSpyPlugin.last_payload = dict(payload)
        if payload.get("post_close_recon"):
            return Result.success(
                {
                    "schema_version": "1.0.0",
                    "post_close_recon": True,
                    "open_positions_count": 0,
                    "open_orders_count": 0,
                }
            )
        return Result.success(
            {
                "schema_version": "1.0.0",
                "pre_close_recon": True,
                "orders_cancelled": 0,
                "stops_verified": 0,
                "positions_at_risk": [],
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


def _reset_spy_plugins() -> None:
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


def _make_config(run_type: str) -> dict[str, Any]:
    return {
        "mode": "PAPER",
        "run_type": run_type,
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


def test_pre_close_recon_sends_post_close_recon_flag(
    orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager],
) -> None:
    """AFTER_MARKET_RECON now sends post_close_recon (reconcile-only, no cancel)."""
    orchestrator, artifact_manager = orchestrator_env
    _reset_spy_plugins()

    result = orchestrator.execute_run(_make_config(RunType.AFTER_MARKET_RECON.value))
    assert result.status is ResultStatus.SUCCESS

    assert ExecutionSpyPlugin.execute_called is True
    assert ExecutionSpyPlugin.last_payload is not None
    assert ExecutionSpyPlugin.last_payload.get("post_close_recon") is True
    assert ExecutionSpyPlugin.last_payload.get("reconcile_first") is True
    assert ExecutionSpyPlugin.last_payload.get("pre_close_recon") is None

    assert UniverseFailPlugin.init_called is False
    assert DataFailPlugin.init_called is False
    assert ScannerFailPlugin.init_called is False

    run_id = _new_run_id(artifact_manager)
    metadata_result = artifact_manager.read_metadata(run_id)
    assert metadata_result.status is ResultStatus.SUCCESS
    assert metadata_result.data is not None
    assert metadata_result.data.run_type is RunType.AFTER_MARKET_RECON


def test_pre_close_cleanup_sends_pre_close_recon_flag(
    orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager],
) -> None:
    """PRE_CLOSE_CLEANUP sends pre_close_recon flag (cancel entries + verify stops)."""
    orchestrator, artifact_manager = orchestrator_env
    _reset_spy_plugins()

    result = orchestrator.execute_run(_make_config(RunType.PRE_CLOSE_CLEANUP.value))
    assert result.status is ResultStatus.SUCCESS

    assert ExecutionSpyPlugin.execute_called is True
    assert ExecutionSpyPlugin.last_payload is not None
    assert ExecutionSpyPlugin.last_payload.get("pre_close_recon") is True
    assert ExecutionSpyPlugin.last_payload.get("reconcile_first") is True
    assert ExecutionSpyPlugin.last_payload.get("post_close_recon") is None

    assert UniverseFailPlugin.init_called is False
    assert DataFailPlugin.init_called is False
    assert ScannerFailPlugin.init_called is False

    run_id = _new_run_id(artifact_manager)
    metadata_result = artifact_manager.read_metadata(run_id)
    assert metadata_result.status is ResultStatus.SUCCESS
    assert metadata_result.data is not None
    assert metadata_result.data.run_type is RunType.PRE_CLOSE_CLEANUP


@dataclass(slots=True)
class _FakeOrderManager:
    mappings: list[IntentOrderMapping]

    cancelled: list[str]
    submitted: list[TradeIntent]

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

    def cancel_order(self, client_order_id: str) -> Result[None]:
        self.cancelled.append(client_order_id)
        return Result.success(data=None)

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


def test_execution_pre_close_cancels_stale_entry_orders() -> None:
    entry_pending = _entry_intent(intent_id="E1", symbol="AAPL")
    entry_live = _entry_intent(intent_id="E2", symbol="MSFT")
    entry_filled = _entry_intent(intent_id="E3", symbol="NVDA")
    stop = _stop_intent(intent_id="S1", symbol="NVDA", parent_intent_id="E3")

    manager = _FakeOrderManager(
        mappings=[
            _mapping_for_intent(intent=entry_pending, client_order_id="C1", state=OrderState.PENDING),
            _mapping_for_intent(intent=entry_live, client_order_id="C2", state=OrderState.ACCEPTED),
            _mapping_for_intent(intent=entry_filled, client_order_id="C3", state=OrderState.FILLED),
            _mapping_for_intent(intent=stop, client_order_id="C4", state=OrderState.ACCEPTED),
        ],
        cancelled=[],
        submitted=[],
    )

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute({"pre_close_recon": True, "reconcile_first": True})
    assert result.status is ResultStatus.SUCCESS
    assert manager.cancelled == ["C1", "C2"]
    assert result.data is not None
    assert result.data["orders_cancelled"] == 2


def test_execution_pre_close_verifies_existing_protective_stops() -> None:
    entry = _entry_intent(intent_id="E1", symbol="AAPL", stop_loss_price=95.0)
    stop = _stop_intent(intent_id="S1", symbol="AAPL", parent_intent_id="E1")

    manager = _FakeOrderManager(
        mappings=[
            _mapping_for_intent(intent=entry, client_order_id="C1", state=OrderState.FILLED),
            _mapping_for_intent(intent=stop, client_order_id="C2", state=OrderState.ACCEPTED),
        ],
        cancelled=[],
        submitted=[],
    )

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute({"pre_close_recon": True, "reconcile_first": False})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["stops_verified"] == 1
    assert result.data["positions_at_risk"] == []


def test_execution_pre_close_generates_missing_protective_stops() -> None:
    entry = _entry_intent(intent_id="E1", symbol="AAPL", stop_loss_price=95.0)

    manager = _FakeOrderManager(
        mappings=[
            _mapping_for_intent(intent=entry, client_order_id="C1", state=OrderState.FILLED),
        ],
        cancelled=[],
        submitted=[],
    )

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute({"pre_close_recon": True, "reconcile_first": False})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["generated_stops"]
    assert manager.submitted
    submitted_intent = manager.submitted[0]
    assert submitted_intent.intent_type is IntentType.STOP_LOSS
    assert submitted_intent.metadata is not None
    assert submitted_intent.metadata.get("order_type") == "STOP_MARKET"
    assert result.data["positions_at_risk"] == []


def test_execution_post_close_recon_repairs_missing_stops() -> None:
    """post_close_recon should reconcile, snapshot, AND repair missing stops for overnight safety."""
    entry_pending = _entry_intent(intent_id="E1", symbol="AAPL")
    entry_filled = _entry_intent(intent_id="E2", symbol="MSFT", stop_loss_price=95.0)

    manager = _FakeOrderManager(
        mappings=[
            _mapping_for_intent(intent=entry_pending, client_order_id="C1", state=OrderState.PENDING),
            _mapping_for_intent(intent=entry_filled, client_order_id="C2", state=OrderState.FILLED),
        ],
        cancelled=[],
        submitted=[],
    )

    plugin = ExecutionPlugin(config={})
    plugin._validated = ExecutionPluginConfig(enabled=True, dry_run=False)
    plugin._manager = manager  # type: ignore[assignment]
    plugin._run_id = "RUN123"

    result = plugin.execute({"post_close_recon": True, "reconcile_first": True})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data["post_close_recon"] is True
    assert result.data["open_positions_count"] == 1
    assert result.data["open_orders_count"] == 1
    # No cancellations (post_close never cancels entry orders)
    assert manager.cancelled == []
    # Filled entry E2 is missing a SL → repair generates one
    assert len(manager.submitted) == 1
    submitted_intent = manager.submitted[0]
    assert submitted_intent.intent_type is IntentType.STOP_LOSS
    assert submitted_intent.symbol == "MSFT"
    assert submitted_intent.stop_loss_price == 95.0
    assert submitted_intent.parent_intent_id == "E2"
    assert len(result.data["generated_stops"]) == 1
