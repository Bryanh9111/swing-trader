from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from common.events import InMemoryEventBus
from common.exceptions import OperationalError, RecoverableError
from common.interface import Result, ResultStatus
from common.logging import StructlogLoggerFactory
from journal import ArtifactManager, JournalWriter
from orchestrator import EODScanOrchestrator, register_real_plugins, register_stub_plugins
from orchestrator.stubs import DataStubPlugin, ScannerStubPlugin
from plugins.registry import PluginRegistry


@pytest.fixture()
def orchestrator_env(tmp_path: Path) -> tuple[EODScanOrchestrator, ArtifactManager]:
    artifact_base = tmp_path / "artifacts"
    artifact_manager = ArtifactManager(base_path=artifact_base)
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_stub_plugins(registry)

    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )

    return orchestrator, artifact_manager


def _stub_config() -> dict[str, Any]:
    return {
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


def _new_run_id(artifact_manager: ArtifactManager) -> str:
    directories = [path for path in artifact_manager.base_path.iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0].name


def test_execute_run_success(orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager]) -> None:
    orchestrator, artifact_manager = orchestrator_env
    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.SUCCESS
    summary = result.data
    assert summary is not None
    assert summary.status is ResultStatus.SUCCESS
    assert summary.candidates_count == 0

    run_id = _new_run_id(artifact_manager)
    metadata_result = artifact_manager.read_metadata(run_id)
    assert metadata_result.status is ResultStatus.SUCCESS
    metadata = metadata_result.data
    assert metadata.status == "completed"

    events_path = artifact_manager.base_path / run_id / "events" / "events.jsonl"
    assert events_path.exists()
    assert events_path.read_text(encoding="utf-8").strip()


def test_execute_run_degraded(monkeypatch: pytest.MonkeyPatch, orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager]) -> None:
    orchestrator, _ = orchestrator_env

    def degraded_execute(self: DataStubPlugin, payload: Any) -> Result[Any]:
        error = RecoverableError(
            "Data snapshot incomplete.",
            module="tests.data_stub",
            reason_code="TEST_DATA_DEGRADED",
        )
        return Result.degraded({"schema_version": "1.0.0", "bars": {}}, error, error.reason_code)

    monkeypatch.setattr(DataStubPlugin, "execute", degraded_execute)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.DEGRADED
    summary = result.data
    assert summary is not None
    assert summary.status is ResultStatus.DEGRADED


def test_market_regime_strategy_overrides_applied(monkeypatch: pytest.MonkeyPatch, orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager]) -> None:
    orchestrator, _ = orchestrator_env

    from orchestrator.stubs import StrategyStubPlugin

    captured: dict[str, Any] = {}

    def capture_validate_config(self: StrategyStubPlugin, config: Any) -> Result[Any]:
        captured["config"] = dict(config) if isinstance(config, dict) else config
        return Result.success(data=None)

    monkeypatch.setattr(StrategyStubPlugin, "validate_config", capture_validate_config)
    monkeypatch.setattr(
        StrategyStubPlugin,
        "execute",
        lambda self, payload: Result.success(_intent_set("AAPL", entry_price=None)),
    )
    monkeypatch.setattr(orchestrator._journal_writer, "persist_complete_run", lambda *args, **kwargs: Result.success(data=None))

    config = _stub_config()
    config.update(
        {
            "strategy_plugin": "strategy_stub",
            "market_regime": {"enabled": True, "mode": "bull"},
        }
    )
    config["plugins"].setdefault("strategies", []).append(
        {"name": "strategy_stub", "config": {"engine": {"tp_sl_ratio": 2.5}}}
    )

    result = orchestrator.execute_run(config)
    assert result.status in (ResultStatus.SUCCESS, ResultStatus.DEGRADED)

    strategy_config = captured.get("config")
    assert isinstance(strategy_config, dict)
    engine = strategy_config.get("engine")
    assert isinstance(engine, dict)
    assert engine.get("take_profit_pct") == 0.10  # bull_market.yaml take_profit_pct: 0.10

    outputs = orchestrator.last_outputs_by_module
    market_regime = outputs.get("market_regime")
    assert isinstance(market_regime, dict)
    applied = market_regime.get("applied_overrides")
    assert isinstance(applied, dict)
    strategy_overrides = applied.get("strategy")
    assert isinstance(strategy_overrides, dict)
    assert strategy_overrides.get("take_profit_pct") == 0.10  # bull_market.yaml take_profit_pct: 0.10


def test_market_regime_risk_gate_overrides_applied(monkeypatch: pytest.MonkeyPatch, orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager]) -> None:
    orchestrator, _ = orchestrator_env

    from orchestrator.stubs import RiskGateStubPlugin, StrategyStubPlugin

    captured: dict[str, Any] = {}
    original_validate_config = RiskGateStubPlugin.validate_config

    def capture_validate_config(self: RiskGateStubPlugin, config: Any) -> Result[Any]:
        captured["config"] = dict(config) if isinstance(config, dict) else config
        return original_validate_config(self, config)

    monkeypatch.setattr(RiskGateStubPlugin, "validate_config", capture_validate_config)
    monkeypatch.setattr(
        StrategyStubPlugin,
        "execute",
        lambda self, payload: Result.success(_intent_set("AAPL", entry_price=None)),
    )
    monkeypatch.setattr(orchestrator._journal_writer, "persist_complete_run", lambda *args, **kwargs: Result.success(data=None))

    config = _stub_config()
    config.update(
        {
            "strategy_plugin": "strategy_stub",
            "risk_gate_plugin": "risk_gate_stub",
            "market_regime": {"enabled": True, "mode": "bull"},
        }
    )
    config["plugins"].setdefault("strategies", []).append({"name": "strategy_stub", "config": {"engine": {}}})
    config["plugins"].setdefault("risk_policies", []).append(
        {"name": "risk_gate_stub", "config": {"portfolio": {"max_total_exposure": 0.85}}}
    )

    result = orchestrator.execute_run(config)
    assert result.status in (ResultStatus.SUCCESS, ResultStatus.DEGRADED)

    risk_gate_config = captured.get("config")
    assert isinstance(risk_gate_config, dict)
    portfolio = risk_gate_config.get("portfolio")
    assert isinstance(portfolio, dict)
    assert portfolio.get("max_leverage") == pytest.approx(1.00)  # bull_market.yaml max_total_exposure: 1.00

    outputs = orchestrator.last_outputs_by_module
    market_regime = outputs.get("market_regime")
    assert isinstance(market_regime, dict)
    applied = market_regime.get("applied_overrides")
    assert isinstance(applied, dict)
    risk_gate_overrides = applied.get("risk_gate")
    assert isinstance(risk_gate_overrides, dict)
    assert risk_gate_overrides.get("max_total_exposure") == pytest.approx(1.00)  # bull_market.yaml: 1.00


def test_execute_run_failure(monkeypatch: pytest.MonkeyPatch, orchestrator_env: tuple[EODScanOrchestrator, ArtifactManager]) -> None:
    orchestrator, artifact_manager = orchestrator_env

    def failing_execute(self: ScannerStubPlugin, payload: Any) -> Result[Any]:
        error = OperationalError(
            "Scanner stub failure.",
            module="tests.scanner_stub",
            reason_code="TEST_SCANNER_FAILED",
        )
        return Result.failed(error, error.reason_code)

    monkeypatch.setattr(ScannerStubPlugin, "execute", failing_execute)

    result = orchestrator.execute_run(_stub_config())

    assert result.status is ResultStatus.FAILED
    assert result.data is None
    assert isinstance(result.error, OperationalError)
    assert "run_summary" in result.error.details  # type: ignore[attr-defined]

    run_id = _new_run_id(artifact_manager)
    metadata_result = artifact_manager.read_metadata(run_id)
    assert metadata_result.status is ResultStatus.SUCCESS
    assert metadata_result.data.status == "failed"


def _price_series(symbol: str, *, close: float = 100.0, bars: int = 21) -> dict[str, Any]:
    import msgspec

    from data.interface import PriceBar, PriceSeriesSnapshot

    now = 1_700_000_000_000_000_000
    series = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="tests",
        asof_timestamp=now,
        symbol=symbol,
        timeframe="1d",
        bars=[
            PriceBar(
                timestamp=now + i * 86_400_000_000_000,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1_000_000,
            )
            for i in range(bars)
        ],
        source="tests",
        quality_flags={},
    )
    return msgspec.to_builtins(series)


def _intent_set(symbol: str, *, entry_price: float | None) -> dict[str, Any]:
    import msgspec

    from strategy.interface import IntentGroup, IntentSnapshot, IntentType, OrderIntentSet, TradeIntent

    now = 1_700_000_000_000_000_000
    intent = TradeIntent(
        intent_id="O-1",
        symbol=symbol,
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=now,
        entry_price=entry_price,
        reduce_only=False,
        reason_codes=["TEST"],
    )
    group = IntentGroup(group_id="G-1", symbol=symbol, intents=[intent], created_at_ns=now)
    intent_set = OrderIntentSet(
        schema_version="3.2.0",
        system_version="tests",
        asof_timestamp=now,
        intent_groups=[group],
        constraints_applied={},
        source_candidates=[symbol],
    )
    snapshot = IntentSnapshot(
        schema_version=IntentSnapshot.SCHEMA_VERSION,
        system_version=intent_set.system_version,
        asof_timestamp=intent_set.asof_timestamp,
        intent_sets=[intent_set],
        degradation_events=[],
    )
    return msgspec.to_builtins(snapshot)


def _read_output(artifact_manager: ArtifactManager, run_id: str, filename: str) -> dict[str, Any]:
    import msgspec.json

    path = artifact_manager.base_path / run_id / "outputs" / filename
    assert path.exists()
    raw = msgspec.json.decode(path.read_bytes())
    assert isinstance(raw, dict)
    return raw


def test_eod_scan_risk_gate_allows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_manager = ArtifactManager(base_path=tmp_path / "artifacts")
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_stub_plugins(registry)
    register_real_plugins(registry)

    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )

    from orchestrator.stubs import StrategyStubPlugin

    monkeypatch.setattr(
        StrategyStubPlugin,
        "execute",
        lambda self, payload: Result.success(_intent_set("AAPL", entry_price=None)),
    )

    config = _stub_config()
    config.update(
        {
            "strategy_plugin": "strategy_stub",
            "risk_gate_plugin": "risk_gate_real",
            "universe_plugin": "universe_stub",
            "data_plugin": "data_stub",
            "scanner_plugin": "scanner_stub",
            "account_equity": 100_000.0,
            "market_data": {"AAPL": _price_series("AAPL", close=100.0)},
        }
    )
    config["plugins"].setdefault("risk_policies", []).append(
        {"name": "risk_gate_real", "config": {
            "state_path": str(tmp_path / "safe_mode.json"),
            "checks": ["reduce_only", "order_count", "rate_limit", "price_band"],
        }}
    )

    result = orchestrator.execute_run(config)

    assert result.status is ResultStatus.SUCCESS

    run_id = _new_run_id(artifact_manager)
    output = _read_output(artifact_manager, run_id, "risk_decisions.json")
    decisions = output["decisions"]["decisions"]
    assert len(decisions) == 1
    assert decisions[0]["decision_type"] == "ALLOW"


def test_eod_scan_risk_gate_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_manager = ArtifactManager(base_path=tmp_path / "artifacts")
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_stub_plugins(registry)
    register_real_plugins(registry)

    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )

    from orchestrator.stubs import StrategyStubPlugin

    monkeypatch.setattr(
        StrategyStubPlugin,
        "execute",
        lambda self, payload: Result.success(_intent_set("AAPL", entry_price=200.0)),
    )

    config = _stub_config()
    config.update(
        {
            "strategy_plugin": "strategy_stub",
            "risk_gate_plugin": "risk_gate_real",
            "universe_plugin": "universe_stub",
            "data_plugin": "data_stub",
            "scanner_plugin": "scanner_stub",
            "account_equity": 100_000.0,
            "market_data": {"AAPL": _price_series("AAPL", close=100.0)},
        }
    )
    config["plugins"].setdefault("risk_policies", []).append(
        {"name": "risk_gate_real", "config": {
            "state_path": str(tmp_path / "safe_mode.json"),
            "checks": ["reduce_only", "order_count", "rate_limit", "price_band"],
        }}
    )

    result = orchestrator.execute_run(config)

    assert result.status is ResultStatus.SUCCESS

    run_id = _new_run_id(artifact_manager)
    output = _read_output(artifact_manager, run_id, "risk_decisions.json")
    decisions = output["decisions"]["decisions"]
    assert len(decisions) == 1
    assert decisions[0]["decision_type"] == "BLOCK"
    assert "SYMBOL.PRICE_BAND" in decisions[0]["reason_codes"]


def test_eod_scan_risk_gate_safe_mode_blocks_new_positions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import msgspec.json

    from risk_gate.interface import SafeModeState
    from risk_gate.safe_mode import SafeModePersistedState

    artifact_manager = ArtifactManager(base_path=tmp_path / "artifacts")
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_stub_plugins(registry)
    register_real_plugins(registry)

    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )

    safe_mode_path = tmp_path / "safe_mode.json"
    safe_mode_path.write_bytes(
        msgspec.json.encode(
            SafeModePersistedState(
                state=SafeModeState.SAFE_REDUCING,
                updated_at_ns=1,
                reason_codes=["TEST_SAFE_MODE"],
                details={},
            )
        )
    )

    from orchestrator.stubs import StrategyStubPlugin

    monkeypatch.setattr(
        StrategyStubPlugin,
        "execute",
        lambda self, payload: Result.success(_intent_set("AAPL", entry_price=None)),
    )

    config = _stub_config()
    config.update(
        {
            "strategy_plugin": "strategy_stub",
            "risk_gate_plugin": "risk_gate_real",
            "universe_plugin": "universe_stub",
            "data_plugin": "data_stub",
            "scanner_plugin": "scanner_stub",
            "account_equity": 100_000.0,
            "market_data": {"AAPL": _price_series("AAPL", close=100.0)},
        }
    )
    config["plugins"].setdefault("risk_policies", []).append(
        {"name": "risk_gate_real", "config": {"state_path": str(safe_mode_path)}}
    )

    result = orchestrator.execute_run(config)

    assert result.status is ResultStatus.DEGRADED

    run_id = _new_run_id(artifact_manager)
    output = _read_output(artifact_manager, run_id, "risk_decisions.json")
    assert output["decisions"]["safe_mode_active"] is True
    decisions = output["decisions"]["decisions"]
    assert len(decisions) == 1
    assert decisions[0]["decision_type"] == "BLOCK"
    assert "SAFE_MODE_REDUCING_ONLY" in decisions[0]["reason_codes"]
