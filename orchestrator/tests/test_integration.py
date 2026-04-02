from __future__ import annotations

from pathlib import Path
from typing import Any

from common.events import InMemoryEventBus
from common.interface import Result, ResultStatus
from common.logging import StructlogLoggerFactory
from journal import ArtifactManager, JournalReader, JournalWriter, ReplayEngine, ReplayMode
from orchestrator import EODScanOrchestrator, register_stub_plugins
from plugins.registry import PluginRegistry


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


def test_end_to_end_run_and_replay(tmp_path: Path) -> None:
    artifact_manager = ArtifactManager(base_path=tmp_path / "artifacts")
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

    run_result = orchestrator.execute_run(_stub_config())
    assert run_result.status is ResultStatus.SUCCESS
    run_summary = run_result.data
    assert run_summary is not None

    run_dirs = [path for path in artifact_manager.base_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_id = run_dirs[0].name

    reader = JournalReader(artifact_manager)

    def executor(metadata: Any, inputs: dict[str, Any], events: list[Any], mode: ReplayMode) -> Result[dict[str, Any]]:
        return Result.success(
            {
                "universe.json": {"schema_version": "1.0.0", "symbols": [], "asof": 0},
                "candidates.json": {"schema_version": "1.0.0", "candidates": []},
            }
        )

    replay_engine = ReplayEngine(
        reader,
        executor=executor,
    )

    replay_result = replay_engine.replay_run(run_id, ReplayMode.VALIDATE_ONLY)
    assert replay_result.status is ResultStatus.SUCCESS

    payload = replay_result.data
    assert payload["outputs_match"] is True
    assert payload["metadata"].run_id == run_id
    assert payload["events"]
