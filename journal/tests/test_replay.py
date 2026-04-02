from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from common.exceptions import OperationalError, RecoverableError, ValidationError
from common.interface import DomainEvent, Result, ResultStatus
from journal import (
    ArtifactManager,
    JournalReader,
    JournalWriter,
    OperatingMode,
    ReplayEngine,
    ReplayMode,
    SnapshotBase,
    RunMetadata,
    RunType,
)


@pytest.fixture
def base_path(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def artifact_manager(base_path: Path) -> ArtifactManager:
    return ArtifactManager(base_path=base_path)


@pytest.fixture
def journal_writer(artifact_manager: ArtifactManager) -> JournalWriter:
    return JournalWriter(artifact_manager=artifact_manager)


@pytest.fixture
def journal_reader(artifact_manager: ArtifactManager) -> JournalReader:
    return JournalReader(artifact_manager=artifact_manager)


@pytest.fixture
def run_id() -> str:
    return "20240203-010203-PRE_MARKET_FULL_SCAN-123e4567-e89b-12d3-a456-426614174777"


@pytest.fixture
def metadata(run_id: str) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        run_type=RunType.PRE_MARKET_FULL_SCAN,
        mode=OperatingMode.DRY_RUN,
        system_version="deadbeef",
        start_time=1,
        end_time=2,
        status="completed",
    )


def _snapshot(schema_version: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "system_version": "deadbeef",
        "asof_timestamp": 100,
    }
    payload.update(extra)
    return payload


def _persist_run(
    journal_writer: JournalWriter,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    inputs = {"input.json": _snapshot("1.0.0", data={"alpha": 1})}
    outputs = {"output.json": _snapshot("1.0.0", data={"omega": 99})}
    events: list[DomainEvent] = []
    journal_writer.persist_complete_run(run_id, metadata, inputs, outputs, events)


def _matching_executor(
    metadata: RunMetadata,
    inputs: dict[str, Any],
    events: list[DomainEvent],
    mode: ReplayMode,
) -> Result[dict[str, Any]]:
    del metadata, inputs, events, mode
    outputs = {"output.json": _snapshot("1.0.0", data={"omega": 99})}
    return Result.success(outputs)


def _mismatch_executor(
    metadata: RunMetadata,
    inputs: dict[str, Any],
    events: list[DomainEvent],
    mode: ReplayMode,
) -> Result[dict[str, Any]]:
    del metadata, inputs, events, mode
    outputs = {"output.json": _snapshot("1.0.0", data={"omega": 100})}
    return Result.success(outputs)


class SampleSnapshot(SnapshotBase, frozen=True, kw_only=True):
    payload: dict[str, Any]


def test_replay_full_success(
    journal_writer: JournalWriter,
    journal_reader: JournalReader,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    _persist_run(journal_writer, run_id, metadata)
    replay_engine = ReplayEngine(journal_reader, _matching_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result.status is ResultStatus.SUCCESS
    payload = result.data
    assert payload["outputs_match"] is True
    assert payload["replay_outputs"]["output.json"]["data"]["omega"] == 99


def test_replay_validate_only_mismatch_fails(
    journal_writer: JournalWriter,
    journal_reader: JournalReader,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    _persist_run(journal_writer, run_id, metadata)
    replay_engine = ReplayEngine(journal_reader, _mismatch_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.VALIDATE_ONLY)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == replay_engine._REPLAY_OUTPUT_MISMATCH_REASON  # type: ignore[attr-defined]


def test_replay_executor_exception_propagates_failure(
    journal_writer: JournalWriter,
    journal_reader: JournalReader,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    _persist_run(journal_writer, run_id, metadata)

    def failing_executor(*args: Any, **kwargs: Any) -> Result[dict[str, Any]]:
        del args, kwargs
        raise RuntimeError("boom")

    replay_engine = ReplayEngine(journal_reader, failing_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == replay_engine._REPLAY_EXECUTION_FAILED_REASON  # type: ignore[attr-defined]


def test_replay_degraded_when_outputs_missing(
    journal_writer: JournalWriter,
    journal_reader: JournalReader,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    _persist_run(journal_writer, run_id, metadata)
    outputs_dir = artifact_manager.base_path / run_id / "outputs"
    shutil.rmtree(outputs_dir)
    replay_engine = ReplayEngine(journal_reader, _matching_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == journal_reader._ARTIFACT_MISSING_REASON  # type: ignore[attr-defined]


def test_replay_returns_reader_failure(
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = OperationalError("load failure", module="tests", reason_code="LOAD_FAIL")
    failure = Result.failed(error, error.reason_code)
    monkeypatch.setattr(journal_reader, "load_run", lambda _: failure)
    replay_engine = ReplayEngine(journal_reader, _matching_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result is failure


def test_replay_missing_metadata_failure(
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"metadata": None, "inputs": {}, "outputs": {}, "events": []}
    success = Result.success(payload)
    monkeypatch.setattr(journal_reader, "load_run", lambda _: success)
    replay_engine = ReplayEngine(journal_reader, _matching_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == replay_engine._REPLAY_INVALID_METADATA_REASON  # type: ignore[attr-defined]


def test_replay_executor_failed_result(
    journal_writer: JournalWriter,
    journal_reader: JournalReader,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    _persist_run(journal_writer, run_id, metadata)
    error = OperationalError("executor failed", module="tests", reason_code="EXEC_FAIL")

    def failing_executor(*_: Any, **__: Any) -> Result[dict[str, Any]]:
        return Result.failed(error, error.reason_code)

    replay_engine = ReplayEngine(journal_reader, failing_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result.status is ResultStatus.FAILED
    assert result.error is error


def test_replay_validation_failure_propagates(
    journal_writer: JournalWriter,
    journal_reader: JournalReader,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _persist_run(journal_writer, run_id, metadata)
    error = ValidationError("invalid outputs", module="tests", reason_code="VAL_FAIL")
    failure = Result.failed(error, error.reason_code)
    replay_engine = ReplayEngine(journal_reader, _matching_executor)
    monkeypatch.setattr(replay_engine, "_validate_outputs", lambda *_: failure)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result is failure


def test_replay_degraded_when_reader_degraded(
    journal_reader: JournalReader,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RecoverableError("partial load", module="tests", reason_code="LOAD_DEG")
    payload = {"metadata": metadata, "inputs": {}, "outputs": {}, "events": []}
    degraded = Result.degraded(payload, error, error.reason_code)
    monkeypatch.setattr(journal_reader, "load_run", lambda _: degraded)
    replay_engine = ReplayEngine(journal_reader, _matching_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result.status is ResultStatus.DEGRADED
    assert result.error is error


def test_replay_degraded_when_executor_degraded(
    journal_writer: JournalWriter,
    journal_reader: JournalReader,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    _persist_run(journal_writer, run_id, metadata)
    error = RecoverableError("executor degraded", module="tests", reason_code="EXEC_DEG")

    def degraded_executor(
        _metadata: RunMetadata,
        inputs: dict[str, Any],
        events: list[DomainEvent],
        mode: ReplayMode,
    ) -> Result[dict[str, Any]]:
        del inputs, events, mode
        return Result.degraded({}, error, error.reason_code)

    replay_engine = ReplayEngine(journal_reader, degraded_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result.status is ResultStatus.DEGRADED
    assert result.error is error


def test_replay_full_mode_detects_mismatch(
    journal_writer: JournalWriter,
    journal_reader: JournalReader,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    _persist_run(journal_writer, run_id, metadata)
    replay_engine = ReplayEngine(journal_reader, _mismatch_executor)

    result = replay_engine.replay_run(run_id, ReplayMode.FULL)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == replay_engine._REPLAY_OUTPUT_MISMATCH_REASON  # type: ignore[attr-defined]


def test_validate_outputs_failure(journal_reader: JournalReader) -> None:
    replay_engine = ReplayEngine(journal_reader, _matching_executor)
    error = ValidationError("schema mismatch", module="tests", reason_code="VAL_FAIL")
    failure = Result.failed(error, error.reason_code)

    class Validator:
        def validate_schema_version(self, *_: Any, **__: Any) -> Result[None]:
            return failure

    replay_engine._validator = Validator()  # type: ignore[assignment]
    outputs = {"output.json": _snapshot("1.0.0")}

    result = replay_engine._validate_outputs(outputs)

    assert result.status is ResultStatus.FAILED
    assert result.error is error


def test_normalise_value_handles_snapshot_structures(journal_reader: JournalReader) -> None:
    replay_engine = ReplayEngine(journal_reader, _matching_executor)
    snapshot = SampleSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        payload={"value": 1},
    )
    outputs = {
        "a": snapshot,
        "b": {"nested": snapshot},
        "c": [snapshot],
        "d": (snapshot,),
    }

    normalised = replay_engine._normalise_outputs(outputs)

    assert isinstance(normalised["a"], dict)
    assert normalised["b"]["nested"]["payload"]["value"] == 1
    assert isinstance(normalised["c"], list)
    assert isinstance(normalised["d"], tuple)
