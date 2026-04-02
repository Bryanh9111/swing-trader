from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import msgspec.json
import pytest

from common.exceptions import OperationalError, RecoverableError
from common.interface import DomainEvent, Result, ResultStatus
from journal import ArtifactManager, JournalWriter, OperatingMode, RunMetadata, RunType
from journal.validator import SnapshotValidator


@pytest.fixture
def base_path(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def artifact_manager(base_path: Path) -> ArtifactManager:
    return ArtifactManager(base_path=base_path)


@pytest.fixture
def journal_writer(artifact_manager: ArtifactManager) -> JournalWriter:
    return JournalWriter(artifact_manager=artifact_manager, validator=SnapshotValidator(), expected_snapshot_major="1")


@pytest.fixture
def run_id() -> str:
    return "20240201-010203-PRE_MARKET_FULL_SCAN-123e4567-e89b-12d3-a456-426614174999"


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
        "asof_timestamp": 123,
    }
    payload.update(extra)
    return payload


def _write_first_line(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return msgspec.json.decode(file.read())


def _read_events(path: Path) -> list[dict[str, Any]]:
    events_file = path / "events" / "events.jsonl"
    if not events_file.exists():
        return []
    raw = events_file.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return [msgspec.json.decode(line.encode("utf-8")) for line in raw.splitlines()]


def test_persist_complete_run_success(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    inputs: Mapping[str, Any] = {"input.json": _snapshot("1.0.0", data={"alpha": 1})}
    outputs: Mapping[str, Any] = {"output.json": _snapshot("1.0.0", data={"omega": 99})}
    events: Sequence[DomainEvent] = [
        DomainEvent(
            event_id="evt-1",
            event_type="test.event",
            run_id=run_id,
            module="journal.tests",
            timestamp_ns=100,
            data={"payload": 1},
        ),
    ]

    result = journal_writer.persist_complete_run(run_id, metadata, inputs, outputs, events)

    assert result.status is ResultStatus.SUCCESS
    run_dir = artifact_manager.base_path / run_id
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "inputs" / "input.json").exists()
    assert (run_dir / "outputs" / "output.json").exists()

    loaded_input = _write_first_line(run_dir / "inputs" / "input.json")
    assert loaded_input["data"]["alpha"] == 1
    loaded_output = _write_first_line(run_dir / "outputs" / "output.json")
    assert loaded_output["data"]["omega"] == 99

    recorded_events = _read_events(run_dir)
    assert len(recorded_events) == 1
    assert recorded_events[0]["event_type"] == "test.event"


def test_persist_complete_run_schema_mismatch(
    journal_writer: JournalWriter,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    inputs = {"input.json": _snapshot("2.0.0")}
    outputs = {}
    events: Sequence[DomainEvent] = []

    result = journal_writer.persist_complete_run(run_id, metadata, inputs, outputs, events)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == SnapshotValidator._SCHEMA_INCOMPATIBLE_REASON  # type: ignore[attr-defined]


def test_persist_complete_run_degraded_on_output_write(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = {"input.json": _snapshot("1.0.0")}
    outputs = {"output.json": _snapshot("1.0.0")}
    events: Sequence[DomainEvent] = []
    degrade_reason = "SIMULATED_DEGRADE"

    def degraded_write_output(*args: Any, **kwargs: Any) -> Result[Any]:
        error = RecoverableError(
            "simulated degrade",
            module="journal.tests",
            reason_code=degrade_reason,
        )
        return error.to_result(data=None, status=ResultStatus.DEGRADED)

    monkeypatch.setattr(artifact_manager, "write_output", degraded_write_output)

    result = journal_writer.persist_complete_run(run_id, metadata, inputs, outputs, events)

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == degrade_reason


def test_persist_complete_run_invalid_arguments(
    journal_writer: JournalWriter,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    result = journal_writer.persist_complete_run(run_id, metadata, inputs=[], outputs={}, events=[])

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == journal_writer._INVALID_ARGUMENT_REASON  # type: ignore[attr-defined]


def test_persist_complete_run_run_directory_failure(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = OperationalError("cannot create run directory", module="tests", reason_code="RUN_DIR_FAIL")
    failure = Result.failed(error, error.reason_code)
    monkeypatch.setattr(artifact_manager, "create_run_directory", lambda _: failure)

    result = journal_writer.persist_complete_run(run_id, metadata, {}, {}, [])

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == error.reason_code
    assert isinstance(result.error, OperationalError)


def test_persist_complete_run_metadata_failure(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_create = artifact_manager.create_run_directory
    run_dir_result = original_create(run_id)
    assert run_dir_result.status is ResultStatus.SUCCESS
    monkeypatch.setattr(
        artifact_manager,
        "create_run_directory",
        lambda *_: Result.success(run_dir_result.data),
    )
    error = OperationalError("cannot write metadata", module="tests", reason_code="META_FAIL")
    failure = Result.failed(error, error.reason_code)
    monkeypatch.setattr(artifact_manager, "write_metadata", lambda *_: failure)

    result = journal_writer.persist_complete_run(run_id, metadata, {}, {}, [])

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == error.reason_code
    assert isinstance(result.error, OperationalError)


def test_persist_complete_run_inputs_degraded(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_create = artifact_manager.create_run_directory
    run_dir_result = original_create(run_id)
    assert run_dir_result.status is ResultStatus.SUCCESS
    monkeypatch.setattr(
        artifact_manager,
        "create_run_directory",
        lambda *_: Result.success(run_dir_result.data),
    )
    inputs = {"input.json": _snapshot("1.0.0")}
    outputs: Mapping[str, Any] = {}
    events: Sequence[DomainEvent] = [
        DomainEvent(
            event_id="evt-1",
            event_type="journal.test",
            run_id=run_id,
            module="journal.tests",
            timestamp_ns=1,
            data=None,
        )
    ]
    error = RecoverableError("inputs degraded", module="tests", reason_code="INP_DEG")
    degrade = Result.degraded(None, error, error.reason_code)

    def patched_persist(
        _run_id: str,
        *,
        artifacts: Mapping[str, Any],
        is_input: bool,
    ) -> Result[None]:
        if is_input:
            return degrade
        return Result.success(data=None)

    monkeypatch.setattr(journal_writer, "_persist_artifacts", patched_persist)
    monkeypatch.setattr(journal_writer, "_append_events", lambda *_: Result.success(data=None))

    result = journal_writer.persist_complete_run(run_id, metadata, inputs, outputs, events)

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == error.reason_code
    assert isinstance(result.error, RecoverableError)


def test_persist_complete_run_outputs_failure(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_create = artifact_manager.create_run_directory
    run_dir_result = original_create(run_id)
    assert run_dir_result.status is ResultStatus.SUCCESS
    monkeypatch.setattr(
        artifact_manager,
        "create_run_directory",
        lambda *_: Result.success(run_dir_result.data),
    )
    inputs = {"input.json": _snapshot("1.0.0")}
    outputs = {"output.json": _snapshot("1.0.0")}
    events: Sequence[DomainEvent] = [
        DomainEvent(
            event_id="evt-1",
            event_type="journal.test",
            run_id=run_id,
            module="journal.tests",
            timestamp_ns=1,
            data=None,
        )
    ]
    error = OperationalError("outputs failed", module="tests", reason_code="OUT_FAIL")
    failure = Result.failed(error, error.reason_code)

    def patched_persist(
        _run_id: str,
        *,
        artifacts: Mapping[str, Any],
        is_input: bool,
    ) -> Result[None]:
        if is_input:
            return Result.success(data=None)
        return failure

    monkeypatch.setattr(journal_writer, "_persist_artifacts", patched_persist)
    monkeypatch.setattr(journal_writer, "_append_events", lambda *_: Result.success(data=None))

    result = journal_writer.persist_complete_run(run_id, metadata, inputs, outputs, events)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == error.reason_code
    assert isinstance(result.error, OperationalError)


def test_persist_complete_run_events_failure(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_create = artifact_manager.create_run_directory
    run_dir_result = original_create(run_id)
    assert run_dir_result.status is ResultStatus.SUCCESS
    monkeypatch.setattr(
        artifact_manager,
        "create_run_directory",
        lambda *_: Result.success(run_dir_result.data),
    )
    inputs = {"input.json": _snapshot("1.0.0")}
    outputs = {"output.json": _snapshot("1.0.0")}
    events: Sequence[DomainEvent] = [
        DomainEvent(
            event_id="evt-1",
            event_type="journal.test",
            run_id=run_id,
            module="journal.tests",
            timestamp_ns=1,
            data=None,
        )
    ]
    error = OperationalError("append failed", module="tests", reason_code="APPEND_FAIL")
    failure = Result.failed(error, error.reason_code)

    def patched_append(*_: Any, **__: Any) -> Result[None]:
        return failure

    monkeypatch.setattr(journal_writer, "_persist_artifacts", lambda *args, **kwargs: Result.success(data=None))
    monkeypatch.setattr(journal_writer, "_append_events", patched_append)

    result = journal_writer.persist_complete_run(run_id, metadata, inputs, outputs, events)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == error.reason_code
    assert isinstance(result.error, OperationalError)


def test_persist_complete_run_events_degraded(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_create = artifact_manager.create_run_directory
    run_dir_result = original_create(run_id)
    assert run_dir_result.status is ResultStatus.SUCCESS
    monkeypatch.setattr(
        artifact_manager,
        "create_run_directory",
        lambda *_: Result.success(run_dir_result.data),
    )
    inputs = {"input.json": _snapshot("1.0.0")}
    outputs = {"output.json": _snapshot("1.0.0")}
    events: Sequence[DomainEvent] = [
        DomainEvent(
            event_id="evt-1",
            event_type="journal.test",
            run_id=run_id,
            module="journal.tests",
            timestamp_ns=1,
            data=None,
        )
    ]
    error = RecoverableError("append degraded", module="tests", reason_code="APPEND_DEG")
    degraded = Result.degraded(None, error, error.reason_code)

    def patched_append(*_: Any, **__: Any) -> Result[None]:
        return degraded

    monkeypatch.setattr(journal_writer, "_persist_artifacts", lambda *args, **kwargs: Result.success(data=None))
    monkeypatch.setattr(journal_writer, "_append_events", patched_append)

    result = journal_writer.persist_complete_run(run_id, metadata, inputs, outputs, events)

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == error.reason_code
    assert isinstance(result.error, RecoverableError)


def test_persist_artifacts_empty_returns_success(
    journal_writer: JournalWriter,
    run_id: str,
) -> None:
    result = journal_writer._persist_artifacts(run_id, artifacts={}, is_input=True)

    assert result.status is ResultStatus.SUCCESS


def test_persist_artifacts_validation_degraded(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs = {"input.json": _snapshot("1.0.0")}
    error = RecoverableError("schema uncertain", module="tests", reason_code="VAL_DEG")
    degraded = Result.degraded(None, error, error.reason_code)

    monkeypatch.setattr(
        journal_writer._validator,
        "validate_schema_version",
        lambda *_: degraded,
    )

    result = journal_writer._persist_artifacts(run_id, artifacts=inputs, is_input=True)

    assert result.status is ResultStatus.DEGRADED
    assert result.error is error


def test_persist_artifacts_write_failure(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    outputs = {"output.json": _snapshot("1.0.0")}
    error = OperationalError("write failed", module="tests", reason_code="WRITE_FAIL")
    failure = Result.failed(error, error.reason_code)

    monkeypatch.setattr(
        artifact_manager,
        "write_output",
        lambda *_: failure,
    )

    result = journal_writer._persist_artifacts(run_id, artifacts=outputs, is_input=False)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == error.reason_code
    assert isinstance(result.error, OperationalError)


def test_append_events_failure(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    error = OperationalError("append failed", module="tests", reason_code="APP_FAIL")
    failure = Result.failed(error, error.reason_code)

    monkeypatch.setattr(artifact_manager, "append_event", lambda *_: failure)

    event = DomainEvent(
        event_id="evt-1",
        event_type="journal.test",
        run_id=run_id,
        module="journal.tests",
        timestamp_ns=1,
        data=None,
    )

    result = journal_writer._append_events(run_id, [event])

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == error.reason_code
    assert isinstance(result.error, OperationalError)


def test_append_events_degraded(
    journal_writer: JournalWriter,
    artifact_manager: ArtifactManager,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    error = RecoverableError("append degraded", module="tests", reason_code="APP_DEG")
    degraded = Result.degraded(None, error, error.reason_code)

    monkeypatch.setattr(artifact_manager, "append_event", lambda *_: degraded)

    event = DomainEvent(
        event_id="evt-1",
        event_type="journal.test",
        run_id=run_id,
        module="journal.tests",
        timestamp_ns=1,
        data=None,
    )

    result = journal_writer._append_events(run_id, [event])

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == error.reason_code
    assert isinstance(result.error, RecoverableError)
