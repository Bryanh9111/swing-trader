from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import msgspec
import msgspec.json
import pytest

from common.exceptions import CriticalError, RecoverableError, ValidationError
from common.interface import DomainEvent, Result, ResultStatus
from journal import (
    ArtifactManager,
    JournalReader,
    JournalWriter,
    OperatingMode,
    RunMetadata,
    RunType,
)
from journal.validator import SnapshotValidator


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
    return "20240202-010203-PRE_MARKET_FULL_SCAN-123e4567-e89b-12d3-a456-426614174888"


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
    journal_writer.persist_complete_run(run_id, metadata, inputs, outputs, [])


def test_load_run_success(
    journal_writer: JournalWriter,
    journal_reader: JournalReader,
    run_id: str,
    metadata: RunMetadata,
) -> None:
    _persist_run(journal_writer, run_id, metadata)

    result = journal_reader.load_run(run_id)

    assert result.status is ResultStatus.SUCCESS
    payload = result.data
    assert payload["metadata"] == metadata
    assert "input.json" in payload["inputs"]
    assert "output.json" in payload["outputs"]


def test_load_inputs_degraded_when_directory_missing(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs_dir = artifact_manager.base_path / run_id / "inputs"
    shutil.rmtree(inputs_dir)

    result = journal_reader.load_inputs(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == journal_reader._ARTIFACT_MISSING_REASON  # type: ignore[attr-defined]
    assert result.data == {}


def test_load_events_decode_error(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    events_path = artifact_manager.base_path / run_id / "events" / "events.jsonl"
    events_path.write_text("{invalid json}", encoding="utf-8")

    result = journal_reader.load_events(run_id)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == journal_reader._EVENTS_DECODE_FAILED_REASON  # type: ignore[attr-defined]


def test_load_outputs_schema_mismatch(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    output_path = artifact_manager.base_path / run_id / "outputs" / "bad.json"
    payload = _snapshot("2.0.0", data={"omega": 1})
    output_path.write_bytes(msgspec.json.encode(payload))

    result = journal_reader.load_outputs(run_id)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == SnapshotValidator._SCHEMA_INCOMPATIBLE_REASON  # type: ignore[attr-defined]


def test_load_run_returns_failed_metadata(
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RecoverableError("metadata missing", module="tests", reason_code="META_FAIL")
    failure = Result.failed(error, error.reason_code)

    monkeypatch.setattr(journal_reader, "load_metadata", lambda _: failure)

    result = journal_reader.load_run(run_id)

    assert result is failure


def test_load_run_returns_failed_inputs(
    journal_reader: JournalReader,
    metadata: RunMetadata,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(journal_reader, "load_metadata", lambda _: Result.success(metadata))
    error = ValidationError("bad inputs", module="tests", reason_code="INPUT_FAIL")
    failure = Result.failed(error, error.reason_code)
    monkeypatch.setattr(journal_reader, "load_inputs", lambda _: failure)

    result = journal_reader.load_run(run_id)

    assert result is failure


def test_load_run_returns_failed_outputs(
    journal_reader: JournalReader,
    metadata: RunMetadata,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(journal_reader, "load_metadata", lambda _: Result.success(metadata))
    monkeypatch.setattr(journal_reader, "load_inputs", lambda _: Result.success({}))
    error = ValidationError("bad outputs", module="tests", reason_code="OUTPUT_FAIL")
    failure = Result.failed(error, error.reason_code)
    monkeypatch.setattr(journal_reader, "load_outputs", lambda _: failure)

    result = journal_reader.load_run(run_id)

    assert result is failure


def test_load_run_returns_failed_events(
    journal_reader: JournalReader,
    metadata: RunMetadata,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(journal_reader, "load_metadata", lambda _: Result.success(metadata))
    monkeypatch.setattr(journal_reader, "load_inputs", lambda _: Result.success({}))
    monkeypatch.setattr(journal_reader, "load_outputs", lambda _: Result.success({}))
    error = ValidationError("bad events", module="tests", reason_code="EVENT_FAIL")
    failure = Result.failed(error, error.reason_code)
    monkeypatch.setattr(journal_reader, "load_events", lambda _: failure)

    result = journal_reader.load_run(run_id)

    assert result is failure


def test_load_run_degraded_when_component_degraded(
    journal_reader: JournalReader,
    metadata: RunMetadata,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    degrade_error = RecoverableError("outputs degraded", module="tests", reason_code="DEG_OUT")
    degraded = Result.degraded({}, degrade_error, degrade_error.reason_code)

    monkeypatch.setattr(journal_reader, "load_metadata", lambda _: Result.success(metadata))
    monkeypatch.setattr(journal_reader, "load_inputs", lambda _: Result.success({"input.json": {}}))
    monkeypatch.setattr(journal_reader, "load_outputs", lambda _: degraded)
    monkeypatch.setattr(journal_reader, "load_events", lambda _: Result.success([]))

    result = journal_reader.load_run(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert result.data["metadata"] == metadata  # type: ignore[index]
    assert result.error is degrade_error
    assert result.reason_code == degrade_error.reason_code


def test_load_metadata_degraded_from_artifact_manager(
    journal_reader: JournalReader,
    artifact_manager: ArtifactManager,
    run_id: str,
    metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    degrade_error = RecoverableError("metadata not readable", module="tests", reason_code="META_DEG")
    degraded = Result.degraded(metadata, degrade_error, degrade_error.reason_code)

    monkeypatch.setattr(artifact_manager, "read_metadata", lambda _: degraded)

    result = journal_reader.load_metadata(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert result.error is degrade_error
    assert result.data == metadata


def test_load_events_permission_error(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    events_path = artifact_manager.base_path / run_id / "events" / "events.jsonl"
    original_read_text = Path.read_text

    def raise_permission(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == events_path:
            raise PermissionError("denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raise_permission)

    result = journal_reader.load_events(run_id)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, CriticalError)
    assert result.reason_code == journal_reader._ARTIFACT_MISSING_REASON  # type: ignore[attr-defined]


def test_load_events_missing_file_degraded(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    events_path = artifact_manager.base_path / run_id / "events" / "events.jsonl"
    events_path.unlink()

    result = journal_reader.load_events(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.data == []


def test_load_events_os_error_degraded(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    events_path = artifact_manager.base_path / run_id / "events" / "events.jsonl"
    original_read_text = Path.read_text

    def raise_os_error(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == events_path:
            raise OSError("disk failure")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", raise_os_error)

    result = journal_reader.load_events(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == journal_reader._ARTIFACT_DECODE_FAILED_REASON  # type: ignore[attr-defined]


def test_load_events_ignores_blank_lines(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    events_path = artifact_manager.base_path / run_id / "events" / "events.jsonl"
    event = DomainEvent(
        event_id="evt-123",
        event_type="journal.test",
        run_id=run_id,
        module="journal.tests",
        timestamp_ns=1,
        data={"value": 1},
    )
    line = msgspec.json.encode(event).decode("utf-8")
    events_path.write_text(f"\n{line}\n", encoding="utf-8")

    result = journal_reader.load_events(run_id)

    assert result.status is ResultStatus.SUCCESS
    assert len(result.data or []) == 1
    assert isinstance(result.data[0], DomainEvent)  # type: ignore[index]


def test_load_snapshot_directory_skips_non_files(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs_dir = artifact_manager.base_path / run_id / "inputs"
    (inputs_dir / "nested").mkdir()

    result = journal_reader._load_snapshot_directory(run_id, subdir="inputs", allow_yaml=True)

    assert result.status is ResultStatus.SUCCESS
    assert result.data == {}


def test_load_snapshot_directory_handles_critical_error(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs_dir = artifact_manager.base_path / run_id / "inputs"
    sample_path = inputs_dir / "input.json"
    sample_path.write_bytes(msgspec.json.encode(_snapshot("1.0.0")))
    critical_error = CriticalError("permission denied", module="tests", reason_code="CRIT")

    def raise_critical(*_: Any, **__: Any) -> dict[str, Any]:
        raise critical_error

    monkeypatch.setattr(journal_reader, "_read_snapshot", raise_critical)

    result = journal_reader._load_snapshot_directory(run_id, subdir="inputs", allow_yaml=True)

    assert result.status is ResultStatus.FAILED
    assert result.error is critical_error


def test_load_snapshot_directory_handles_recoverable_error(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs_dir = artifact_manager.base_path / run_id / "inputs"
    sample_path = inputs_dir / "input.json"
    sample_path.write_bytes(msgspec.json.encode(_snapshot("1.0.0")))
    recoverable_error = RecoverableError("corrupt payload", module="tests", reason_code="REC")

    def raise_recoverable(*_: Any, **__: Any) -> dict[str, Any]:
        raise recoverable_error

    monkeypatch.setattr(journal_reader, "_read_snapshot", raise_recoverable)

    result = journal_reader._load_snapshot_directory(run_id, subdir="inputs", allow_yaml=True)

    assert result.status is ResultStatus.DEGRADED
    assert result.error is recoverable_error
    assert result.data == {}


def test_load_snapshot_directory_handles_validation_error(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs_dir = artifact_manager.base_path / run_id / "inputs"
    sample_path = inputs_dir / "input.json"
    sample_path.write_bytes(msgspec.json.encode(_snapshot("1.0.0")))
    validation_error = ValidationError("invalid schema", module="tests", reason_code="VAL")

    def raise_validation(*_: Any, **__: Any) -> dict[str, Any]:
        raise validation_error

    monkeypatch.setattr(journal_reader, "_read_snapshot", raise_validation)

    result = journal_reader._load_snapshot_directory(run_id, subdir="inputs", allow_yaml=True)

    assert result.status is ResultStatus.FAILED
    assert result.error is validation_error


def test_read_snapshot_supports_yaml(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs_dir = artifact_manager.base_path / run_id / "inputs"
    yaml_path = inputs_dir / "config.yaml"
    yaml_path.write_text(
        "schema_version: '1.0.0'\nsystem_version: 'deadbeef'\nasof_timestamp: 100\nvalue: 1\n",
        encoding="utf-8",
    )

    payload = journal_reader._read_snapshot(yaml_path, allow_yaml=True)

    assert isinstance(payload, dict)
    assert payload["value"] == 1


def test_read_snapshot_permission_error_raises_critical(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs_dir = artifact_manager.base_path / run_id / "inputs"
    json_path = inputs_dir / "input.json"
    json_path.write_bytes(msgspec.json.encode(_snapshot("1.0.0")))
    original_read_bytes = Path.read_bytes

    def raise_permission(self: Path) -> bytes:
        if self == json_path:
            raise PermissionError("denied")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", raise_permission)

    with pytest.raises(CriticalError):
        journal_reader._read_snapshot(json_path, allow_yaml=False)


def test_read_snapshot_file_missing_raises_recoverable(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs_dir = artifact_manager.base_path / run_id / "inputs"
    missing_path = inputs_dir / "missing.json"

    with pytest.raises(RecoverableError):
        journal_reader._read_snapshot(missing_path, allow_yaml=False)


def test_read_snapshot_decode_error_raises_recoverable(
    artifact_manager: ArtifactManager,
    journal_reader: JournalReader,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)
    inputs_dir = artifact_manager.base_path / run_id / "inputs"
    json_path = inputs_dir / "input.json"
    json_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(RecoverableError):
        journal_reader._read_snapshot(json_path, allow_yaml=False)
