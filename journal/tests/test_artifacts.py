from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import msgspec
import msgspec.json
import pytest
import yaml

from common.exceptions import CriticalError, OperationalError, RecoverableError
from common.interface import DomainEvent, Result, ResultStatus
from journal import ArtifactManager, OperatingMode, RunMetadata, RunType
import journal.artifacts as artifacts_module

if not hasattr(msgspec.json, "DecodeError"):
    setattr(msgspec.json, "DecodeError", msgspec.DecodeError)
if not hasattr(msgspec.json, "EncodeError"):
    setattr(msgspec.json, "EncodeError", msgspec.EncodeError)


@pytest.fixture
def base_path(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def artifact_manager(base_path: Path) -> ArtifactManager:
    return ArtifactManager(base_path=base_path)


@pytest.fixture
def run_id() -> str:
    return "20240101-010203-PRE_MARKET_FULL_SCAN-123e4567-e89b-12d3-a456-426614174000"


@pytest.fixture
def sample_metadata(run_id: str) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        run_type=RunType.PRE_MARKET_FULL_SCAN,
        mode=OperatingMode.DRY_RUN,
        system_version="deadbeef",
        start_time=1,
        end_time=None,
        status="running",
    )


@pytest.fixture
def sample_event(run_id: str) -> DomainEvent:
    return DomainEvent(
        event_id="evt-1",
        event_type="journal.test",
        run_id=run_id,
        module="journal.tests",
        timestamp_ns=100,
        data={"payload": 1},
    )


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    return {"alpha": 1, "beta": [1, 2, 3]}


def _read_json(path: Path, type_: Any) -> Any:
    return msgspec.json.decode(path.read_bytes(), type=type_)


def test_init_creates_base_path(tmp_path: Path) -> None:
    base_path = tmp_path / "journal_artifacts"
    ArtifactManager(base_path=base_path)

    assert base_path.exists()
    assert base_path.is_dir()


def test_create_run_directory_success(artifact_manager: ArtifactManager, run_id: str) -> None:
    result = artifact_manager.create_run_directory(run_id)

    assert result.status is ResultStatus.SUCCESS
    run_path = result.data
    assert isinstance(run_path, Path)
    assert run_path.exists()
    assert (run_path / "inputs").is_dir()
    assert (run_path / "outputs").is_dir()
    assert (run_path / "events").is_dir()
    assert (run_path / "events" / "events.jsonl").is_file()
    assert (run_path / "logs").is_dir()
    assert (run_path / "logs" / "run.log").is_file()


def test_create_run_directory_already_exists(artifact_manager: ArtifactManager, run_id: str) -> None:
    first = artifact_manager.create_run_directory(run_id)
    assert first.status is ResultStatus.SUCCESS

    second = artifact_manager.create_run_directory(run_id)

    assert second.status is ResultStatus.FAILED
    assert isinstance(second.error, OperationalError)
    assert second.reason_code == ArtifactManager._RUN_DIR_EXISTS_REASON


def test_create_run_directory_atomic(artifact_manager: ArtifactManager, run_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def tracking_replace(src: str | bytes | Path, dst: str | bytes | Path) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        calls.append((src_path, dst_path))
        original_replace(src, dst)

    monkeypatch.setattr("journal.artifacts.os.replace", tracking_replace)

    result = artifact_manager.create_run_directory(run_id)

    assert result.status is ResultStatus.SUCCESS
    assert calls, "os.replace should be invoked for atomic directory creation"
    temp_path, final_path = calls[0]
    assert temp_path.parent == artifact_manager.base_path
    assert temp_path.name.startswith(f".{run_id}-")
    assert final_path == artifact_manager.base_path / run_id
    assert not temp_path.exists()
    assert final_path.exists()


def test_create_run_directory_cleanup_on_error(
    artifact_manager: ArtifactManager,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dirs: list[Path] = []
    original_mkdtemp = artifacts_module.tempfile.mkdtemp

    def tracking_mkdtemp(*args: Any, **kwargs: Any) -> str:
        path = Path(original_mkdtemp(*args, **kwargs))
        temp_dirs.append(path)
        return str(path)

    def failing_replace(*_: Any, **__: Any) -> None:
        raise OSError("simulated failure")

    monkeypatch.setattr(artifacts_module.tempfile, "mkdtemp", tracking_mkdtemp)
    monkeypatch.setattr("journal.artifacts.os.replace", failing_replace)

    result = artifact_manager.create_run_directory(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == ArtifactManager._RUN_DIR_CREATION_FAILED_REASON
    assert temp_dirs, "mkdtemp should have been invoked"
    assert not temp_dirs[0].exists()
    assert not (artifact_manager.base_path / run_id).exists()
    assert list(artifact_manager.base_path.iterdir()) == []


def test_create_run_directory_permission_denied(
    artifact_manager: ArtifactManager,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_dirs: list[Path] = []
    original_mkdtemp = artifacts_module.tempfile.mkdtemp

    def tracking_mkdtemp(*args: Any, **kwargs: Any) -> str:
        path = Path(original_mkdtemp(*args, **kwargs))
        temp_dirs.append(path)
        return str(path)

    def failing_initialise(self: ArtifactManager, _: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(artifacts_module.tempfile, "mkdtemp", tracking_mkdtemp)
    monkeypatch.setattr(ArtifactManager, "_initialise_structure", failing_initialise)

    result = artifact_manager.create_run_directory(run_id)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, CriticalError)
    assert result.reason_code == ArtifactManager._RUN_DIR_PERMISSION_DENIED_REASON
    assert temp_dirs, "mkdtemp should have been invoked"
    assert not temp_dirs[0].exists()
    assert not (artifact_manager.base_path / run_id).exists()


def test_write_metadata_success(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_metadata: RunMetadata,
) -> None:
    artifact_manager.create_run_directory(run_id)

    result = artifact_manager.write_metadata(run_id, sample_metadata)

    assert result.status is ResultStatus.SUCCESS
    metadata_path = result.data
    assert isinstance(metadata_path, Path)
    stored = _read_json(metadata_path, RunMetadata)
    assert stored == sample_metadata


def test_write_metadata_missing_run(
    artifact_manager: ArtifactManager,
    sample_metadata: RunMetadata,
) -> None:
    result = artifact_manager.write_metadata("missing-run", sample_metadata)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, OperationalError)
    assert result.reason_code == ArtifactManager._ARTIFACT_MISSING_REASON


def test_read_metadata_success(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_metadata: RunMetadata,
) -> None:
    artifact_manager.create_run_directory(run_id)
    artifact_manager.write_metadata(run_id, sample_metadata)

    result = artifact_manager.read_metadata(run_id)

    assert result.status is ResultStatus.SUCCESS
    assert isinstance(result.data, RunMetadata)
    assert result.data == sample_metadata


def test_read_metadata_not_found(
    artifact_manager: ArtifactManager,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)

    result = artifact_manager.read_metadata(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == ArtifactManager._METADATA_NOT_FOUND_REASON


def test_read_metadata_invalid_json(
    artifact_manager: ArtifactManager,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    metadata_path = artifact_manager.base_path / run_id / ArtifactManager._METADATA_FILENAME
    metadata_path.write_text("{ invalid json", encoding="utf-8")

    result = artifact_manager.read_metadata(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == ArtifactManager._METADATA_DECODE_FAILED_REASON


def test_read_metadata_schema_missing(
    artifact_manager: ArtifactManager,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    metadata_path = artifact_manager.base_path / run_id / ArtifactManager._METADATA_FILENAME
    metadata_payload = {"run_id": run_id}
    metadata_path.write_bytes(msgspec.json.encode(metadata_payload))

    result = artifact_manager.read_metadata(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == ArtifactManager._METADATA_DECODE_FAILED_REASON


def test_read_metadata_schema_incompatible(
    artifact_manager: ArtifactManager,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    metadata_path = artifact_manager.base_path / run_id / ArtifactManager._METADATA_FILENAME
    metadata_path.write_bytes(msgspec.json.encode({"schema_version": "2.0.0"}))

    result = artifact_manager.read_metadata(run_id)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, OperationalError)
    assert result.reason_code == ArtifactManager._METADATA_SCHEMA_INCOMPATIBLE_REASON


@pytest.mark.parametrize(
    ("exception_factory", "expected_status", "expected_reason", "expected_error_type"),
    [
        (PermissionError, ResultStatus.FAILED, ArtifactManager._METADATA_PERMISSION_DENIED_REASON, CriticalError),
        (OSError, ResultStatus.DEGRADED, ArtifactManager._METADATA_NOT_FOUND_REASON, RecoverableError),
    ],
)
def test_read_metadata_io_errors(
    artifact_manager: ArtifactManager,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
    exception_factory: type[BaseException],
    expected_status: ResultStatus,
    expected_reason: str,
    expected_error_type: type[BaseException],
) -> None:
    artifact_manager.create_run_directory(run_id)
    metadata_path = artifact_manager.base_path / run_id / ArtifactManager._METADATA_FILENAME
    metadata_path.write_bytes(b"{}")

    def failing_read_bytes(self: Path) -> bytes:
        raise exception_factory("simulated failure")

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)

    result = artifact_manager.read_metadata(run_id)

    assert result.status is expected_status
    assert isinstance(result.error, expected_error_type)
    assert result.reason_code == expected_reason


def test_read_metadata_schema_non_mapping(
    artifact_manager: ArtifactManager,
    run_id: str,
) -> None:
    artifact_manager.create_run_directory(run_id)
    metadata_path = artifact_manager.base_path / run_id / ArtifactManager._METADATA_FILENAME
    metadata_path.write_text("[]", encoding="utf-8")

    result = artifact_manager.read_metadata(run_id)

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == ArtifactManager._METADATA_DECODE_FAILED_REASON


def test_write_input_json(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_payload: dict[str, Any],
) -> None:
    artifact_manager.create_run_directory(run_id)

    result = artifact_manager.write_input(run_id, "prices.json", sample_payload)

    assert result.status is ResultStatus.SUCCESS
    target_path = result.data
    assert isinstance(target_path, Path)
    assert _read_json(target_path, dict) == sample_payload


def test_write_input_invalid_filename(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_payload: dict[str, Any],
) -> None:
    artifact_manager.create_run_directory(run_id)

    result = artifact_manager.write_input(run_id, "nested/path.json", sample_payload)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, OperationalError)
    assert result.reason_code == "INVALID_ARTIFACT_FILENAME"


@pytest.mark.parametrize(
    ("exception_factory", "expected_status", "expected_reason", "expected_error_type"),
    [
        (PermissionError, ResultStatus.FAILED, ArtifactManager._ARTIFACT_PERMISSION_DENIED_REASON, CriticalError),
        (OSError, ResultStatus.DEGRADED, ArtifactManager._ARTIFACT_WRITE_FAILED_REASON, RecoverableError),
    ],
)
def test_write_input_yaml_io_errors(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    exception_factory: type[BaseException],
    expected_status: ResultStatus,
    expected_reason: str,
    expected_error_type: type[BaseException],
) -> None:
    artifact_manager.create_run_directory(run_id)

    def failing_open(*_: Any, **__: Any) -> None:
        raise exception_factory("simulated failure")

    monkeypatch.setattr("builtins.open", failing_open)

    result = artifact_manager.write_input(run_id, "config.yaml", sample_payload)

    assert result.status is expected_status
    assert isinstance(result.error, expected_error_type)
    assert result.reason_code == expected_reason


def test_write_input_yaml(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_payload: dict[str, Any],
) -> None:
    artifact_manager.create_run_directory(run_id)

    result = artifact_manager.write_input(run_id, "config.yaml", sample_payload)

    assert result.status is ResultStatus.SUCCESS
    target_path = result.data
    assert isinstance(target_path, Path)
    loaded = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    assert loaded == sample_payload


def test_write_output_missing_run(
    artifact_manager: ArtifactManager,
    sample_payload: dict[str, Any],
) -> None:
    result = artifact_manager.write_output("missing-run", "output.json", sample_payload)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, OperationalError)
    assert result.reason_code == ArtifactManager._ARTIFACT_MISSING_REASON


def test_write_output_json(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_payload: dict[str, Any],
) -> None:
    artifact_manager.create_run_directory(run_id)

    result = artifact_manager.write_output(run_id, "signals.json", sample_payload)

    assert result.status is ResultStatus.SUCCESS
    target_path = result.data
    assert isinstance(target_path, Path)
    assert _read_json(target_path, dict) == sample_payload


def test_write_output_unsupported_extension(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_payload: dict[str, Any],
) -> None:
    artifact_manager.create_run_directory(run_id)

    result = artifact_manager.write_output(run_id, "summary.txt", sample_payload)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, OperationalError)
    assert result.reason_code == "UNSUPPORTED_ARTIFACT_EXTENSION"


def test_write_output_json_permission_denied(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)

    def failing_open(*_: Any, **__: Any) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", failing_open)

    result = artifact_manager.write_output(run_id, "signals.json", sample_payload)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, CriticalError)
    assert result.reason_code == ArtifactManager._ARTIFACT_PERMISSION_DENIED_REASON


def test_write_output_json_os_error(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)

    def failing_open(*_: Any, **__: Any) -> None:
        raise OSError("io failure")

    monkeypatch.setattr("builtins.open", failing_open)

    result = artifact_manager.write_output(run_id, "signals.json", sample_payload)

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == ArtifactManager._ARTIFACT_WRITE_FAILED_REASON


def test_append_event(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_event: DomainEvent,
) -> None:
    artifact_manager.create_run_directory(run_id)

    result = artifact_manager.append_event(run_id, sample_event)

    assert result.status is ResultStatus.SUCCESS
    events_path = artifact_manager.base_path / run_id / "events" / ArtifactManager._EVENT_LOG_FILENAME
    lines = [line for line in events_path.read_bytes().splitlines() if line]
    assert len(lines) == 1
    stored_event = msgspec.json.decode(lines[0], type=DomainEvent)
    assert stored_event == sample_event


def test_append_event_multiple(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_event: DomainEvent,
) -> None:
    artifact_manager.create_run_directory(run_id)
    second_event = DomainEvent(
        event_id="evt-2",
        event_type="journal.test",
        run_id=run_id,
        module="journal.tests",
        timestamp_ns=200,
        data={"payload": 2},
    )

    artifact_manager.append_event(run_id, sample_event)
    artifact_manager.append_event(run_id, second_event)

    events_path = artifact_manager.base_path / run_id / "events" / ArtifactManager._EVENT_LOG_FILENAME
    lines = [line for line in events_path.read_bytes().splitlines() if line]
    decoded = [msgspec.json.decode(line, type=DomainEvent) for line in lines]
    assert decoded == [sample_event, second_event]


def test_append_event_missing_run(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_event: DomainEvent,
) -> None:
    result = artifact_manager.append_event(run_id, sample_event)

    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == ArtifactManager._ARTIFACT_WRITE_FAILED_REASON


def test_append_event_permission_denied(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_event: DomainEvent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_manager.create_run_directory(run_id)

    def failing_open(*_: Any, **__: Any) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", failing_open)

    result = artifact_manager.append_event(run_id, sample_event)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, CriticalError)
    assert result.reason_code == ArtifactManager._ARTIFACT_PERMISSION_DENIED_REASON


def test_list_runs_empty(artifact_manager: ArtifactManager) -> None:
    result = artifact_manager.list_runs()

    assert result.status is ResultStatus.SUCCESS
    assert result.data == []


def test_list_runs_multiple(
    artifact_manager: ArtifactManager,
    run_id: str,
) -> None:
    other_run_ids = [
        "20240102-010203-PRE_MARKET_FULL_SCAN-123e4567-e89b-12d3-a456-426614174001",
        "20240103-010203-PRE_MARKET_FULL_SCAN-123e4567-e89b-12d3-a456-426614174002",
    ]

    all_run_ids: Iterable[str] = [run_id, *other_run_ids]
    for value in all_run_ids:
        result = artifact_manager.create_run_directory(value)
        assert result.status is ResultStatus.SUCCESS

    listing = artifact_manager.list_runs()

    assert listing.status is ResultStatus.SUCCESS
    assert listing.data == sorted(all_run_ids)


@pytest.mark.parametrize(
    ("exception_factory", "expected_status", "expected_reason", "expected_error_type"),
    [
        (PermissionError, ResultStatus.FAILED, ArtifactManager._ARTIFACT_PERMISSION_DENIED_REASON, CriticalError),
        (OSError, ResultStatus.DEGRADED, ArtifactManager._ARTIFACT_ENUMERATION_FAILED_REASON, RecoverableError),
    ],
)
def test_list_runs_io_errors(
    artifact_manager: ArtifactManager,
    monkeypatch: pytest.MonkeyPatch,
    exception_factory: type[BaseException],
    expected_status: ResultStatus,
    expected_reason: str,
    expected_error_type: type[BaseException],
) -> None:
    def failing_iterdir(self: Path) -> Iterable[Path]:
        raise exception_factory("simulated failure")

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)

    result = artifact_manager.list_runs()

    assert result.status is expected_status
    assert isinstance(result.error, expected_error_type)
    assert result.reason_code == expected_reason


def test_write_metadata_missing_run_directory_error_includes_details(
    artifact_manager: ArtifactManager,
    sample_metadata: RunMetadata,
) -> None:
    result = artifact_manager.write_metadata("missing-run", sample_metadata)

    assert isinstance(result.error, OperationalError)
    assert result.error.details["run_id"] == "missing-run"


@pytest.mark.parametrize(
    ("exception_factory", "expected_status", "expected_reason", "expected_error_type"),
    [
        (PermissionError, ResultStatus.FAILED, ArtifactManager._METADATA_PERMISSION_DENIED_REASON, CriticalError),
        (OSError, ResultStatus.DEGRADED, ArtifactManager._METADATA_WRITE_FAILED_REASON, RecoverableError),
    ],
)
def test_write_metadata_io_errors(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_metadata: RunMetadata,
    monkeypatch: pytest.MonkeyPatch,
    exception_factory: type[BaseException],
    expected_status: ResultStatus,
    expected_reason: str,
    expected_error_type: type[BaseException],
) -> None:
    artifact_manager.create_run_directory(run_id)
    metadata_path = artifact_manager.base_path / run_id / ArtifactManager._METADATA_FILENAME

    def failing_write_bytes(self: Path, _: bytes) -> None:
        raise exception_factory("simulated failure")

    monkeypatch.setattr(Path, "write_bytes", failing_write_bytes)

    result = artifact_manager.write_metadata(run_id, sample_metadata)

    assert result.status is expected_status
    assert isinstance(result.error, expected_error_type)
    assert result.reason_code == expected_reason
    assert metadata_path.exists() is False


def test_result_types(
    artifact_manager: ArtifactManager,
    run_id: str,
    sample_metadata: RunMetadata,
    sample_event: DomainEvent,
    sample_payload: dict[str, Any],
) -> None:
    results: list[Result[Any]] = []

    results.append(artifact_manager.create_run_directory(run_id))
    results.append(artifact_manager.write_metadata(run_id, sample_metadata))
    results.append(artifact_manager.read_metadata(run_id))
    results.append(artifact_manager.write_input(run_id, "input.json", sample_payload))
    results.append(artifact_manager.write_output(run_id, "output.json", sample_payload))
    results.append(artifact_manager.append_event(run_id, sample_event))
    results.append(artifact_manager.list_runs())

    assert all(isinstance(entry, Result) for entry in results)
