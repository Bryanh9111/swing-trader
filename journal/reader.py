"""High-level artifact reader for replay and audit workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgspec
import msgspec.json
import yaml

from common.exceptions import CriticalError, RecoverableError, ValidationError
from common.interface import DomainEvent, Result, ResultStatus

from .artifacts import ArtifactManager
from .interface import RunMetadata
from .validator import SnapshotValidator

__all__ = ["JournalReader"]

JSON_DECODE_ERROR = getattr(msgspec.json, "DecodeError", msgspec.DecodeError)


class JournalReader:
    """Load journalled run artifacts with schema validation."""

    _MODULE_NAME = "journal.reader"
    _ARTIFACT_MISSING_REASON = "JOURNAL_ARTIFACT_MISSING"
    _ARTIFACT_DECODE_FAILED_REASON = "JOURNAL_ARTIFACT_DECODE_FAILED"
    _ARTIFACT_SCHEMA_INCOMPATIBLE_REASON = "JOURNAL_ARTIFACT_SCHEMA_INCOMPATIBLE"
    _EVENTS_DECODE_FAILED_REASON = "JOURNAL_EVENTS_DECODE_FAILED"

    def __init__(
        self,
        artifact_manager: ArtifactManager,
        *,
        validator: SnapshotValidator | None = None,
        expected_snapshot_major: str = "1",
    ) -> None:
        """Initialise the reader with storage and validation helpers."""

        self._artifact_manager = artifact_manager
        self._validator = validator or SnapshotValidator()
        self._expected_snapshot_major = str(expected_snapshot_major)

    def load_run(self, run_id: str) -> Result[dict[str, Any]]:
        """Return the complete run payload (metadata, inputs, outputs, events)."""

        metadata_result = self.load_metadata(run_id)
        if metadata_result.status is ResultStatus.FAILED:
            return metadata_result  # type: ignore[return-value]
        metadata = metadata_result.data

        inputs_result = self.load_inputs(run_id)
        if inputs_result.status is ResultStatus.FAILED:
            return inputs_result  # type: ignore[return-value]
        inputs = inputs_result.data or {}

        outputs_result = self.load_outputs(run_id)
        if outputs_result.status is ResultStatus.FAILED:
            return outputs_result  # type: ignore[return-value]
        outputs = outputs_result.data or {}

        events_result = self.load_events(run_id)
        if events_result.status is ResultStatus.FAILED:
            return events_result  # type: ignore[return-value]
        events = events_result.data or []

        payload = {
            "run_id": run_id,
            "metadata": metadata,
            "inputs": inputs,
            "outputs": outputs,
            "events": events,
        }

        degraded_result = self._first_degraded(
            metadata_result,
            inputs_result,
            outputs_result,
            events_result,
        )

        if degraded_result is not None:
            error = degraded_result.error or RecoverableError(
                "Journal read degraded without error context.",
                module=self._MODULE_NAME,
                reason_code=degraded_result.reason_code or self._ARTIFACT_DECODE_FAILED_REASON,
                details={"run_id": run_id},
            )
            reason = degraded_result.reason_code or getattr(error, "reason_code", None) or self._ARTIFACT_DECODE_FAILED_REASON
            return Result.degraded(payload, error, reason)

        return Result.success(payload)

    def load_metadata(self, run_id: str) -> Result[RunMetadata]:
        """Load persisted :class:`RunMetadata` for ``run_id``."""

        result = self._artifact_manager.read_metadata(run_id)
        if result.status is ResultStatus.FAILED:
            return result
        if result.status is ResultStatus.DEGRADED:
            error = result.error or RecoverableError(
                "Metadata degraded without error context.",
                module=self._MODULE_NAME,
                reason_code=result.reason_code or self._ARTIFACT_MISSING_REASON,
                details={"run_id": run_id},
            )
            return Result.degraded(result.data, error, result.reason_code or self._ARTIFACT_MISSING_REASON)
        return result

    def load_inputs(self, run_id: str) -> Result[dict[str, Any]]:
        """Load all input snapshots for ``run_id``."""

        return self._load_snapshot_directory(run_id, subdir="inputs", allow_yaml=True)

    def load_outputs(self, run_id: str) -> Result[dict[str, Any]]:
        """Load all output snapshots for ``run_id``."""

        return self._load_snapshot_directory(run_id, subdir="outputs", allow_yaml=False)

    def load_events(self, run_id: str) -> Result[list[DomainEvent]]:
        """Load the domain event log for ``run_id``."""

        events_path = (
            self._artifact_manager.base_path
            / run_id
            / "events"
            / "events.jsonl"
        )

        try:
            raw_lines = events_path.read_text(encoding="utf-8").splitlines()
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_MISSING_REASON,
                details={"run_id": run_id, "path": str(events_path)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)
        except FileNotFoundError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_MISSING_REASON,
                details={"run_id": run_id, "path": str(events_path)},
            )
            return Result.degraded([], error, error.reason_code)
        except OSError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_DECODE_FAILED_REASON,
                details={"run_id": run_id, "path": str(events_path)},
            )
            return Result.degraded([], error, error.reason_code)

        events: list[DomainEvent] = []
        for index, line in enumerate(raw_lines, start=1):
            if not line.strip():
                continue
            try:
                event = msgspec.json.decode(line.encode("utf-8"), type=DomainEvent)
            except (JSON_DECODE_ERROR, msgspec.ValidationError) as exc:
                error = ValidationError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code=self._EVENTS_DECODE_FAILED_REASON,
                    details={"run_id": run_id, "path": str(events_path), "line": index},
                )
                return error.to_result(data=None, status=ResultStatus.FAILED)
            events.append(event)

        return Result.success(events)

    def _load_snapshot_directory(
        self,
        run_id: str,
        *,
        subdir: str,
        allow_yaml: bool,
    ) -> Result[dict[str, Any]]:
        """Load and validate all snapshot files in ``subdir``."""

        directory = self._artifact_manager.base_path / run_id / subdir
        if not directory.is_dir():
            error = RecoverableError(
                f"{subdir} directory missing for run {run_id}.",
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_MISSING_REASON,
                details={"run_id": run_id, "path": str(directory)},
            )
            return error.to_result(data={}, status=ResultStatus.DEGRADED)

        payloads: dict[str, Any] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            try:
                payload = self._read_snapshot(path, allow_yaml=allow_yaml)
            except CriticalError as error:
                return error.to_result(data=None, status=ResultStatus.FAILED)
            except RecoverableError as error:
                return error.to_result(data=payloads, status=ResultStatus.DEGRADED)
            except ValidationError as error:
                return error.to_result(data=None, status=ResultStatus.FAILED)

            validation_result = self._validator.validate_schema_version(
                payload,
                self._expected_snapshot_major,
            )
            if validation_result.status is ResultStatus.FAILED:
                error = validation_result.error or ValidationError(
                    "Snapshot schema version incompatible.",
                    module=self._MODULE_NAME,
                    reason_code=self._ARTIFACT_SCHEMA_INCOMPATIBLE_REASON,
                    details={"run_id": run_id, "path": str(path)},
                )
                reason = validation_result.reason_code or self._ARTIFACT_SCHEMA_INCOMPATIBLE_REASON
                return Result.failed(error, reason)

            payloads[path.name] = payload

        return Result.success(payloads)

    def _read_snapshot(self, path: Path, *, allow_yaml: bool) -> Any:
        """Read a single snapshot file, returning native Python data."""

        try:
            if allow_yaml and path.suffix.lower() in {".yaml", ".yml"}:
                with path.open("r", encoding="utf-8") as file:
                    return yaml.safe_load(file)

            raw = path.read_bytes()
            return msgspec.json.decode(raw)
        except PermissionError as exc:
            raise CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_MISSING_REASON,
                details={"path": str(path)},
            )
        except FileNotFoundError as exc:
            raise RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_MISSING_REASON,
                details={"path": str(path)},
            )
        except (OSError, yaml.YAMLError, JSON_DECODE_ERROR) as exc:
            raise RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_DECODE_FAILED_REASON,
                details={"path": str(path)},
            )

    def _first_degraded(self, *results: Result[Any]) -> Result[Any] | None:
        """Return the first degraded result in ``results`` if present."""

        for result in results:
            if result.status is ResultStatus.DEGRADED:
                return result
        return None
