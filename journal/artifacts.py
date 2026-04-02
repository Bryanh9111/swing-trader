"""Filesystem-backed artifact management for journalled automation runs.

This module implements the ``ArtifactManager`` responsible for preparing and
maintaining the deterministic directory layout used to persist run metadata,
inputs, outputs, events, and logs.  It follows the conventions established in
``docs/sessions/session-2025-12-17-run-metadata-artifacts.md`` and returns
``Result`` instances for all operations so callers can distinguish between
success, recoverable faults, and critical failures.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import msgspec.json
import yaml

from common.exceptions import CriticalError, OperationalError, RecoverableError
from common.interface import DomainEvent, Result, ResultStatus

from .interface import RunMetadata

__all__ = ["ArtifactManager"]


class ArtifactManager:
    """Manage artifact directory lifecycles for journalled runs.

    The manager encapsulates filesystem layout creation, serialization helpers,
    and append-only event logging while surfacing structured ``Result`` values
    so that orchestrators can gracefully handle degraded states or escalate
    critical faults.
    """

    _MODULE_NAME = "journal.artifacts"
    _METADATA_FILENAME = "metadata.json"
    _INPUTS_DIRNAME = "inputs"
    _OUTPUTS_DIRNAME = "outputs"
    _EVENTS_DIRNAME = "events"
    _LOGS_DIRNAME = "logs"
    _EVENT_LOG_FILENAME = "events.jsonl"
    _RUN_LOG_FILENAME = "run.log"

    _RUN_DIR_EXISTS_REASON = "RUN_DIRECTORY_EXISTS"
    _RUN_DIR_CREATION_FAILED_REASON = "RUN_DIRECTORY_CREATION_FAILED"
    _RUN_DIR_PERMISSION_DENIED_REASON = "RUN_DIRECTORY_PERMISSION_DENIED"
    _METADATA_WRITE_FAILED_REASON = "METADATA_WRITE_FAILED"
    _METADATA_PERMISSION_DENIED_REASON = "METADATA_PERMISSION_DENIED"
    _METADATA_NOT_FOUND_REASON = "METADATA_NOT_FOUND"
    _METADATA_DECODE_FAILED_REASON = "METADATA_DECODE_FAILED"
    _METADATA_SCHEMA_INCOMPATIBLE_REASON = "METADATA_SCHEMA_INCOMPATIBLE"
    _ARTIFACT_PERMISSION_DENIED_REASON = "ARTIFACT_PERMISSION_DENIED"
    _ARTIFACT_WRITE_FAILED_REASON = "ARTIFACT_WRITE_FAILED"
    _ARTIFACT_MISSING_REASON = "ARTIFACT_DIRECTORY_MISSING"
    _ARTIFACT_ENUMERATION_FAILED_REASON = "ARTIFACT_ENUMERATION_FAILED"

    _SUPPORTED_METADATA_SCHEMA_MAJOR = "1"

    def __init__(self, base_path: str | Path = "artifacts") -> None:
        """Initialise the manager with the base artifacts directory.

        Args:
            base_path: Root directory containing per-run artifact trees.

        Raises:
            CriticalError: If the base directory cannot be created because of
                insufficient permissions.
            RecoverableError: If the base directory cannot be created due to an
                IO-related failure.
        """

        self.base_path = Path(base_path)

        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:  # pragma: no cover - depends on fs perms.
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_PERMISSION_DENIED_REASON,
                details={"base_path": str(self.base_path)},
            )
            raise error
        except OSError as exc:  # pragma: no cover - depends on fs state.
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._RUN_DIR_CREATION_FAILED_REASON,
                details={"base_path": str(self.base_path)},
            )
            raise error

    def create_run_directory(self, run_id: str) -> Result[Path]:
        """Create the artifact directory tree for ``run_id`` atomically.

        Args:
            run_id: Unique run identifier.

        Returns:
            Result containing the final run directory path on success.
        """

        run_path = self.base_path / run_id

        if run_path.exists():
            error = OperationalError(
                f"Run directory already exists for run_id={run_id}.",
                module=self._MODULE_NAME,
                reason_code=self._RUN_DIR_EXISTS_REASON,
                details={"run_id": run_id, "path": str(run_path)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        temp_dir_path: Path | None = None
        try:
            temp_dir_path = Path(
                tempfile.mkdtemp(prefix=f".{run_id}-", dir=str(self.base_path)),
            )

            self._initialise_structure(temp_dir_path)
            os.replace(temp_dir_path, run_path)
        except PermissionError as exc:
            if temp_dir_path is not None:
                shutil.rmtree(temp_dir_path, ignore_errors=True)
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._RUN_DIR_PERMISSION_DENIED_REASON,
                details={"run_id": run_id, "path": str(run_path)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)
        except OSError as exc:
            if temp_dir_path is not None:
                shutil.rmtree(temp_dir_path, ignore_errors=True)
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._RUN_DIR_CREATION_FAILED_REASON,
                details={"run_id": run_id, "path": str(run_path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        return Result.success(run_path)

    def write_metadata(self, run_id: str, metadata: RunMetadata) -> Result[Path]:
        """Persist run metadata for downstream inspection.

        Args:
            run_id: Identifier of the run whose metadata is being recorded.
            metadata: Structured :class:`RunMetadata` payload.

        Returns:
            Result carrying the path to ``metadata.json`` on success.
        """

        run_path = self.base_path / run_id
        if not run_path.is_dir():
            error = OperationalError(
                f"Run directory missing for run_id={run_id}.",
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_MISSING_REASON,
                details={"run_id": run_id},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        metadata_path = run_path / self._METADATA_FILENAME
        try:
            payload = msgspec.json.encode(metadata)
            metadata_path.write_bytes(payload)
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._METADATA_PERMISSION_DENIED_REASON,
                details={"run_id": run_id, "path": str(metadata_path)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)
        except OSError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._METADATA_WRITE_FAILED_REASON,
                details={"run_id": run_id, "path": str(metadata_path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        return Result.success(metadata_path)

    def read_metadata(self, run_id: str) -> Result[RunMetadata]:
        """Load the persisted metadata for a run and enforce schema compatibility.

        Args:
            run_id: Identifier of the run to load.

        Returns:
            Result containing the decoded :class:`RunMetadata` instance.
        """

        metadata_path = self.base_path / run_id / self._METADATA_FILENAME

        try:
            raw_bytes = metadata_path.read_bytes()
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._METADATA_PERMISSION_DENIED_REASON,
                details={"run_id": run_id, "path": str(metadata_path)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)
        except FileNotFoundError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._METADATA_NOT_FOUND_REASON,
                details={"run_id": run_id, "path": str(metadata_path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)
        except OSError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._METADATA_NOT_FOUND_REASON,
                details={"run_id": run_id, "path": str(metadata_path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        try:
            decoded_any = msgspec.json.decode(raw_bytes)
        except msgspec.json.DecodeError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._METADATA_DECODE_FAILED_REASON,
                details={"run_id": run_id, "path": str(metadata_path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        schema_version = self._extract_schema_version(decoded_any)
        if schema_version is not None and not self._is_schema_version_supported(
            schema_version,
        ):
            error = OperationalError(
                f"Incompatible run metadata schema version: {schema_version!r}.",
                module=self._MODULE_NAME,
                reason_code=self._METADATA_SCHEMA_INCOMPATIBLE_REASON,
                details={
                    "run_id": run_id,
                    "path": str(metadata_path),
                    "schema_version": schema_version,
                    "supported_major": self._SUPPORTED_METADATA_SCHEMA_MAJOR,
                },
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        try:
            metadata = msgspec.json.decode(raw_bytes, type=RunMetadata)
        except msgspec.json.DecodeError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._METADATA_DECODE_FAILED_REASON,
                details={"run_id": run_id, "path": str(metadata_path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        return Result.success(metadata)

    def write_input(self, run_id: str, filename: str, data: Any) -> Result[Path]:
        """Write a snapshot of input data beneath ``inputs/``.

        Args:
            run_id: Identifier of the run receiving the artifact.
            filename: Target filename (``.json``, ``.yaml``, or ``.yml``).
            data: Serializable payload to persist.

        Returns:
            Result containing the path to the written artifact.
        """

        return self._write_payload(
            run_id=run_id,
            directory_name=self._INPUTS_DIRNAME,
            filename=filename,
            data=data,
            allow_yaml=True,
        )

    def write_output(self, run_id: str, filename: str, data: Any) -> Result[Path]:
        """Persist generated outputs beneath ``outputs/`` in JSON format.

        Args:
            run_id: Identifier of the run receiving the artifact.
            filename: Target JSON filename.
            data: Serializable payload to persist.

        Returns:
            Result containing the path to the written artifact.
        """

        return self._write_payload(
            run_id=run_id,
            directory_name=self._OUTPUTS_DIRNAME,
            filename=filename,
            data=data,
            allow_yaml=False,
        )

    def append_event(self, run_id: str, event: DomainEvent) -> Result[None]:
        """Append a domain event to ``events/events.jsonl``.

        Args:
            run_id: Identifier of the run receiving the event record.
            event: Event payload to append.

        Returns:
            Result with ``None`` data when the append succeeds.
        """

        events_path = (
            self.base_path
            / run_id
            / self._EVENTS_DIRNAME
            / self._EVENT_LOG_FILENAME
        )

        try:
            encoded = msgspec.json.encode(event)
            with open(events_path, "ab") as file:
                file.write(encoded + b"\n")
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_PERMISSION_DENIED_REASON,
                details={"run_id": run_id, "path": str(events_path)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)
        except OSError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_WRITE_FAILED_REASON,
                details={"run_id": run_id, "path": str(events_path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        return Result.success(data=None)

    def list_runs(self) -> Result[list[str]]:
        """Enumerate known run identifiers in the artifacts root.

        Returns:
            Result containing a lexicographically sorted list of run IDs.
        """

        try:
            run_ids = [
                path.name
                for path in self.base_path.iterdir()
                if path.is_dir()
            ]
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_PERMISSION_DENIED_REASON,
                details={"base_path": str(self.base_path)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)
        except OSError as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_ENUMERATION_FAILED_REASON,
                details={"base_path": str(self.base_path)},
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        run_ids.sort()
        return Result.success(run_ids)

    def _write_payload(
        self,
        *,
        run_id: str,
        directory_name: str,
        filename: str,
        data: Any,
        allow_yaml: bool,
    ) -> Result[Path]:
        """Write ``data`` within ``run_id`` under ``directory_name``.

        Args:
            run_id: Identifier of the run receiving the payload.
            directory_name: Either ``inputs`` or ``outputs``.
            filename: Basename for the artifact file.
            data: Serializable payload.
            allow_yaml: Whether YAML formats are permitted (for inputs only).

        Returns:
            Result containing the path to the persisted artifact.
        """

        run_path = self.base_path / run_id
        if not run_path.is_dir():
            error = OperationalError(
                f"Run directory missing for run_id={run_id}.",
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_MISSING_REASON,
                details={"run_id": run_id},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        if "/" in filename or "\\" in filename:
            error = OperationalError(
                "Nested filenames are not allowed for artifact writes.",
                module=self._MODULE_NAME,
                reason_code="INVALID_ARTIFACT_FILENAME",
                details={"run_id": run_id, "filename": filename},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        target_path = run_path / directory_name / filename
        suffix = target_path.suffix.lower()

        if allow_yaml and suffix in {".yaml", ".yml"}:
            try:
                with open(target_path, "w", encoding="utf-8") as file:
                    yaml.safe_dump(data, file, sort_keys=False)
            except PermissionError as exc:
                error = CriticalError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code=self._ARTIFACT_PERMISSION_DENIED_REASON,
                    details={
                        "run_id": run_id,
                        "path": str(target_path),
                        "directory": directory_name,
                    },
                )
                return error.to_result(data=None, status=ResultStatus.FAILED)
            except OSError as exc:
                error = RecoverableError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code=self._ARTIFACT_WRITE_FAILED_REASON,
                    details={
                        "run_id": run_id,
                        "path": str(target_path),
                        "directory": directory_name,
                    },
                )
                return error.to_result(data=None, status=ResultStatus.DEGRADED)

            return Result.success(target_path)

        if suffix != ".json":
            error = OperationalError(
                f"Unsupported artifact extension for {filename!r}.",
                module=self._MODULE_NAME,
                reason_code="UNSUPPORTED_ARTIFACT_EXTENSION",
                details={
                    "run_id": run_id,
                    "filename": filename,
                    "allowed": ["*.json", "*.yaml", "*.yml"] if allow_yaml else ["*.json"],
                },
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        try:
            payload = msgspec.json.encode(data)
            with open(target_path, "wb") as file:
                file.write(payload)
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_PERMISSION_DENIED_REASON,
                details={
                    "run_id": run_id,
                    "path": str(target_path),
                    "directory": directory_name,
                },
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)
        except (OSError, msgspec.json.EncodeError) as exc:
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._ARTIFACT_WRITE_FAILED_REASON,
                details={
                    "run_id": run_id,
                    "path": str(target_path),
                    "directory": directory_name,
                },
            )
            return error.to_result(data=None, status=ResultStatus.DEGRADED)

        return Result.success(target_path)

    def _initialise_structure(self, root: Path) -> None:
        """Create the directory skeleton and placeholder files for a run."""

        (root / self._INPUTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / self._OUTPUTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        events_dir = root / self._EVENTS_DIRNAME
        events_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = root / self._LOGS_DIRNAME
        logs_dir.mkdir(parents=True, exist_ok=True)

        (events_dir / self._EVENT_LOG_FILENAME).touch(exist_ok=True)
        (logs_dir / self._RUN_LOG_FILENAME).touch(exist_ok=True)

    def _extract_schema_version(self, payload: Any) -> str | None:
        """Return ``schema_version`` from the decoded raw payload if present."""

        if isinstance(payload, dict):
            schema = payload.get("schema_version")
            return schema if isinstance(schema, str) else None
        return None

    def _is_schema_version_supported(self, schema_version: str) -> bool:
        """Return ``True`` when ``schema_version`` matches the supported major."""

        major = schema_version.split(".", 1)[0]
        return major == self._SUPPORTED_METADATA_SCHEMA_MAJOR
