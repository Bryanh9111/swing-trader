"""High-level journaling writer for Automated Swing Trader runs.

``JournalWriter`` coordinates the lower-level :class:`ArtifactManager` APIs and
the :class:`SnapshotValidator` to persist complete run snapshots atomically.
All operations surface structured :class:`Result` instances so that callers can
distinguish between success, degraded states, and fatal failures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from common.exceptions import OperationalError, RecoverableError, ValidationError
from common.interface import DomainEvent, Result, ResultStatus

from .artifacts import ArtifactManager
from .interface import RunMetadata
from .validator import SnapshotValidator

__all__ = ["JournalWriter"]


class JournalWriter:
    """Persist complete run artifacts using ``ArtifactManager`` primitives."""

    _MODULE_NAME = "journal.writer"
    _INVALID_ARGUMENT_REASON = "JOURNAL_INVALID_ARGUMENT"
    _PERSIST_DEGRADED_REASON = "JOURNAL_PERSIST_DEGRADED"
    _PERSIST_FAILED_REASON = "JOURNAL_PERSIST_FAILED"

    def __init__(
        self,
        artifact_manager: ArtifactManager,
        *,
        validator: SnapshotValidator | None = None,
        expected_snapshot_major: str = "1",
    ) -> None:
        """Initialise the writer with dependencies and schema expectations."""

        self._artifact_manager = artifact_manager
        self._validator = validator or SnapshotValidator()
        self._expected_snapshot_major = str(expected_snapshot_major)

    def persist_complete_run(
        self,
        run_id: str,
        metadata: RunMetadata,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        events: Sequence[DomainEvent],
    ) -> Result[None]:
        """Persist metadata, input/output snapshots, and event log atomically."""

        if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
            error = ValidationError(
                "inputs and outputs must be mapping instances.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_ARGUMENT_REASON,
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        run_dir_result = self._artifact_manager.create_run_directory(run_id)
        converted = self._convert_result(run_dir_result)
        if converted.status is ResultStatus.FAILED:
            return converted
        degraded_result: Result[None] | None = (
            converted if converted.status is ResultStatus.DEGRADED else None
        )

        metadata_result = self._artifact_manager.write_metadata(run_id, metadata)
        converted = self._convert_result(metadata_result)
        if converted.status is ResultStatus.FAILED:
            return converted
        if degraded_result is None and converted.status is ResultStatus.DEGRADED:
            degraded_result = converted

        converted = self._persist_artifacts(
            run_id,
            artifacts=inputs,
            is_input=True,
        )
        if converted.status is ResultStatus.FAILED:
            return converted
        if degraded_result is None and converted.status is ResultStatus.DEGRADED:
            degraded_result = converted

        converted = self._persist_artifacts(
            run_id,
            artifacts=outputs,
            is_input=False,
        )
        if converted.status is ResultStatus.FAILED:
            return converted
        if degraded_result is None and converted.status is ResultStatus.DEGRADED:
            degraded_result = converted

        converted = self._append_events(run_id, events)
        if converted.status is ResultStatus.FAILED:
            return converted
        if degraded_result is None and converted.status is ResultStatus.DEGRADED:
            degraded_result = converted

        if degraded_result is not None:
            return degraded_result

        return Result.success(data=None)

    def _persist_artifacts(
        self,
        run_id: str,
        *,
        artifacts: Mapping[str, Any],
        is_input: bool,
    ) -> Result[None]:
        """Persist a mapping of artifact filenames to payloads."""

        if not artifacts:
            return Result.success(data=None)

        degraded_result: Result[None] | None = None
        writer = (
            self._artifact_manager.write_input
            if is_input
            else self._artifact_manager.write_output
        )

        for filename in sorted(artifacts):
            payload = artifacts[filename]

            validation_result = self._validator.validate_schema_version(
                payload,
                self._expected_snapshot_major,
            )
            converted = self._convert_result(validation_result)
            if converted.status is ResultStatus.FAILED:
                return converted
            if degraded_result is None and converted.status is ResultStatus.DEGRADED:
                degraded_result = converted
                # continue processing remaining artifacts to preserve coverage.

            write_result = writer(run_id, filename, payload)
            converted = self._convert_result(write_result)
            if converted.status is ResultStatus.FAILED:
                return converted
            if degraded_result is None and converted.status is ResultStatus.DEGRADED:
                degraded_result = converted

        if degraded_result is not None:
            return degraded_result

        return Result.success(data=None)

    def _append_events(self, run_id: str, events: Sequence[DomainEvent]) -> Result[None]:
        """Append each domain event to the journalled run."""

        degraded_result: Result[None] | None = None

        for event in events:
            append_result = self._artifact_manager.append_event(run_id, event)
            converted = self._convert_result(append_result)
            if converted.status is ResultStatus.FAILED:
                return converted
            if degraded_result is None and converted.status is ResultStatus.DEGRADED:
                degraded_result = converted

        if degraded_result is not None:
            return degraded_result

        return Result.success(data=None)

    def _convert_result(self, result: Result[Any]) -> Result[None]:
        """Normalise dependency results to ``Result[None]`` for aggregation."""

        if result.status is ResultStatus.SUCCESS:
            return Result.success(data=None, reason_code=result.reason_code or "OK")

        if result.status is ResultStatus.DEGRADED:
            error = result.error or RecoverableError(
                "Journal dependency degraded without error context.",
                module=self._MODULE_NAME,
                reason_code=result.reason_code or self._PERSIST_DEGRADED_REASON,
            )
            reason = result.reason_code or getattr(error, "reason_code", None) or self._PERSIST_DEGRADED_REASON
            return Result.degraded(data=None, error=error, reason_code=reason)

        error = result.error or OperationalError(
            "Journal dependency failed without error context.",
            module=self._MODULE_NAME,
            reason_code=result.reason_code or self._PERSIST_FAILED_REASON,
        )
        reason = result.reason_code or getattr(error, "reason_code", None) or self._PERSIST_FAILED_REASON
        return Result.failed(error, reason)

