"""Deterministic replay utilities for journalled runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any

import msgspec

from common.exceptions import OperationalError, RecoverableError, ValidationError
from common.interface import DomainEvent, Result, ResultStatus

from .interface import RunMetadata, SnapshotBase
from .reader import JournalReader
from .validator import SnapshotValidator

__all__ = ["ReplayEngine", "ReplayMode"]

ReplayExecutor = Callable[[RunMetadata, dict[str, Any], list[DomainEvent], "ReplayMode"], Result[dict[str, Any]]]


class ReplayMode(str, Enum):
    """Replay behaviour modes for the :class:`ReplayEngine`."""

    FULL = "full"
    VALIDATE_ONLY = "validate_only"


class ReplayEngine:
    """Re-execute journalled runs and ensure deterministic outputs."""

    _MODULE_NAME = "journal.replay"
    _REPLAY_EXECUTION_FAILED_REASON = "REPLAY_EXECUTION_FAILED"
    _REPLAY_DEGRADED_REASON = "REPLAY_DEGRADED"
    _REPLAY_OUTPUT_MISMATCH_REASON = "REPLAY_OUTPUT_MISMATCH"
    _REPLAY_INVALID_METADATA_REASON = "REPLAY_INVALID_METADATA"

    def __init__(
        self,
        reader: JournalReader,
        executor: ReplayExecutor,
        *,
        validator: SnapshotValidator | None = None,
        expected_snapshot_major: str = "1",
    ) -> None:
        """Initialise the replay engine with its dependencies."""

        self._reader = reader
        self._executor = executor
        self._validator = validator or SnapshotValidator()
        self._expected_snapshot_major = str(expected_snapshot_major)

    def replay_run(self, run_id: str, replay_mode: ReplayMode) -> Result[dict[str, Any]]:
        """Replay ``run_id`` and validate deterministic outputs."""

        load_result = self._reader.load_run(run_id)
        if load_result.status is ResultStatus.FAILED:
            return load_result

        payload = load_result.data or {}
        metadata = payload.get("metadata")
        inputs = payload.get("inputs") or {}
        stored_outputs = payload.get("outputs") or {}
        events = payload.get("events") or []

        if not isinstance(metadata, RunMetadata):
            error = ValidationError(
                "Replay requires RunMetadata to be present.",
                module=self._MODULE_NAME,
                reason_code=self._REPLAY_INVALID_METADATA_REASON,
                details={"run_id": run_id},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        try:
            execution_result = self._executor(metadata, dict(inputs), list(events), replay_mode)
        except Exception as exc:  # noqa: BLE001 - executor is third-party.
            error = OperationalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._REPLAY_EXECUTION_FAILED_REASON,
                details={"run_id": run_id},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        if execution_result.status is ResultStatus.FAILED:
            error = execution_result.error or OperationalError(
                "Replay executor failed without error context.",
                module=self._MODULE_NAME,
                reason_code=self._REPLAY_EXECUTION_FAILED_REASON,
                details={"run_id": run_id},
            )
            reason = execution_result.reason_code or getattr(error, "reason_code", None) or self._REPLAY_EXECUTION_FAILED_REASON
            return Result.failed(error, reason)

        replay_outputs = execution_result.data or {}
        validation_result = self._validate_outputs(replay_outputs)
        if validation_result.status is ResultStatus.FAILED:
            return validation_result

        stored_normalised = self._normalise_outputs(stored_outputs)
        replay_normalised = self._normalise_outputs(replay_outputs)
        outputs_match = stored_normalised == replay_normalised

        if not outputs_match and replay_mode is ReplayMode.VALIDATE_ONLY:
            error = ValidationError(
                "Replay outputs differ from stored artifacts.",
                module=self._MODULE_NAME,
                reason_code=self._REPLAY_OUTPUT_MISMATCH_REASON,
                details={"run_id": run_id},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        result_payload = {
            "run_id": run_id,
            "metadata": metadata,
            "inputs": inputs,
            "stored_outputs": stored_outputs,
            "replay_outputs": replay_outputs,
            "events": events,
            "mode": replay_mode,
            "outputs_match": outputs_match,
        }

        if load_result.status is ResultStatus.DEGRADED:
            error = load_result.error or RecoverableError(
                "Replay degraded due to partial artifact load.",
                module=self._MODULE_NAME,
                reason_code=load_result.reason_code or self._REPLAY_DEGRADED_REASON,
                details={"run_id": run_id},
            )
            reason = load_result.reason_code or getattr(error, "reason_code", None) or self._REPLAY_DEGRADED_REASON
            return Result.degraded(result_payload, error, reason)

        if execution_result.status is ResultStatus.DEGRADED:
            error = execution_result.error or RecoverableError(
                "Replay executor degraded without error context.",
                module=self._MODULE_NAME,
                reason_code=execution_result.reason_code or self._REPLAY_DEGRADED_REASON,
                details={"run_id": run_id},
            )
            reason = execution_result.reason_code or getattr(error, "reason_code", None) or self._REPLAY_DEGRADED_REASON
            return Result.degraded(result_payload, error, reason)

        if not outputs_match and replay_mode is ReplayMode.FULL:
            error = ValidationError(
                "Replay outputs differ from stored artifacts.",
                module=self._MODULE_NAME,
                reason_code=self._REPLAY_OUTPUT_MISMATCH_REASON,
                details={"run_id": run_id},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        return Result.success(result_payload)

    def _validate_outputs(self, outputs: Mapping[str, Any]) -> Result[None]:
        """Validate schema versions of replayed outputs."""

        for name, payload in sorted(outputs.items()):
            validation_result = self._validator.validate_schema_version(
                payload,
                self._expected_snapshot_major,
            )
            if validation_result.status is ResultStatus.FAILED:
                error = validation_result.error or ValidationError(
                    "Replay output schema incompatible.",
                    module=self._MODULE_NAME,
                    reason_code=self._REPLAY_OUTPUT_MISMATCH_REASON,
                    details={"artifact": name},
                )
                reason = validation_result.reason_code or getattr(error, "reason_code", None) or self._REPLAY_OUTPUT_MISMATCH_REASON
                return Result.failed(error, reason)
        return Result.success(data=None)

    def _normalise_outputs(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        """Return a deterministic representation of outputs for comparison."""

        normalised: dict[str, Any] = {}
        for key in sorted(outputs):
            normalised[key] = self._normalise_value(outputs[key])
        return normalised

    def _normalise_value(self, value: Any) -> Any:
        """Convert msgspec structs into builtins recursively for comparison."""

        if isinstance(value, SnapshotBase):
            return self._normalise_value(msgspec.to_builtins(value))  # type: ignore[arg-type]
        if isinstance(value, dict):
            return {k: self._normalise_value(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [self._normalise_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._normalise_value(item) for item in value)
        return value

