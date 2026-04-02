"""Snapshot validation helpers for journalled artifacts.

This module centralises schema version compatibility checks so that both the
writer and reader components can enforce deterministic semantics regardless of
the storage backend.  The validator understands ``SemVer`` major version
compatibility, handles conversion into typed ``msgspec.Struct`` instances, and
emits structured :class:`Result` objects to integrate with the wider AST error
handling conventions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

import msgspec

from common.exceptions import ValidationError
from common.interface import Result, ResultStatus

from .interface import SnapshotBase

SnapshotT = TypeVar("SnapshotT", bound=SnapshotBase)

__all__ = ["SnapshotValidator"]


class SnapshotValidator:
    """Validate snapshot payloads and enforce schema version compatibility."""

    _MODULE_NAME = "journal.validator"
    _SCHEMA_MISSING_REASON = "SNAPSHOT_SCHEMA_MISSING"
    _SCHEMA_INVALID_REASON = "SNAPSHOT_SCHEMA_INVALID"
    _SCHEMA_INCOMPATIBLE_REASON = "SNAPSHOT_SCHEMA_INCOMPATIBLE"
    _INVALID_SNAPSHOT_REASON = "SNAPSHOT_INVALID_PAYLOAD"
    _INVALID_SCHEMA_CLASS_REASON = "SNAPSHOT_INVALID_SCHEMA_CLASS"

    def validate_schema_version(
        self,
        snapshot: SnapshotBase | Mapping[str, Any],
        expected_major_version: str,
    ) -> Result[None]:
        """Verify that ``snapshot`` is compatible with ``expected_major_version``.

        Args:
            snapshot: Snapshot payload or mapping containing ``schema_version``.
            expected_major_version: SemVer major version required by the caller.

        Returns:
            ``Result.success(None)`` when the snapshot major version matches
            ``expected_major_version``.
        """

        schema_version = self._extract_schema_version(snapshot)
        if schema_version is None:
            error = ValidationError(
                "Snapshot payload is missing schema_version.",
                module=self._MODULE_NAME,
                reason_code=self._SCHEMA_MISSING_REASON,
                details={"snapshot_type": type(snapshot).__name__},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        snapshot_major = self._extract_major(schema_version)
        expected_major = self._extract_major(str(expected_major_version))

        if snapshot_major is None or expected_major is None:
            error = ValidationError(
                "Snapshot schema_version is not a valid SemVer string.",
                module=self._MODULE_NAME,
                reason_code=self._SCHEMA_INVALID_REASON,
                details={
                    "snapshot_version": schema_version,
                    "expected_version": str(expected_major_version),
                },
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        if snapshot_major != expected_major:
            error = ValidationError(
                "Snapshot schema major version is incompatible with reader.",
                module=self._MODULE_NAME,
                reason_code=self._SCHEMA_INCOMPATIBLE_REASON,
                details={
                    "snapshot_major": snapshot_major,
                    "expected_major": expected_major,
                },
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        return Result.success(data=None)

    def validate_snapshot(
        self,
        snapshot_dict: Mapping[str, Any],
        schema_class: type[SnapshotT],
    ) -> Result[SnapshotT]:
        """Convert ``snapshot_dict`` into ``schema_class`` and validate version."""

        if not isinstance(schema_class, type) or not issubclass(schema_class, SnapshotBase):
            error = ValidationError(
                "schema_class must be a SnapshotBase subclass.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_SCHEMA_CLASS_REASON,
                details={"schema_class": repr(schema_class)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        try:
            snapshot = msgspec.convert(snapshot_dict, type=schema_class)
        except (msgspec.ValidationError, TypeError, ValueError) as exc:
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._INVALID_SNAPSHOT_REASON,
                details={"schema_class": schema_class.__name__},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        expected_major = (
            self._infer_expected_major(schema_class)
            or self._extract_major(snapshot.schema_version)
        )

        if expected_major is None:
            error = ValidationError(
                "Unable to determine expected schema major version.",
                module=self._MODULE_NAME,
                reason_code=self._SCHEMA_INVALID_REASON,
                details={"schema_class": schema_class.__name__},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        version_result = self.validate_schema_version(snapshot, expected_major)
        if version_result.status is ResultStatus.SUCCESS:
            return Result.success(snapshot)

        error = version_result.error or ValidationError(
            "Snapshot schema version incompatible.",
            module=self._MODULE_NAME,
            reason_code=self._SCHEMA_INCOMPATIBLE_REASON,
            details={"schema_class": schema_class.__name__},
        )
        reason_code = version_result.reason_code or self._SCHEMA_INCOMPATIBLE_REASON
        return Result.failed(error, reason_code)

    def _extract_schema_version(
        self,
        snapshot: SnapshotBase | Mapping[str, Any],
    ) -> str | None:
        """Return the schema_version string from ``snapshot`` if present."""

        if isinstance(snapshot, Mapping):
            value = snapshot.get("schema_version")
            return value if isinstance(value, str) else None

        return getattr(snapshot, "schema_version", None)

    def _extract_major(self, version: str) -> str | None:
        """Return the major component of a SemVer string if valid."""

        if not version:
            return None
        return version.split(".", 1)[0] or None

    def _infer_expected_major(self, schema_class: type[SnapshotT]) -> str | None:
        """Infer the expected major version from ``schema_class`` metadata."""

        if hasattr(schema_class, "SCHEMA_MAJOR_VERSION"):
            value = getattr(schema_class, "SCHEMA_MAJOR_VERSION")
            if isinstance(value, (str, int)):
                return str(value)

        if hasattr(schema_class, "SCHEMA_VERSION"):
            value = getattr(schema_class, "SCHEMA_VERSION")
            if isinstance(value, str):
                return self._extract_major(value)

        return None

