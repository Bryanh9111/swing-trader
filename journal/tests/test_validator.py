from __future__ import annotations

from typing import Any, Mapping, Type

import pytest

from common.interface import ResultStatus
from journal.interface import SnapshotBase
from journal.validator import SnapshotValidator


class SnapshotWithVersion(SnapshotBase, frozen=True, kw_only=True):
    """Sample snapshot embedding SCHEMA_VERSION metadata."""

    SCHEMA_VERSION = "1.5.0"

    payload: dict[str, Any]


class SnapshotWithMajor(SnapshotBase, frozen=True, kw_only=True):
    """Sample snapshot embedding SCHEMA_MAJOR_VERSION metadata."""

    SCHEMA_MAJOR_VERSION = "1"

    payload: dict[str, Any]


class SnapshotNoMetadata(SnapshotBase, frozen=True, kw_only=True):
    """Snapshot without schema metadata for expected major inference."""

    payload: dict[str, Any]


@pytest.fixture
def validator() -> SnapshotValidator:
    return SnapshotValidator()


def test_validate_schema_version_success_with_struct(validator: SnapshotValidator) -> None:
    snapshot = SnapshotWithVersion(
        schema_version="1.9.0",
        system_version="abc",
        asof_timestamp=1,
        payload={"value": 1},
    )

    result = validator.validate_schema_version(snapshot, expected_major_version="1")

    assert result.status is ResultStatus.SUCCESS


def test_validate_schema_version_missing(validator: SnapshotValidator) -> None:
    result = validator.validate_schema_version({}, expected_major_version="1")

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._SCHEMA_MISSING_REASON  # type: ignore[attr-defined]


def test_validate_schema_version_mismatch(validator: SnapshotValidator) -> None:
    snapshot = {"schema_version": "2.0.0"}

    result = validator.validate_schema_version(snapshot, expected_major_version="1")

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._SCHEMA_INCOMPATIBLE_REASON  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("schema_class", "schema_version"),
    [
        (SnapshotWithVersion, "1.7.0"),
        (SnapshotWithMajor, "1.1.0"),
    ],
)
def test_validate_snapshot_success(
    validator: SnapshotValidator,
    schema_class: Type[SnapshotBase],
    schema_version: str,
) -> None:
    payload: Mapping[str, Any] = {
        "schema_version": schema_version,
        "system_version": "abc",
        "asof_timestamp": 1,
        "payload": {"value": 42},
    }

    result = validator.validate_snapshot(payload, schema_class)  # type: ignore[arg-type]

    assert result.status is ResultStatus.SUCCESS
    assert isinstance(result.data, schema_class)


def test_validate_snapshot_schema_incompatible(validator: SnapshotValidator) -> None:
    payload = {
        "schema_version": "2.0.0",
        "system_version": "abc",
        "asof_timestamp": 1,
        "payload": {"value": 42},
    }

    result = validator.validate_snapshot(payload, SnapshotWithMajor)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._SCHEMA_INCOMPATIBLE_REASON  # type: ignore[attr-defined]


def test_validate_snapshot_invalid_schema_class(validator: SnapshotValidator) -> None:
    result = validator.validate_snapshot({}, type("Dummy", (), {}))  # type: ignore[arg-type]

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._INVALID_SCHEMA_CLASS_REASON  # type: ignore[attr-defined]


def test_validate_schema_version_invalid_semver(validator: SnapshotValidator) -> None:
    snapshot = {"schema_version": "", "system_version": "abc", "asof_timestamp": 1}

    result = validator.validate_schema_version(snapshot, expected_major_version="1")

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._SCHEMA_INVALID_REASON  # type: ignore[attr-defined]


def test_validate_snapshot_conversion_error(validator: SnapshotValidator) -> None:
    payload = {
        "schema_version": "1.0.0",
        "system_version": "abc",
        # missing asof_timestamp to trigger conversion failure
    }

    result = validator.validate_snapshot(payload, SnapshotWithMajor)  # type: ignore[arg-type]

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._INVALID_SNAPSHOT_REASON  # type: ignore[attr-defined]


def test_validate_snapshot_missing_expected_major(validator: SnapshotValidator) -> None:
    payload = {
        "schema_version": "",
        "system_version": "abc",
        "asof_timestamp": 1,
        "payload": {"value": 1},
    }

    result = validator.validate_snapshot(payload, SnapshotNoMetadata)  # type: ignore[arg-type]

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._SCHEMA_INVALID_REASON  # type: ignore[attr-defined]
