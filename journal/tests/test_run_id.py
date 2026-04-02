from __future__ import annotations

import re
from uuid import UUID

import pytest

from journal import RunIDGenerator, RunType

_RUN_ID_REGEX = re.compile(
    r"^(?P<date>\d{8})-"
    r"(?P<time>\d{6})-"
    r"(?P<run_type>[A-Z0-9_]+)-"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
)


def test_generate_run_id_format() -> None:
    run_id = RunIDGenerator.generate_run_id(RunType.PRE_MARKET_FULL_SCAN)

    match = _RUN_ID_REGEX.fullmatch(run_id)
    assert match is not None
    assert match.group("run_type") == RunType.PRE_MARKET_FULL_SCAN.value
    UUID(match.group("uuid"))


def test_generate_run_id_unique() -> None:
    run_ids = {RunIDGenerator.generate_run_id(RunType.PRE_MARKET_FULL_SCAN) for _ in range(5)}

    assert len(run_ids) == 5


def test_generate_run_id_invalid_type() -> None:
    with pytest.raises(ValueError):
        RunIDGenerator.generate_run_id("PRE_MARKET_FULL_SCAN")  # type: ignore[arg-type]


def test_parse_run_id_valid() -> None:
    run_id = "20240101-010203-EOD_SCAN-123e4567-e89b-12d3-a456-426614174000"

    parsed = RunIDGenerator.parse_run_id(run_id)

    assert parsed["date"] == "20240101"
    assert parsed["time"] == "010203"
    assert parsed["run_type"] is RunType.PRE_MARKET_FULL_SCAN
    assert parsed["uuid"] == "123e4567-e89b-12d3-a456-426614174000"


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "20240101-010203-EOD_SCAN",
        "20240101-010203-123e4567-e89b-12d3-a456-426614174000",
        "20240101010203-EOD_SCAN-123e4567-e89b-12d3-a456-426614174000",
        "20240101-010203-EOD_SCAN-123e4567e89b12d3a456426614174000",
    ],
)
def test_parse_run_id_invalid_format(run_id: str) -> None:
    with pytest.raises(ValueError):
        RunIDGenerator.parse_run_id(run_id)


def test_parse_run_id_invalid_timestamp() -> None:
    run_id = "20230230-250000-EOD_SCAN-123e4567-e89b-12d3-a456-426614174000"

    with pytest.raises(ValueError):
        RunIDGenerator.parse_run_id(run_id)


def test_parse_run_id_invalid_uuid() -> None:
    run_id = "20240101-010203-EOD_SCAN-not-a-uuid"

    with pytest.raises(ValueError):
        RunIDGenerator.parse_run_id(run_id)


def test_parse_run_id_unknown_run_type() -> None:
    run_id = "20240101-010203-UNKNOWN_TYPE-123e4567-e89b-12d3-a456-426614174000"

    with pytest.raises(ValueError):
        RunIDGenerator.parse_run_id(run_id)


def test_get_system_version(monkeypatch: pytest.MonkeyPatch) -> None:
    import journal.run_id as run_id

    run_id._get_system_version_cached.cache_clear()
    monkeypatch.setenv("AST_SYSTEM_VERSION", "deadbeefcafebabe")

    assert RunIDGenerator.get_system_version() == "deadbeefcafebabe"
