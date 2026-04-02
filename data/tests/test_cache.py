from __future__ import annotations

import os
import time as time_module
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from common.interface import ResultStatus
from data.cache import FileCache
from data.interface import PriceBar, PriceSeriesSnapshot


def _ns_at_utc_midnight(day: date) -> int:
    dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return int(dt.timestamp()) * 1_000_000_000


def _bar_for_day(day: date, *, close: float = 1.0, timestamp_ns: int | None = None) -> PriceBar:
    return PriceBar(
        timestamp=int(_ns_at_utc_midnight(day) if timestamp_ns is None else timestamp_ns),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1,
    )


@pytest.fixture()
def sample_snapshot() -> PriceSeriesSnapshot:
    bar = PriceBar(timestamp=1, open=1.0, high=2.0, low=0.5, close=1.5, volume=10)
    return PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=123,
        symbol="AAPL",
        timeframe="1D",
        bars=[bar],
        source="polygon",
        quality_flags={"availability": True, "completeness": True, "freshness": True, "stability": True},
    )


def test_file_cache_initialization_creates_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = FileCache(cache_dir=cache_dir, ttl_seconds=60)
    assert cache.cache_dir == cache_dir
    assert cache_dir.exists()
    assert cache.ttl_seconds == 60


def test_set_and_get_round_trip(tmp_path: Path, sample_snapshot: PriceSeriesSnapshot) -> None:
    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    key = cache.make_key("AAPL", "1D", "2025-01-01", "2025-01-02")

    set_result = cache.set(key, sample_snapshot)
    assert set_result.status is ResultStatus.SUCCESS
    assert set_result.reason_code == "CACHE_WRITE_OK"

    get_result = cache.get(key)
    assert get_result.status is ResultStatus.SUCCESS
    assert get_result.reason_code == "CACHE_HIT"
    assert get_result.data == sample_snapshot


def test_ttl_expiry_returns_none_and_removes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = FileCache(cache_dir=tmp_path, ttl_seconds=1)
    key = cache.make_key("AAPL", "1D", "2025-01-01", "2025-01-02")

    monkeypatch.setattr(time_module, "time_ns", lambda: 1_000_000_000)
    snapshot = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=123,
        symbol="AAPL",
        timeframe="1D",
        bars=[],
        source="polygon",
        quality_flags={},
    )
    assert cache.set(key, snapshot).status is ResultStatus.SUCCESS
    cache_file = cache.cache_dir / Path(key).name
    assert cache_file.exists()

    monkeypatch.setattr(time_module, "time_ns", lambda: 3_000_000_001)
    get_result = cache.get(key)
    assert get_result.status is ResultStatus.SUCCESS
    assert get_result.reason_code == "CACHE_EXPIRED"
    assert get_result.data is None
    assert not cache_file.exists()


def test_cache_key_generation_sanitizes_and_uppercases(tmp_path: Path) -> None:
    cache = FileCache(cache_dir=tmp_path)
    symbol = f"aa{os.sep}bb"
    timeframe = f"1{os.sep}d"
    key = cache.make_key(symbol, timeframe, "2025-01-01", "2025-01-02")
    assert os.sep not in key
    assert key.startswith("AA_BB_1_D_2025-01-01_2025-01-02")
    assert key.endswith(".json")


def test_invalidate_removes_entry(tmp_path: Path, sample_snapshot: PriceSeriesSnapshot) -> None:
    cache = FileCache(cache_dir=tmp_path)
    key = cache.make_key("AAPL", "1D", "2025-01-01", "2025-01-02")
    assert cache.set(key, sample_snapshot).status is ResultStatus.SUCCESS
    cache_file = cache.cache_dir / Path(key).name
    assert cache_file.exists()

    invalidate_result = cache.invalidate(key)
    assert invalidate_result.status is ResultStatus.SUCCESS
    assert invalidate_result.reason_code == "CACHE_INVALIDATE_OK"
    assert not cache_file.exists()


def test_get_handles_read_decode_errors(tmp_path: Path) -> None:
    cache = FileCache(cache_dir=tmp_path)
    key = cache.make_key("AAPL", "1D", "2025-01-01", "2025-01-02")
    (cache.cache_dir / Path(key).name).write_text("not-json", encoding="utf-8")

    result = cache.get(key)
    assert result.status is ResultStatus.DEGRADED
    assert result.data is None
    assert result.reason_code == "CACHE_READ_FAILED"


@pytest.mark.parametrize("exc_type", [PermissionError, OSError])
def test_set_handles_write_errors(
    tmp_path: Path,
    sample_snapshot: PriceSeriesSnapshot,
    monkeypatch: pytest.MonkeyPatch,
    exc_type: type[Exception],
) -> None:
    cache = FileCache(cache_dir=tmp_path)
    key = cache.make_key("AAPL", "1D", "2025-01-01", "2025-01-02")

    def raising_write_bytes(self: Path, data: bytes) -> int:
        raise exc_type("nope")

    monkeypatch.setattr(Path, "write_bytes", raising_write_bytes)
    result = cache.set(key, sample_snapshot)
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "CACHE_WRITE_FAILED"


@pytest.mark.parametrize("exc_type", [PermissionError, OSError])
def test_invalidate_handles_unlink_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc_type: type[Exception],
) -> None:
    cache = FileCache(cache_dir=tmp_path)
    key = cache.make_key("AAPL", "1D", "2025-01-01", "2025-01-02")

    def raising_unlink(self: Path, missing_ok: bool = False) -> None:
        raise exc_type("nope")

    monkeypatch.setattr(Path, "unlink", raising_unlink)
    result = cache.invalidate(key)
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "CACHE_INVALIDATE_FAILED"


def test_make_symbol_key(tmp_path: Path) -> None:
    """Symbol-level keys are stable and safe for filenames."""

    cache = FileCache(cache_dir=tmp_path)
    assert cache.make_symbol_key("AAPL", "1D") == "AAPL_1D.json"

    symbol = f"aa{os.sep}bb"
    timeframe = f"1{os.sep}d"
    key = cache.make_symbol_key(symbol, timeframe)
    assert os.sep not in key
    assert key == "AA_BB_1_D.json"


def test_get_incremental_cache_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No cache entry -> full requested range is missing (and handles edge cases)."""

    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    start = date(2025, 1, 6)
    end = date(2025, 1, 10)

    result = cache.get_incremental("AAPL", "1D", start, end)
    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "CACHE_MISS"
    cached_bars, missing_ranges = result.data  # type: ignore[misc]
    assert cached_bars == []
    assert missing_ranges == [(start, end)]

    invalid = cache.get_incremental("AAPL", "1D", end, start)
    assert invalid.status is ResultStatus.FAILED
    assert invalid.reason_code == "INVALID_RANGE"

    from common.interface import Result

    monkeypatch.setattr(cache, "get", lambda _key: Result.degraded(None, RuntimeError("read failed"), "CACHE_READ_FAILED"))
    degraded = cache.get_incremental("AAPL", "1D", start, end)
    assert degraded.status is ResultStatus.DEGRADED
    assert degraded.reason_code == "CACHE_IGNORED"
    degraded_bars, degraded_ranges = degraded.data  # type: ignore[misc]
    assert degraded_bars == []
    assert degraded_ranges == [(start, end)]


def test_get_incremental_cache_hit_full_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache fully covers requested weekdays -> no missing ranges and bars are filtered."""

    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    start = date(2025, 1, 6)  # Mon
    end = date(2025, 1, 10)  # Fri
    bars = [_bar_for_day(start + timedelta(days=offset), close=100 + offset) for offset in range(5)]
    snapshot = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=123,
        symbol="AAPL",
        timeframe="1D",
        bars=bars,
        source="polygon",
        quality_flags={"availability": True, "completeness": True, "freshness": True, "stability": True},
    )

    key = cache.make_symbol_key("AAPL", "1D")
    assert cache.set(key, snapshot).status is ResultStatus.SUCCESS

    result = cache.get_incremental("AAPL", "1D", start, end)
    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "CACHE_HIT"
    cached_bars, missing_ranges = result.data  # type: ignore[misc]
    assert missing_ranges == []
    assert cached_bars == bars


def test_get_incremental_cache_partial_coverage(tmp_path: Path) -> None:
    """Cache covers only part of the window -> missing ranges are detected."""

    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    start = date(2025, 1, 6)  # Mon
    end = date(2025, 1, 10)  # Fri
    cached_days = [start + timedelta(days=offset) for offset in range(3)]
    cached_bars = [_bar_for_day(day, close=100 + idx) for idx, day in enumerate(cached_days)]
    snapshot = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=123,
        symbol="AAPL",
        timeframe="1D",
        bars=cached_bars,
        source="polygon",
        quality_flags={"availability": True, "completeness": True, "freshness": True, "stability": True},
    )
    assert cache.set(cache.make_symbol_key("AAPL", "1D"), snapshot).status is ResultStatus.SUCCESS

    result = cache.get_incremental("AAPL", "1D", start, end)
    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "CACHE_PARTIAL"
    returned_bars, missing_ranges = result.data  # type: ignore[misc]
    assert returned_bars == cached_bars
    assert missing_ranges == [(date(2025, 1, 9), date(2025, 1, 10))]

    # Also cover multiple disjoint missing ranges collapsing.
    cached_bars_2 = [
        _bar_for_day(date(2025, 1, 6), close=1.0),
        _bar_for_day(date(2025, 1, 8), close=2.0),
    ]
    snapshot_2 = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=123,
        symbol="AAPL",
        timeframe="1D",
        bars=cached_bars_2,
        source="polygon",
        quality_flags={},
    )
    assert cache.set(cache.make_symbol_key("AAPL", "1D"), snapshot_2).status is ResultStatus.SUCCESS
    result_2 = cache.get_incremental("AAPL", "1D", start, end)
    assert result_2.status is ResultStatus.SUCCESS
    assert result_2.reason_code == "CACHE_PARTIAL"
    _, missing_ranges_2 = result_2.data  # type: ignore[misc]
    assert missing_ranges_2 == [(date(2025, 1, 7), date(2025, 1, 7)), (date(2025, 1, 9), date(2025, 1, 10))]


def test_get_incremental_excludes_weekends(tmp_path: Path) -> None:
    """Missing ranges should not include weekend-only gaps."""

    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    start = date(2025, 1, 10)  # Fri
    end = date(2025, 1, 13)  # Mon
    cached_bars = [_bar_for_day(start, close=100.0)]
    snapshot = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=123,
        symbol="AAPL",
        timeframe="1D",
        bars=cached_bars,
        source="polygon",
        quality_flags={"availability": True, "completeness": True, "freshness": True, "stability": True},
    )
    assert cache.set(cache.make_symbol_key("AAPL", "1D"), snapshot).status is ResultStatus.SUCCESS

    result = cache.get_incremental("AAPL", "1D", start, end)
    assert result.status is ResultStatus.SUCCESS
    returned_bars, missing_ranges = result.data  # type: ignore[misc]
    assert returned_bars == cached_bars
    assert missing_ranges == [(end, end)]
    assert all(day.weekday() < 5 for s, e in missing_ranges for day in (s, e))


def test_get_incremental_ttl_expired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Expired cache should behave like a miss for incremental reads."""

    cache = FileCache(cache_dir=tmp_path, ttl_seconds=1)
    start = date(2025, 1, 6)
    end = date(2025, 1, 10)
    key = cache.make_symbol_key("AAPL", "1D")

    monkeypatch.setattr(time_module, "time_ns", lambda: 1_000_000_000)
    snapshot = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=123,
        symbol="AAPL",
        timeframe="1D",
        bars=[_bar_for_day(start)],
        source="polygon",
        quality_flags={},
    )
    assert cache.set(key, snapshot).status is ResultStatus.SUCCESS

    monkeypatch.setattr(time_module, "time_ns", lambda: 3_000_000_001)
    result = cache.get_incremental("AAPL", "1D", start, end)
    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "CACHE_MISS"
    cached_bars, missing_ranges = result.data  # type: ignore[misc]
    assert cached_bars == []
    assert missing_ranges == [(start, end)]


def test_merge_and_save_new_cache(tmp_path: Path) -> None:
    """First-time merge should save all bars and return a snapshot."""

    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    start = date(2025, 1, 6)
    new_bars = [_bar_for_day(start + timedelta(days=offset), close=10 + offset) for offset in range(3)]

    result = cache.merge_and_save("AAPL", "1D", new_bars)
    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "CACHE_MERGE_OK"
    assert result.data is not None
    assert result.data.bars == new_bars

    persisted = cache.get(cache.make_symbol_key("AAPL", "1D"))
    assert persisted.status is ResultStatus.SUCCESS
    assert persisted.data is not None
    assert persisted.data.bars == new_bars


def test_merge_and_save_dedup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate timestamps are deduped (new wins) and bars remain sorted."""

    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    existing = [
        _bar_for_day(date(2025, 1, 6), close=101.0, timestamp_ns=1),
        _bar_for_day(date(2025, 1, 7), close=102.0, timestamp_ns=2),
    ]
    snapshot = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=123,
        symbol="AAPL",
        timeframe="1D",
        bars=existing,
        source="polygon",
        quality_flags={},
    )
    assert cache.set(cache.make_symbol_key("AAPL", "1D"), snapshot).status is ResultStatus.SUCCESS

    captured: dict[str, Any] = {}

    class _Logger:
        def info(self, event: str, **kwargs: Any) -> None:
            captured["event"] = event
            captured.update(kwargs)

    monkeypatch.setattr(cache, "_logger", _Logger())

    new_bars = [
        _bar_for_day(date(2025, 1, 7), close=999.0, timestamp_ns=2),
        _bar_for_day(date(2025, 1, 8), close=103.0, timestamp_ns=3),
    ]
    result = cache.merge_and_save("AAPL", "1D", new_bars)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert [bar.timestamp for bar in result.data.bars] == [1, 2, 3]
    assert next(bar for bar in result.data.bars if bar.timestamp == 2).close == 999.0
    assert captured.get("event") == "cache.merge.saved"
    assert captured.get("deduped") == 1


def test_merge_and_save_with_max_age(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old bars beyond max_age_days are removed relative to newest bar."""

    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    newest = date(2025, 2, 10)
    # 45 daily bars ending at newest; max_age_days=30 should drop the oldest ~14 bars (inclusive cutoff).
    all_days = [newest - timedelta(days=offset) for offset in range(45)]
    existing_bars = [_bar_for_day(day, close=100.0 + idx) for idx, day in enumerate(sorted(all_days))]
    snapshot = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=123,
        symbol="AAPL",
        timeframe="1D",
        bars=existing_bars,
        source="polygon",
        quality_flags={},
    )
    assert cache.set(cache.make_symbol_key("AAPL", "1D"), snapshot).status is ResultStatus.SUCCESS

    captured: dict[str, Any] = {}

    class _Logger:
        def info(self, event: str, **kwargs: Any) -> None:
            captured["event"] = event
            captured.update(kwargs)

    monkeypatch.setattr(cache, "_logger", _Logger())

    result = cache.merge_and_save("AAPL", "1D", [], max_age_days=30)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    cutoff = newest - timedelta(days=30)
    assert all(datetime.fromtimestamp(bar.timestamp / 1_000_000_000, tz=UTC).date() >= cutoff for bar in result.data.bars)
    assert captured.get("event") == "cache.merge.saved"
    assert captured.get("removed", 0) > 0


def test_merge_and_save_cache_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If cache write fails, return DEGRADED but keep the merged snapshot payload."""

    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    bars = [_bar_for_day(date(2025, 1, 6), close=101.0)]

    class _Logger:
        def info(self, event: str, **kwargs: Any) -> None:
            raise RuntimeError("logger down")

    monkeypatch.setattr(cache, "_logger", _Logger())

    def _degraded_set(key: str, snapshot: PriceSeriesSnapshot):  # type: ignore[no-untyped-def]
        from common.interface import Result

        return Result.degraded(None, PermissionError("nope"), reason_code="CACHE_WRITE_FAILED")

    monkeypatch.setattr(cache, "set", _degraded_set)
    result = cache.merge_and_save("AAPL", "1D", bars)
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "CACHE_MERGE_WRITE_FAILED"
    assert result.data is not None
    assert result.data.bars == bars
