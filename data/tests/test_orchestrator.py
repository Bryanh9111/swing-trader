from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from common.interface import Result, ResultStatus
from data.interface import DataLayerStats, PriceBar, PriceSeriesSnapshot
from data.orchestrator import DataOrchestrator


@dataclass(slots=True)
class _StubAdapter:
    source: str
    results: list[Result[PriceSeriesSnapshot]]
    calls: list[tuple[str, date, date]] = field(default_factory=list, init=False)

    def get_source_name(self) -> str:
        return self.source

    def fetch_price_series(self, symbol: str, start_date: Any, end_date: Any) -> Result[PriceSeriesSnapshot]:
        self.calls.append((symbol, start_date, end_date))
        if not self.results:
            return Result.failed(RuntimeError("no stub result configured"), reason_code="TEST_STUB_EMPTY")
        return self.results.pop(0)


class _StubCache:
    def __init__(self) -> None:
        self.store: dict[str, PriceSeriesSnapshot] = {}
        self.get_calls: list[str] = []
        self.set_calls: list[str] = []

    def make_key(self, symbol: str, timeframe: str, start: str, end: str) -> str:
        return f"{symbol.upper()}_{timeframe}_{start}_{end}.json"

    def get(self, key: str) -> Result[PriceSeriesSnapshot | None]:
        self.get_calls.append(key)
        if key not in self.store:
            return Result.success(None, reason_code="CACHE_MISS")
        return Result.success(self.store[key], reason_code="CACHE_HIT")

    def set(self, key: str, snapshot: PriceSeriesSnapshot) -> Result[None]:
        self.set_calls.append(key)
        self.store[key] = snapshot
        return Result.success(None, reason_code="CACHE_WRITE_OK")


def _clone_snapshot(snapshot: PriceSeriesSnapshot, **overrides: Any) -> PriceSeriesSnapshot:
    return PriceSeriesSnapshot(
        schema_version=overrides.get("schema_version", snapshot.schema_version),
        system_version=overrides.get("system_version", snapshot.system_version),
        asof_timestamp=overrides.get("asof_timestamp", snapshot.asof_timestamp),
        symbol=overrides.get("symbol", snapshot.symbol),
        timeframe=overrides.get("timeframe", snapshot.timeframe),
        bars=overrides.get("bars", snapshot.bars),
        source=overrides.get("source", snapshot.source),
        quality_flags=overrides.get("quality_flags", snapshot.quality_flags),
    )


@pytest.fixture()
def good_snapshot() -> PriceSeriesSnapshot:
    bar = PriceBar(timestamp=1, open=1.0, high=2.0, low=0.5, close=1.5, volume=1)
    return PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        symbol="AAPL",
        timeframe="1D",
        bars=[bar],
        source="polygon",
        quality_flags={"availability": True, "completeness": True, "freshness": True, "stability": True},
    )


def _ns_at_utc_midnight(day: date) -> int:
    dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return int(dt.timestamp()) * 1_000_000_000


def _bars_for_days(start: date, days: int, *, close_start: float = 1.0) -> list[PriceBar]:
    bars: list[PriceBar] = []
    for offset in range(days):
        bars.append(
            PriceBar(
                timestamp=_ns_at_utc_midnight(start + timedelta(days=offset)),
                open=close_start + offset,
                high=close_start + offset,
                low=close_start + offset,
                close=close_start + offset,
                volume=1,
            )
        )
    return bars


def _snapshot(symbol: str, bars: list[PriceBar], *, source: str = "polygon", quality_flags: dict[str, Any] | None = None) -> PriceSeriesSnapshot:
    return PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1,
        symbol=symbol,
        timeframe="1D",
        bars=bars,
        source=source,
        quality_flags=quality_flags
        or {"availability": True, "completeness": True, "freshness": True, "stability": True},
    )


class _StubIncrementalCache:
    def __init__(self) -> None:
        self.snapshot_by_key: dict[str, PriceSeriesSnapshot] = {}
        self.get_calls: list[str] = []
        self.get_incremental_calls: list[tuple[str, str, date, date]] = []
        self.merge_calls: list[tuple[str, str, int]] = []
        self.set_calls: list[tuple[str, int]] = []
        self.incremental_payload: tuple[list[PriceBar], list[tuple[date, date]]] | None = None
        self.incremental_status: ResultStatus = ResultStatus.SUCCESS
        self.merge_result: Result[PriceSeriesSnapshot] | None = None

    def make_symbol_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}_{timeframe}.json"

    def get(self, key: str) -> Result[PriceSeriesSnapshot | None]:
        self.get_calls.append(key)
        return Result.success(self.snapshot_by_key.get(key), reason_code="CACHE_HIT" if key in self.snapshot_by_key else "CACHE_MISS")

    def get_incremental(self, symbol: str, timeframe: str, requested_start: date, requested_end: date) -> Result[tuple[list[PriceBar], list[tuple[date, date]]]]:
        self.get_incremental_calls.append((symbol, timeframe, requested_start, requested_end))
        payload = self.incremental_payload or ([], [(requested_start, requested_end)])
        if self.incremental_status == ResultStatus.DEGRADED:
            return Result.degraded(payload, RuntimeError("cache degraded"), reason_code="CACHE_IGNORED")
        return Result.success(payload, reason_code="CACHE_MISS" if payload[1] else "CACHE_HIT")

    def set(self, key: str, snapshot: PriceSeriesSnapshot) -> Result[None]:
        self.snapshot_by_key[key] = snapshot
        self.set_calls.append((key, len(snapshot.bars)))
        return Result.success(None, reason_code="CACHE_WRITE_OK")

    def merge_and_save(self, symbol: str, timeframe: str, new_bars: list[PriceBar], max_age_days: int = 365) -> Result[PriceSeriesSnapshot]:
        self.merge_calls.append((symbol, timeframe, len(new_bars)))
        if self.merge_result is not None:
            return self.merge_result
        key = self.make_symbol_key(symbol, timeframe)
        existing = self.snapshot_by_key.get(key)
        merged = [*(existing.bars if existing else []), *new_bars]
        snapshot = _snapshot(symbol, merged, source=(existing.source if existing else "polygon"))
        self.snapshot_by_key[key] = snapshot
        return Result.success(snapshot, reason_code="CACHE_MERGE_OK")


def test_data_orchestrator_initialization() -> None:
    primary = _StubAdapter("polygon", results=[])
    orch = DataOrchestrator(primary_adapter=primary)
    assert orch._primary is primary  # noqa: SLF001 - construction invariant


def test_fetch_with_cache_cache_hit(good_snapshot: PriceSeriesSnapshot) -> None:
    cache = _StubCache()
    cache_key = cache.make_key("AAPL", "1D", "2025-01-01", "2025-01-02")
    cache.store[cache_key] = good_snapshot

    primary = _StubAdapter("polygon", results=[])
    orch = DataOrchestrator(primary_adapter=primary, cache=cache)
    result = orch.fetch_with_cache("AAPL", "2025-01-01", "2025-01-02")

    assert result.status is ResultStatus.SUCCESS
    assert result.data == good_snapshot
    assert primary.calls == []


def test_fetch_with_cache_cache_miss_primary_success(good_snapshot: PriceSeriesSnapshot) -> None:
    cache = _StubCache()
    primary = _StubAdapter("polygon", results=[Result.success(good_snapshot)])
    orch = DataOrchestrator(primary_adapter=primary, cache=cache)

    result = orch.fetch_with_cache("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.SUCCESS
    assert len(primary.calls) == 1
    assert cache.set_calls


def test_fetch_with_cache_primary_fail_secondary_success(good_snapshot: PriceSeriesSnapshot) -> None:
    cache = _StubCache()
    primary = _StubAdapter("polygon", results=[Result.failed(RuntimeError("primary down"), "PRIMARY_FAILED")])
    secondary = _StubAdapter("yahoo", results=[Result.success(good_snapshot)])
    orch = DataOrchestrator(primary_adapter=primary, secondary_adapter=secondary, cache=cache)

    result = orch.fetch_with_cache("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.SUCCESS
    assert len(primary.calls) == 1
    assert len(secondary.calls) == 1


def test_fetch_with_cache_all_sources_degraded_returns_degraded(good_snapshot: PriceSeriesSnapshot) -> None:
    degraded_snapshot = _clone_snapshot(
        good_snapshot,
        quality_flags={**good_snapshot.quality_flags, "freshness": False},
    )
    primary = _StubAdapter("polygon", results=[Result.degraded(degraded_snapshot, RuntimeError("partial"), "PARTIAL")])
    secondary = _StubAdapter("yahoo", results=[Result.degraded(degraded_snapshot, RuntimeError("partial"), "PARTIAL")])
    orch = DataOrchestrator(primary_adapter=primary, secondary_adapter=secondary)

    result = orch.fetch_with_cache("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "QUALITY_DEGRADED"


def test_validate_quality_success(good_snapshot: PriceSeriesSnapshot) -> None:
    orch = DataOrchestrator(primary_adapter=_StubAdapter("polygon", results=[]))
    result = orch.validate_quality(good_snapshot)
    assert result.status is ResultStatus.SUCCESS


@pytest.mark.parametrize(
    ("flags", "expected_status", "expected_reason"),
    [
        ({"availability": False}, ResultStatus.FAILED, "QUALITY_UNAVAILABLE"),
        ({"availability": True, "completeness": False}, ResultStatus.DEGRADED, "QUALITY_DEGRADED"),
        ({"availability": True, "freshness": False}, ResultStatus.DEGRADED, "QUALITY_DEGRADED"),
        ({"availability": True, "stability": False}, ResultStatus.DEGRADED, "QUALITY_DEGRADED"),
    ],
)
def test_validate_quality_degraded_and_failed_paths(
    good_snapshot: PriceSeriesSnapshot,
    flags: dict[str, Any],
    expected_status: ResultStatus,
    expected_reason: str,
) -> None:
    snapshot = _clone_snapshot(good_snapshot, quality_flags=flags)
    orch = DataOrchestrator(primary_adapter=_StubAdapter("polygon", results=[]))
    result = orch.validate_quality(snapshot)
    assert result.status is expected_status
    assert result.reason_code == expected_reason


def test_single_source_per_run_enforced_even_on_cache_mismatch(good_snapshot: PriceSeriesSnapshot) -> None:
    cache = _StubCache()
    primary = _StubAdapter("polygon", results=[Result.success(good_snapshot), Result.failed(RuntimeError("down"), "DOWN")])
    secondary = _StubAdapter("yahoo", results=[Result.success(good_snapshot)])
    orch = DataOrchestrator(primary_adapter=primary, secondary_adapter=secondary, cache=cache)

    first = orch.fetch_with_cache("AAPL", "2025-01-01", "2025-01-02")
    assert first.status is ResultStatus.SUCCESS
    assert len(primary.calls) == 1

    cache_key = cache.make_key("AAPL", "1D", "2025-01-01", "2025-01-02")
    cache.store[cache_key] = _clone_snapshot(good_snapshot, source="yahoo")

    second = orch.fetch_with_cache("AAPL", "2025-01-01", "2025-01-02")
    assert second.status is ResultStatus.FAILED
    assert len(primary.calls) == 2
    assert secondary.calls == []


def test_fetch_with_incremental_cache_full_miss() -> None:
    """No symbol cache -> fetch the full range once and persist symbol-level snapshot."""

    start = date(2025, 1, 1)
    end = date(2025, 4, 30)
    bars = _bars_for_days(start, 120)
    adapter_snapshot = _snapshot("AAPL", bars, source="polygon")

    cache = _StubIncrementalCache()
    cache.incremental_payload = ([], [(start, end)])

    primary = _StubAdapter("polygon", results=[Result.success(adapter_snapshot)])
    orch = DataOrchestrator(primary_adapter=primary, cache=cache)

    result = orch.fetch_with_incremental_cache("AAPL", start, end)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert len(primary.calls) == 1
    assert primary.calls[0] == ("AAPL", start, end)
    assert cache.set_calls and cache.set_calls[0][1] == 120
    assert len(result.data.bars) == 120


def test_fetch_with_incremental_cache_full_hit() -> None:
    """Full symbol-cache coverage -> no adapter calls."""

    start = date(2025, 1, 6)
    end = date(2025, 1, 10)
    cached_bars = _bars_for_days(start, 5)
    cached_snapshot = _snapshot("AAPL", cached_bars, source="polygon")

    cache = _StubIncrementalCache()
    cache_key = cache.make_symbol_key("AAPL", "1D")
    cache.snapshot_by_key[cache_key] = cached_snapshot
    cache.incremental_payload = (cached_bars, [])

    primary = _StubAdapter("polygon", results=[])
    orch = DataOrchestrator(primary_adapter=primary, cache=cache)

    result = orch.fetch_with_incremental_cache("AAPL", start, end)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert primary.calls == []
    assert result.data.bars == cached_bars


def test_fetch_with_incremental_cache_partial_fetch() -> None:
    """Partial coverage -> fetch only the gap and merge."""

    start = date(2025, 1, 1)
    end = date(2025, 1, 10)
    gap_start = date(2025, 1, 6)
    gap_end = date(2025, 1, 10)

    cached_bars = _bars_for_days(start, 5, close_start=10.0)
    gap_bars = _bars_for_days(gap_start, 5, close_start=20.0)
    cached_snapshot = _snapshot("AAPL", cached_bars, source="polygon")

    cache = _StubIncrementalCache()
    cache_key = cache.make_symbol_key("AAPL", "1D")
    cache.snapshot_by_key[cache_key] = cached_snapshot
    cache.incremental_payload = (cached_bars, [(gap_start, gap_end)])

    primary = _StubAdapter("polygon", results=[Result.success(_snapshot("AAPL", gap_bars, source="polygon"))])
    orch = DataOrchestrator(primary_adapter=primary, cache=cache)

    result = orch.fetch_with_incremental_cache("AAPL", start, end)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert len(primary.calls) == 1
    assert primary.calls[0] == ("AAPL", gap_start, gap_end)
    assert cache.merge_calls == [("AAPL", "1D", len(gap_bars))]
    assert len(result.data.bars) == 10


def test_fetch_with_incremental_cache_adapter_failure() -> None:
    """Gap fetch failures return cached data as DEGRADED (do not block the run)."""

    start = date(2025, 1, 1)
    end = date(2025, 1, 10)
    gap_start = date(2025, 1, 6)
    gap_end = date(2025, 1, 10)

    cached_bars = _bars_for_days(start, 5)
    cached_snapshot = _snapshot("AAPL", cached_bars, source="polygon")

    cache = _StubIncrementalCache()
    cache_key = cache.make_symbol_key("AAPL", "1D")
    cache.snapshot_by_key[cache_key] = cached_snapshot
    cache.incremental_payload = (cached_bars, [(gap_start, gap_end)])

    primary = _StubAdapter("polygon", results=[Result.failed(RuntimeError("down"), reason_code="ADAPTER_FAILED")])
    orch = DataOrchestrator(primary_adapter=primary, cache=cache)

    result = orch.fetch_with_incremental_cache("AAPL", start, end)
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "INCREMENTAL_FETCH_FAILED"
    assert result.data is not None
    assert result.data.bars == cached_bars
    assert primary.calls == [("AAPL", gap_start, gap_end)]


def test_fetch_with_incremental_cache_quality_validation() -> None:
    """Quality validation runs on the final window snapshot."""

    start = date(2025, 1, 1)
    end = date(2025, 1, 3)
    bars = _bars_for_days(start, 3)
    degraded_flags = {"availability": True, "completeness": True, "freshness": False, "stability": True}

    cache = _StubIncrementalCache()
    cache.incremental_payload = ([], [(start, end)])

    primary = _StubAdapter("polygon", results=[Result.success(_snapshot("AAPL", bars, source="polygon", quality_flags=degraded_flags))])
    orch = DataOrchestrator(primary_adapter=primary, cache=cache)

    result = orch.fetch_with_incremental_cache("AAPL", start, end)
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "QUALITY_DEGRADED"
    assert result.data is not None


def test_fetch_with_incremental_cache_merge_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache merge/write issues should not block returning the fetched window."""

    start = date(2025, 1, 1)
    end = date(2025, 1, 10)
    gap_start = date(2025, 1, 6)
    gap_end = date(2025, 1, 10)

    cached_bars = _bars_for_days(start, 5, close_start=10.0)
    gap_bars = _bars_for_days(gap_start, 5, close_start=20.0)
    cached_snapshot = _snapshot("AAPL", cached_bars, source="polygon")

    cache = _StubIncrementalCache()
    cache_key = cache.make_symbol_key("AAPL", "1D")
    cache.snapshot_by_key[cache_key] = cached_snapshot
    cache.incremental_payload = (cached_bars, [(gap_start, gap_end)])

    merged_snapshot = _snapshot("AAPL", [*cached_bars, *gap_bars], source="polygon")
    cache.merge_result = Result.degraded(merged_snapshot, RuntimeError("write failed"), reason_code="CACHE_MERGE_WRITE_FAILED")

    warnings: list[tuple[str, dict[str, Any]]] = []

    class _Logger:
        def warning(self, event: str, **kwargs: Any) -> None:
            warnings.append((event, kwargs))

        def info(self, event: str, **kwargs: Any) -> None:
            pass

    primary = _StubAdapter("polygon", results=[Result.success(_snapshot("AAPL", gap_bars, source="polygon"))])
    orch = DataOrchestrator(primary_adapter=primary, cache=cache)
    monkeypatch.setattr(orch, "_logger", _Logger())

    result = orch.fetch_with_incremental_cache("AAPL", start, end)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert len(result.data.bars) == 10
    assert any(event == "data.cache.merge_degraded" for event, _ in warnings)


class TestDataOrchestrator:
    """Test suite for DataOrchestrator statistics tracking."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.orchestrator = DataOrchestrator(primary_adapter=_StubAdapter("polygon", results=[]))

    def test_get_run_stats_initial(self) -> None:
        """get_run_stats() returns zero-initialized stats on fresh orchestrator."""

        stats = self.orchestrator.get_run_stats()
        assert stats.total_symbols == 0
        assert stats.freshness_pass_count == 0
        assert stats.cache_hit_count == 0
        assert stats.backfill_attempt_count == 0

    def test_reset_run_source_clears_stats(self) -> None:
        """reset_run_source() resets all statistics counters."""

        # Simulate some statistics (we can't easily trigger real stats without complex setup)
        self.orchestrator._stats_total_symbols = 10  # noqa: SLF001 - test seeding counters
        self.orchestrator._stats_cache_hit = 5  # noqa: SLF001 - test seeding counters

        self.orchestrator.reset_run_source()

        stats = self.orchestrator.get_run_stats()
        assert stats.total_symbols == 0
        assert stats.cache_hit_count == 0


class TestDataLayerStats:
    """Test suite for DataLayerStats properties."""

    def test_freshness_rate_zero_symbols(self) -> None:
        """Freshness rate is 0.0 when total_symbols=0."""

        stats = DataLayerStats()
        assert stats.freshness_rate == 0.0

    def test_freshness_rate_all_pass(self) -> None:
        """Freshness rate is 1.0 when all symbols pass."""

        stats = DataLayerStats(total_symbols=10, freshness_pass_count=10)
        assert stats.freshness_rate == 1.0

    def test_freshness_rate_partial(self) -> None:
        """Freshness rate calculates correctly for partial pass."""

        stats = DataLayerStats(total_symbols=10, freshness_pass_count=7)
        assert stats.freshness_rate == 0.7

    def test_backfill_success_rate_no_attempts(self) -> None:
        """Backfill success rate is 1.0 when no backfills attempted."""

        stats = DataLayerStats()
        assert stats.backfill_success_rate == 1.0

    def test_backfill_success_rate_all_success(self) -> None:
        """Backfill success rate is 1.0 when all backfills succeed."""

        stats = DataLayerStats(backfill_attempt_count=5, backfill_success_count=5)
        assert stats.backfill_success_rate == 1.0

    def test_backfill_success_rate_partial(self) -> None:
        """Backfill success rate calculates correctly for partial success."""

        stats = DataLayerStats(backfill_attempt_count=10, backfill_success_count=6)
        assert stats.backfill_success_rate == 0.6

    def test_cache_hit_rate_no_requests(self) -> None:
        """Cache hit rate is 0.0 when no cache requests."""

        stats = DataLayerStats()
        assert stats.cache_hit_rate == 0.0

    def test_cache_hit_rate_all_hits(self) -> None:
        """Cache hit rate is 1.0 when all requests hit cache."""

        stats = DataLayerStats(cache_hit_count=10, cache_miss_count=0)
        assert stats.cache_hit_rate == 1.0

    def test_cache_hit_rate_partial(self) -> None:
        """Cache hit rate calculates correctly for partial hits."""

        stats = DataLayerStats(cache_hit_count=3, cache_miss_count=1)
        assert stats.cache_hit_rate == 0.75
