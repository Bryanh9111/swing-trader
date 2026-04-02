"""Orchestrator for fetching market data with caching and strict failover."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
import time
from typing import Any

import structlog

from common.interface import Result, ResultStatus

from .cache import FileCache
from .interface import DataSourceAdapter, DataLayerStats, DateLike, PriceBar, PriceSeriesSnapshot, normalize_date

__all__ = ["DataOrchestrator"]

EventEmitter = Callable[[str, str, Mapping[str, Any] | None], None]


class DataQualityError(RuntimeError):
    """Raised when snapshot quality is too poor to proceed."""


@dataclass(slots=True)
class _RunSourceLock:
    source: str
    adapter: DataSourceAdapter


class DataOrchestrator:
    """Fetch price series snapshots using one source per run.

    The orchestrator enforces run-boundary failover semantics:
    - A run locks onto the first successful source (cache or adapter).
    - If the primary fails before the lock is established, the orchestrator may
      switch to the secondary adapter and lock it for the remainder of the run.
    - After locking, no per-request fallback occurs (avoids mixed sources) unless
      a call provides an explicit ``source_hint`` for hybrid date routing.
    """

    _MODULE_NAME = "data.orchestrator"

    def __init__(
        self,
        *,
        primary_adapter: DataSourceAdapter,
        secondary_adapter: DataSourceAdapter | None = None,
        cache: FileCache | None = None,
        emit_event: EventEmitter | None = None,
    ) -> None:
        """Create a data orchestrator.

        Args:
            primary_adapter: Preferred adapter for the run.
            secondary_adapter: Run-boundary fallback when the primary fails
                before the source lock is established.
            cache: Optional cache implementation.
            emit_event: Optional event emitter callback.
        """

        self._primary = primary_adapter
        self._secondary = secondary_adapter
        self._cache = cache
        self._emit_event = emit_event
        self._logger = structlog.get_logger(__name__).bind(module=self._MODULE_NAME)
        self._run_lock: _RunSourceLock | None = None
        self._stats_total_symbols: int = 0
        self._stats_freshness_pass: int = 0
        self._stats_freshness_fail: int = 0
        self._stats_backfill_attempt: int = 0
        self._stats_backfill_success: int = 0
        self._stats_backfill_fail: int = 0
        self._stats_cache_hit: int = 0
        self._stats_cache_miss: int = 0

    def reset_run_source(self) -> None:
        """Clear the run source lock (useful between orchestrator runs)."""

        self._run_lock = None
        self._stats_total_symbols = 0
        self._stats_freshness_pass = 0
        self._stats_freshness_fail = 0
        self._stats_backfill_attempt = 0
        self._stats_backfill_success = 0
        self._stats_backfill_fail = 0
        self._stats_cache_hit = 0
        self._stats_cache_miss = 0

    def get_run_stats(self) -> DataLayerStats:
        """Return a snapshot of Data Layer run-level statistics.

        Returns:
            DataLayerStats with current counter values and computed rates.
        """

        return DataLayerStats(
            total_symbols=self._stats_total_symbols,
            freshness_pass_count=self._stats_freshness_pass,
            freshness_fail_count=self._stats_freshness_fail,
            backfill_attempt_count=self._stats_backfill_attempt,
            backfill_success_count=self._stats_backfill_success,
            backfill_fail_count=self._stats_backfill_fail,
            cache_hit_count=self._stats_cache_hit,
            cache_miss_count=self._stats_cache_miss,
        )

    def fetch_with_cache(self, symbol: str, start: DateLike, end: DateLike) -> Result[PriceSeriesSnapshot]:
        """Fetch a snapshot using cache, enforcing single-source runs."""

        start_date = normalize_date(start)
        end_date = normalize_date(end)
        if end_date < start_date:
            return Result.failed(ValueError("end must be >= start."), reason_code="INVALID_RANGE")

        timeframe = "1D"
        cache_key = None

        if self._cache is not None:
            cache_key = self._cache.make_key(symbol, timeframe, start_date.isoformat(), end_date.isoformat())
            cached = self._cache.get(cache_key)
            if cached.status == ResultStatus.SUCCESS and cached.data is not None:
                snapshot = cached.data
                if self._run_lock is None:
                    resolved = self._adapter_for_source(snapshot.source)
                    if resolved is None:
                        self._emit(
                            "data.cache.ignored",
                            {"reason": "unknown_source", "cache_source": snapshot.source},
                        )
                        snapshot = None
                    else:
                        self._lock_source(snapshot.source, resolved)
                        self._emit("data.source.locked", {"source": snapshot.source, "origin": "cache"})
                if snapshot is None:
                    pass
                elif self._run_lock is not None and snapshot.source != self._run_lock.source:
                    self._emit(
                        "data.cache.ignored",
                        {"cache_source": snapshot.source, "run_source": self._run_lock.source},
                    )
                else:
                    self._emit("data.cache.hit", {"symbol": symbol, "source": snapshot.source})
                    return self.validate_quality(snapshot)

        adapter_result = self._fetch_from_adapters(symbol, start_date, end_date)
        if adapter_result.status == ResultStatus.FAILED or adapter_result.data is None:
            return adapter_result

        validated = self.validate_quality(adapter_result.data)
        if validated.status == ResultStatus.FAILED or validated.data is None:
            return validated

        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, validated.data)

        return validated

    def fetch_with_incremental_cache(self, symbol: str, start: DateLike, end: DateLike) -> Result[PriceSeriesSnapshot]:
        """Fetch with incremental caching (reduces redundant data pulls).

        Flow:
            1. Read symbol-level cache (``{symbol}_{timeframe}.json``) and detect missing ranges.
            2. If all required dates are cached: return filtered cached bars.
            3. If ranges are missing: fetch only gaps from the adapters.
            4. Merge new bars into cache and return the requested window.

        Error handling:
            - Cache read issues: ignore cache and fetch full range.
            - Incremental gap fetch failures: degrade to full-range fetch.
            - Cache merge/write failures: still return fetched data.
        """

        self._stats_total_symbols += 1

        start_date = normalize_date(start)
        end_date = normalize_date(end)
        if end_date < start_date:
            return Result.failed(ValueError("end must be >= start."), reason_code="INVALID_RANGE")

        def _track_freshness(result: Result[PriceSeriesSnapshot]) -> None:
            if result.data is None:
                return
            freshness = bool((result.data.quality_flags or {}).get("freshness", True))
            if freshness:
                self._stats_freshness_pass += 1
            else:
                self._stats_freshness_fail += 1

        timeframe = "1D"

        def _normalize_source_hint(source_hint: str | None) -> str | None:
            if source_hint is None:
                return None
            if self._adapter_for_source(source_hint) is None:
                self._emit(
                    "data.source_hint.ignored",
                    {"symbol": symbol, "reason": "adapter_missing", "source_hint": source_hint},
                )
                return None
            return source_hint

        if self._cache is None:
            # Hybrid fetching: prefer S3 for historical ranges, Polygon for current-year ranges.
            _, preferred_source = self._split_date_range_by_source(start_date, end_date)
            adapter_result = self._fetch_from_adapters(
                symbol,
                start_date,
                end_date,
                source_hint=_normalize_source_hint(preferred_source),
            )
            if adapter_result.status == ResultStatus.FAILED or adapter_result.data is None:
                return adapter_result
            validated = self.validate_quality(adapter_result.data)
            _track_freshness(validated)
            return validated

        cache_key = self._cache.make_symbol_key(symbol, timeframe)
        cached_snapshot_result = self._cache.get(cache_key)
        cached_snapshot = cached_snapshot_result.data if cached_snapshot_result.status != ResultStatus.DEGRADED else None

        incremental_result = self._cache.get_incremental(symbol, timeframe, start_date, end_date)
        if incremental_result.status == ResultStatus.DEGRADED or incremental_result.data is None:
            self._emit("data.cache.ignored", {"symbol": symbol, "reason": "incremental_degraded"})
            _, preferred_source = self._split_date_range_by_source(start_date, end_date)
            adapter_result = self._fetch_from_adapters(
                symbol,
                start_date,
                end_date,
                source_hint=_normalize_source_hint(preferred_source),
            )
            if adapter_result.status == ResultStatus.FAILED or adapter_result.data is None:
                return adapter_result
            validated = self.validate_quality(adapter_result.data)
            _track_freshness(validated)
            if validated.status != ResultStatus.FAILED and validated.data is not None:
                self._cache.set(cache_key, validated.data)
            return validated

        cached_bars, missing_ranges = incremental_result.data

        if cached_snapshot is not None:
            if self._run_lock is None:
                resolved = self._adapter_for_source(cached_snapshot.source)
                if resolved is None:
                    self._emit(
                        "data.cache.ignored",
                        {"reason": "unknown_source", "cache_source": cached_snapshot.source},
                    )
                    cached_snapshot = None
                    cached_bars = []
                    missing_ranges = [(start_date, end_date)]
                else:
                    # URGENT FIX (hybrid fetch vs run lock conflict):
                    #
                    # Historically, the orchestrator locks the entire run to the first successful
                    # source, including cache hits. This is correct for single-source runs, but it
                    # breaks hybrid fetching where one run intentionally mixes sources by date:
                    #   - S3 for historical ranges (<= last year)
                    #   - Polygon for current-year ranges
                    #
                    # When a cache hit comes from S3 and we lock the run to "s3", later requests
                    # that *must* use Polygon (via `source_hint="polygon"`) can end up being
                    # influenced by the lock/failover semantics, causing expensive and futile S3
                    # attempts against current-year dates (403s/timeouts).
                    #
                    # To avoid this, only lock from cache when the preferred source for the
                    # requested window matches the cache source. Otherwise, skip locking and allow
                    # subsequent gap fetching to be driven by `source_hint`.
                    _, preferred_source = self._split_date_range_by_source(start_date, end_date)
                    if preferred_source == cached_snapshot.source:
                        self._lock_source(cached_snapshot.source, resolved)
                        self._emit("data.source.locked", {"source": cached_snapshot.source, "origin": "cache"})
                    else:
                        self._emit(
                            "data.cache.lock_skipped",
                            {
                                "cache_source": cached_snapshot.source,
                                "preferred_source": preferred_source,
                                "reason": "hybrid_fetch_required",
                            },
                        )
            elif cached_snapshot.source != self._run_lock.source:
                self._emit(
                    "data.cache.ignored",
                    {"cache_source": cached_snapshot.source, "run_source": self._run_lock.source},
                )
                cached_snapshot = None
                cached_bars = []
                missing_ranges = [(start_date, end_date)]

        if not missing_ranges and cached_snapshot is not None:
            snapshot = PriceSeriesSnapshot(
                schema_version=cached_snapshot.schema_version,
                system_version=cached_snapshot.system_version,
                asof_timestamp=cached_snapshot.asof_timestamp,
                symbol=cached_snapshot.symbol,
                timeframe=cached_snapshot.timeframe,
                bars=cached_bars,
                source=cached_snapshot.source,
                quality_flags=cached_snapshot.quality_flags or {},
            )
            self._stats_cache_hit += 1
            self._emit("data.cache.hit", {"symbol": symbol, "source": snapshot.source, "key": cache_key})
            validated = self.validate_quality(snapshot)
            _track_freshness(validated)
            return validated

        self._stats_cache_miss += 1
        self._emit(
            "data.cache.miss",
            {
                "symbol": symbol,
                "key": cache_key,
                "missing_ranges": [(s.isoformat(), e.isoformat()) for s, e in missing_ranges],
            },
        )

        gap_bars: list[PriceBar] = []
        last_gap_snapshot: PriceSeriesSnapshot | None = None
        hard_failure: Result[PriceSeriesSnapshot] | None = None

        for gap_start, gap_end in missing_ranges:
            # Hybrid fetching: split cross-year gaps once into (historical=S3, current=Polygon).
            split_ranges, _ = self._split_date_range_by_source(gap_start, gap_end)

            for range_start, range_end, source_hint in split_ranges:
                self._stats_backfill_attempt += 1
                effective_hint = _normalize_source_hint(source_hint)
                gap_result = self._fetch_from_adapters(
                    symbol,
                    range_start,
                    range_end,
                    source_hint=effective_hint,
                )
                if gap_result.status == ResultStatus.FAILED or gap_result.data is None:
                    self._stats_backfill_fail += 1
                    if gap_result.reason_code == "NO_DATA":
                        self._logger.info(
                            "data.incremental.gap_no_data",
                            symbol=symbol,
                            start=range_start.isoformat(),
                            end=range_end.isoformat(),
                            reason_code=gap_result.reason_code,
                            source_hint=effective_hint,
                        )
                        continue
                    hard_failure = gap_result
                    break

                self._stats_backfill_success += 1
                last_gap_snapshot = gap_result.data
                gap_bars.extend(gap_result.data.bars)

            if hard_failure is not None:
                break

        if hard_failure is not None:
            self._logger.warning(
                "data.incremental.failed",
                symbol=symbol,
                reason_code=hard_failure.reason_code,
                error=str(hard_failure.error) if hard_failure.error else None,
            )
            if cached_snapshot is not None:
                cached_window_bars = [
                    bar for bar in cached_bars if start_date <= self._utc_date_from_ns(bar.timestamp) <= end_date
                ]
                cached_window_snapshot = PriceSeriesSnapshot(
                    schema_version=cached_snapshot.schema_version,
                    system_version=cached_snapshot.system_version,
                    asof_timestamp=cached_snapshot.asof_timestamp,
                    symbol=cached_snapshot.symbol,
                    timeframe=cached_snapshot.timeframe,
                    bars=cached_window_bars,
                    source=cached_snapshot.source,
                    quality_flags=cached_snapshot.quality_flags or {},
                )
                validated = self.validate_quality(cached_window_snapshot)
                if validated.status == ResultStatus.FAILED or validated.data is None:
                    return validated
                _track_freshness(validated)
                return Result.degraded(
                    validated.data,
                    hard_failure.error or RuntimeError("Incremental gap fetch failed."),
                    reason_code="INCREMENTAL_FETCH_FAILED",
                )

            _, preferred_source = self._split_date_range_by_source(start_date, end_date)
            adapter_result = self._fetch_from_adapters(
                symbol,
                start_date,
                end_date,
                source_hint=_normalize_source_hint(preferred_source),
            )
            if adapter_result.status == ResultStatus.FAILED or adapter_result.data is None:
                return adapter_result
            validated = self.validate_quality(adapter_result.data)
            _track_freshness(validated)
            if validated.status != ResultStatus.FAILED and validated.data is not None:
                self._cache.set(cache_key, validated.data)
            return validated

        merged_snapshot: PriceSeriesSnapshot
        if cached_snapshot is None:
            if last_gap_snapshot is None:
                return Result.failed(RuntimeError("No cached data and no fetched data available."), reason_code="NO_DATA")
            merged_snapshot = PriceSeriesSnapshot(
                schema_version=last_gap_snapshot.schema_version,
                system_version=last_gap_snapshot.system_version,
                asof_timestamp=last_gap_snapshot.asof_timestamp,
                symbol=last_gap_snapshot.symbol,
                timeframe=last_gap_snapshot.timeframe,
                bars=self._dedupe_sort_bars(gap_bars),
                source=last_gap_snapshot.source,
                quality_flags=last_gap_snapshot.quality_flags or {},
            )
            write = self._cache.set(cache_key, merged_snapshot)
            if write.status == ResultStatus.DEGRADED:
                self._logger.warning("data.cache.write_degraded", symbol=symbol, key=cache_key, reason_code=write.reason_code)
        else:
            # For backtest (historical) requests, disable age filter entirely.
            # The age filter is based on the NEWEST bar in cache after merge,
            # which can be much newer than the requested window (e.g., cache has
            # 2026 data, request is for 2023). This would incorrectly delete all
            # historical bars that fall outside the filter window.
            #
            # Detection: if end_date is more than 7 days before today, assume backtest.
            today = date.today()
            is_historical_request = (today - end_date).days > 7

            if is_historical_request:
                # Backtest mode: disable age filter to preserve all historical data
                max_age_days = 0
            else:
                # Live/recent mode: use dynamic age filter
                data_range_days = (end_date - start_date).days
                max_age_days = data_range_days + 365  # Add 1-year buffer

            merge_result = self._cache.merge_and_save(symbol, timeframe, gap_bars, max_age_days=max_age_days)
            if merge_result.status == ResultStatus.FAILED or merge_result.data is None:
                self._logger.warning(
                    "data.cache.merge_failed",
                    symbol=symbol,
                    reason_code=merge_result.reason_code,
                    error=str(merge_result.error) if merge_result.error else None,
                )
                merged_snapshot = PriceSeriesSnapshot(
                    schema_version=cached_snapshot.schema_version,
                    system_version=cached_snapshot.system_version,
                    asof_timestamp=time.time_ns(),
                    symbol=cached_snapshot.symbol,
                    timeframe=cached_snapshot.timeframe,
                    bars=self._dedupe_sort_bars([*cached_snapshot.bars, *gap_bars]),
                    source=cached_snapshot.source,
                    quality_flags=cached_snapshot.quality_flags or {},
                )
            else:
                if merge_result.status == ResultStatus.DEGRADED:
                    self._logger.warning(
                        "data.cache.merge_degraded",
                        symbol=symbol,
                        reason_code=merge_result.reason_code,
                        error=str(merge_result.error) if merge_result.error else None,
                    )
                merged = merge_result.data
                if last_gap_snapshot is not None:
                    merged_snapshot = PriceSeriesSnapshot(
                        schema_version=merged.schema_version,
                        system_version=merged.system_version,
                        asof_timestamp=merged.asof_timestamp,
                        symbol=merged.symbol,
                        timeframe=merged.timeframe,
                        bars=merged.bars,
                        source=last_gap_snapshot.source,
                        quality_flags=last_gap_snapshot.quality_flags or merged.quality_flags or {},
                    )
                else:
                    merged_snapshot = merged

        window_bars = [bar for bar in merged_snapshot.bars if start_date <= self._utc_date_from_ns(bar.timestamp) <= end_date]

        window_snapshot = PriceSeriesSnapshot(
            schema_version=merged_snapshot.schema_version,
            system_version=merged_snapshot.system_version,
            asof_timestamp=merged_snapshot.asof_timestamp,
            symbol=merged_snapshot.symbol,
            timeframe=merged_snapshot.timeframe,
            bars=window_bars,
            source=merged_snapshot.source,
            quality_flags=merged_snapshot.quality_flags or {},
        )
        validated = self.validate_quality(window_snapshot)
        _track_freshness(validated)
        return validated

    def validate_quality(self, snapshot: PriceSeriesSnapshot) -> Result[PriceSeriesSnapshot]:
        """Validate snapshot quality flags, preferring DEGRADED over FAILED."""

        flags = snapshot.quality_flags or {}
        availability = bool(flags.get("availability", True))
        completeness = bool(flags.get("completeness", True))
        freshness = bool(flags.get("freshness", True))
        stability = bool(flags.get("stability", True))

        if not availability:
            return Result.failed(DataQualityError("Data unavailable."), reason_code="QUALITY_UNAVAILABLE")

        issues: list[str] = []
        if not completeness:
            issues.append("completeness")
        if not freshness:
            issues.append("freshness")
        if not stability:
            issues.append("stability")

        if issues:
            return Result.degraded(
                snapshot,
                DataQualityError(f"Quality degraded: {', '.join(issues)}"),
                reason_code="QUALITY_DEGRADED",
            )

        return Result.success(snapshot, reason_code="OK")

    def _split_date_range_by_source(self, start: date, end: date) -> tuple[list[tuple[date, date, str]], str]:
        """Split a date range into historical vs current-year segments.

        This supports hybrid fetching when the historical adapter (S3) only
        contains data up to last year while the live adapter (Polygon) has current
        year data.

        Returns:
            (ranges, preferred_source)
            - ranges: [(start, end, source_hint)] where source_hint is "s3" or "polygon"
            - preferred_source: primary source name to prefer for run-level locking

        Logic:
            - s3_cutoff = Dec 31 of (current_year - 1)
            - if end <= s3_cutoff: all historical -> S3
            - if start > s3_cutoff: all current-year -> Polygon
            - else: cross-year -> split once into (historical=S3, current=Polygon),
              and prefer Polygon for freshness.
        """

        from datetime import datetime, UTC, timedelta

        current_year = datetime.now(UTC).year
        s3_cutoff = date(current_year - 1, 12, 31)

        if end <= s3_cutoff:
            return [(start, end, "s3")], "s3"
        if start > s3_cutoff:
            return [(start, end, "polygon")], "polygon"
        return [
            (start, s3_cutoff, "s3"),
            (s3_cutoff + timedelta(days=1), end, "polygon"),
        ], "polygon"

    def _fetch_from_adapters(
        self,
        symbol: str,
        start: date,
        end: date,
        source_hint: str | None = None,
    ) -> Result[PriceSeriesSnapshot]:
        """Fetch from adapters with optional forced source selection.

        Backward compatible behavior (no hint):
            - If a run is already locked -> use the lock.
            - Otherwise, try primary then secondary (existing failover), locking to the
              first successful adapter for the remainder of the run.

        Hybrid-fetch behavior (hint provided):
            - `source_hint` is treated as authoritative: ONLY use the hinted adapter.
            - Do NOT fallback to primary/secondary or the current run lock if the hinted
              adapter returns empty/failed data.

        This is critical for cross-year hybrid fetching where we *know* one adapter is
        unable to serve the other adapter's date segment (e.g., S3 has no current-year
        data, Polygon has no deep historical data). Any fallback in those cases is both
        slow and noisy (403s/timeouts) and can extend runs by hours.
        """

        # When `source_hint` is provided, bypass the run lock and failover logic entirely.
        # This prevents an earlier cache hit (e.g., from S3) from forcing later, current-year
        # requests onto the wrong adapter.
        if source_hint is not None:
            hinted_adapter = self._adapter_for_source(source_hint)
            if hinted_adapter is None:
                return Result.failed(
                    RuntimeError(f"Source hint '{source_hint}' does not match any available adapter"),
                    reason_code="INVALID_SOURCE_HINT",
                )

            result = hinted_adapter.fetch_price_series(symbol, start, end)
            self._emit_fetch_result(
                hinted_adapter.get_source_name(),
                symbol,
                start,
                end,
                result,
                source_hint=source_hint,
            )

            # Only establish a run lock on success and only if the run is not already locked.
            if self._run_lock is None and result.status != ResultStatus.FAILED and result.data is not None:
                self._lock_source(hinted_adapter.get_source_name(), hinted_adapter)
                self._emit(
                    "data.source.locked",
                    {"source": hinted_adapter.get_source_name(), "origin": "hint_forced", "hint": source_hint},
                )

            return result

        if self._run_lock is not None:
            locked = self._run_lock
            result = locked.adapter.fetch_price_series(symbol, start, end)
            self._emit_fetch_result(locked.source, symbol, start, end, result, source_hint=source_hint)
            return result

        primary_adapter = self._primary
        secondary_adapter = self._secondary

        result = primary_adapter.fetch_price_series(symbol, start, end)
        self._emit_fetch_result(primary_adapter.get_source_name(), symbol, start, end, result, source_hint=source_hint)
        if result.status != ResultStatus.FAILED and result.data is not None:
            self._lock_source(primary_adapter.get_source_name(), primary_adapter)
            self._emit(
                "data.source.locked",
                {"source": primary_adapter.get_source_name(), "origin": "primary_with_hint", "hint": source_hint},
            )
            return result

        if secondary_adapter is None:
            return result

        self._emit(
            "data.failover.attempt",
            {"from": primary_adapter.get_source_name(), "to": secondary_adapter.get_source_name(), "hint": source_hint},
        )
        secondary_result = secondary_adapter.fetch_price_series(symbol, start, end)
        self._emit_fetch_result(
            secondary_adapter.get_source_name(),
            symbol,
            start,
            end,
            secondary_result,
            source_hint=source_hint,
        )
        if secondary_result.status != ResultStatus.FAILED and secondary_result.data is not None:
            self._lock_source(secondary_adapter.get_source_name(), secondary_adapter)
            self._emit(
                "data.source.locked",
                {"source": secondary_adapter.get_source_name(), "origin": "secondary_fallback", "hint": source_hint},
            )
        return secondary_result

    def _adapter_for_source(self, source: str) -> DataSourceAdapter | None:
        """Resolve a source name to an adapter, if available."""

        if source == self._primary.get_source_name():
            return self._primary
        if self._secondary is not None and source == self._secondary.get_source_name():
            return self._secondary
        return None

    def _lock_source(self, source: str, adapter: DataSourceAdapter) -> None:
        self._run_lock = _RunSourceLock(source=source, adapter=adapter)

    def _emit(self, event_type: str, data: Mapping[str, Any] | None = None) -> None:
        if self._emit_event is None:
            return
        self._emit_event(event_type, self._MODULE_NAME, data)

    def _emit_fetch_result(
        self,
        source: str,
        symbol: str,
        start: date,
        end: date,
        result: Result[PriceSeriesSnapshot],
        *,
        source_hint: str | None = None,
    ) -> None:
        payload = {
            "source": source,
            "symbol": symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "status": result.status.value,
            "reason_code": result.reason_code,
        }
        if source_hint is not None:
            payload["source_hint"] = source_hint
        if result.status == ResultStatus.FAILED and result.error is not None:
            self._logger.warning("data.fetch_failed", **payload, error=str(result.error))
            self._emit("data.fetch.failed", payload)
            return
        self._logger.info("data.fetch_completed", **payload)
        self._emit("data.fetch.completed", payload)

    @staticmethod
    def _utc_date_from_ns(timestamp_ns: int) -> date:
        dt = datetime.fromtimestamp(int(timestamp_ns) / 1_000_000_000, tz=UTC)
        return dt.date()

    @staticmethod
    def _dedupe_sort_bars(bars: list[PriceBar]) -> list[PriceBar]:
        by_ts: dict[int, PriceBar] = {}
        for bar in bars:
            by_ts[int(bar.timestamp)] = bar
        return [by_ts[timestamp] for timestamp in sorted(by_ts)]
