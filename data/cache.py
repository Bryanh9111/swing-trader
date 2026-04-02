"""File-based caching layer for replay-safe price snapshots."""

from __future__ import annotations

import os
import time
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import structlog

from common.interface import Result, ResultStatus
from journal.run_id import RunIDGenerator

from .interface import PriceBar, PriceSeriesSnapshot

if TYPE_CHECKING:
    from common.market_calendar import MarketCalendar

__all__ = ["FileCache"]


class CacheEntry(msgspec.Struct, frozen=True, kw_only=True):
    """Serialized cache entry wrapper for replay safety + TTL enforcement."""

    saved_at_ns: int
    snapshot: PriceSeriesSnapshot


class FileCache:
    """A JSON file cache for ``PriceSeriesSnapshot`` payloads."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/data",
        ttl_seconds: int = 86400,
        market_calendar: MarketCalendar | None = None,
    ) -> None:
        """Create a file cache.

        Args:
            cache_dir: Directory for cache entries.
            ttl_seconds: Entry TTL in seconds. Set to ``0`` to disable expiry.
        """

        self._cache_dir = Path(cache_dir)
        self._ttl_seconds = int(ttl_seconds)
        self._market_calendar = market_calendar
        self._decoder = msgspec.json.Decoder(type=CacheEntry)
        self._encoder = msgspec.json.Encoder()
        self._logger = structlog.get_logger(__name__).bind(module="data.cache", cache_dir=str(cache_dir))

        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def make_key(self, symbol: str, timeframe: str, start: str, end: str) -> str:
        """Create a stable cache key for a price series request."""

        safe_symbol = symbol.replace(os.sep, "_").upper()
        safe_timeframe = timeframe.replace(os.sep, "_").upper()
        return f"{safe_symbol}_{safe_timeframe}_{start}_{end}.json"

    def make_symbol_key(self, symbol: str, timeframe: str) -> str:
        """Create cache key for symbol-level aggregated cache."""

        safe_symbol = symbol.replace(os.sep, "_").upper()
        safe_timeframe = timeframe.replace(os.sep, "_").upper()
        return f"{safe_symbol}_{safe_timeframe}.json"

    def get(self, key: str) -> Result[PriceSeriesSnapshot | None]:
        """Retrieve a snapshot from cache if present and not expired."""

        path = self._path_for_key(key)
        if not path.exists():
            return Result.success(None, reason_code="CACHE_MISS")

        try:
            entry = self._decoder.decode(path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - cache should be resilient.
            return Result.degraded(None, exc, reason_code="CACHE_READ_FAILED")

        if self._ttl_seconds > 0:
            age_seconds = (time.time_ns() - entry.saved_at_ns) / 1_000_000_000
            if age_seconds > self._ttl_seconds:
                with suppress(Exception):
                    path.unlink(missing_ok=True)
                return Result.success(None, reason_code="CACHE_EXPIRED")

        return Result.success(entry.snapshot, reason_code="CACHE_HIT")

    def get_incremental(
        self,
        symbol: str,
        timeframe: str,
        requested_start: date,
        requested_end: date,
    ) -> Result[tuple[list[PriceBar], list[tuple[date, date]]]]:
        """Get cached data and detect missing date ranges.

        Returns:
            Result containing:
            - list[PriceBar]: Cached bars (may be empty)
            - list[tuple[date, date]]: Missing date ranges to fetch
              Example: [(2025-01-01, 2025-01-15), (2025-06-20, 2025-06-30)]
              Empty list if all data is cached

        Logic:
            1. Read cached snapshot using symbol-level key.
            2. Extract existing bars and their observed dates.
            3. Detect gaps for weekday dates in [requested_start, requested_end].
            4. Handle TTL expiry/read errors by treating as no cache.
        """

        if requested_end < requested_start:
            return Result.failed(ValueError("requested_end must be >= requested_start."), reason_code="INVALID_RANGE")

        key = self.make_symbol_key(symbol, timeframe)
        cached = self.get(key)

        if cached.status == ResultStatus.DEGRADED:
            payload = ([], [(requested_start, requested_end)])
            try:
                self._logger.warning(
                    "cache.incremental.ignored",
                    symbol=symbol,
                    timeframe=timeframe,
                    reason_code=cached.reason_code,
                    error=str(cached.error) if cached.error else None,
                )
            except Exception:  # noqa: BLE001
                pass
            return Result.degraded(payload, cached.error or RuntimeError("Cache read degraded."), "CACHE_IGNORED")

        snapshot = cached.data
        if snapshot is None or not snapshot.bars:
            try:
                self._logger.info(
                    "cache.incremental.miss",
                    symbol=symbol,
                    timeframe=timeframe,
                    reason_code=cached.reason_code,
                )
            except Exception:  # noqa: BLE001
                pass
            return Result.success(([], [(requested_start, requested_end)]), reason_code="CACHE_MISS")

        cached_bars = list(snapshot.bars)
        dates_present = {self._utc_date_from_ns(bar.timestamp) for bar in cached_bars}
        missing_dates = [
            d
            for d in self._weekday_dates(requested_start, requested_end)
            if d not in dates_present and self._is_trading_day(d)
        ]
        missing_ranges = self._collapse_dates_to_ranges(missing_dates)

        if not missing_ranges:
            filtered = [bar for bar in cached_bars if requested_start <= self._utc_date_from_ns(bar.timestamp) <= requested_end]
            try:
                self._logger.info(
                    "cache.incremental.hit",
                    symbol=symbol,
                    timeframe=timeframe,
                    cached_bars=len(cached_bars),
                    filtered_bars=len(filtered),
                )
            except Exception:  # noqa: BLE001
                pass
            return Result.success((filtered, []), reason_code="CACHE_HIT")

        try:
            self._logger.info(
                "cache.incremental.partial",
                symbol=symbol,
                timeframe=timeframe,
                cached_bars=len(cached_bars),
                missing_ranges=[(s.isoformat(), e.isoformat()) for s, e in missing_ranges],
            )
        except Exception:  # noqa: BLE001
            pass
        return Result.success((cached_bars, missing_ranges), reason_code="CACHE_PARTIAL")

    def set(self, key: str, snapshot: PriceSeriesSnapshot) -> Result[None]:
        """Store a snapshot in cache as JSON."""

        path = self._path_for_key(key)
        entry = CacheEntry(saved_at_ns=time.time_ns(), snapshot=snapshot)
        try:
            payload = self._encoder.encode(entry)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_bytes(payload)
            tmp_path.replace(path)
        except Exception as exc:  # noqa: BLE001
            return Result.degraded(None, exc, reason_code="CACHE_WRITE_FAILED")
        return Result.success(None, reason_code="CACHE_WRITE_OK")

    def merge_and_save(
        self,
        symbol: str,
        timeframe: str,
        new_bars: list[PriceBar],
        max_age_days: int = 365,
    ) -> Result[PriceSeriesSnapshot]:
        """Merge new bars into cache and save.

        Args:
            symbol: Stock symbol.
            timeframe: Timeframe (e.g., "1D").
            new_bars: New bars to merge.
            max_age_days: Remove bars older than this (0 = keep all).

        Returns:
            Result containing the merged snapshot.

        Logic:
            1. Read existing cache.
            2. Merge new_bars with existing bars (dedupe by timestamp, new wins).
            3. Sort by timestamp.
            4. Remove bars older than max_age_days (if > 0, relative to newest bar).
            5. Create a new snapshot.
            6. Save to cache.
            7. Return snapshot (DEGRADED if cache save fails).
        """

        key = self.make_symbol_key(symbol, timeframe)
        existing_result = self.get(key)
        existing_snapshot = existing_result.data if existing_result.status != ResultStatus.DEGRADED else None

        existing_bars = list(existing_snapshot.bars) if existing_snapshot is not None else []
        merged_by_ts: dict[int, PriceBar] = {int(bar.timestamp): bar for bar in existing_bars}
        for bar in new_bars:
            merged_by_ts[int(bar.timestamp)] = bar

        merged = [merged_by_ts[timestamp] for timestamp in sorted(merged_by_ts)]
        removed = 0
        if max_age_days > 0 and merged:
            newest_date = self._utc_date_from_ns(merged[-1].timestamp)
            cutoff = newest_date - timedelta(days=int(max_age_days))
            before = len(merged)
            merged = [bar for bar in merged if self._utc_date_from_ns(bar.timestamp) >= cutoff]
            removed = before - len(merged)

        deduped = max(0, len(existing_bars) + len(new_bars) - len(merged_by_ts))
        added = max(0, len(merged) - len(existing_bars))

        quality_flags: dict = {}
        source = "unknown"
        if existing_snapshot is not None:
            quality_flags = dict(existing_snapshot.quality_flags or {})
            source = existing_snapshot.source
        else:
            quality_flags = {"availability": True, "completeness": True, "freshness": True, "stability": True}

        snapshot = PriceSeriesSnapshot(
            schema_version=getattr(existing_snapshot, "schema_version", "1.0.0") if existing_snapshot is not None else "1.0.0",
            system_version=getattr(existing_snapshot, "system_version", RunIDGenerator.get_system_version())
            if existing_snapshot is not None
            else RunIDGenerator.get_system_version(),
            asof_timestamp=time.time_ns(),
            symbol=symbol.upper(),
            timeframe=timeframe,
            bars=merged,
            source=source,
            quality_flags=quality_flags,
        )

        try:
            self._logger.info(
                "cache.merge.saved",
                symbol=symbol,
                timeframe=timeframe,
                existing_bars=len(existing_bars),
                new_bars=len(new_bars),
                merged_bars=len(merged),
                deduped=deduped,
                removed=removed,
                added=added,
            )
        except Exception:  # noqa: BLE001
            pass

        write = self.set(key, snapshot)
        if write.status == ResultStatus.DEGRADED:
            return Result.degraded(snapshot, write.error or RuntimeError("Cache write degraded."), "CACHE_MERGE_WRITE_FAILED")
        return Result.success(snapshot, reason_code="CACHE_MERGE_OK")

    def invalidate(self, key: str) -> Result[None]:
        """Invalidate a cache entry, if it exists."""

        path = self._path_for_key(key)
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            return Result.degraded(None, exc, reason_code="CACHE_INVALIDATE_FAILED")
        return Result.success(None, reason_code="CACHE_INVALIDATE_OK")

    def _path_for_key(self, key: str) -> Path:
        filename = Path(key).name
        return self._cache_dir / filename

    @staticmethod
    def _utc_date_from_ns(timestamp_ns: int) -> date:
        dt = datetime.fromtimestamp(int(timestamp_ns) / 1_000_000_000, tz=UTC)
        return dt.date()

    def _is_trading_day(self, target: date) -> bool:
        """Check if date is a trading day (not a holiday). Returns True if calendar unavailable or API fails."""

        if self._market_calendar is None:
            return True

        result = self._market_calendar.is_trading_day(target)
        if result.status is ResultStatus.FAILED:
            return True
        return bool(result.data)

    @staticmethod
    def _weekday_dates(start: date, end: date) -> list[date]:
        if end < start:
            return []
        days = (end - start).days
        dates: list[date] = []
        for offset in range(days + 1):
            current = start + timedelta(days=offset)
            if current.weekday() < 5:
                dates.append(current)
        return dates

    @staticmethod
    def _collapse_dates_to_ranges(dates: list[date]) -> list[tuple[date, date]]:
        if not dates:
            return []
        sorted_dates = sorted(dates)
        ranges: list[tuple[date, date]] = []
        range_start = sorted_dates[0]
        prev = sorted_dates[0]
        for current in sorted_dates[1:]:
            if current == prev + timedelta(days=1):
                prev = current
                continue
            ranges.append((range_start, prev))
            range_start = current
            prev = current
        ranges.append((range_start, prev))
        return ranges
