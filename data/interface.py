"""Interfaces and schemas for the Phase 2.2 price data layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol, TypeAlias

from msgspec import Struct

from common.interface import Result
from journal.interface import SnapshotBase

__all__ = [
    "DateLike",
    "PriceBar",
    "PriceSeriesSnapshot",
    "DataSourceAdapter",
    "DataLayerStats",
    "QualityAssessment",
    "normalize_date",
]

DateLike: TypeAlias = date | datetime | str


def normalize_date(value: DateLike) -> date:
    """Normalize a date-like value to a ``datetime.date`` instance."""

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(UTC).date()
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(value)!r}")


class PriceBar(Struct, frozen=True, kw_only=True):
    """OHLCV data for a single bar (nanosecond timestamps)."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int


class PriceSeriesSnapshot(SnapshotBase, frozen=True, kw_only=True):
    """Normalized OHLCV time series snapshot for one symbol."""

    symbol: str
    timeframe: str
    bars: list[PriceBar]
    source: str
    quality_flags: dict[str, Any]


class DataSourceAdapter(Protocol):
    """Protocol for pluggable price data source adapters."""

    def fetch_price_series(self, symbol: str, start_date: DateLike, end_date: DateLike) -> Result[PriceSeriesSnapshot]:
        """Fetch and normalize a price series for the provided symbol/date range."""

    def get_source_name(self) -> str:
        """Return a stable identifier for the data source (e.g. ``\"polygon\"``)."""


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Helper for building quality flag payloads."""

    availability: bool
    completeness: bool
    freshness: bool
    stability: bool
    details: dict[str, Any]

    def to_flags(self) -> dict[str, Any]:
        """Convert the assessment into a ``quality_flags`` mapping."""

        return {
            "availability": self.availability,
            "completeness": self.completeness,
            "freshness": self.freshness,
            "stability": self.stability,
            **self.details,
        }


@dataclass(frozen=True, slots=True)
class DataLayerStats:
    """Run-level statistics for Data Layer fetch operations.

    Tracks data quality metrics across all symbols fetched in a run,
    including freshness compliance and backfill success rates.

    Attributes:
        total_symbols: Total number of unique symbols fetched.
        freshness_pass_count: Symbols with fresh data (within threshold).
        freshness_fail_count: Symbols with stale data.
        backfill_attempt_count: Symbols requiring backfill (gap filling).
        backfill_success_count: Successful backfill operations.
        backfill_fail_count: Failed backfill operations.
        cache_hit_count: Symbols served from cache.
        cache_miss_count: Symbols fetched from adapters.
    """

    total_symbols: int = 0
    freshness_pass_count: int = 0
    freshness_fail_count: int = 0
    backfill_attempt_count: int = 0
    backfill_success_count: int = 0
    backfill_fail_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0

    @property
    def freshness_rate(self) -> float:
        """Percentage of symbols with fresh data."""

        if self.total_symbols == 0:
            return 0.0
        return self.freshness_pass_count / self.total_symbols

    @property
    def backfill_success_rate(self) -> float:
        """Percentage of successful backfill operations."""

        if self.backfill_attempt_count == 0:
            return 1.0  # No backfills needed = 100% success
        return self.backfill_success_count / self.backfill_attempt_count

    @property
    def cache_hit_rate(self) -> float:
        """Percentage of cache hits."""

        total_requests = self.cache_hit_count + self.cache_miss_count
        if total_requests == 0:
            return 0.0
        return self.cache_hit_count / total_requests
