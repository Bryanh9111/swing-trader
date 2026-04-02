"""Event data acquisition sources for Event Guard.

This module implements:

1) Polygon (Massive SDK) integration as the primary event source.
2) Manual file ingestion (CSV/JSON) as a fallback and override layer.
3) A conservative fallback orchestrator returning ``Result[EventSnapshot]``.

All timestamps are normalized to UTC nanosecond epoch integers.
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests
import structlog

from common.interface import Result, ResultStatus
from journal.run_id import RunIDGenerator

from .interface import EventDataSourceAdapter, EventGuardConfig, EventSnapshot, EventType, MarketEvent, RiskLevel

__all__ = [
    "PolygonEventSource",
    "PolygonRestEventSource",
    "ManualFileSource",
    "YFinanceEventSource",
    "fetch_events_with_fallback",
]

_SCHEMA_VERSION = "1.0.0"
_DEFAULT_POLYGON_TIMEOUT_SECONDS = 15.0

try:
    from massive import RESTClient  # type: ignore[import-not-found]

    HAS_MASSIVE_SDK = True
except ImportError:  # pragma: no cover
    HAS_MASSIVE_SDK = False
    RESTClient = None  # type: ignore[assignment]

try:
    import yfinance as yf  # type: ignore[import-not-found]

    HAS_YFINANCE = True
except ImportError:  # pragma: no cover
    HAS_YFINANCE = False
    yf = None  # type: ignore[assignment]


def _now_snapshot_base_kwargs() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "system_version": RunIDGenerator.get_system_version(),
        "asof_timestamp": time.time_ns(),
    }


def _normalize_symbol(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    value = str(symbol).strip().upper()
    return value or None


def _coerce_enum(value: Any, enum_type: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, enum_type):
        return value
    with suppress(Exception):
        return enum_type(str(value).strip().upper())
    return None


def _default_risk_for_event_type(event_type: EventType) -> RiskLevel:
    if event_type == EventType.EARNINGS:
        return RiskLevel.HIGH
    if event_type in (EventType.STOCK_SPLIT, EventType.REVERSE_SPLIT):
        return RiskLevel.HIGH
    if event_type == EventType.DIVIDEND:
        return RiskLevel.LOW
    return RiskLevel.MEDIUM


def _parse_date_utc_ns(value: str | date | datetime) -> int | None:
    """Parse a date/datetime-like value into UTC midnight nanoseconds.

    Polygon reference endpoints (splits/dividends) generally provide calendar
    dates. To keep normalization deterministic, those dates are interpreted as
    00:00:00 UTC on the given day.
    """

    if isinstance(value, datetime):
        dt = value.astimezone(UTC)
        return int(dt.timestamp() * 1_000_000_000)

    if isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=UTC)
        return int(dt.timestamp() * 1_000_000_000)

    text = str(value).strip()
    if not text:
        return None

    # Common Polygon payload: YYYY-MM-DD
    with suppress(ValueError):
        parsed_date = datetime.strptime(text[:10], "%Y-%m-%d").date()
        dt = datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=UTC)
        return int(dt.timestamp() * 1_000_000_000)

    # ISO8601 datetime (manual files commonly).
    iso = text.replace("Z", "+00:00")
    with suppress(ValueError):
        dt2 = datetime.fromisoformat(iso)
        if dt2.tzinfo is None:
            dt2 = dt2.replace(tzinfo=UTC)
        return int(dt2.astimezone(UTC).timestamp() * 1_000_000_000)

    return None


def _parse_timestamp_ns(value: Any) -> int | None:
    """Parse numeric or ISO-ish values into UTC nanosecond epoch."""

    if value is None:
        return None

    if isinstance(value, (datetime, date, str)):
        return _parse_date_utc_ns(value)  # includes ISO datetime support

    if isinstance(value, (int, float)):
        number = int(value)
        if number <= 0:
            return None
        # Heuristic: seconds / ms / us / ns.
        if number < 1_000_000_000_000:  # < 1e12 => seconds
            return number * 1_000_000_000
        if number < 1_000_000_000_000_000:  # < 1e15 => milliseconds
            return number * 1_000_000
        if number < 1_000_000_000_000_000_000:  # < 1e18 => microseconds
            return number * 1_000
        return number

    # Fallback string parse (e.g. numbers stored as strings).
    text = str(value).strip()
    if not text:
        return None
    with suppress(ValueError):
        return _parse_timestamp_ns(int(text))
    return _parse_date_utc_ns(text)


def _in_range(ts_ns: int, start_ns: int, end_ns: int) -> bool:
    return start_ns <= ts_ns <= end_ns


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("size must be > 0.")
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _merge_events(primary: list[MarketEvent], overlay: list[MarketEvent]) -> list[MarketEvent]:
    """Merge events with overlay priority.

    Manual overrides take precedence over API results. Two events are
    considered identical if they share (symbol, event_type, event_date).
    """

    merged: dict[tuple[str, EventType, int], MarketEvent] = {}
    for event in primary:
        merged[(event.symbol, event.event_type, event.event_date)] = event
    for event in overlay:
        merged[(event.symbol, event.event_type, event.event_date)] = event
    return sorted(merged.values(), key=lambda e: (e.symbol, e.event_date, e.event_type.value))


@dataclass(slots=True)
class PolygonEventSource(EventDataSourceAdapter):
    """Polygon (Massive SDK) event source for earnings, dividends and splits.

    This uses the Massive SDK already referenced elsewhere in the repo
    (see ``universe/data_sources.py``) to avoid manual HTTP plumbing.
    """

    api_key: str
    requests_timeout_seconds: float = _DEFAULT_POLYGON_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        self._logger = structlog.get_logger(__name__).bind(source="polygon", component="event_guard")
        self._timeout_seconds = float(self.requests_timeout_seconds)
        self._client = self._build_client()

    def _build_client(self) -> Any:
        if not HAS_MASSIVE_SDK or RESTClient is None:  # pragma: no cover
            return None

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        timeout = float(self.requests_timeout_seconds)
        with suppress(Exception):
            params = inspect.signature(RESTClient).parameters
            if "timeout" in params:
                client_kwargs["timeout"] = timeout
            else:
                if "connect_timeout" in params:
                    client_kwargs["connect_timeout"] = timeout
                if "read_timeout" in params:
                    client_kwargs["read_timeout"] = timeout
        return RESTClient(**client_kwargs)

    def get_source_name(self) -> str:
        return "polygon"

    def fetch_events(self, symbols: list[str], start_ns: int, end_ns: int) -> Result[EventSnapshot]:
        """Fetch earnings calendar + corporate actions and normalize to MarketEvent.

        Args:
            symbols: List of ticker symbols to fetch events for.
            start_ns: Inclusive UTC nanosecond start timestamp.
            end_ns: Inclusive UTC nanosecond end timestamp.

        Returns:
            ``Result[EventSnapshot]`` containing normalized events.
        """

        normalized_symbols = [s for s in (_normalize_symbol(sym) for sym in symbols) if s]
        if not normalized_symbols:
            snapshot = EventSnapshot(
                **_now_snapshot_base_kwargs(),
                events=[],
                source=self.get_source_name(),
                symbols_covered=[],
            )
            return Result.success(snapshot, reason_code="OK")

        if start_ns > end_ns:
            return Result.failed(ValueError("start_ns must be <= end_ns."), reason_code="INVALID_RANGE")

        if not self.api_key:
            return Result.failed(RuntimeError("POLYGON_API_KEY is not set."), reason_code="MISSING_API_KEY")

        if not HAS_MASSIVE_SDK or self._client is None:
            return Result.failed(RuntimeError("massive SDK is not available."), reason_code="NOT_AVAILABLE")

        start_date = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC).date()
        end_date = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC).date()

        events: list[MarketEvent] = []

        endpoints = [
            "/stocks/financials/v1/income-statements",  # Primary earnings source
            "/v2/reference/news",  # Supplementary earnings source
            "/benzinga/v1/earnings",  # Fallback earnings source
            "/v3/reference/dividends",
            "/v3/reference/splits",
        ]
        logger = self._logger.bind(endpoints=endpoints, symbols=len(normalized_symbols))

        try:
            # Primary: Fetch earnings from Income Statements API
            financials_earnings = self._fetch_earnings_from_financials(
                normalized_symbols, start_date, end_date, start_ns, end_ns
            )
            events.extend(financials_earnings)

            # Supplementary: Fetch earnings from News API (real-time detection)
            news_earnings: list[MarketEvent] = []
            try:
                news_earnings = self._fetch_earnings_from_news(
                    normalized_symbols, start_date, end_date, start_ns, end_ns
                )
                events.extend(news_earnings)
            except Exception as news_exc:  # noqa: BLE001
                exc_name = type(news_exc).__name__
                logger.warning(
                    "polygon.earnings.news_fetch_failed",
                    error_type=exc_name,
                    error=str(news_exc)[:200],
                )
                # Continue without news earnings - not critical

            # Fallback: Try Benzinga if both financials and news returned no data
            if not financials_earnings and not news_earnings:
                logger.info(
                    "polygon.earnings.financials_empty_trying_benzinga",
                    symbols=len(normalized_symbols),
                )
                try:
                    benzinga_earnings = self._fetch_earnings_from_benzinga(
                        normalized_symbols, start_date, end_date, start_ns, end_ns
                    )
                    events.extend(benzinga_earnings)
                except Exception as benzinga_exc:  # noqa: BLE001
                    exc_name = type(benzinga_exc).__name__
                    logger.warning(
                        "polygon.earnings.benzinga_fallback_failed",
                        error_type=exc_name,
                        error=str(benzinga_exc)[:200],
                    )
                    # Continue without Benzinga earnings - not critical

            events.extend(self._fetch_dividends(normalized_symbols, start_date, end_date, start_ns, end_ns))
            events.extend(self._fetch_splits(normalized_symbols, start_date, end_date, start_ns, end_ns))
        except Exception as exc:  # noqa: BLE001
            exc_name = type(exc).__name__
            logger.warning("polygon.events.sdk_error", error_type=exc_name, error=str(exc)[:200])
            if exc_name == "AuthError":
                return Result.failed(exc, reason_code="MISSING_API_KEY")
            if exc_name == "BadResponse":
                return Result.failed(exc, reason_code="HTTP_ERROR")
            return Result.failed(exc, reason_code="HTTP_ERROR")

        # De-duplicate events (same symbol/event_type/date from different sources)
        events_before = len(events)
        events = self._dedup_events(events)
        if events_before > len(events):
            logger.info(
                "polygon.events.deduped",
                events_before=events_before,
                events_after=len(events),
                removed=events_before - len(events),
            )

        snapshot = EventSnapshot(
            **_now_snapshot_base_kwargs(),
            events=events,  # Already sorted by _dedup_events()
            source=self.get_source_name(),
            symbols_covered=normalized_symbols,
        )
        return Result.success(snapshot, reason_code="OK")

    def _fetch_earnings_from_financials(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        start_ns: int,
        end_ns: int,
    ) -> list[MarketEvent]:
        """Fetch historical earnings from income statements filing dates.

        Args:
            symbols: List of ticker symbols.
            start_date: Start date for filing_date filter.
            end_date: End date for filing_date filter.
            start_ns: Start timestamp in nanoseconds (for range check).
            end_ns: End timestamp in nanoseconds (for range check).

        Returns:
            List of MarketEvent objects representing earnings events.
        """

        earnings_events: list[MarketEvent] = []

        # Batch query for efficiency (Income Statements API supports ticker.any_of)
        for batch in _chunked(symbols, size=50):
            ticker_any_of = ",".join(batch)

            try:
                url = "https://api.polygon.io/stocks/financials/v1/income-statements"
                params = {
                    "tickers.any_of": ticker_any_of,
                    "timeframe": "quarterly",
                    "filing_date.gte": start_date.isoformat(),
                    "filing_date.lte": end_date.isoformat(),
                    "limit": 1000,
                    "apiKey": getattr(self._client, "api_key", self.api_key),
                }

                response = requests.get(url, params=params, timeout=self._timeout_seconds)
                response.raise_for_status()
            except requests.RequestException as exc:
                self._logger.warning(
                    "polygon.earnings.financials_batch_failed",
                    batch_size=len(batch),
                    ticker_any_of=ticker_any_of,
                    error=str(exc)[:200],
                )
                continue
            try:
                data = response.json()
            except ValueError as exc:
                self._logger.warning(
                    "polygon.earnings.financials_decode_failed",
                    batch_size=len(batch),
                    ticker_any_of=ticker_any_of,
                    error=str(exc)[:200],
                )
                continue

            items = data.get("results", []) if isinstance(data, dict) else []
            if not isinstance(items, list):
                self._logger.warning(
                    "polygon.earnings.financials_decode_failed",
                    batch_size=len(batch),
                    ticker_any_of=ticker_any_of,
                    error="unexpected_response_shape",
                )
                continue

            for stmt in items:
                # Extract ticker (income statements return tickers as list)
                if not isinstance(stmt, dict):
                    continue
                tickers = stmt.get("tickers")
                if not tickers or not isinstance(tickers, list):
                    continue
                symbol = _normalize_symbol(tickers[0])
                if not symbol:
                    continue

                # Use filing_date as earnings event date
                filing_date = stmt.get("filing_date")
                event_ns = _parse_date_utc_ns(filing_date) if filing_date else None

                if event_ns is None or not _in_range(event_ns, start_ns, end_ns):
                    continue

                # Extract metadata
                metadata: dict[str, str] = {}
                fiscal_year = stmt.get("fiscal_year")
                if fiscal_year is not None:
                    metadata["fiscal_year"] = str(fiscal_year)

                fiscal_quarter = stmt.get("fiscal_quarter")
                if fiscal_quarter is not None:
                    metadata["fiscal_quarter"] = str(fiscal_quarter)

                if filing_date is not None:
                    metadata["filing_date"] = str(filing_date)

                period_end = stmt.get("period_end")
                if period_end is not None:
                    metadata["period_end"] = str(period_end)

                eps = stmt.get("basic_earnings_per_share")
                if eps is not None:
                    metadata["eps"] = str(eps)

                earnings_events.append(
                    MarketEvent(
                        symbol=symbol,
                        event_type=EventType.EARNINGS,
                        event_date=event_ns,
                        risk_level=RiskLevel.HIGH,
                        source=f"{self.get_source_name()}_financials",
                        metadata=metadata or None,
                    )
                )

        return earnings_events

    def _fetch_earnings_from_benzinga(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        start_ns: int,
        end_ns: int,
    ) -> list[MarketEvent]:
        """Fetch earnings from Benzinga API (fallback source when financials fail).

        Note: This endpoint requires Benzinga entitlement and may return NOT_AUTHORIZED
        if the API key does not have access. This is kept as a fallback source.

        Args:
            symbols: List of ticker symbols.
            start_date: Start date for earnings filter.
            end_date: End date for earnings filter.
            start_ns: Start timestamp in nanoseconds.
            end_ns: End timestamp in nanoseconds.

        Returns:
            List of MarketEvent objects representing earnings events.
        """

        earnings_events: list[MarketEvent] = []

        # Benzinga endpoint supports ticker_any_of, so batch to reduce calls.
        for batch in _chunked(symbols, size=50):
            ticker_any_of = ",".join(batch)
            items = self._client.list_benzinga_earnings(  # type: ignore[no-any-return]
                ticker_any_of=ticker_any_of,
                date_gte=start_date,
                date_lte=end_date,
                limit=1000,
            )
            for earning in items:
                symbol = _normalize_symbol(getattr(earning, "ticker", None))
                if not symbol:
                    continue
                raw_date = getattr(earning, "date", None)
                event_ns = _parse_date_utc_ns(raw_date) if raw_date else None
                if event_ns is None or not _in_range(event_ns, start_ns, end_ns):
                    continue

                metadata: dict[str, str] = {}
                for key in ("time", "fiscal_period", "fiscal_year", "importance", "date_status", "company_name", "currency"):
                    value = getattr(earning, key, None)
                    if value is not None and str(value).strip():
                        metadata[key] = str(value).strip()
                provider_id = getattr(earning, "benzinga_id", None)
                if provider_id is not None and str(provider_id).strip():
                    metadata["provider_event_id"] = str(provider_id).strip()

                earnings_events.append(
                    MarketEvent(
                        symbol=symbol,
                        event_type=EventType.EARNINGS,
                        event_date=event_ns,
                        risk_level=RiskLevel.HIGH,
                        source=self.get_source_name(),
                        metadata=metadata or None,
                    )
                )

        return earnings_events

    def _fetch_earnings_from_news(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        start_ns: int,
        end_ns: int,
    ) -> list[MarketEvent]:
        """Fetch earnings events from news articles (real-time detection).

        This method searches for news articles containing earnings-related keywords
        and extracts sentiment information. It serves as a supplementary source
        for detecting earnings announcements in real-time.

        Args:
            symbols: List of ticker symbols.
            start_date: Start date for published_utc filter.
            end_date: End date for published_utc filter.
            start_ns: Start timestamp in nanoseconds (for range check).
            end_ns: End timestamp in nanoseconds (for range check).

        Returns:
            List of MarketEvent objects representing earnings events detected from news.
        """

        earnings_events: list[MarketEvent] = []

        # Earnings keywords for filtering
        earnings_keywords = {
            "earnings",
            "quarterly results",
            "q1 results",
            "q2 results",
            "q3 results",
            "q4 results",
            "financial results",
            "earnings report",
        }

        for symbol in symbols:
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                "ticker": symbol,
                "published_utc.gte": start_date.isoformat(),
                "published_utc.lte": end_date.isoformat(),
                "limit": 100,
                "order": "desc",
                "apiKey": getattr(self._client, "api_key", self.api_key),
            }

            try:
                response = requests.get(url, params=params, timeout=self._timeout_seconds)
                response.raise_for_status()
            except requests.RequestException as exc:
                # Log but continue - partial failure acceptable
                self._logger.warning(
                    "polygon.earnings.news_fetch_failed",
                    symbol=symbol,
                    error=str(exc)[:200],
                )
                continue

            try:
                data = response.json()
            except ValueError as exc:
                self._logger.warning(
                    "polygon.earnings.news_decode_failed",
                    symbol=symbol,
                    error=str(exc)[:200],
                )
                continue

            news_items = data.get("results", [])
            if not isinstance(news_items, list):
                news_items = []

            for article in news_items:
                if not isinstance(article, dict):
                    continue
                # Filter earnings-related news
                title = (article.get("title", "") or "").lower()

                # Check if title contains earnings keywords
                if not any(keyword in title for keyword in earnings_keywords):
                    continue

                # Extract published timestamp
                published_utc = article.get("published_utc")
                event_ns = _parse_timestamp_ns(published_utc)

                if event_ns is None or not _in_range(event_ns, start_ns, end_ns):
                    continue

                # Extract sentiment from insights
                sentiment = "neutral"
                insights = article.get("insights")
                if insights and isinstance(insights, list):
                    for insight in insights:
                        if not isinstance(insight, dict):
                            continue
                        if insight.get("ticker", "") == symbol:
                            sentiment = str(insight.get("sentiment", "neutral"))
                            break

                # Extract metadata
                metadata: dict[str, str] = {}
                article_title = article.get("title")
                if article_title:
                    metadata["title"] = str(article_title)

                metadata["sentiment"] = sentiment

                publisher = article.get("publisher")
                if publisher and isinstance(publisher, dict):
                    publisher_name = publisher.get("name", "")
                    if publisher_name:
                        metadata["publisher"] = str(publisher_name)

                article_url = article.get("article_url", "")
                if article_url:
                    metadata["article_url"] = str(article_url)

                earnings_events.append(
                    MarketEvent(
                        symbol=symbol,
                        event_type=EventType.EARNINGS,
                        event_date=event_ns,
                        risk_level=RiskLevel.HIGH,
                        source=f"{self.get_source_name()}_news",
                        metadata=metadata or None,
                    )
                )

        return earnings_events

    @staticmethod
    def _dedup_events(events: list[MarketEvent]) -> list[MarketEvent]:
        """De-duplicate events with same (symbol, event_type, event_date).

        Priority: financials > news > benzinga > manual
        Rationale: Filing dates are more authoritative than news publish dates.

        Args:
            events: List of MarketEvent objects (may contain duplicates).

        Returns:
            De-duplicated list of MarketEvent objects, sorted by (symbol, event_date, event_type).
        """

        # Source priority mapping (lower number = higher priority)
        source_priority = {
            "polygon_financials": 1,
            "polygon_news": 2,
            "polygon": 3,  # Benzinga via Polygon
            "manual": 4,
            "merged": 5,
            "degraded": 6,
        }

        # Group events by (symbol, event_type, event_date)
        grouped: dict[tuple[str, EventType, int], list[MarketEvent]] = {}
        for event in events:
            key = (event.symbol, event.event_type, event.event_date)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(event)

        # For each group, keep only the highest priority event
        deduped: list[MarketEvent] = []
        for event_list in grouped.values():
            # Sort by priority (lower priority number = higher priority)
            event_list.sort(key=lambda e: source_priority.get(e.source, 99))
            # Keep the first (highest priority) event
            deduped.append(event_list[0])

        # Sort final list by (symbol, event_date, event_type)
        return sorted(deduped, key=lambda e: (e.symbol, e.event_date, e.event_type.value))

    def _fetch_dividends(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        start_ns: int,
        end_ns: int,
    ) -> list[MarketEvent]:
        dividend_events: list[MarketEvent] = []
        for symbol in symbols:
            items = self._client.list_dividends(  # type: ignore[no-any-return]
                ticker=symbol,
                ex_dividend_date_gte=start_date,
                ex_dividend_date_lte=end_date,
                limit=1000,
            )
            for dividend in items:
                sym = _normalize_symbol(getattr(dividend, "ticker", None)) or symbol
                raw_date = getattr(dividend, "ex_dividend_date", None)
                event_ns = _parse_date_utc_ns(raw_date) if raw_date else None
                if event_ns is None or not _in_range(event_ns, start_ns, end_ns):
                    continue

                metadata: dict[str, str] = {}
                for key in ("pay_date", "record_date", "declaration_date", "frequency", "cash_amount", "dividend_type", "currency"):
                    value = getattr(dividend, key, None)
                    if value is not None and str(value).strip():
                        metadata[key] = str(value).strip()

                dividend_events.append(
                    MarketEvent(
                        symbol=sym,
                        event_type=EventType.DIVIDEND,
                        event_date=event_ns,
                        risk_level=RiskLevel.LOW,
                        source=self.get_source_name(),
                        metadata=metadata or None,
                    )
                )
        return dividend_events

    def _fetch_splits(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        start_ns: int,
        end_ns: int,
    ) -> list[MarketEvent]:
        split_events: list[MarketEvent] = []
        for symbol in symbols:
            items = self._client.list_splits(  # type: ignore[no-any-return]
                ticker=symbol,
                execution_date_gte=start_date,
                execution_date_lte=end_date,
                limit=1000,
            )
            for split in items:
                sym = _normalize_symbol(getattr(split, "ticker", None)) or symbol
                raw_date = getattr(split, "execution_date", None)
                event_ns = _parse_date_utc_ns(raw_date) if raw_date else None
                if event_ns is None or not _in_range(event_ns, start_ns, end_ns):
                    continue

                split_from = getattr(split, "split_from", None)
                split_to = getattr(split, "split_to", None)
                reverse = False
                with suppress(Exception):
                    if split_from is not None and split_to is not None:
                        reverse = int(split_from) > int(split_to)

                event_type = EventType.REVERSE_SPLIT if reverse else EventType.STOCK_SPLIT
                metadata: dict[str, str] = {}
                if split_from is not None:
                    metadata["split_from"] = str(split_from)
                if split_to is not None:
                    metadata["split_to"] = str(split_to)

                split_events.append(
                    MarketEvent(
                        symbol=sym,
                        event_type=event_type,
                        event_date=event_ns,
                        risk_level=_default_risk_for_event_type(event_type),
                        source=self.get_source_name(),
                        metadata=metadata or None,
                    )
                )
        return split_events


@dataclass(slots=True)
class PolygonRestEventSource(EventDataSourceAdapter):
    """Polygon REST API event source for dividends and splits (no SDK dependency).

    Uses direct HTTP calls to Polygon's /stocks/v1/dividends and /stocks/v1/splits
    endpoints, which are available on the current subscription plan.
    """

    api_key: str
    requests_timeout_seconds: float = _DEFAULT_POLYGON_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        self._logger = structlog.get_logger(__name__).bind(source="polygon_rest", component="event_guard")
        self._timeout_seconds = float(self.requests_timeout_seconds)
        self._BASE_URL = "https://api.polygon.io"

    def get_source_name(self) -> str:
        return "polygon_rest"

    def fetch_events(self, symbols: list[str], start_ns: int, end_ns: int) -> Result[EventSnapshot]:
        """Fetch dividends and splits from Polygon REST APIs.

        Args:
            symbols: List of ticker symbols to fetch events for.
            start_ns: Inclusive UTC nanosecond start timestamp.
            end_ns: Inclusive UTC nanosecond end timestamp.

        Returns:
            Result[EventSnapshot] containing normalized dividend and split events.
        """

        normalized_symbols = [s for s in (_normalize_symbol(sym) for sym in symbols) if s]
        if not normalized_symbols:
            snapshot = EventSnapshot(
                **_now_snapshot_base_kwargs(),
                events=[],
                source=self.get_source_name(),
                symbols_covered=[],
            )
            return Result.success(snapshot, reason_code="OK")

        if start_ns > end_ns:
            return Result.failed(ValueError("start_ns must be <= end_ns."), reason_code="INVALID_RANGE")

        if not self.api_key:
            return Result.failed(RuntimeError("POLYGON_API_KEY is not set."), reason_code="MISSING_API_KEY")

        start_date = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC).date()
        end_date = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC).date()

        logger = self._logger.bind(
            op="fetch_events",
            endpoints=["/stocks/v1/dividends", "/stocks/v1/splits"],
            symbols=len(normalized_symbols),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        logger.info("polygon_rest.fetch_start")

        events: list[MarketEvent] = []

        try:
            dividend_events = self._fetch_dividends_rest(normalized_symbols, start_date, end_date, start_ns, end_ns)
            events.extend(dividend_events)

            split_events = self._fetch_splits_rest(normalized_symbols, start_date, end_date, start_ns, end_ns)
            events.extend(split_events)
        except Exception as exc:  # noqa: BLE001
            exc_name = type(exc).__name__
            logger.warning("polygon_rest.events.fetch_error", error_type=exc_name, error=str(exc)[:200])
            return Result.failed(exc, reason_code="HTTP_ERROR")

        events_before = len(events)
        events = self._dedup_events(events)
        if events_before > len(events):
            logger.info(
                "polygon_rest.events.deduped",
                events_before=events_before,
                events_after=len(events),
                removed=events_before - len(events),
            )

        snapshot = EventSnapshot(
            **_now_snapshot_base_kwargs(),
            events=events,
            source=self.get_source_name(),
            symbols_covered=normalized_symbols,
        )
        logger.info("polygon_rest.fetch_complete", events=len(events))
        return Result.success(snapshot, reason_code="OK")

    def _get_paginated_results(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        batch_logger: Any,
        max_pages: int = 25,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        url = f"{self._BASE_URL}{endpoint}"
        params = dict(params)
        params["apiKey"] = self.api_key

        page = 0
        next_url: str | None = None
        next_url_has_api_key = False
        while page < max_pages:
            page += 1
            request_url = next_url or url
            request_params = params if next_url is None else ({} if next_url_has_api_key else {"apiKey": self.api_key})
            try:
                response = requests.get(request_url, params=request_params, timeout=self._timeout_seconds)
                response.raise_for_status()
            except requests.RequestException as exc:
                batch_logger.warning(
                    "polygon_rest.http_failed",
                    endpoint=endpoint,
                    page=page,
                    error=str(exc)[:200],
                )
                break

            try:
                data = response.json()
            except ValueError as exc:
                batch_logger.warning(
                    "polygon_rest.http_decode_failed",
                    endpoint=endpoint,
                    page=page,
                    error=str(exc)[:200],
                )
                break

            items = data.get("results", []) if isinstance(data, dict) else []
            if not isinstance(items, list):
                batch_logger.warning(
                    "polygon_rest.http_unexpected_shape",
                    endpoint=endpoint,
                    page=page,
                    error="unexpected_response_shape",
                )
                break

            for item in items:
                if isinstance(item, dict):
                    results.append(item)

            next_url_value = data.get("next_url") if isinstance(data, dict) else None
            if not next_url_value or not isinstance(next_url_value, str) or not next_url_value.strip():
                break
            next_url = next_url_value.strip()
            if next_url.startswith("/"):
                next_url = f"{self._BASE_URL}{next_url}"
            elif not next_url.startswith("http"):
                next_url = f"{self._BASE_URL}/{next_url}"
            next_url_has_api_key = "apiKey=" in next_url

        if page >= max_pages:
            batch_logger.warning(
                "polygon_rest.http_max_pages_reached",
                endpoint=endpoint,
                max_pages=max_pages,
            )

        return results

    def _fetch_dividends_rest(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        start_ns: int,
        end_ns: int,
    ) -> list[MarketEvent]:
        """Fetch dividend events via Polygon's /stocks/v1/dividends REST endpoint."""

        dividend_events: list[MarketEvent] = []
        for batch in _chunked(symbols, size=50):
            ticker_any_of = ",".join(batch)
            batch_logger = self._logger.bind(op="dividends", batch_size=len(batch), ticker_any_of=ticker_any_of)
            batch_logger.debug("polygon_rest.batch_start")
            batch_events_before = len(dividend_events)

            params = {
                "ticker.any_of": ticker_any_of,
                "ex_dividend_date.gte": start_date.isoformat(),
                "ex_dividend_date.lte": end_date.isoformat(),
                "limit": 1000,
            }

            items = self._get_paginated_results("/stocks/v1/dividends", params, batch_logger=batch_logger)
            for item in items:
                symbol = _normalize_symbol(item.get("ticker"))
                if not symbol:
                    continue

                raw_date = item.get("ex_dividend_date")
                event_ns = _parse_date_utc_ns(raw_date) if raw_date else None
                if event_ns is None or not _in_range(event_ns, start_ns, end_ns):
                    continue

                metadata: dict[str, str] = {}
                for key in ("cash_amount", "currency", "pay_date", "record_date", "declaration_date", "frequency"):
                    value = item.get(key)
                    if value is not None and str(value).strip():
                        metadata[key] = str(value).strip()

                distribution_type = item.get("distribution_type")
                if distribution_type is None:
                    distribution_type = item.get("dividend_type")
                if distribution_type is not None and str(distribution_type).strip():
                    metadata["distribution_type"] = str(distribution_type).strip()

                dividend_events.append(
                    MarketEvent(
                        symbol=symbol,
                        event_type=EventType.DIVIDEND,
                        event_date=event_ns,
                        risk_level=RiskLevel.LOW,
                        source=self.get_source_name(),
                        metadata=metadata or None,
                    )
                )

            batch_logger.debug("polygon_rest.batch_complete", events_found=len(dividend_events) - batch_events_before)

        return dividend_events

    def _fetch_splits_rest(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        start_ns: int,
        end_ns: int,
    ) -> list[MarketEvent]:
        """Fetch split events via Polygon's /stocks/v1/splits REST endpoint."""

        split_events: list[MarketEvent] = []
        for batch in _chunked(symbols, size=50):
            ticker_any_of = ",".join(batch)
            batch_logger = self._logger.bind(op="splits", batch_size=len(batch), ticker_any_of=ticker_any_of)
            batch_logger.debug("polygon_rest.batch_start")
            batch_events_before = len(split_events)

            params = {
                "ticker.any_of": ticker_any_of,
                "execution_date.gte": start_date.isoformat(),
                "execution_date.lte": end_date.isoformat(),
                "limit": 1000,
            }

            items = self._get_paginated_results("/stocks/v1/splits", params, batch_logger=batch_logger)
            for item in items:
                symbol = _normalize_symbol(item.get("ticker"))
                if not symbol:
                    continue

                raw_date = item.get("execution_date")
                event_ns = _parse_date_utc_ns(raw_date) if raw_date else None
                if event_ns is None or not _in_range(event_ns, start_ns, end_ns):
                    continue

                adjustment_type = item.get("adjustment_type")
                adjustment_type_norm = str(adjustment_type).strip().lower() if adjustment_type is not None else ""
                if adjustment_type_norm == "reverse_split":
                    event_type = EventType.REVERSE_SPLIT
                elif adjustment_type_norm == "forward_split":
                    event_type = EventType.STOCK_SPLIT
                else:
                    split_from = item.get("split_from")
                    split_to = item.get("split_to")
                    reverse = False
                    with suppress(Exception):
                        if split_from is not None and split_to is not None:
                            reverse = int(split_from) > int(split_to)
                    event_type = EventType.REVERSE_SPLIT if reverse else EventType.STOCK_SPLIT

                metadata: dict[str, str] = {}
                for key in ("split_from", "split_to", "adjustment_type", "historical_adjustment_factor"):
                    value = item.get(key)
                    if value is not None and str(value).strip():
                        metadata[key] = str(value).strip()

                split_events.append(
                    MarketEvent(
                        symbol=symbol,
                        event_type=event_type,
                        event_date=event_ns,
                        risk_level=RiskLevel.HIGH,
                        source=self.get_source_name(),
                        metadata=metadata or None,
                    )
                )

            batch_logger.debug("polygon_rest.batch_complete", events_found=len(split_events) - batch_events_before)

        return split_events

    @staticmethod
    def _dedup_events(events: list[MarketEvent]) -> list[MarketEvent]:
        """De-duplicate events by (symbol, event_type, event_date) and sort."""

        deduped: dict[tuple[str, EventType, int], MarketEvent] = {}
        for event in events:
            key = (event.symbol, event.event_type, event.event_date)
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = event
                continue

            existing_meta_len = len(existing.metadata or {})
            incoming_meta_len = len(event.metadata or {})
            if incoming_meta_len > existing_meta_len:
                deduped[key] = event

        return sorted(deduped.values(), key=lambda e: (e.symbol, e.event_date, e.event_type.value))


@dataclass(slots=True)
class YFinanceEventSource(EventDataSourceAdapter):
    """Yahoo Finance event source for earnings calendar.

    Provides:
    - Future earnings dates (calendar)
    - Historical earnings dates (earnings_dates)
    - Completely free

    Limitations:
    - Requires per-ticker queries (slower than bulk APIs)
    - Rate limiting required
    - Less reliable than paid sources
    """

    max_workers: int = 10  # Concurrent query threads
    request_delay_seconds: float = 0.05  # Rate limiting
    cache_ttl_hours: int = 24  # Cache validity

    def __post_init__(self) -> None:
        self._logger = structlog.get_logger(__name__).bind(source="yfinance", component="event_guard")

    def get_source_name(self) -> str:
        return "yfinance"

    def fetch_events(self, symbols: list[str], start_ns: int, end_ns: int) -> Result[EventSnapshot]:
        """Fetch earnings events from Yahoo Finance.

        Args:
            symbols: List of ticker symbols.
            start_ns: Inclusive UTC nanosecond start timestamp.
            end_ns: Inclusive UTC nanosecond end timestamp.

        Returns:
            Result[EventSnapshot] containing earnings events.
        """

        if not HAS_YFINANCE or yf is None:
            return Result.failed(
                RuntimeError("yfinance is not installed. Run: pip install yfinance"),
                reason_code="NOT_AVAILABLE",
            )

        normalized_symbols = [s for s in (_normalize_symbol(sym) for sym in symbols) if s]
        if not normalized_symbols:
            snapshot = EventSnapshot(
                **_now_snapshot_base_kwargs(),
                events=[],
                source=self.get_source_name(),
                symbols_covered=[],
            )
            return Result.success(snapshot, reason_code="OK")

        if start_ns > end_ns:
            return Result.failed(ValueError("start_ns must be <= end_ns."), reason_code="INVALID_RANGE")

        start_date = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC).date()
        end_date = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC).date()

        # Parallel fetch with ThreadPoolExecutor
        earnings_events: list[MarketEvent] = []
        failed_count = 0

        self._logger.info(
            "yfinance.fetch_start",
            symbols=len(normalized_symbols),
            max_workers=self.max_workers,
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(
                    self._fetch_earnings_for_symbol,
                    symbol,
                    start_date,
                    end_date,
                    start_ns,
                    end_ns,
                ): symbol
                for symbol in normalized_symbols
            }

            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    events = future.result()
                    if events:
                        earnings_events.extend(events)
                        self._logger.debug(
                            "yfinance.symbol_fetched",
                            symbol=symbol,
                            events_count=len(events),
                        )
                except Exception as exc:  # noqa: BLE001
                    failed_count += 1
                    self._logger.warning(
                        "yfinance.symbol_failed",
                        symbol=symbol,
                        error=str(exc)[:200],
                    )

        # Sort events
        earnings_events.sort(key=lambda e: (e.symbol, e.event_date, e.event_type.value))

        self._logger.info(
            "yfinance.fetch_complete",
            total_symbols=len(normalized_symbols),
            failed_symbols=failed_count,
            events_found=len(earnings_events),
        )

        snapshot = EventSnapshot(
            **_now_snapshot_base_kwargs(),
            events=earnings_events,
            source=self.get_source_name(),
            symbols_covered=normalized_symbols,
        )

        return Result.success(snapshot, reason_code="OK")

    @staticmethod
    @lru_cache(maxsize=2048)
    def _ticker(symbol: str) -> Any:
        if yf is None:  # pragma: no cover
            raise RuntimeError("yfinance is not available.")
        return yf.Ticker(symbol)

    def _fetch_earnings_for_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        start_ns: int,
        end_ns: int,
    ) -> list[MarketEvent]:
        """Fetch earnings for a single symbol (called by ThreadPool)."""

        import time

        del start_date, end_date

        earnings_events: list[MarketEvent] = []

        # Rate limiting
        time.sleep(self.request_delay_seconds)

        try:
            ticker = self._ticker(symbol)

            # Try to get calendar (future earnings)
            try:
                calendar = ticker.calendar
                if calendar is not None and not calendar.empty:
                    # calendar is a DataFrame with potential 'Earnings Date' column
                    if "Earnings Date" in calendar.columns or "Earnings Date" in calendar.index:
                        earnings_date = None

                        # Handle both column and index cases
                        if "Earnings Date" in calendar.columns:
                            earnings_date = calendar["Earnings Date"].iloc[0] if len(calendar) > 0 else None
                        elif "Earnings Date" in calendar.index:
                            earnings_date = (
                                calendar.loc["Earnings Date"].iloc[0]
                                if hasattr(calendar.loc["Earnings Date"], "iloc")
                                else calendar.loc["Earnings Date"]
                            )

                        if earnings_date is not None:
                            event_ns = _parse_timestamp_ns(earnings_date)
                            if event_ns and _in_range(event_ns, start_ns, end_ns):
                                earnings_events.append(
                                    MarketEvent(
                                        symbol=symbol,
                                        event_type=EventType.EARNINGS,
                                        event_date=event_ns,
                                        risk_level=RiskLevel.HIGH,
                                        source=f"{self.get_source_name()}_calendar",
                                        metadata={"source_type": "future_calendar"},
                                    )
                                )
            except Exception:
                pass  # Calendar may not be available for all tickers

            # Try to get earnings_dates (historical earnings)
            try:
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    # earnings_dates is a DataFrame with datetime index
                    for idx, row in earnings_dates.iterrows():
                        event_ns = _parse_timestamp_ns(idx)
                        if event_ns and _in_range(event_ns, start_ns, end_ns):
                            # Extract EPS if available
                            metadata: dict[str, str] = {"source_type": "historical"}
                            if "EPS Estimate" in row and row["EPS Estimate"] is not None:
                                metadata["eps_estimate"] = str(row["EPS Estimate"])
                            if "Reported EPS" in row and row["Reported EPS"] is not None:
                                metadata["eps_reported"] = str(row["Reported EPS"])

                            earnings_events.append(
                                MarketEvent(
                                    symbol=symbol,
                                    event_type=EventType.EARNINGS,
                                    event_date=event_ns,
                                    risk_level=RiskLevel.HIGH,
                                    source=f"{self.get_source_name()}_historical",
                                    metadata=metadata or None,
                                )
                            )
            except Exception:
                pass  # earnings_dates may not be available

        except Exception as exc:
            # Re-raise to be caught by ThreadPoolExecutor
            raise RuntimeError(f"yfinance fetch failed for {symbol}: {exc}") from exc

        return earnings_events


@dataclass(slots=True)
class ManualFileSource(EventDataSourceAdapter):
    """Manual event source from a CSV/JSON file.

    Supported formats:
    - CSV header: ``symbol,event_type,event_date,risk_level,source`` (``source`` optional)
    - JSON list of objects: ``[{\"symbol\": \"AAPL\", \"event_type\": \"EARNINGS\", ...}]``
    """

    file_path: str

    def __post_init__(self) -> None:
        self._logger = structlog.get_logger(__name__).bind(source="manual", component="event_guard")

    def get_source_name(self) -> str:
        return "manual"

    def fetch_events(self, symbols: list[str], start_ns: int, end_ns: int) -> Result[EventSnapshot]:
        """Load events from a CSV/JSON file and normalize to MarketEvent."""

        normalized_symbols = {s for s in (_normalize_symbol(sym) for sym in symbols) if s}
        if start_ns > end_ns:
            return Result.failed(ValueError("start_ns must be <= end_ns."), reason_code="INVALID_RANGE")

        path = Path(self.file_path)
        if not path.exists():
            return Result.failed(FileNotFoundError(str(path)), reason_code="FILE_NOT_FOUND")

        try:
            if path.suffix.lower() == ".csv":
                events = self._load_csv(path)
            elif path.suffix.lower() == ".json":
                events = self._load_json(path)
            else:
                # Conservative: refuse unknown formats.
                return Result.failed(ValueError(f"Unsupported file format: {path.suffix}"), reason_code="UNSUPPORTED_FORMAT")
        except json.JSONDecodeError as exc:
            self._logger.warning("manual.events.json_decode_error", path=str(path), error=str(exc)[:200])
            return Result.failed(exc, reason_code="DECODE_ERROR")
        except csv.Error as exc:
            self._logger.warning("manual.events.csv_decode_error", path=str(path), error=str(exc)[:200])
            return Result.failed(exc, reason_code="DECODE_ERROR")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("manual.events.read_failed", path=str(path), error=str(exc)[:200])
            return Result.failed(exc, reason_code="READ_ERROR")

        filtered: list[MarketEvent] = []
        for event in events:
            if normalized_symbols and event.symbol not in normalized_symbols:
                continue
            if not _in_range(event.event_date, start_ns, end_ns):
                continue
            filtered.append(event)

        symbols_covered = sorted(normalized_symbols) if normalized_symbols else sorted({event.symbol for event in filtered})
        snapshot = EventSnapshot(
            **_now_snapshot_base_kwargs(),
            events=sorted(filtered, key=lambda e: (e.symbol, e.event_date, e.event_type.value)),
            source=self.get_source_name(),
            symbols_covered=symbols_covered,
        )
        return Result.success(snapshot, reason_code="OK")

    def _load_csv(self, path: Path) -> list[MarketEvent]:
        events: list[MarketEvent] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV is missing headers.")

            for row in reader:
                event = self._row_to_event(row)
                if event is not None:
                    events.append(event)
        return events

    def _load_json(self, path: Path) -> list[MarketEvent]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("JSON payload must be a list of objects.")
        events: list[MarketEvent] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            event = self._row_to_event(dict(item))
            if event is not None:
                events.append(event)
        return events

    def _row_to_event(self, row: Mapping[str, Any]) -> MarketEvent | None:
        symbol = _normalize_symbol(row.get("symbol"))
        if not symbol:
            return None

        raw_event_type = row.get("event_type")
        event_type = _coerce_enum(raw_event_type, EventType) or EventType.OTHER

        ts = _parse_timestamp_ns(row.get("event_date"))
        if ts is None:
            return None

        raw_risk = row.get("risk_level")
        risk = _coerce_enum(raw_risk, RiskLevel)
        if risk is None:
            # Conservative fallback if missing/invalid.
            risk = _default_risk_for_event_type(event_type)

        source = str(row.get("source") or self.get_source_name()).strip().lower() or self.get_source_name()

        metadata: dict[str, str] = {}
        raw_metadata = row.get("metadata")
        if isinstance(raw_metadata, Mapping):
            for k, v in raw_metadata.items():
                if k is None or v is None:
                    continue
                key = str(k).strip()
                val = str(v).strip()
                if key and val:
                    metadata[key] = val

        return MarketEvent(
            symbol=symbol,
            event_type=event_type,
            event_date=ts,
            risk_level=risk,
            source=source,
            metadata=metadata or None,
        )


def fetch_events_with_fallback(
    config: EventGuardConfig,
    symbols: list[str],
    start_ns: int,
    end_ns: int,
) -> Result[EventSnapshot]:
    """Fetch events using Polygon + manual overrides with conservative fallbacks.

    Strategy:
    1) Try Polygon (Tier 1/2) if configured and available.
       - Polygon fetch includes internal Benzinga fallback for earnings when needed.
    2) If Polygon fails, try YFinance (Tier 3) when enabled.
    3) If both API sources fail, fall back to manual file alone (Tier 4).
    4) Always attempt to apply manual overrides when configured.
    5) If all fail, return DEGRADED with an empty snapshot.

    Manual file events take precedence over API events on identical keys
    (symbol, event_type, event_date).
    """

    logger = structlog.get_logger(__name__).bind(component="event_guard", op="fetch_events_with_fallback")

    sources_cfg = getattr(config, "sources", None)
    if not isinstance(sources_cfg, Mapping):
        sources_cfg = {}
    yfinance_cfg = sources_cfg.get("yfinance", {})
    if not isinstance(yfinance_cfg, Mapping):
        yfinance_cfg = {}
    yfinance_enabled = bool(yfinance_cfg.get("enabled", False))
    yfinance_max_workers = 10
    yfinance_request_delay_seconds = 0.05
    with suppress(Exception):
        yfinance_max_workers = int(yfinance_cfg.get("max_workers", yfinance_max_workers))
    with suppress(Exception):
        yfinance_request_delay_seconds = float(
            yfinance_cfg.get("request_delay_seconds", yfinance_request_delay_seconds)
        )
    if yfinance_max_workers <= 0:
        yfinance_max_workers = 10
    if yfinance_request_delay_seconds < 0:
        yfinance_request_delay_seconds = 0.0

    # Read polygon_rest configuration
    polygon_rest_cfg = sources_cfg.get("polygon_rest", {})
    if not isinstance(polygon_rest_cfg, Mapping):
        polygon_rest_cfg = {}
    polygon_rest_enabled = bool(polygon_rest_cfg.get("enabled", False))
    polygon_rest_priority = int(polygon_rest_cfg.get("priority", 999))
    polygon_rest_timeout = 30.0
    with suppress(Exception):
        polygon_rest_timeout = float(polygon_rest_cfg.get("request_timeout_seconds", 30.0))

    # Read polygon (SDK) configuration - check if enabled in sources_cfg
    polygon_sdk_cfg = sources_cfg.get("polygon", {})
    if not isinstance(polygon_sdk_cfg, Mapping):
        polygon_sdk_cfg = {}
    polygon_sdk_enabled = bool(polygon_sdk_cfg.get("enabled", False))
    polygon_sdk_priority = int(polygon_sdk_cfg.get("priority", 999))

    # Read yfinance priority
    yfinance_priority = int(yfinance_cfg.get("priority", 999))
    logger = logger.bind(yfinance_priority=yfinance_priority)

    manual_path = config.manual_events_file
    manual_source = ManualFileSource(manual_path) if manual_path else None

    polygon_result: Result[EventSnapshot] | None = None
    yfinance_result: Result[EventSnapshot] | None = None
    manual_result: Result[EventSnapshot] | None = None

    # Determine which Polygon source to use based on priority
    # Lower number = higher priority
    polygon_result = None

    # Choose the highest priority Polygon source (polygon_rest vs polygon SDK)
    if polygon_rest_enabled and polygon_rest_priority <= polygon_sdk_priority:
        # Use PolygonRestEventSource (direct REST API, no SDK)
        api_key = os.environ.get("POLYGON_API_KEY", "")
        if api_key:
            polygon_source = PolygonRestEventSource(
                api_key=api_key,
                requests_timeout_seconds=polygon_rest_timeout,
            )
            polygon_result = polygon_source.fetch_events(symbols, start_ns, end_ns)
            logger.info(
                "event_guard.using_polygon_rest",
                priority=polygon_rest_priority,
                timeout=polygon_rest_timeout,
            )
        else:
            logger.warning("event_guard.polygon_rest_no_api_key")
            polygon_result = Result.failed(
                ValueError("POLYGON_API_KEY not found in environment"),
                "MISSING_API_KEY",
            )
    elif polygon_sdk_enabled:
        # Use PolygonEventSource (Massive SDK)
        api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY") or ""
        if api_key:
            polygon_source = PolygonEventSource(api_key=api_key)
            polygon_result = polygon_source.fetch_events(symbols, start_ns, end_ns)
            logger.info(
                "event_guard.using_polygon_sdk",
                priority=polygon_sdk_priority,
            )
        else:
            logger.warning("event_guard.polygon_sdk_no_api_key")
            polygon_result = Result.failed(
                ValueError("POLYGON_API_KEY not found in environment"),
                "MISSING_API_KEY",
            )
    else:
        # Neither polygon_rest nor polygon SDK is enabled
        # Check legacy primary_source configuration for backward compatibility
        primary = (config.primary_source or "").strip().lower()
        if primary in ("polygon", "massive"):
            api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY") or ""
            if api_key:
                polygon_source = PolygonEventSource(api_key=api_key)
                polygon_result = polygon_source.fetch_events(symbols, start_ns, end_ns)
                logger.info("event_guard.using_polygon_legacy_config")
            else:
                polygon_result = Result.failed(
                    ValueError("POLYGON_API_KEY not found in environment"),
                    "MISSING_API_KEY",
                )
        else:
            polygon_result = Result.failed(
                ValueError("No Polygon sources enabled in configuration"),
                "NO_POLYGON_SOURCE",
            )

    if manual_source is not None:
        manual_result = manual_source.fetch_events(symbols, start_ns, end_ns)

    # Primary succeeded: optionally overlay manual.
    if polygon_result.status == ResultStatus.SUCCESS and polygon_result.data is not None:
        if manual_result is None:
            return polygon_result

        if manual_result.status == ResultStatus.SUCCESS and manual_result.data is not None:
            merged_events = _merge_events(polygon_result.data.events, manual_result.data.events)
            snapshot = EventSnapshot(
                **_now_snapshot_base_kwargs(),
                events=merged_events,
                source="merged",
                symbols_covered=polygon_result.data.symbols_covered,
            )
            return Result.success(snapshot, reason_code="OK")

        # Could not apply manual overrides; keep Polygon data but mark degraded.
        if manual_result.error is not None:
            logger.warning(
                "event_guard.manual_overrides_failed",
                path=manual_path,
                reason_code=manual_result.reason_code,
                error=str(manual_result.error)[:200],
            )
            return Result.degraded(polygon_result.data, manual_result.error, reason_code="MANUAL_OVERRIDE_FAILED")

        return polygon_result

    # Primary failed: try YFinance (Tier 3), then manual-only (Tier 4).
    primary_error = polygon_result.error or RuntimeError("Polygon events unavailable.")

    if yfinance_enabled:
        if not HAS_YFINANCE:
            logger.info(
                "event_guard.yfinance_not_available",
                reason="yfinance_not_installed",
            )
        else:
            yfinance_source = YFinanceEventSource(
                max_workers=yfinance_max_workers,
                request_delay_seconds=yfinance_request_delay_seconds,
            )
            yfinance_result = yfinance_source.fetch_events(symbols, start_ns, end_ns)

    if yfinance_result is not None and yfinance_result.status == ResultStatus.SUCCESS and yfinance_result.data is not None:
        logger.warning(
            "event_guard.polygon_failed_using_yfinance",
            polygon_reason_code=polygon_result.reason_code,
            polygon_error=str(primary_error)[:200],
            yfinance_max_workers=yfinance_max_workers,
            yfinance_request_delay_seconds=yfinance_request_delay_seconds,
        )

        if manual_result is None:
            return Result.degraded(yfinance_result.data, primary_error, reason_code="FALLBACK_YFINANCE")

        if manual_result.status == ResultStatus.SUCCESS and manual_result.data is not None:
            merged_events = _merge_events(yfinance_result.data.events, manual_result.data.events)
            snapshot = EventSnapshot(
                **_now_snapshot_base_kwargs(),
                events=merged_events,
                source="merged",
                symbols_covered=yfinance_result.data.symbols_covered,
            )
            return Result.degraded(snapshot, primary_error, reason_code="FALLBACK_YFINANCE")

        if manual_result.error is not None:
            logger.warning(
                "event_guard.manual_overrides_failed",
                path=manual_path,
                reason_code=manual_result.reason_code,
                error=str(manual_result.error)[:200],
            )
            return Result.degraded(yfinance_result.data, manual_result.error, reason_code="MANUAL_OVERRIDE_FAILED")

        return Result.degraded(yfinance_result.data, primary_error, reason_code="FALLBACK_YFINANCE")

    # API sources failed/unavailable: try manual-only.
    if manual_result is not None and manual_result.status == ResultStatus.SUCCESS and manual_result.data is not None:
        last_error = primary_error
        if yfinance_enabled and yfinance_result is not None and yfinance_result.error is not None:
            last_error = yfinance_result.error

        logger.warning(
            "event_guard.api_failed_using_manual",
            polygon_reason_code=polygon_result.reason_code,
            polygon_error=str(primary_error)[:200],
            yfinance_enabled=yfinance_enabled,
            yfinance_reason_code=yfinance_result.reason_code if yfinance_result else None,
            yfinance_error=str(yfinance_result.error)[:200] if yfinance_result and yfinance_result.error else None,
        )
        return Result.degraded(manual_result.data, last_error, reason_code="FALLBACK_MANUAL")

    # Both unavailable -> conservative degraded empty.
    err = (
        polygon_result.error
        or (yfinance_result.error if yfinance_result else None)
        or (manual_result.error if manual_result else None)
        or RuntimeError("Event sources unavailable.")
    )
    logger.warning(
        "event_guard.events_degraded_empty",
        polygon_reason_code=polygon_result.reason_code,
        yfinance_reason_code=yfinance_result.reason_code if yfinance_result else None,
        manual_reason_code=manual_result.reason_code if manual_result else None,
        error=str(err)[:200],
    )
    snapshot = EventSnapshot(
        **_now_snapshot_base_kwargs(),
        events=[],
        source="degraded",
        symbols_covered=[s for s in (_normalize_symbol(sym) for sym in symbols) if s],
    )
    return Result.degraded(snapshot, err, reason_code="DEGRADED_EMPTY")
