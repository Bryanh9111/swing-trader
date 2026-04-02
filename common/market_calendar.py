"""Market calendar service for holiday detection and market status.

This module integrates with Polygon's market status endpoints to determine:

- Upcoming market holidays and early-close sessions.
- Current market state (open/closed/extended-hours).
- Whether trading actions (order execution) should be enabled today.
"""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Optional, cast

try:
    import pandas_market_calendars as mcal
except ModuleNotFoundError:  # pragma: no cover
    mcal = None  # type: ignore[assignment]
import requests
import structlog
from msgspec import Struct

from .interface import Result, ResultStatus

__all__ = ["MarketCalendar", "MarketHoliday", "MarketStatus"]


class MarketHoliday(Struct, frozen=True, kw_only=True):
    """Market holiday information."""

    date: str  # ISO date string "YYYY-MM-DD"
    exchange: str  # NYSE, NASDAQ, OTC
    name: str
    status: str  # "closed" or "early-close"
    open_time: Optional[str] = None  # ISO datetime string (early-close only)
    close_time: Optional[str] = None  # ISO datetime string (early-close only)


class MarketStatus(Struct, frozen=True, kw_only=True):
    """Current market status."""

    is_market_open: bool
    is_holiday: bool
    is_early_close: bool
    current_status: str  # "open", "closed", "extended-hours"
    server_time: str  # ISO datetime string
    nyse_status: str
    nasdaq_status: str


def _today_utc() -> date:
    return datetime.now(tz=UTC).date()


def _parse_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    with suppress(ValueError):
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    return None


@dataclass(slots=True)
class MarketCalendar:
    """Market calendar service for holiday detection and market status."""

    api_key: str
    cache_ttl_hours: int = 24
    requests_timeout_seconds: int = 30

    _BASE_URL: str = "https://api.polygon.io"
    _logger: Any = field(init=False, repr=False)
    _cache: dict[str, Any] = field(init=False, repr=False, default_factory=dict)
    _cache_timestamp: int = field(init=False, repr=False, default=0)
    _nyse: Any = field(init=False, repr=False)
    _trading_days_cache: dict[int, set[date]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_logger", structlog.get_logger(__name__).bind(component="market_calendar")
        )
        if mcal is None:  # pragma: no cover
            self._logger.error("market_calendar.missing_pandas_market_calendars")
            object.__setattr__(self, "_nyse", None)
        else:
            object.__setattr__(self, "_nyse", cast(Any, mcal).get_calendar("NYSE"))

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid (within TTL)."""
        if not self._cache_timestamp:
            return False

        now_ns = time.time_ns()
        age_hours = (now_ns - self._cache_timestamp) / (3600 * 1_000_000_000)
        return age_hours < float(self.cache_ttl_hours)

    def _update_cache(self, holidays: list[MarketHoliday]) -> None:
        """Update cache with new holiday data."""
        self._cache["holidays"] = holidays
        self._cache_timestamp = time.time_ns()

    def _fetch_holidays_http(self) -> list[dict[str, Any]]:
        """Fetch holidays via HTTP."""
        url = f"{self._BASE_URL}/v1/marketstatus/upcoming"
        params = {"apiKey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=float(self.requests_timeout_seconds))
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            self._logger.error("market_calendar.fetch_holidays_failed", error=str(exc)[:200])
            raise

        if not isinstance(payload, list):
            raise ValueError("Unexpected Polygon holidays payload type.")
        return payload  # type: ignore[return-value]

    def _fetch_status_http(self) -> dict[str, Any]:
        """Fetch market status via HTTP."""
        url = f"{self._BASE_URL}/v1/marketstatus/now"
        params = {"apiKey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=float(self.requests_timeout_seconds))
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            self._logger.error("market_calendar.fetch_status_failed", error=str(exc)[:200])
            raise

        if not isinstance(payload, dict):
            raise ValueError("Unexpected Polygon market status payload type.")
        return payload  # type: ignore[return-value]

    def _parse_holidays_payload(self, payload: list[dict[str, Any]]) -> list[MarketHoliday]:
        holidays: list[MarketHoliday] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            date_str = str(item.get("date", "")).strip()
            exchange = str(item.get("exchange", "")).strip()
            name = str(item.get("name", "")).strip()
            status = str(item.get("status", "")).strip()
            if not (date_str and exchange and name and status):
                continue
            open_time = item.get("open")
            close_time = item.get("close")
            holidays.append(
                MarketHoliday(
                    date=date_str,
                    exchange=exchange,
                    name=name,
                    status=status,
                    open_time=str(open_time).strip() if open_time else None,
                    close_time=str(close_time).strip() if close_time else None,
                )
            )
        return holidays

    def fetch_upcoming_holidays(self, days_ahead: int = 90) -> Result[list[MarketHoliday]]:
        """Fetch upcoming market holidays from Polygon API.

        Args:
            days_ahead: Number of days ahead to fetch (default 90)

        Returns:
            Result with list of MarketHoliday objects
        """
        if not self.api_key:
            return Result.failed(RuntimeError("Polygon api_key is required."), reason_code="MISSING_API_KEY")

        cached = self._cache.get("holidays")
        if self._is_cache_valid() and isinstance(cached, list):
            return Result.success(self._filter_days_ahead(cached, days_ahead), reason_code="CACHE_HIT")

        try:
            raw = self._fetch_holidays_http()
            holidays = self._parse_holidays_payload(raw)
        except requests.RequestException as exc:
            return Result.degraded([], exc, reason_code="HOLIDAYS_API_FAILED")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("market_calendar.parse_holidays_failed", error=str(exc)[:200])
            return Result.degraded([], exc, reason_code="HOLIDAYS_PARSE_FAILED")

        self._update_cache(holidays)
        return Result.success(self._filter_days_ahead(holidays, days_ahead), reason_code="OK")

    def _filter_days_ahead(self, holidays: list[MarketHoliday], days_ahead: int) -> list[MarketHoliday]:
        if days_ahead <= 0:
            return []
        start = _today_utc()
        end = start + timedelta(days=int(days_ahead))
        filtered: list[MarketHoliday] = []
        for holiday in holidays:
            parsed = _parse_iso_date(holiday.date)
            if parsed is None:
                continue
            if start <= parsed <= end:
                filtered.append(holiday)
        return filtered

    def _holiday_flags_for_date(self, target: date, holidays: list[MarketHoliday]) -> tuple[bool, bool]:
        target_str = target.isoformat()
        is_holiday = any(h.date == target_str and h.status == "closed" for h in holidays)
        is_early_close = any(h.date == target_str and h.status == "early-close" for h in holidays)
        return is_holiday, is_early_close

    def get_current_status(self) -> Result[MarketStatus]:
        """Get current market status from Polygon API.

        Returns:
            Result with MarketStatus object
        """
        if not self.api_key:
            return Result.failed(RuntimeError("Polygon api_key is required."), reason_code="MISSING_API_KEY")

        try:
            payload = self._fetch_status_http()
        except requests.RequestException as exc:
            return Result.failed(exc, reason_code="STATUS_API_FAILED")
        except Exception as exc:  # noqa: BLE001
            return Result.failed(exc, reason_code="STATUS_PARSE_FAILED")

        market_value = str(payload.get("market", "unknown")).strip() or "unknown"
        exchanges = payload.get("exchanges") or {}
        if not isinstance(exchanges, dict):
            exchanges = {}
        nyse_status = str(exchanges.get("nyse", "unknown")).strip() or "unknown"
        nasdaq_status = str(exchanges.get("nasdaq", "unknown")).strip() or "unknown"

        current_status = market_value
        if current_status not in {"open", "closed", "extended-hours"}:
            current_status = "closed" if market_value == "close" else current_status

        is_market_open = current_status in {"open", "extended-hours"}
        server_time = str(payload.get("serverTime") or datetime.now(tz=UTC).isoformat())

        holiday_result = self.fetch_upcoming_holidays()
        holidays = holiday_result.data or []
        today = _today_utc()
        is_holiday, is_early_close = self._holiday_flags_for_date(today, holidays)

        status_obj = MarketStatus(
            is_market_open=is_market_open,
            is_holiday=is_holiday,
            is_early_close=is_early_close,
            current_status=current_status,
            server_time=server_time,
            nyse_status=nyse_status,
            nasdaq_status=nasdaq_status,
        )

        if holiday_result.status is not ResultStatus.SUCCESS:
            err = holiday_result.error or RuntimeError("Holiday lookup degraded.")
            return Result.degraded(status_obj, err, reason_code="HOLIDAY_LOOKUP_DEGRADED")

        return Result.success(status_obj, reason_code="OK")

    def is_trading_day(self, target_date: date | None = None) -> Result[bool]:
        """Check if target date is a trading day (not a holiday).

        Args:
            target_date: Date to check (default: today)

        Returns:
            Result[bool]: True if trading day, False if holiday
        """
        target = target_date or _today_utc()
        year = target.year

        if self._nyse is None:  # pragma: no cover
            err = ModuleNotFoundError("pandas_market_calendars is required for is_trading_day()")
            return Result.degraded(True, err, reason_code="CALENDAR_LIB_MISSING")

        if year not in self._trading_days_cache:
            try:
                schedule = self._nyse.schedule(
                    start_date=f"{year}-01-01",
                    end_date=f"{year}-12-31",
                )
                self._trading_days_cache[year] = {d.date() for d in schedule.index}
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "market_calendar.load_trading_days_failed",
                    year=year,
                    error=str(exc)[:200],
                )
                return Result.degraded(True, exc, reason_code="CALENDAR_LOAD_FAILED")

        is_trading = target in self._trading_days_cache[year]
        return Result.success(is_trading, reason_code="OK")

    def should_execute_orders(self) -> Result[bool]:
        """Check if orders should be executed today.

        Returns:
            Result[bool]: False on holidays, True on trading days
        """
        status_result = self.get_current_status()
        status_obj = status_result.data

        if status_obj is None:
            self._logger.warning(
                "market_calendar.api_failed_assuming_trading_day",
                error=str(status_result.error)[:200] if status_result.error else None,
            )
            return Result.success(True, reason_code="API_FAILED_CONSERVATIVE")

        should_execute = (not status_obj.is_holiday) and status_obj.current_status != "closed"

        if status_result.status is ResultStatus.DEGRADED:
            err = status_result.error or RuntimeError("Market status degraded.")
            return Result.degraded(should_execute, err, reason_code="STATUS_DEGRADED")

        if not should_execute:
            reason = "HOLIDAY" if status_obj.is_holiday else "MARKET_CLOSED"
            return Result.success(False, reason_code=reason)
        return Result.success(True, reason_code="OK")
