"""Polygon.io data source adapter for US equity OHLCV series."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
import math
from datetime import UTC, date, datetime, timedelta
from typing import Any

import msgspec
import requests
import structlog

from common.interface import Result
from journal.run_id import RunIDGenerator

from .interface import PriceBar, PriceSeriesSnapshot, normalize_date
from .rate_limiter import TokenBucketRateLimiter

__all__ = ["PolygonDataAdapter"]


class PolygonAPIError(RuntimeError):
    """Raised for Polygon API responses that indicate failure."""


@dataclass(frozen=True, slots=True)
class _PolygonAgg:
    t: int
    o: float
    h: float
    low: float
    c: float
    v: float | int


class PolygonDataAdapter:
    """Polygon API adapter implementing the ``DataSourceAdapter`` protocol."""

    _BASE_URL = "https://api.polygon.io"
    _SOURCE_NAME = "polygon"
    _SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
        calls_per_minute: int = 5,
        requests_timeout_seconds: float = 15.0,
    ) -> None:
        """Create a Polygon adapter.

        Args:
            api_key: Polygon API key. When omitted, the adapter reads
                ``POLYGON_API_KEY`` from the environment.
            rate_limiter: Shared limiter instance for rate limiting Polygon requests.
            calls_per_minute: Rate limiter refill rate used when `rate_limiter` is not
                provided. Default is tuned for the free tier (5). For unlimited API
                keys, a value like 1000 is typically safe.
            requests_timeout_seconds: Per-request timeout.
        """

        self._api_key = api_key or os.environ.get("POLYGON_API_KEY")
        self._rate_limiter = rate_limiter or TokenBucketRateLimiter(calls_per_minute=float(calls_per_minute))
        self._timeout = float(requests_timeout_seconds)
        self._logger = structlog.get_logger(__name__).bind(source=self._SOURCE_NAME)
        self._decoder = msgspec.json.Decoder(type=dict[str, Any])

    def get_source_name(self) -> str:
        """Return the adapter source identifier (``\"polygon\"``)."""

        return self._SOURCE_NAME

    def fetch_price_series(
        self,
        symbol: str,
        start_date: date | datetime | str,
        end_date: date | datetime | str,
    ) -> Result[PriceSeriesSnapshot]:
        """Fetch daily OHLCV bars from Polygon and normalize them to a snapshot."""

        if not self._api_key:
            return Result.failed(RuntimeError("POLYGON_API_KEY is not set."), reason_code="MISSING_API_KEY")

        start = normalize_date(start_date)
        end = normalize_date(end_date)
        if end < start:
            return Result.failed(ValueError("end_date must be >= start_date."), reason_code="INVALID_RANGE")

        acquired = self._rate_limiter.acquire(timeout_seconds=65.0)
        if not acquired:
            return Result.failed(TimeoutError("Polygon rate limiter timed out."), reason_code="RATE_LIMIT")

        from_str = start.isoformat()
        to_str = end.isoformat()
        url = f"{self._BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{from_str}/{to_str}"
        params = {"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": self._api_key}

        request_started_ns = time.time_ns()
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
        except requests.RequestException as exc:
            self._logger.warning("polygon.request_failed", symbol=symbol, error=str(exc))
            return Result.failed(exc, reason_code="NETWORK_ERROR")

        latency_ms = (time.time_ns() - request_started_ns) / 1_000_000
        if response.status_code != 200:
            error = PolygonAPIError(f"Polygon HTTP {response.status_code}: {response.text[:200]}")
            self._logger.warning(
                "polygon.http_error",
                symbol=symbol,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
            return Result.failed(error, reason_code="HTTP_ERROR")

        try:
            payload = self._decoder.decode(response.content)
        except Exception as exc:  # noqa: BLE001
            return Result.failed(exc, reason_code="DECODE_ERROR")

        status = payload.get("status")
        if status not in ("OK", "DELAYED"):
            message = payload.get("error") or payload.get("message") or "Unknown Polygon error"
            error = PolygonAPIError(str(message))
            self._logger.warning("polygon.api_error", symbol=symbol, status=status, message=str(message))
            return Result.failed(error, reason_code="API_ERROR")

        results = payload.get("results") or []
        if not isinstance(results, list):
            return Result.failed(PolygonAPIError("Unexpected results type."), reason_code="API_ERROR")

        bars: list[PriceBar] = []
        completeness_ok = True
        for item in results:
            if not isinstance(item, Mapping):
                completeness_ok = False
                continue
            try:
                agg = _PolygonAgg(
                    t=int(item["t"]),
                    o=float(item["o"]),
                    h=float(item["h"]),
                    low=float(item["l"]),
                    c=float(item["c"]),
                    v=float(item.get("v", 0)),
                )
            except Exception:  # noqa: BLE001
                completeness_ok = False
                continue
            bars.append(
                PriceBar(
                    timestamp=agg.t * 1_000_000,
                    open=agg.o,
                    high=agg.h,
                    low=agg.low,
                    close=agg.c,
                    volume=int(agg.v),
                )
            )

        availability_ok = len(bars) > 0
        # --- Gapfill stage (use prev endpoint) ---
        gapfill_triggered = False
        prev_bar_used = False
        prev_metadata: dict[str, Any] = {
            "prev_status": "NOT_TRIGGERED",
            "prev_latency_ms": 0.0,
            "prev_error": None,
        }

        days_from_now = (date.today() - end).days
        if not bars and days_from_now <= 7:
            gapfill_triggered = True

        if bars:
            _freshness_ok, _ = _assess_freshness(bars, end)
            if not _freshness_ok:
                gapfill_triggered = True

        if gapfill_triggered:
            prev_bar, prev_metadata = self._call_prev_endpoint(symbol)
            if prev_bar:
                prev_date = datetime.fromtimestamp(prev_bar.timestamp / 1_000_000_000, tz=UTC).date()
                in_window = start <= prev_date <= end
                is_duplicate = any(b.timestamp == prev_bar.timestamp for b in bars)
                if in_window and not is_duplicate:
                    bars.append(prev_bar)
                    bars.sort(key=lambda b: b.timestamp)
                    prev_bar_used = True
                    self._logger.info(
                        "polygon.prev_gapfill_used",
                        symbol=symbol,
                        prev_date=prev_date.isoformat(),
                        prev_bar_count=1,
                    )

        # --- Mid-window gapfill (use open-close endpoint) ---
        midgap_filled_count = 0
        midgap_metadata: dict[str, Any] = {
            "midgap_fill_attempted": False,
            "midgap_fill_count": 0,
            "midgap_fill_failed": 0,
            "open_close_status": "NOT_TRIGGERED",
            "open_close_latency_ms": 0.0,
            "open_close_error": None,
        }

        if bars and len(bars) >= 2:
            missing_dates = self._detect_missing_dates(bars, start, end, max_gap_days=5)
            if missing_dates:
                midgap_metadata["midgap_fill_attempted"] = True
                self._logger.info(
                    "polygon.midgap_detected",
                    symbol=symbol,
                    missing_count=len(missing_dates),
                    missing_dates=[d.isoformat() for d in missing_dates[:5]],
                )

                for missing_date in missing_dates:
                    open_close_bar, oc_metadata = self._call_open_close_endpoint(symbol, missing_date)
                    midgap_metadata.update(oc_metadata)
                    if open_close_bar:
                        is_duplicate = any(b.timestamp == open_close_bar.timestamp for b in bars)
                        if not is_duplicate:
                            bars.append(open_close_bar)
                            midgap_filled_count += 1
                            self._logger.debug(
                                "polygon.midgap_filled",
                                symbol=symbol,
                                date=missing_date.isoformat(),
                            )
                    else:
                        midgap_metadata["midgap_fill_failed"] += 1
                        self._logger.debug(
                            "polygon.midgap_fill_failed",
                            symbol=symbol,
                            date=missing_date.isoformat(),
                            status=oc_metadata.get("open_close_status"),
                            error=oc_metadata.get("open_close_error"),
                        )

                if midgap_filled_count > 0:
                    bars.sort(key=lambda b: b.timestamp)

                midgap_metadata["midgap_fill_count"] = midgap_filled_count

        def _assess_completeness(series: list[PriceBar]) -> bool:
            for bar in series:
                if not isinstance(bar.timestamp, int) or bar.timestamp <= 0:
                    return False
                if not isinstance(bar.volume, int) or bar.volume < 0:
                    return False
                floats = (bar.open, bar.high, bar.low, bar.close)
                if any(not isinstance(value, float) or not math.isfinite(value) for value in floats):
                    return False
                if bar.high < bar.low:
                    return False
            return True

        # Re-evaluate after gapfill stages
        availability_ok = len(bars) > 0
        completeness_ok = completeness_ok and _assess_completeness(bars)
        freshness_ok, freshness_details = _assess_freshness(bars, end)

        quality_flags: dict[str, Any] = {
            "availability": availability_ok,
            "completeness": completeness_ok,
            "freshness": freshness_ok,
            "stability": True,
            **freshness_details,
            "polygon_status": status,
            "polygon_latency_ms": latency_ms,
            "gapfill_prev_used": prev_bar_used,
            **prev_metadata,
            **midgap_metadata,
        }

        snapshot = PriceSeriesSnapshot(
            schema_version=self._SCHEMA_VERSION,
            system_version=RunIDGenerator.get_system_version(),
            asof_timestamp=time.time_ns(),
            symbol=symbol.upper(),
            timeframe="1D",
            bars=bars,
            source=self._SOURCE_NAME,
            quality_flags=quality_flags,
        )

        if not availability_ok:
            return Result.failed(PolygonAPIError("No bars returned."), reason_code="NO_DATA")

        if not (completeness_ok and freshness_ok):
            if midgap_filled_count > 0 and prev_bar_used:
                reason = "QUALITY_DEGRADED_PREV_MIDGAP_USED"
            elif midgap_filled_count > 0:
                reason = "QUALITY_DEGRADED_MIDGAP_USED"
            elif prev_bar_used:
                reason = "QUALITY_DEGRADED_PREV_USED"
            else:
                reason = "QUALITY_DEGRADED"
            return Result.degraded(snapshot, RuntimeError("Polygon data quality degraded."), reason)

        return Result.success(snapshot, reason_code="OK")

    def _call_prev_endpoint(
        self,
        symbol: str,
    ) -> tuple[PriceBar | None, dict[str, Any]]:
        """Call GET /v2/aggs/ticker/{ticker}/prev to fetch the most recent completed trading day bar.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (PriceBar | None, metadata dict)
            - PriceBar: The prev bar if successful
            - metadata: {"prev_status": str, "prev_latency_ms": float, "prev_error": str | None}
        """

        url = f"{self._BASE_URL}/v2/aggs/ticker/{symbol}/prev"
        params = {"adjusted": "true", "apiKey": self._api_key}

        acquired = self._rate_limiter.acquire(timeout_seconds=65.0)
        if not acquired:
            return None, {
                "prev_status": "RATE_LIMIT",
                "prev_latency_ms": 0.0,
                "prev_error": "Polygon rate limiter timed out.",
            }

        request_started_ns = time.time_ns()
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
        except requests.RequestException as exc:
            latency_ms = (time.time_ns() - request_started_ns) / 1_000_000
            self._logger.warning("polygon.prev_request_failed", symbol=symbol, error=str(exc))
            return None, {
                "prev_status": "NETWORK_ERROR",
                "prev_latency_ms": latency_ms,
                "prev_error": str(exc),
            }

        latency_ms = (time.time_ns() - request_started_ns) / 1_000_000
        if response.status_code != 200:
            error = f"Polygon HTTP {response.status_code}: {response.text[:200]}"
            self._logger.warning(
                "polygon.prev_http_error",
                symbol=symbol,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
            return None, {
                "prev_status": "HTTP_ERROR",
                "prev_latency_ms": latency_ms,
                "prev_error": error,
            }

        try:
            payload = self._decoder.decode(response.content)
        except Exception as exc:  # noqa: BLE001
            return None, {
                "prev_status": "DECODE_ERROR",
                "prev_latency_ms": latency_ms,
                "prev_error": str(exc),
            }

        status = payload.get("status")
        if status not in ("OK", "DELAYED"):
            message = payload.get("error") or payload.get("message") or "Unknown Polygon error"
            error = str(message)
            self._logger.warning("polygon.prev_api_error", symbol=symbol, status=status, message=error)
            return None, {
                "prev_status": str(status),
                "prev_latency_ms": latency_ms,
                "prev_error": error,
            }

        raw_results = payload.get("results")
        item: Any | None = None
        if isinstance(raw_results, list) and raw_results:
            item = raw_results[0]
        elif isinstance(raw_results, Mapping):
            item = raw_results

        if not isinstance(item, Mapping):
            return None, {
                "prev_status": "NO_RESULTS",
                "prev_latency_ms": latency_ms,
                "prev_error": "No results returned.",
            }

        try:
            agg = _PolygonAgg(
                t=int(item["t"]),
                o=float(item["o"]),
                h=float(item["h"]),
                low=float(item["l"]),
                c=float(item["c"]),
                v=float(item.get("v", 0)),
            )
        except Exception as exc:  # noqa: BLE001
            return None, {
                "prev_status": "PARSE_ERROR",
                "prev_latency_ms": latency_ms,
                "prev_error": str(exc),
            }

        bar = PriceBar(
            timestamp=agg.t * 1_000_000,
            open=agg.o,
            high=agg.h,
            low=agg.low,
            close=agg.c,
            volume=int(agg.v),
        )
        return bar, {
            "prev_status": str(status),
            "prev_latency_ms": latency_ms,
            "prev_error": None,
        }

    def _call_open_close_endpoint(
        self,
        symbol: str,
        date: date,
    ) -> tuple[PriceBar | None, dict[str, Any]]:
        """Call GET /v1/open-close/{ticker}/{date} to fetch a specific day's OHLCV bar.

        Args:
            symbol: Stock symbol
            date: Specific date to fetch (date object)

        Returns:
            Tuple of (PriceBar | None, metadata dict)
            - PriceBar: The bar for the specified date if successful
            - metadata: {"open_close_status": str, "open_close_latency_ms": float, "open_close_error": str | None}
        """

        url = f"{self._BASE_URL}/v1/open-close/{symbol}/{date.isoformat()}"
        params = {"adjusted": "true", "apiKey": self._api_key}

        acquired = self._rate_limiter.acquire(timeout_seconds=65.0)
        if not acquired:
            return None, {
                "open_close_status": "RATE_LIMIT",
                "open_close_latency_ms": 0.0,
                "open_close_error": "Polygon rate limiter timed out.",
            }

        request_started_ns = time.time_ns()
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
        except requests.RequestException as exc:
            latency_ms = (time.time_ns() - request_started_ns) / 1_000_000
            self._logger.warning("polygon.open_close_request_failed", symbol=symbol, date=date.isoformat(), error=str(exc))
            return None, {
                "open_close_status": "NETWORK_ERROR",
                "open_close_latency_ms": latency_ms,
                "open_close_error": str(exc),
            }

        latency_ms = (time.time_ns() - request_started_ns) / 1_000_000
        if response.status_code == 404:
            return None, {
                "open_close_status": "NOT_FOUND",
                "open_close_latency_ms": latency_ms,
                "open_close_error": "Not found.",
            }

        if response.status_code != 200:
            error = f"Polygon HTTP {response.status_code}: {response.text[:200]}"
            self._logger.warning(
                "polygon.open_close_http_error",
                symbol=symbol,
                date=date.isoformat(),
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
            return None, {
                "open_close_status": "HTTP_ERROR",
                "open_close_latency_ms": latency_ms,
                "open_close_error": error,
            }

        try:
            payload = self._decoder.decode(response.content)
        except Exception as exc:  # noqa: BLE001
            return None, {
                "open_close_status": "DECODE_ERROR",
                "open_close_latency_ms": latency_ms,
                "open_close_error": str(exc),
            }

        status = payload.get("status")
        if status not in ("OK", "DELAYED"):
            message = payload.get("error") or payload.get("message") or "Unknown Polygon error"
            error = str(message)
            self._logger.debug(
                "polygon.open_close_api_error",
                symbol=symbol,
                date=date.isoformat(),
                status=status,
                message=error,
            )
            return None, {
                "open_close_status": str(status),
                "open_close_latency_ms": latency_ms,
                "open_close_error": error,
            }

        try:
            open_px = float(payload["open"])
            high_px = float(payload["high"])
            low_px = float(payload["low"])
            close_px = float(payload["close"])
            volume = int(payload.get("volume", 0))
        except Exception as exc:  # noqa: BLE001
            return None, {
                "open_close_status": "PARSE_ERROR",
                "open_close_latency_ms": latency_ms,
                "open_close_error": str(exc),
            }

        timestamp_ns = int(datetime(date.year, date.month, date.day, tzinfo=UTC).timestamp() * 1_000_000_000)
        bar = PriceBar(
            timestamp=timestamp_ns,
            open=open_px,
            high=high_px,
            low=low_px,
            close=close_px,
            volume=volume,
        )
        return bar, {
            "open_close_status": str(status),
            "open_close_latency_ms": latency_ms,
            "open_close_error": None,
        }

    @staticmethod
    def _detect_missing_dates(
        bars: list[PriceBar],
        start_date: date,
        end_date: date,
        max_gap_days: int = 5,
    ) -> list[date]:
        """Detect missing trading days in the middle of the bar series.

        Args:
            bars: List of price bars (assumed sorted by timestamp)
            start_date: Expected start date of window
            end_date: Expected end date of window
            max_gap_days: Maximum gap (in days) to consider for filling (avoid filling long halts)

        Returns:
            List of missing dates that should be filled (excluding weekends)
        """

        if len(bars) < 2 or max_gap_days < 2:
            return []

        bar_dates: list[date] = [
            datetime.fromtimestamp(bar.timestamp / 1_000_000_000, tz=UTC).date() for bar in bars
        ]

        missing: set[date] = set()
        for prev_date, next_date in zip(bar_dates, bar_dates[1:], strict=False):
            gap_days = (next_date - prev_date).days
            if gap_days <= 1 or gap_days > max_gap_days:
                continue

            cursor = prev_date + timedelta(days=1)
            while cursor < next_date:
                if start_date <= cursor <= end_date and cursor.weekday() < 5:
                    missing.add(cursor)
                cursor += timedelta(days=1)

        return sorted(missing)


def _assess_freshness(bars: list[PriceBar], end_date: date) -> tuple[bool, dict[str, Any]]:
    if not bars:
        return False, {"freshness_age_hours": None}

    last_ts_ns = bars[-1].timestamp
    last_dt = datetime.fromtimestamp(last_ts_ns / 1_000_000_000, tz=UTC)
    last_date = last_dt.date()

    diff_days = (end_date - last_date).days
    freshness_ok = diff_days <= 3
    age_hours = max(0.0, (datetime.now(tz=UTC) - last_dt).total_seconds() / 3600.0)
    return freshness_ok, {"freshness_days_diff": diff_days, "freshness_age_hours": age_hours}
