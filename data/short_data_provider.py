"""Short interest and short volume data provider with multi-tier fallbacks.

Tier strategy:
    1) Polygon API (when available)
    2) FINRA public data
    3) Graceful degradation (return None)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import msgspec
import requests
import structlog

__all__ = [
    "ShortInterestData",
    "ShortVolumeData",
    "ShortDataProvider",
]


@dataclass(frozen=True, slots=True)
class ShortInterestData:
    ticker: str
    short_interest: int
    shares_outstanding: int
    short_percent: float
    short_ratio: float
    settlement_date: str


@dataclass(frozen=True, slots=True)
class ShortVolumeData:
    ticker: str
    date: str
    short_volume: int
    total_volume: int
    short_volume_ratio: float


class ShortDataProvider:
    _POLYGON_BASE_URL = "https://api.polygon.io"
    _FINRA_REGSHO_BASE_URL = "https://cdn.finra.org/equity/regsho/daily"

    def __init__(
        self,
        *,
        polygon_api_key: str | None = None,
        requests_timeout_seconds: float = 15.0,
    ) -> None:
        self._polygon_api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY")
        self._timeout = float(requests_timeout_seconds)
        self._decoder = msgspec.json.Decoder(type=dict[str, Any])
        self._logger = structlog.get_logger(__name__)
        self._interest_cache: dict[str, ShortInterestData | None] = {}
        self._volume_cache: dict[tuple[str, int], list[ShortVolumeData] | None] = {}

    def fetch_short_interest(self, ticker: str) -> ShortInterestData | None:
        symbol = str(ticker).upper().strip()
        if not symbol:
            return None

        if symbol in self._interest_cache:
            return self._interest_cache[symbol]

        result = None
        try:
            result = self._fetch_short_interest_polygon(symbol)
        except Exception as exc:  # noqa: BLE001 - graceful fallback
            self._logger.info("short_interest.polygon_failed", ticker=symbol, error=type(exc).__name__)

        if result is None:
            try:
                result = self._fetch_short_interest_finra(symbol)
            except Exception as exc:  # noqa: BLE001 - graceful fallback
                self._logger.info("short_interest.finra_failed", ticker=symbol, error=type(exc).__name__)

        self._interest_cache[symbol] = result
        return result

    def fetch_short_volume(self, ticker: str, days: int = 30) -> list[ShortVolumeData] | None:
        symbol = str(ticker).upper().strip()
        days = int(days or 0)
        if not symbol or days <= 0:
            return None

        cache_key = (symbol, days)
        if cache_key in self._volume_cache:
            return self._volume_cache[cache_key]

        result = None
        try:
            result = self._fetch_short_volume_polygon(symbol, days=days)
        except Exception as exc:  # noqa: BLE001 - graceful fallback
            self._logger.info("short_volume.polygon_failed", ticker=symbol, error=type(exc).__name__)

        if result is None:
            try:
                result = self._fetch_short_volume_finra(symbol, days=days)
            except Exception as exc:  # noqa: BLE001 - graceful fallback
                self._logger.info("short_volume.finra_failed", ticker=symbol, error=type(exc).__name__)

        self._volume_cache[cache_key] = result
        return result

    def _fetch_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        request_started_ns = time.time_ns()
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        try:
            decoded = self._decoder.decode(response.content)
        except Exception:  # noqa: BLE001
            return None
        latency_ms = (time.time_ns() - request_started_ns) / 1_000_000
        if isinstance(decoded, dict):
            decoded.setdefault("_meta", {})  # internal
            if isinstance(decoded["_meta"], dict):
                decoded["_meta"]["latency_ms"] = latency_ms
        return decoded if isinstance(decoded, dict) else None

    def _fetch_short_interest_polygon(self, ticker: str) -> ShortInterestData | None:
        if not self._polygon_api_key:
            return None

        url = f"{self._POLYGON_BASE_URL}/v3/reference/short-interest"
        payload = self._fetch_json(url, params={"ticker": ticker, "apiKey": self._polygon_api_key, "limit": 1, "sort": "desc"})
        if not payload:
            return None

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None

        first = results[0]
        if not isinstance(first, dict):
            return None

        short_interest_raw = (
            first.get("short_interest")
            or first.get("shortInterest")
            or first.get("short_interest_shares")
            or first.get("shortInterestShares")
        )
        shares_outstanding_raw = (
            first.get("shares_outstanding")
            or first.get("sharesOutstanding")
            or first.get("float")
            or first.get("floatShares")
        )
        settlement_date = str(first.get("settlement_date") or first.get("settlementDate") or first.get("date") or "")
        if not settlement_date:
            settlement_date = datetime.now(tz=UTC).date().isoformat()

        try:
            short_interest = int(short_interest_raw)
        except Exception:  # noqa: BLE001
            return None

        shares_outstanding = 0
        try:
            if shares_outstanding_raw is not None:
                shares_outstanding = int(shares_outstanding_raw)
        except Exception:  # noqa: BLE001
            shares_outstanding = 0

        short_percent = float(short_interest) / float(shares_outstanding) if shares_outstanding > 0 else 0.0

        avg_daily_volume = self._fetch_avg_daily_volume_polygon(ticker, days=30)
        short_ratio = float(short_interest) / float(avg_daily_volume) if avg_daily_volume > 0 else 0.0

        return ShortInterestData(
            ticker=ticker,
            short_interest=short_interest,
            shares_outstanding=shares_outstanding,
            short_percent=float(short_percent),
            short_ratio=float(short_ratio),
            settlement_date=settlement_date,
        )

    def _fetch_avg_daily_volume_polygon(self, ticker: str, *, days: int = 30) -> float:
        if not self._polygon_api_key:
            return 0.0

        today = datetime.now(tz=UTC).date()
        start = today - timedelta(days=max(5, int(days) * 2))
        url = f"{self._POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{today.isoformat()}"
        payload = self._fetch_json(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": self._polygon_api_key},
        )
        if not payload:
            return 0.0

        results = payload.get("results") or []
        if not isinstance(results, list) or not results:
            return 0.0

        volumes: list[float] = []
        for item in results[-int(days) :]:
            if not isinstance(item, dict):
                continue
            try:
                volumes.append(float(item.get("v", 0.0) or 0.0))
            except Exception:  # noqa: BLE001
                continue

        if not volumes:
            return 0.0
        return float(sum(volumes) / len(volumes))

    def _fetch_short_volume_polygon(self, ticker: str, *, days: int = 30) -> list[ShortVolumeData] | None:
        if not self._polygon_api_key:
            return None

        url = f"{self._POLYGON_BASE_URL}/v1/markets/stocks/short-volume"
        payload = self._fetch_json(
            url,
            params={"ticker": ticker, "apiKey": self._polygon_api_key, "limit": int(days), "sort": "desc"},
        )
        if not payload:
            return None

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None

        parsed: list[ShortVolumeData] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            dt = str(item.get("date") or item.get("settlement_date") or item.get("asof") or "")
            if not dt:
                continue
            short_vol_raw = item.get("short_volume") or item.get("shortVolume") or item.get("short")
            total_vol_raw = item.get("total_volume") or item.get("totalVolume") or item.get("volume")
            try:
                short_vol = int(short_vol_raw)
                total_vol = int(total_vol_raw)
            except Exception:  # noqa: BLE001
                continue
            ratio = float(short_vol) / float(total_vol) if total_vol > 0 else 0.0
            parsed.append(
                ShortVolumeData(
                    ticker=ticker,
                    date=dt,
                    short_volume=short_vol,
                    total_volume=total_vol,
                    short_volume_ratio=float(ratio),
                )
            )

        if not parsed:
            return None

        parsed.sort(key=lambda row: row.date)
        return parsed[-int(days) :]

    def _fetch_short_volume_finra(self, ticker: str, *, days: int = 30) -> list[ShortVolumeData] | None:
        end = datetime.now(tz=UTC).date()
        start = end - timedelta(days=max(5, int(days) * 2))
        results: list[ShortVolumeData] = []

        day = start
        while day <= end:
            if day.weekday() >= 5:
                day += timedelta(days=1)
                continue

            payload = self._fetch_finra_regsho_file(day)
            if payload is None:
                day += timedelta(days=1)
                continue

            record = self._parse_finra_regsho_for_symbol(payload, ticker)
            if record is not None:
                results.append(record)

            day += timedelta(days=1)

        if not results:
            return None

        results.sort(key=lambda row: row.date)
        return results[-int(days) :]

    def _fetch_finra_regsho_file(self, trading_day: date) -> str | None:
        filename = f"CNMSshvol{trading_day.strftime('%Y%m%d')}.txt"
        url = f"{self._FINRA_REGSHO_BASE_URL}/{filename}"
        try:
            response = requests.get(url, timeout=self._timeout)
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        try:
            return response.content.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return None

    def _parse_finra_regsho_for_symbol(self, text: str, ticker: str) -> ShortVolumeData | None:
        symbol = str(ticker).upper().strip()
        if not symbol:
            return None

        for line in text.splitlines():
            if not line or line.startswith("Date|"):
                continue
            parts = line.split("|")
            if len(parts) < 5:
                continue
            if parts[1].strip().upper() != symbol:
                continue
            dt = parts[0].strip()
            try:
                short_vol = int(parts[2])
                total_vol = int(parts[4])
            except Exception:  # noqa: BLE001
                return None
            ratio = float(short_vol) / float(total_vol) if total_vol > 0 else 0.0
            return ShortVolumeData(
                ticker=symbol,
                date=dt,
                short_volume=short_vol,
                total_volume=total_vol,
                short_volume_ratio=float(ratio),
            )

        return None

    def _fetch_short_interest_finra(self, ticker: str) -> ShortInterestData | None:
        url = "https://api.finra.org/data/group/otcMarket/name/shortInterest"
        payload = self._fetch_json(url, params={"symbol": ticker})
        if not payload:
            return None

        results = payload.get("results") or payload.get("data") or payload.get("rows")
        if not isinstance(results, list) or not results:
            return None

        first = results[0]
        if not isinstance(first, dict):
            return None

        short_interest_raw = first.get("shortInterestQuantity") or first.get("short_interest") or first.get("shortInterest")
        settlement_date = str(first.get("settlementDate") or first.get("settlement_date") or first.get("date") or "")
        shares_outstanding_raw = first.get("sharesOutstanding") or first.get("shares_outstanding") or first.get("float")
        try:
            short_interest = int(short_interest_raw)
        except Exception:  # noqa: BLE001
            return None

        shares_outstanding = 0
        try:
            if shares_outstanding_raw is not None:
                shares_outstanding = int(shares_outstanding_raw)
        except Exception:  # noqa: BLE001
            shares_outstanding = 0

        short_percent = float(short_interest) / float(shares_outstanding) if shares_outstanding > 0 else 0.0
        short_ratio = 0.0
        if not settlement_date:
            settlement_date = datetime.now(tz=UTC).date().isoformat()

        return ShortInterestData(
            ticker=ticker,
            short_interest=short_interest,
            shares_outstanding=shares_outstanding,
            short_percent=float(short_percent),
            short_ratio=float(short_ratio),
            settlement_date=settlement_date,
        )
