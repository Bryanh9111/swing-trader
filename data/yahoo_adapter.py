"""Yahoo Finance fallback adapter using ``yfinance``."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import structlog

from common.interface import Result
from journal.run_id import RunIDGenerator

from .interface import PriceBar, PriceSeriesSnapshot, normalize_date

__all__ = ["YahooDataAdapter"]


class YahooDataAdapter:
    """Yahoo Finance adapter implementing the ``DataSourceAdapter`` protocol."""

    _SOURCE_NAME = "yahoo"
    _SCHEMA_VERSION = "1.0.0"

    def __init__(self, *, requests_timeout_seconds: float = 20.0) -> None:
        """Create a Yahoo Finance adapter (no API key required)."""

        self._timeout = float(requests_timeout_seconds)
        self._logger = structlog.get_logger(__name__).bind(source=self._SOURCE_NAME)

    def get_source_name(self) -> str:
        """Return the adapter source identifier (``\"yahoo\"``)."""

        return self._SOURCE_NAME

    def fetch_price_series(
        self,
        symbol: str,
        start_date: date | datetime | str,
        end_date: date | datetime | str,
    ) -> Result[PriceSeriesSnapshot]:
        """Fetch daily OHLCV bars via ``yfinance`` and normalize them to a snapshot."""

        try:
            import yfinance as yf
        except Exception as exc:  # noqa: BLE001
            return Result.failed(exc, reason_code="MISSING_DEPENDENCY")

        start = normalize_date(start_date)
        end = normalize_date(end_date)
        if end < start:
            return Result.failed(ValueError("end_date must be >= start_date."), reason_code="INVALID_RANGE")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                interval="1d",
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=False,
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("yahoo.request_failed", symbol=symbol, error=str(exc))
            return Result.failed(exc, reason_code="NETWORK_ERROR")

        if df is None or len(df) == 0:
            return Result.failed(RuntimeError("No bars returned."), reason_code="NO_DATA")

        bars: list[PriceBar] = []
        completeness_ok = True

        ts_ns = None
        try:
            idx = df.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC")
            ts_ns = idx.view("int64")
        except Exception:  # noqa: BLE001
            ts_ns = None

        for i, row in enumerate(df.itertuples()):
            try:
                timestamp = int(ts_ns[i]) if ts_ns is not None else _timestamp_from_index(row.Index)
                open_v = float(row.Open)
                high_v = float(row.High)
                low_v = float(row.Low)
                close_v = float(row.Close)
                volume_v = int(getattr(row, "Volume", 0))
            except Exception:  # noqa: BLE001
                completeness_ok = False
                continue
            bars.append(
                PriceBar(
                    timestamp=timestamp,
                    open=open_v,
                    high=high_v,
                    low=low_v,
                    close=close_v,
                    volume=volume_v,
                )
            )

        availability_ok = len(bars) > 0
        freshness_ok, freshness_details = _assess_freshness(bars, end)

        quality_flags: dict[str, Any] = {
            "availability": availability_ok,
            "completeness": completeness_ok,
            "freshness": freshness_ok,
            "stability": True,
            **freshness_details,
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
            return Result.failed(RuntimeError("No bars returned."), reason_code="NO_DATA")
        if not (completeness_ok and freshness_ok):
            return Result.degraded(snapshot, RuntimeError("Yahoo data quality degraded."), "QUALITY_DEGRADED")
        return Result.success(snapshot, reason_code="OK")


def _timestamp_from_index(index_value: Any) -> int:
    if isinstance(index_value, datetime):
        dt = index_value if index_value.tzinfo is not None else index_value.replace(tzinfo=UTC)
        return int(dt.astimezone(UTC).timestamp() * 1_000_000_000)
    if isinstance(index_value, date):
        dt = datetime(index_value.year, index_value.month, index_value.day, tzinfo=UTC)
        return int(dt.timestamp() * 1_000_000_000)
    raise TypeError("Unable to infer timestamp from dataframe index.")


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
