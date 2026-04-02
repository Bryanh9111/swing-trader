"""S3 historical data adapter for Polygon paid data."""

from __future__ import annotations

import csv
import gzip
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3
import structlog
from botocore.config import Config
from botocore.exceptions import ClientError

from common.interface import Result
from journal.run_id import RunIDGenerator

from .interface import PriceBar, PriceSeriesSnapshot, normalize_date

__all__ = ["S3DataAdapter"]


@dataclass(slots=True)
class S3DataAdapter:
    """Adapter for fetching historical price data from S3.

    Attributes:
        endpoint_url: S3 endpoint URL (e.g., ``https://your-s3-endpoint.example.com``)
        bucket_name: S3 bucket name (e.g., ``your-bucket``)
        access_key_id: S3 access key ID (reads ``S3_ACCESS_KEY_ID`` if empty)
        secret_access_key: S3 secret access key (reads ``S3_SECRET_ACCESS_KEY`` if empty)
        path_prefix: S3 path prefix (defaults to Polygon flatfiles layout)
        local_cache_dir: Local cache directory mirroring Polygon flatfiles layout
        max_workers: Max worker threads for parallel downloads
    """

    endpoint_url: str
    bucket_name: str
    access_key_id: str = ""
    secret_access_key: str = ""
    path_prefix: str = "us_stocks_sip/day_aggs_v1"
    local_cache_dir: str | None = ".cache/historical"
    symbol_cache_dir: str | None = ".cache/by_symbol"
    max_workers: int = 15
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 60.0
    max_attempts: int = 5
    tcp_keepalive: bool = True

    _logger: Any = field(init=False, repr=False)
    _s3_client: Any = field(init=False, repr=False, default=None)

    _SOURCE_NAME = "s3"
    _SCHEMA_VERSION = "1.0.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_logger", structlog.get_logger(__name__).bind(source=self._SOURCE_NAME))
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        # Lazy initialization of the S3 client (created on first use).

    def get_source_name(self) -> str:
        """Return the adapter source identifier (``\"s3\"``)."""

        return self._SOURCE_NAME

    def _get_s3_client(self) -> Any:
        """Get or create S3 client."""

        if self._s3_client is None:
            access_key = self.access_key_id or os.environ.get("S3_ACCESS_KEY_ID") or ""
            secret_key = self.secret_access_key or os.environ.get("S3_SECRET_ACCESS_KEY") or ""
            if not access_key or not secret_key:
                raise RuntimeError("Missing S3 credentials (S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY).")

            # Without explicit timeouts, a single hung socket read can block the entire
            # ThreadPoolExecutor map (and therefore the backtest) indefinitely.
            client_config = Config(
                max_pool_connections=self.max_workers,
                connect_timeout=float(self.connect_timeout_seconds),
                read_timeout=float(self.read_timeout_seconds),
                retries={"max_attempts": int(self.max_attempts), "mode": "standard"},
                tcp_keepalive=bool(self.tcp_keepalive),
            )
            object.__setattr__(
                self,
                "_s3_client",
                boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    config=client_config,
                ),
            )
        return self._s3_client

    def _parse_gzip_csv_bytes(self, gzip_content: bytes, symbol: str) -> PriceBar | None:
        csv_content = gzip.decompress(gzip_content).decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_content))
        for row in reader:
            if (row.get("ticker") or "").upper() != symbol:
                continue

            return PriceBar(
                timestamp=int(row["window_start"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )

        return None

    def _read_symbol_cache(self, cache_path: Path, symbol: str) -> list[PriceBar]:
        bars: list[PriceBar] = []
        with gzip.open(cache_path, mode="rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                if (row.get("ticker") or "").upper() != symbol:
                    continue

                bars.append(
                    PriceBar(
                        timestamp=int(row["window_start"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                    )
                )
        return bars

    def _fetch_single_date(self, s3: Any, symbol: str, current: date) -> PriceBar | None:
        year = current.year
        date_str = current.isoformat()
        s3_key = f"{self.path_prefix}/{year}/{current.month:02d}/{date_str}.csv.gz"

        cache_path: str | None = None
        if self.local_cache_dir is not None:
            cache_path = os.path.join(
                self.local_cache_dir,
                str(year),
                f"{current.month:02d}",
                f"{date_str}.csv.gz",
            )
            if os.path.exists(cache_path):
                self._logger.debug("s3.cache_hit", date=date_str, path=cache_path)
                try:
                    with open(cache_path, "rb") as handle:
                        gzip_content = handle.read()
                    return self._parse_gzip_csv_bytes(gzip_content, symbol)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "s3.cache_read_failed",
                        date=date_str,
                        path=cache_path,
                        error=str(exc)[:200],
                    )
            else:
                self._logger.debug("s3.cache_miss", date=date_str)

        try:
            response = s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            gzip_content = response["Body"].read()

            if cache_path is not None:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    tmp_path = f"{cache_path}.part"
                    with open(tmp_path, "wb") as handle:
                        handle.write(gzip_content)
                    os.replace(tmp_path, cache_path)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "s3.cache_write_failed",
                        date=date_str,
                        path=cache_path,
                        error=str(exc)[:200],
                    )

            return self._parse_gzip_csv_bytes(gzip_content, symbol)

        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", "Unknown"))
            if error_code in {"NoSuchKey", "404"}:
                self._logger.debug("s3.file_not_found", date=date_str, key=s3_key)
            else:
                self._logger.warning(
                    "s3.fetch_failed",
                    date=date_str,
                    key=s3_key,
                    error_code=error_code,
                    error=str(exc)[:200],
                )
            return None
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("s3.parse_failed", date=date_str, key=s3_key, error=str(exc)[:200])
            return None

    def fetch_price_series(
        self,
        symbol: str,
        start_date: date | str,
        end_date: date | str,
        timeframe: str = "1D",
    ) -> Result[PriceSeriesSnapshot]:
        """Fetch historical daily OHLCV bars from S3.

        Args:
            symbol: Stock symbol (e.g., ``AAPL``)
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            timeframe: Timeframe (only ``1D`` supported)
        """

        if timeframe != "1D":
            return Result.failed(
                ValueError(f"S3 adapter only supports 1D timeframe, got {timeframe}"),
                reason_code="UNSUPPORTED_TIMEFRAME",
            )

        start = normalize_date(start_date)
        end = normalize_date(end_date)
        if end < start:
            return Result.failed(ValueError("end_date must be >= start_date."), reason_code="INVALID_RANGE")

        symbol = symbol.upper()
        bars: list[PriceBar] = []

        symbol_cache_path: Path | None = None
        used_symbol_cache = False
        if self.symbol_cache_dir:
            symbol_cache_path = Path(self.symbol_cache_dir) / f"{symbol}.csv.gz"
            if symbol_cache_path.exists():
                self._logger.debug("symbol_cache_hit", symbol=symbol, path=str(symbol_cache_path))
                try:
                    all_bars = self._read_symbol_cache(symbol_cache_path, symbol)
                    used_symbol_cache = True
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "symbol_cache_read_failed",
                        symbol=symbol,
                        path=str(symbol_cache_path),
                        error=str(exc)[:200],
                    )
                    all_bars = []

                start_timestamp = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp()) * 1_000_000_000
                end_exclusive = end + timedelta(days=1)
                end_timestamp = (
                    int(datetime(end_exclusive.year, end_exclusive.month, end_exclusive.day, tzinfo=UTC).timestamp())
                    * 1_000_000_000
                    - 1
                )
                bars = [bar for bar in all_bars if start_timestamp <= bar.timestamp <= end_timestamp]
            else:
                self._logger.debug("symbol_cache_miss", symbol=symbol, path=str(symbol_cache_path))

        if used_symbol_cache:
            if not bars:
                return Result.failed(RuntimeError(f"No data found for {symbol} in symbol cache"), reason_code="NO_DATA")
            snapshot = PriceSeriesSnapshot(
                schema_version=self._SCHEMA_VERSION,
                system_version=RunIDGenerator.get_system_version(),
                asof_timestamp=time.time_ns(),
                symbol=symbol,
                timeframe=timeframe,
                bars=bars,
                source=self._SOURCE_NAME,
                quality_flags={
                    "availability": True,
                    "completeness": True,
                    "freshness": True,
                    "stability": True,
                },
            )
            return Result.success(snapshot, reason_code="OK")

        try:
            s3 = self._get_s3_client()
        except Exception as exc:  # noqa: BLE001
            return Result.failed(exc, reason_code="S3_CLIENT_ERROR")

        dates: list[date] = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)

        if dates:
            # When reading from local cache, serial execution is faster than ThreadPoolExecutor overhead
            # Only use threading if cache miss rate is high (network I/O bound)
            if self.max_workers == 1 or self.local_cache_dir:
                # Serial execution (no threading overhead)
                for d in dates:
                    bar = self._fetch_single_date(s3, symbol, d)
                    if bar is not None:
                        bars.append(bar)
            else:
                # Parallel execution (for network-heavy workloads)
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(dates))) as executor:
                    for bar in executor.map(lambda d: self._fetch_single_date(s3, symbol, d), dates):
                        if bar is not None:
                            bars.append(bar)

        if not bars:
            return Result.failed(RuntimeError(f"No data found for {symbol} in S3"), reason_code="NO_DATA")

        snapshot = PriceSeriesSnapshot(
            schema_version=self._SCHEMA_VERSION,
            system_version=RunIDGenerator.get_system_version(),
            asof_timestamp=time.time_ns(),
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
            source=self._SOURCE_NAME,
            quality_flags={
                "availability": True,
                "completeness": True,
                "freshness": True,
                "stability": True,
            },
        )

        return Result.success(snapshot, reason_code="OK")
