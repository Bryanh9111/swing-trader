#!/usr/bin/env python3
"""Bulk pre-download Polygon S3 historical daily CSV.gz files.

Downloads files for each U.S. equity trading day in the requested range and
saves them under the local cache directory mirroring the Polygon flatfiles
layout:

    .cache/historical/{year}/{month:02d}/{date}.csv.gz

Usage:
    python scripts/download_historical_data.py
    python scripts/download_historical_data.py --start 2021-01-01 --end 2025-12-31 --workers 32
    python scripts/download_historical_data.py --start 2024-01-01 --end 2024-12-31 --force
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import structlog
from botocore.exceptions import ClientError
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from dotenv import load_dotenv

from calendar_data import MarketCalendar
from common.interface import Result, ResultStatus
from common.logging import init_logging
from data.s3_adapter import S3DataAdapter


@dataclass(slots=True, frozen=True)
class DownloadArgs:
    """CLI arguments for the downloader."""

    start: date
    end: date
    workers: int
    force: bool


@dataclass(slots=True, frozen=True)
class S3SourceConfig:
    """Resolved S3 config from `config/config.yaml`."""

    endpoint_url: str
    bucket_name: str
    path_prefix: str


@dataclass(slots=True, frozen=True)
class DownloadTask:
    """Single trading-day download task."""

    trading_day: date
    s3_key: str
    local_path: Path


@dataclass(slots=True, frozen=True)
class DownloadOutcome:
    """Outcome of a download attempt."""

    task: DownloadTask
    skipped: bool


def _parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'; expected YYYY-MM-DD") from exc


def _parse_args(argv: list[str] | None = None) -> DownloadArgs:
    parser = argparse.ArgumentParser(description="Bulk download Polygon historical CSV.gz files from S3.")
    parser.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD). Default: 2021-01-01")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument("--workers", type=int, default=20, help="Max concurrent downloads. Default: 20")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    ns = parser.parse_args(argv)

    start = _parse_iso_date(str(ns.start))
    end = _parse_iso_date(str(ns.end))
    if end < start:
        raise ValueError("--end must be >= --start")

    workers = int(ns.workers)
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    return DownloadArgs(start=start, end=end, workers=workers, force=bool(ns.force))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must be a mapping, got {type(loaded).__name__}")
    return loaded


def _init_structlog_from_config(config: Mapping[str, Any]) -> None:
    system_cfg = config.get("system")
    if not isinstance(system_cfg, Mapping):
        return

    try:
        import msgspec
        from common.config import SystemConfig

        system = msgspec.convert(dict(system_cfg), type=SystemConfig)
        init_logging(system)
    except Exception:
        return


def _get_s3_source_config(config: Mapping[str, Any]) -> S3SourceConfig:
    data_sources_cfg = config.get("data_sources")
    if not isinstance(data_sources_cfg, Mapping):
        raise KeyError("Missing `data_sources` section in config/config.yaml")

    s3_cfg = data_sources_cfg.get("s3")
    if not isinstance(s3_cfg, Mapping):
        raise KeyError("Missing `data_sources.s3` section in config/config.yaml")

    endpoint_url = str(s3_cfg.get("endpoint_url") or "").strip()
    bucket_name = str(s3_cfg.get("bucket_name") or "").strip()
    path_prefix = str(s3_cfg.get("path_prefix") or "").strip() or "us_stocks_sip/day_aggs_v1"

    missing: list[str] = []
    if not endpoint_url:
        missing.append("endpoint_url")
    if not bucket_name:
        missing.append("bucket_name")
    if missing:
        raise ValueError(f"Missing S3 config fields in data_sources.s3: {', '.join(missing)}")

    return S3SourceConfig(endpoint_url=endpoint_url, bucket_name=bucket_name, path_prefix=path_prefix)


def _get_trading_days(start: date, end: date, calendar: MarketCalendar) -> list[date]:
    try:
        holidays = calendar.get_all_holidays(start.year, end.year)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load market holidays: {exc}") from exc

    trading_days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in holidays:
            trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


def _build_tasks(
    trading_days: Iterable[date],
    *,
    path_prefix: str,
    local_root: Path,
) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    for trading_day in trading_days:
        date_str = trading_day.isoformat()
        s3_key = f"{path_prefix}/{trading_day.year}/{trading_day.month:02d}/{date_str}.csv.gz"
        local_path = local_root / str(trading_day.year) / f"{trading_day.month:02d}" / f"{date_str}.csv.gz"
        tasks.append(DownloadTask(trading_day=trading_day, s3_key=s3_key, local_path=local_path))
    return tasks


def _download_one(
    *,
    adapter: S3DataAdapter,
    task: DownloadTask,
    force: bool,
    logger: Any,
) -> Result[DownloadOutcome]:
    if task.local_path.exists() and not force:
        logger.debug(
            "s3.download_skipped",
            date=task.trading_day.isoformat(),
            key=task.s3_key,
            path=str(task.local_path),
        )
        return Result.success(DownloadOutcome(task=task, skipped=True), reason_code="SKIPPED_EXISTS")

    task.local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(task.local_path) + ".part")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:  # noqa: BLE001
            pass

    try:
        s3 = adapter._get_s3_client()
        response = s3.get_object(Bucket=adapter.bucket_name, Key=task.s3_key)
        body = response["Body"]

        with tmp_path.open("wb") as handle:
            while True:
                chunk = body.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

        tmp_path.replace(task.local_path)
        logger.debug(
            "s3.download_ok",
            date=task.trading_day.isoformat(),
            key=task.s3_key,
            path=str(task.local_path),
        )
        return Result.success(DownloadOutcome(task=task, skipped=False), reason_code="DOWNLOADED")

    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", "Unknown"))
        logger.warning(
            "s3.download_failed",
            date=task.trading_day.isoformat(),
            key=task.s3_key,
            error_code=error_code,
            error=str(exc)[:200],
        )
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:  # noqa: BLE001
            pass

        outcome = DownloadOutcome(task=task, skipped=False)
        if error_code in {"NoSuchKey", "404"}:
            return Result.degraded(outcome, exc, reason_code="S3_NO_SUCH_KEY")
        return Result.degraded(outcome, exc, reason_code="S3_CLIENT_ERROR")

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "s3.download_exception",
            date=task.trading_day.isoformat(),
            key=task.s3_key,
            error=str(exc)[:200],
        )
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:  # noqa: BLE001
            pass
        return Result.degraded(DownloadOutcome(task=task, skipped=False), exc, reason_code="UNEXPECTED_ERROR")


def _iter_results_parallel(
    tasks: list[DownloadTask],
    *,
    adapter: S3DataAdapter,
    workers: int,
    force: bool,
    logger: Any,
) -> Iterable[Result[DownloadOutcome]]:
    max_workers = min(max(1, int(workers)), max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[Future[Result[DownloadOutcome]]] = [
            executor.submit(_download_one, adapter=adapter, task=task, force=force, logger=logger)
            for task in tasks
        ]
        for future in as_completed(futures):
            yield future.result()


def main(argv: list[str] | None = None) -> int:
    config_path = project_root / "config" / "config.yaml"
    dotenv_path = project_root / "config" / "secrets.env"
    local_root = project_root / ".cache" / "historical"

    try:
        args = _parse_args(argv)
    except Exception as exc:  # noqa: BLE001
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    try:
        config = _load_yaml_mapping(config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    _init_structlog_from_config(config)
    logger = (
        structlog.get_logger(__name__)
        .bind(script="download_historical_data")
        .bind(start=args.start.isoformat(), end=args.end.isoformat(), workers=args.workers, force=args.force)
    )

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
        logger.info("env.loaded", path=str(dotenv_path))
    else:
        logger.warning("env.missing", path=str(dotenv_path))

    try:
        s3_cfg = _get_s3_source_config(config)
    except Exception as exc:  # noqa: BLE001
        logger.error("config.s3_missing", error=str(exc))
        print(f"S3 config error: {exc}", file=sys.stderr)
        return 2

    logger = logger.bind(
        endpoint_url=s3_cfg.endpoint_url,
        bucket_name=s3_cfg.bucket_name,
        path_prefix=s3_cfg.path_prefix,
        cache_root=str(local_root),
    )
    logger.info("download.start")

    adapter = S3DataAdapter(
        endpoint_url=s3_cfg.endpoint_url,
        bucket_name=s3_cfg.bucket_name,
        path_prefix=s3_cfg.path_prefix,
        max_workers=max(1, int(args.workers)),
    )

    try:
        adapter._get_s3_client()
    except Exception as exc:  # noqa: BLE001
        logger.error("s3.client_init_failed", error=str(exc)[:200])
        print(f"S3 client init failed: {exc}", file=sys.stderr)
        return 2

    calendar = MarketCalendar()
    try:
        trading_days = _get_trading_days(args.start, args.end, calendar)
    except Exception as exc:  # noqa: BLE001
        logger.error("calendar.failed", error=str(exc))
        print(f"Calendar error: {exc}", file=sys.stderr)
        return 2

    if not trading_days:
        logger.info("download.no_trading_days")
        print("No trading days in range; nothing to download.")
        return 0

    tasks = _build_tasks(trading_days, path_prefix=s3_cfg.path_prefix, local_root=local_root)
    logger.info("download.planned", trading_days=len(trading_days), tasks=len(tasks))

    downloaded = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    progress = tqdm(total=len(tasks), desc="Downloading", unit="day", dynamic_ncols=True)
    try:
        for result in _iter_results_parallel(tasks, adapter=adapter, workers=args.workers, force=args.force, logger=logger):
            if result.status is ResultStatus.SUCCESS and result.data is not None:
                if result.data.skipped:
                    skipped += 1
                else:
                    downloaded += 1
            else:
                failed_date = result.data.task.trading_day.isoformat() if result.data is not None else "unknown"
                failed_reason = result.reason_code or (type(result.error).__name__ if result.error is not None else "UNKNOWN")
                failed.append((failed_date, failed_reason))
            progress.update(1)
    finally:
        progress.close()

    total = len(tasks)
    failed_count = total - downloaded - skipped
    logger.info(
        "download.finished",
        total=total,
        downloaded=downloaded,
        skipped=skipped,
        failed=failed_count,
    )

    print("=" * 72)
    print("S3 Historical Data Pre-download Summary")
    print("=" * 72)
    print(f"Range:      {args.start.isoformat()} -> {args.end.isoformat()}")
    print(f"Cache root: {local_root}")
    print(f"Total:      {total}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped} (already exists)")
    print(f"Failed:     {failed_count}")

    if failed_count:
        print("\nFailed dates (YYYY-MM-DD -> reason_code):")
        for day, reason in failed[:50]:
            print(f"  - {day} -> {reason}")
        if len(failed) > 50:
            print(f"  ... ({len(failed) - 50} more)")
        print("\nFailures detected. Re-run to retry (use --force to overwrite existing files).")
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
