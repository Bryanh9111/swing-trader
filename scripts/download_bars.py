#!/usr/bin/env python3
"""Download bars from Polygon REST API for all symbols in by_symbol cache.

Supports any timespan: minute, hour, day, week, month, quarter, year.

Saves to: .cache/historical_{timespan}s/{start}_{end}/{SYMBOL}.csv.gz

Usage:
    python scripts/download_bars.py --timespan week --start 2021-03-01 --end 2026-01-31
    python scripts/download_bars.py --timespan month --start 2021-03-01 --end 2026-01-31 --workers 3
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import sys
import time
import traceback
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

VALID_TIMESPANS = ("minute", "hour", "day", "week", "month", "quarter", "year")


@dataclass(slots=True, frozen=True)
class DownloadArgs:
    timespan: str
    start: date
    end: date
    workers: int
    force: bool
    output_dir: str | None


def _parse_args(argv: list[str] | None = None) -> DownloadArgs:
    parser = argparse.ArgumentParser(description="Download bars from Polygon.")
    parser.add_argument("--timespan", required=True, choices=VALID_TIMESPANS, help="Bar timespan")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent workers. Default: 3")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    ns = parser.parse_args(argv)
    start = datetime.strptime(ns.start, "%Y-%m-%d").date()
    end = datetime.strptime(ns.end, "%Y-%m-%d").date()
    return DownloadArgs(timespan=ns.timespan, start=start, end=end, workers=max(1, ns.workers), force=ns.force, output_dir=ns.output_dir)


def _get_symbols(by_symbol_dir: Path) -> list[str]:
    symbols = []
    for f in sorted(by_symbol_dir.iterdir()):
        if f.name.endswith(".csv.gz"):
            symbols.append(f.name.replace(".csv.gz", ""))
    return symbols


def _fetch_bars(symbol: str, timespan: str, start: date, end: date, api_key: str) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []
    base_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}"
        f"/{start.isoformat()}/{end.isoformat()}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    url = base_url

    while url:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AST/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1 * (attempt + 1))

        results = data.get("results") or []
        all_results.extend(results)

        next_url = data.get("next_url")
        if next_url:
            url = f"{next_url}&apiKey={api_key}"
        else:
            url = ""

    return all_results


def _save_csv_gz(bars: list[dict[str, Any]], output_path: Path) -> int:
    if not bars:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["t", "o", "h", "l", "c", "v", "vw", "n"]
    buf = io.BytesIO()
    with gzip.open(buf, "wt", compresslevel=6) as gz:
        writer = csv.DictWriter(gz, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for bar in bars:
            writer.writerow(bar)
    output_path.write_bytes(buf.getvalue())
    return len(bars)


def _download_one(
    symbol: str, timespan: str, start: date, end: date, api_key: str, output_dir: Path, force: bool,
) -> tuple[str, int, str]:
    output_path = output_dir / f"{symbol}.csv.gz"
    if output_path.exists() and not force:
        return (symbol, 0, "skipped")
    try:
        bars = _fetch_bars(symbol, timespan, start, end, api_key)
        if not bars:
            return (symbol, 0, "no_data")
        count = _save_csv_gz(bars, output_path)
        return (symbol, count, "ok")
    except Exception as exc:
        return (symbol, 0, f"error:{str(exc)[:100]}")


def main(argv: list[str] | None = None) -> int:
    dotenv_path = project_root / "config" / "secrets.env"
    by_symbol_dir = project_root / ".cache" / "by_symbol"
    args = _parse_args(argv)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / ".cache" / f"historical_{args.timespan}s" / f"{args.start}_{args.end}"

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("AST_POLYGON_API_KEY") or ""
    if not api_key:
        print("Error: POLYGON_API_KEY not set", file=sys.stderr)
        return 2

    if not by_symbol_dir.exists():
        print(f"Error: {by_symbol_dir} not found", file=sys.stderr)
        return 2

    symbols = _get_symbols(by_symbol_dir)
    if not symbols:
        print("No symbols found in by_symbol cache.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading 1-{args.timespan} bars: {len(symbols)} symbols")
    print(f"Range: {args.start} -> {args.end}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")
    print()

    downloaded = 0
    skipped = 0
    no_data = 0
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future, str] = {}
        for symbol in symbols:
            fut = executor.submit(
                _download_one, symbol, args.timespan, args.start, args.end, api_key, output_dir, args.force
            )
            futures[fut] = symbol

        progress = tqdm(total=len(symbols), desc="Downloading", unit="sym", dynamic_ncols=True)
        try:
            for fut in as_completed(futures):
                sym, count, status = fut.result()
                if status == "ok":
                    downloaded += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "no_data":
                    no_data += 1
                else:
                    errors.append((sym, status))
                progress.update(1)
        finally:
            progress.close()

    print()
    print("=" * 60)
    print(f"{args.timespan.title()} Bar Download Summary")
    print("=" * 60)
    print(f"Total symbols:  {len(symbols)}")
    print(f"Downloaded:     {downloaded}")
    print(f"Skipped:        {skipped} (already exists)")
    print(f"No data:        {no_data}")
    print(f"Errors:         {len(errors)}")

    if errors:
        print(f"\nFirst 20 errors:")
        for sym, err in errors[:20]:
            print(f"  {sym}: {err}")
        if len(errors) > 20:
            print(f"  ... ({len(errors) - 20} more)")

    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
