#!/usr/bin/env python3
"""Pivot historical daily cache from date-partitioned files to per-symbol files.

Input:
  - Source directory: .cache/historical/**/*.csv.gz
  - Each CSV.gz contains single-day rows for many tickers.

Output:
  - Target directory: .cache/by_symbol/
  - One gzip CSV per symbol: {SYMBOL}.csv.gz
  - Same CSV columns as input.

Usage:
  python3 scripts/pivot_historical_cache.py
  python3 scripts/pivot_historical_cache.py --workers 4
  python3 scripts/pivot_historical_cache.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    print("[WARN] tqdm is not installed; progress bar disabled.", file=sys.stderr)

    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable

FIELDNAMES: tuple[str, ...] = (
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
)


@dataclass(frozen=True)
class PivotArgs:
    source_dir: Path
    output_dir: Path
    flush_every: int
    workers: int
    dry_run: bool


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _parse_args(argv: list[str] | None = None) -> PivotArgs:
    parser = argparse.ArgumentParser(description="Pivot .cache/historical date files into per-symbol CSV.gz files.")
    parser.add_argument("--source", default=".cache/historical", help="Source directory. Default: .cache/historical")
    parser.add_argument("--output", default=".cache/by_symbol", help="Output directory. Default: .cache/by_symbol")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=100,
        help="Flush buffers every N date files (memory control). Default: 100",
    )
    parser.add_argument("--workers", type=int, default=1, help="Process workers for reading. Default: 1")
    parser.add_argument("--dry-run", action="store_true", help="Only compute statistics; do not write output.")
    ns = parser.parse_args(argv)

    flush_every = int(ns.flush_every)
    if flush_every < 1:
        raise ValueError("--flush-every must be >= 1")

    workers = int(ns.workers)
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    return PivotArgs(
        source_dir=Path(str(ns.source)),
        output_dir=Path(str(ns.output)),
        flush_every=flush_every,
        workers=workers,
        dry_run=bool(ns.dry_run),
    )


def _iter_source_files(source_dir: Path) -> list[Path]:
    files = sorted(source_dir.rglob("*.csv.gz"))
    return files


def _read_rows_from_gz_csv(path_str: str) -> tuple[str, list[tuple[str, ...]] | None, str | None]:
    """Worker-safe reader.

    Returns: (path_str, rows, error)
    """
    path = Path(path_str)
    try:
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            rows: list[tuple[str, ...]] = []
            for row in reader:
                if not row:
                    continue
                ticker = (row.get("ticker") or "").strip()
                if not ticker:
                    continue
                row_tuple = tuple((row.get(name) or "") for name in FIELDNAMES)
                rows.append(row_tuple)
        return (path_str, rows, None)
    except Exception as exc:  # noqa: BLE001 - intended: corruption shouldn't abort the run
        return (path_str, None, f"{type(exc).__name__}: {exc}")


def _ingest_single_file(
    path: Path,
    buffers: DefaultDict[str, list[tuple[str, ...]]],
    symbols_seen: set[str],
) -> int:
    rows_ingested = 0
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row:
                continue
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                continue
            symbols_seen.add(ticker)
            row_tuple = tuple((row.get(name) or "") for name in FIELDNAMES)
            buffers[ticker].append(row_tuple)
            rows_ingested += 1
    return rows_ingested


def _flush_buffers(
    output_dir: Path,
    buffers: DefaultDict[str, list[tuple[str, ...]]],
    written_symbols: set[str],
    dry_run: bool,
) -> int:
    if not buffers:
        return 0
    if dry_run:
        buffers.clear()
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for symbol, rows in buffers.items():
        if not rows:
            continue
        out_path = output_dir / f"{symbol}.csv.gz"
        mode = "wt" if symbol not in written_symbols else "at"
        with gzip.open(out_path, mode=mode, encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            if mode == "wt":
                writer.writeheader()
                written_symbols.add(symbol)
            for row_tuple in rows:
                writer.writerow(dict(zip(FIELDNAMES, row_tuple)))
                total_written += 1

    buffers.clear()
    return total_written


def _pivot(args: PivotArgs) -> int:
    source_dir = args.source_dir
    output_dir = args.output_dir

    files = _iter_source_files(source_dir)
    total_files = len(files)
    if total_files == 0:
        print(f"No input files found under: {source_dir}")
        return 0

    buffers: DefaultDict[str, list[tuple[str, ...]]] = defaultdict(list)
    written_symbols: set[str] = set()
    symbols_seen: set[str] = set()
    errors: list[tuple[Path, str]] = []

    processed_files = 0
    skipped_files = 0
    total_rows = 0
    total_rows_written = 0

    start = time.perf_counter()

    if args.workers == 1:
        for index, path in enumerate(tqdm(files, desc="Reading daily files", unit="file"), start=1):
            try:
                total_rows += _ingest_single_file(path, buffers=buffers, symbols_seen=symbols_seen)
                processed_files += 1
            except Exception as exc:  # noqa: BLE001
                skipped_files += 1
                errors.append((path, f"{type(exc).__name__}: {exc}"))
                print(f"[WARN] Skipping corrupted file: {path} ({type(exc).__name__}: {exc})", file=sys.stderr)

            if index % args.flush_every == 0:
                total_rows_written += _flush_buffers(
                    output_dir=output_dir,
                    buffers=buffers,
                    written_symbols=written_symbols,
                    dry_run=args.dry_run,
                )
    else:
        path_strs = [str(p) for p in files]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            iterator: Iterable[tuple[str, list[tuple[str, ...]] | None, str | None]] = executor.map(
                _read_rows_from_gz_csv,
                path_strs,
                chunksize=1,
            )
            for index, (path_str, rows, error) in enumerate(
                tqdm(iterator, total=total_files, desc="Reading daily files", unit="file"),
                start=1,
            ):
                path = Path(path_str)
                if error is not None or rows is None:
                    skipped_files += 1
                    errors.append((path, error or "Unknown error"))
                    print(f"[WARN] Skipping corrupted file: {path} ({error})", file=sys.stderr)
                else:
                    processed_files += 1
                    total_rows += len(rows)
                    for row_tuple in rows:
                        ticker = (row_tuple[0] or "").strip()
                        if not ticker:
                            continue
                        symbols_seen.add(ticker)
                        buffers[ticker].append(row_tuple)

                if index % args.flush_every == 0:
                    total_rows_written += _flush_buffers(
                        output_dir=output_dir,
                        buffers=buffers,
                        written_symbols=written_symbols,
                        dry_run=args.dry_run,
                    )

    total_rows_written += _flush_buffers(
        output_dir=output_dir,
        buffers=buffers,
        written_symbols=written_symbols,
        dry_run=args.dry_run,
    )

    elapsed = time.perf_counter() - start

    output_size = _dir_size_bytes(output_dir)
    symbol_file_count = len(symbols_seen) if args.dry_run else len(written_symbols)

    print("\nDone.")
    print(f"- Processed files: {processed_files}/{total_files} (skipped {skipped_files})")
    print(f"- Symbols: {symbol_file_count}")
    print(f"- Rows read: {total_rows}")
    if not args.dry_run:
        print(f"- Rows written: {total_rows_written}")
    print(f"- Elapsed: {elapsed:.2f}s")
    print(f"- Disk usage (output): {output_dir} -> {_human_bytes(output_size)}")
    if errors:
        print(f"- Errors: {len(errors)} (see stderr for warnings)")
    return 0


def main() -> int:
    try:
        args = _parse_args()
        return _pivot(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
