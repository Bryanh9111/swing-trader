#!/usr/bin/env python3
"""Merge incremental per-symbol CSV.gz files into the main by_symbol directory.

This script takes a source directory of per-symbol CSV.gz files (e.g., from a
partial pivot of new months) and merges them into the existing by_symbol
directory. It deduplicates rows by the window_start timestamp column (index 6)
and keeps the output sorted by timestamp.

Typical workflow:
  1. Download new month(s) to .cache/historical/
  2. Pivot ONLY the new files to a temp directory
  3. Use this script to merge temp into .cache/by_symbol/

Usage:
  python3 scripts/merge_by_symbol.py --source .cache/by_symbol_new_tmp --target .cache/by_symbol
  python3 scripts/merge_by_symbol.py --source .cache/by_symbol_new_tmp --target .cache/by_symbol --dry-run
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path


def _read_gz_csv(path: Path) -> list[list[str]]:
    """Read all rows from a gzip CSV file."""
    rows: list[list[str]] = []
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows


def _write_gz_csv(path: Path, rows: list[list[str]]) -> None:
    """Write rows to a gzip CSV file."""
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def merge_symbol_file(
    source_file: Path, target_dir: Path, *, dry_run: bool = False
) -> tuple[str, int, int, bool]:
    """Merge a single source symbol file into target directory.

    Returns (symbol, rows_before, rows_after, is_new).
    """
    symbol = source_file.stem.replace(".csv", "")  # Handle .csv.gz
    target_file = target_dir / source_file.name

    source_rows = _read_gz_csv(source_file)

    if target_file.exists():
        existing_rows = _read_gz_csv(target_file)
        is_new = False
    else:
        existing_rows = []
        is_new = True

    rows_before = len(existing_rows)

    # Deduplicate by window_start timestamp (column index 6)
    seen_timestamps: set[str] = set()
    merged: list[list[str]] = []

    for row in existing_rows + source_rows:
        if len(row) > 6:
            ts = row[6]
            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                merged.append(row)
        else:
            merged.append(row)

    # Sort by window_start timestamp (column 6)
    merged.sort(key=lambda r: int(r[6]) if len(r) > 6 and r[6].isdigit() else 0)

    rows_after = len(merged)

    if not dry_run:
        _write_gz_csv(target_file, merged)

    return symbol, rows_before, rows_after, is_new


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge incremental per-symbol CSV.gz into main by_symbol directory."
    )
    parser.add_argument(
        "--source", required=True, help="Source directory with new per-symbol CSV.gz files"
    )
    parser.add_argument(
        "--target", default=".cache/by_symbol", help="Target by_symbol directory. Default: .cache/by_symbol"
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report; do not write.")
    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)

    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)
    if not target_dir.exists():
        print(f"ERROR: Target directory does not exist: {target_dir}", file=sys.stderr)
        sys.exit(1)

    source_files = sorted(source_dir.glob("*.csv.gz"))
    if not source_files:
        print("No .csv.gz files found in source directory.")
        sys.exit(0)

    print(f"Source: {source_dir} ({len(source_files)} files)")
    print(f"Target: {target_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    appended = 0
    created = 0
    total_new_rows = 0

    for i, src in enumerate(source_files, 1):
        symbol, before, after, is_new = merge_symbol_file(src, target_dir, dry_run=args.dry_run)
        added = after - before
        total_new_rows += added

        if is_new:
            created += 1
        else:
            appended += 1

        if i % 2000 == 0 or i == len(source_files):
            print(f"  Progress: {i}/{len(source_files)} files processed")

    print(f"\nDone: {appended} appended, {created} new, {total_new_rows} total new rows added")


if __name__ == "__main__":
    main()
