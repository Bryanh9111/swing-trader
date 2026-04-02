#!/usr/bin/env python3
"""Run PRE_MARKET_FULL_SCAN in paper trading mode.

This script runs the full PRE_MARKET_FULL_SCAN pipeline in paper trading mode:
1. Build universe of tradable stocks
2. Fetch market data
3. Scan for platform-phase candidates
4. Apply event guard constraints
5. Generate trade intents
6. Apply risk gate decisions
7. Submit orders to IBKR paper account

Usage:
    # Dry run (no orders submitted, test the flow)
    python scripts/run_paper_eod_scan.py --dry-run

    # Paper trading (submit orders to IBKR paper account)
    python scripts/run_paper_eod_scan.py --paper

    # Full paper trading with verbose output
    python scripts/run_paper_eod_scan.py --paper --verbose
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from secrets.env
from dotenv import load_dotenv
load_dotenv(project_root / "config" / "secrets.env")

import shutil
import subprocess
import tempfile
from datetime import date, timedelta


def check_trading_day() -> bool:
    """Return True if today is a US market trading day (not weekend/holiday).

    Uses pandas_market_calendars (NYSE) for offline check — no network needed.
    """
    try:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE")
        today = date.today()
        schedule = nyse.schedule(
            start_date=today.isoformat(),
            end_date=today.isoformat(),
        )
        return len(schedule) > 0
    except Exception as exc:
        # If calendar check fails, assume trading day (fail-open for safety)
        print(f"   ⚠️  Trading day check failed ({exc}), assuming trading day")
        return True

from common.config import _deep_merge
from common.events import InMemoryEventBus
from common.interface import ResultStatus
from common.logging import StructlogLoggerFactory
from journal import ArtifactManager, JournalWriter
from journal.interface import OperatingMode, RunType
from notifier.telegram import TelegramNotifier
from orchestrator import EODScanOrchestrator, register_real_plugins
from plugins.registry import PluginRegistry
from reporting.run_report import RunReportGenerator


def update_daily_cache() -> bool:
    """Download latest trading day bar from S3 and merge into .cache/by_symbol/.

    Three-step incremental: download → pivot (temp) → merge.
    Non-blocking: failure prints a warning and returns False.
    """
    from calendar_data import MarketCalendar

    target_dir = project_root / ".cache" / "by_symbol"
    if not target_dir.exists():
        print("   ⚠️  .cache/by_symbol/ not found — skipping cache update")
        return False

    # Find last completed trading day
    cal = MarketCalendar()
    today = date.today()
    candidate = today - timedelta(days=1)
    holidays = cal.get_all_holidays(candidate.year, candidate.year)
    while candidate.weekday() >= 5 or candidate in holidays:
        candidate -= timedelta(days=1)
        if candidate.year < today.year - 1:
            break
        if candidate.year != (candidate + timedelta(days=1)).year:
            holidays = cal.get_all_holidays(candidate.year, candidate.year)

    last_td = candidate
    date_str = last_td.isoformat()

    cache_file = (
        project_root / ".cache" / "historical"
        / str(last_td.year) / f"{last_td.month:02d}" / f"{date_str}.csv.gz"
    )

    # Step 1: Download from S3 if missing
    if not cache_file.exists():
        print(f"   Downloading S3 data for {date_str}...")
        r = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "download_historical_data.py"),
             "--start", date_str, "--end", date_str, "--workers", "3"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0 or not cache_file.exists():
            print(f"   ⚠️  S3 download failed for {date_str} (may not be available yet)")
            return False
        print(f"   ✅ Downloaded {date_str}")
    else:
        print(f"   📦 {date_str} already in local cache")

    # Step 2: Pivot single day to temp dir → Step 3: Merge into by_symbol
    with tempfile.TemporaryDirectory(prefix="ast_pivot_") as tmp_dir:
        tmp_source = Path(tmp_dir) / "src" / str(last_td.year) / f"{last_td.month:02d}"
        tmp_source.mkdir(parents=True)
        shutil.copy2(cache_file, tmp_source / cache_file.name)

        tmp_output = Path(tmp_dir) / "pivoted"
        tmp_output.mkdir()

        r = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "pivot_historical_cache.py"),
             "--source", str(Path(tmp_dir) / "src"), "--output", str(tmp_output)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            print(f"   ⚠️  Pivot failed: {(r.stderr or '')[:200]}")
            return False

        pivoted_files = list(tmp_output.glob("*.csv.gz"))
        if not pivoted_files:
            print("   ⚠️  Pivot produced no output files")
            return False

        r = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "merge_by_symbol.py"),
             "--source", str(tmp_output), "--target", str(target_dir)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            print(f"   ⚠️  Merge failed: {(r.stderr or '')[:200]}")
            return False

    print(f"   ✅ Cache updated: {date_str} → .cache/by_symbol/ ({len(pivoted_files)} symbols)")
    return True


def load_config(mode: str) -> dict:
    """Load configuration for the specified mode."""
    import yaml

    # Load base config
    config_path = project_root / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load mode-specific config (paper or live)
    # dry_run shares paper config (same infra: market_calendar, data sources)
    effective_mode = "paper" if mode.lower() == "dry_run" else mode.lower()
    mode_config_path = project_root / "config" / f"config.{effective_mode}.yaml"
    if mode_config_path.exists():
        with open(mode_config_path) as f:
            mode_config = yaml.safe_load(f)
            # Deep merge mode config into base config
            config = _deep_merge(config, mode_config)

    return config


def build_run_config(config: dict, mode: OperatingMode) -> dict:
    """Build orchestrator run configuration."""
    # Get Polygon API key from environment
    polygon_api_key = os.getenv("POLYGON_API_KEY")

    # Base configuration
    run_config = {
        "mode": mode.value,
        "run_type": RunType.PRE_MARKET_FULL_SCAN.value,

        # Market calendar configuration (for backfill optimization)
        "market_calendar": config.get("market_calendar", {}),

        # Hoist market_regime to top-level for orchestrator regime detection
        "market_regime": config.get("scanner", {}).get("market_regime", {}),

        # Plugin configuration
        "plugins": {
            "data_sources": [
                {
                    "name": "universe_real",
                    "config": {
                        **config.get("universe_builder", {}),
                        "polygon_api_key": polygon_api_key,
                    },
                },
                {
                    "name": "data_real",
                    "config": {
                        # Use the full `data` config node so DataPlugin picks up primary/secondary sources and
                        # any other adapter settings (avoids falling back to defaults like primary_source="yahoo").
                        **config.get("data", {}),
                        "polygon_api_key": polygon_api_key,
                        "polygon_unlimited": True,
                    },
                },
            ],
            "scanners": [
                {
                    "name": "scanner_real",
                    "config": config.get("scanner", {}),
                },
            ],
            "risk_policies": [
                {
                    "name": "event_guard_real",
                    "config": config.get("event_guard", {}),
                },
                {
                    "name": "risk_gate_real",
                    "config": config.get("risk_gate", {}),
                },
            ],
            "strategies": [
                {
                    "name": "strategy_real",
                    "config": config.get("strategy", {}),
                },
            ],
            "signals": [
                {
                    "name": "execution_real",
                    "config": config.get("execution", {}),
                },
            ],
        },

        # Plugin names for orchestrator
        "event_guard_plugin": "event_guard_real",
        "strategy_plugin": "strategy_real",
        "risk_gate_plugin": "risk_gate_real",
        "execution_plugin": "execution_real",
    }

    return run_config


def print_summary(result, run_id: str):
    """Print run summary."""
    print("\n" + "=" * 70)
    print("PRE_MARKET_FULL_SCAN RUN SUMMARY")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Status: {result.status.value}")

    if result.status is ResultStatus.SUCCESS:
        print("✅ Run completed successfully!")

        if result.data:
            summary = result.data
            print(f"\nPipeline Status: {summary.status.value}")
            print(f"  Candidates: {summary.candidates_count}")
            print(f"  Execution Reports: {summary.execution_reports_count}")
            print(f"  Failures: {summary.failures}")
            print(f"  Duration: {summary.duration_ms}ms")

    elif result.status is ResultStatus.DEGRADED:
        print("⚠️  Run completed with degraded status")
        print(f"Reason: {result.reason_code}")
        if result.error:
            print(f"Error: {result.error}")

    else:
        print("❌ Run failed!")
        print(f"Reason: {result.reason_code}")
        if result.error:
            print(f"Error: {result.error}")

    print("=" * 70)

    # Print artifact location
    artifact_path = project_root / "journal" / run_id
    if artifact_path.exists():
        print(f"\n📁 Artifacts saved to: {artifact_path}")
        print("   View detailed logs and outputs in the journal directory")

    print()


def main():
    parser = argparse.ArgumentParser(description="Run PRE_MARKET_FULL_SCAN in paper trading mode")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in DRY_RUN mode (no orders submitted)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in PAPER mode (submit orders to IBKR paper account)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Override account equity (USD). Overrides config.paper.yaml value.",
    )

    args = parser.parse_args()

    # Skip non-trading days (weekends + US holidays)
    if not check_trading_day():
        print("📅 Market closed today (weekend or US holiday). Skipping run.")
        sys.exit(0)

    # Determine mode
    if args.paper:
        mode = OperatingMode.PAPER
        mode_str = "paper"
        print("\n🚀 Running PRE_MARKET_FULL_SCAN in PAPER TRADING mode")
        print("   Orders will be submitted to IBKR paper account")
    elif args.dry_run:
        mode = OperatingMode.DRY_RUN
        mode_str = "dry_run"
        print("\n🧪 Running PRE_MARKET_FULL_SCAN in DRY_RUN mode")
        print("   No orders will be submitted (testing flow only)")
    else:
        print("❌ Error: Must specify either --dry-run or --paper")
        parser.print_help()
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("SETUP")
    print('=' * 70)

    # Load configuration
    print("1. Loading configuration...")
    config = load_config(mode_str)
    effective_cfg = "paper" if mode_str == "dry_run" else mode_str
    print(f"   ✅ Config loaded from config.yaml + config.{effective_cfg}.yaml")

    # Apply --equity CLI override if provided
    if args.equity is not None:
        config.setdefault("strategy", {})["account_equity"] = args.equity
        config.setdefault("risk_gate", {})["account_equity"] = args.equity
        config["account_equity"] = args.equity
        print(f"   ✅ Equity override from CLI: ${args.equity:,.0f}")

    # Print effective account equity
    effective_equity = config.get("strategy", {}).get("account_equity", 100_000.0)
    print(f"   💰 Effective account equity: ${effective_equity:,.0f}")

    # Verify Polygon API key
    polygon_key = os.getenv("POLYGON_API_KEY")
    if polygon_key:
        print(f"   ✅ Polygon API key found: {polygon_key[:8]}...{polygon_key[-4:]}")
    else:
        print("   ⚠️  Polygon API key not found in environment")

    # Update daily cache (download + pivot + merge latest trading day into .cache/by_symbol/)
    print("\n1b. Updating daily cache...")
    try:
        update_daily_cache()
    except Exception as exc:
        print(f"   ⚠️  Cache update failed (non-blocking): {exc}")

    # Create infrastructure
    print("\n2. Creating infrastructure...")
    artifact_base = project_root / "journal"
    artifact_manager = ArtifactManager(base_path=artifact_base)
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()

    # Register real plugins
    print("   Registering real plugins...")
    register_real_plugins(registry)
    print(f"   ✅ {len(registry.list_plugins())} plugins registered")

    # Create report generator and notifier
    report_gen = RunReportGenerator(base_dir=project_root / "runtime" / "reports")
    notifier = TelegramNotifier()

    def post_run_hook(*, run_id, metadata, module_results, outputs, events):
        outputs["_configured_equity"] = config.get("strategy", {}).get("account_equity", 100_000.0)
        report = report_gen.generate(run_id, metadata, module_results, outputs, events)
        report_gen.save(report)
        if notifier.enabled:
            notifier.send_run_summary(report)
            if report.trades and report.trades.fills:
                notifier.send_fills_summary(report)

    # Create orchestrator
    print("\n3. Creating orchestrator...")
    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
        report_hook=post_run_hook,
    )
    print("   ✅ Orchestrator created")

    # Build run config
    run_config = build_run_config(config, mode)

    # Pre-flight: verify IB Gateway connectivity (paper mode only)
    if mode is OperatingMode.PAPER:
        from ibkr_watchdog import ensure_ibkr_ready

        broker_cfg = config.get("execution", {}).get("broker", {})
        ibkr_ok = ensure_ibkr_ready(
            host=broker_cfg.get("host", "127.0.0.1"),
            port=broker_cfg.get("port", 4002),
            notifier=notifier,
        )
        if ibkr_ok:
            print("   ✅ IB Gateway connectivity verified")
        else:
            print("   ⚠️  IB Gateway unreachable — run will proceed in degraded mode")

    # Execute run
    print(f"\n{'=' * 70}")
    print("EXECUTING PRE_MARKET_FULL_SCAN")
    print('=' * 70)
    print(f"Mode: {mode.value}")
    print(f"Run Type: {RunType.PRE_MARKET_FULL_SCAN.value}")
    print()

    start_time = time.time()
    # Defensive handling: orchestrator.execute_run() may (in rare edge cases) crash with an
    # unhandled exception or return None; handle both to avoid masking the root cause.
    try:
        result = orchestrator.execute_run(run_config)
    except Exception as e:
        duration = time.time() - start_time
        tb = traceback.format_exc()
        notifier.send_p0_alert(
            run_id="unknown", stage="orchestrator",
            error=str(e), log_tail=tb[-500:],
        )
        print(f"\n{'=' * 70}")
        print("❌ CRITICAL ERROR: Orchestrator crashed with unhandled exception")
        print('=' * 70)
        print(f"Exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print(f"\nExecution time before crash: {duration:.2f}s")
        print("\nFull traceback:")
        traceback.print_exc()
        print('=' * 70)
        sys.exit(1)
    duration = time.time() - start_time

    # Get run_id from result
    run_id = "unknown"
    if result is None:
        # Defensive check: result should never be None; fail fast with a clear error.
        print("\n" + "=" * 70)
        print("❌ CRITICAL ERROR: orchestrator.execute_run() returned None")
        print("=" * 70)
        print("This should never happen. Check orchestrator implementation.")
        print(f"Execution time: {duration:.2f}s")
        print("=" * 70)
        sys.exit(1)
    elif hasattr(result, "data") and result.data and hasattr(result.data, 'metadata'):
        run_id = result.data.metadata.run_id

    # Print summary
    print_summary(result, run_id)

    print(f"⏱️  Total execution time: {duration:.2f}s")

    # Exit with appropriate code
    if result.status is ResultStatus.SUCCESS:
        print("\n✅ PRE_MARKET_FULL_SCAN completed successfully!")
        sys.exit(0)
    elif result.status is ResultStatus.DEGRADED:
        print("\n⚠️  PRE_MARKET_FULL_SCAN completed with warnings")
        sys.exit(0)  # Still exit 0 for degraded
    else:
        notifier.send_p0_alert(
            run_id=run_id, stage="result",
            error=str(result.error) if result.error else result.reason_code or "unknown",
        )
        print("\n❌ PRE_MARKET_FULL_SCAN failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
