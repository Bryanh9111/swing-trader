#!/usr/bin/env python3
"""Run AFTER_MARKET_RECON in paper trading mode.

This script runs the pre-close reconciliation:
- Cancel open entry orders (keep protective stops)
- Reconcile positions
- Generate daily report

The orchestrator already has a AFTER_MARKET_RECON short-circuit path
that skips universe/scanner/strategy/risk_gate and only runs execution.

Usage:
    # Dry run
    python scripts/run_paper_pre_close_recon.py --dry-run

    # Paper trading
    python scripts/run_paper_pre_close_recon.py --paper
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

from common.events import InMemoryEventBus
from common.interface import ResultStatus
from common.logging import StructlogLoggerFactory
from journal import ArtifactManager, JournalWriter
from journal.interface import OperatingMode, RunType
from notifier.telegram import TelegramNotifier
from orchestrator import EODScanOrchestrator, register_real_plugins
from plugins.registry import PluginRegistry
from reporting.run_report import RunReportGenerator

# Reuse config loading from eod_scan
from run_paper_eod_scan import load_config, build_run_config, print_summary, check_trading_day


def main():
    parser = argparse.ArgumentParser(description="Run AFTER_MARKET_RECON in paper trading mode")
    parser.add_argument("--dry-run", action="store_true", help="Run in DRY_RUN mode")
    parser.add_argument("--paper", action="store_true", help="Run in PAPER mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--equity", type=float, default=None, help="Override account equity (USD)")

    args = parser.parse_args()

    # Skip non-trading days (weekends + US holidays)
    if not check_trading_day():
        print("📅 Market closed today (weekend or US holiday). Skipping run.")
        sys.exit(0)

    # Determine mode
    if args.paper:
        mode = OperatingMode.PAPER
        mode_str = "paper"
    elif args.dry_run:
        mode = OperatingMode.DRY_RUN
        mode_str = "dry_run"
    else:
        print("Error: Must specify either --dry-run or --paper")
        parser.print_help()
        sys.exit(1)

    print(f"\nRunning AFTER_MARKET_RECON in {mode.value} mode")

    # Load configuration
    config = load_config(mode_str)

    if args.equity is not None:
        config.setdefault("strategy", {})["account_equity"] = args.equity
        config.setdefault("risk_gate", {})["account_equity"] = args.equity
        config["account_equity"] = args.equity

    # Build run config and override run_type
    run_config = build_run_config(config, mode)
    run_config["run_type"] = RunType.AFTER_MARKET_RECON.value

    # Create infrastructure
    artifact_manager = ArtifactManager(base_path=project_root / "journal")
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_real_plugins(registry)

    # Report + notifier
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

    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
        report_hook=post_run_hook,
    )

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

    # Execute
    start_time = time.time()
    try:
        result = orchestrator.execute_run(run_config)
    except Exception as e:
        duration = time.time() - start_time
        tb = traceback.format_exc()
        notifier.send_p0_alert(
            run_id="unknown", stage="orchestrator",
            error=str(e), log_tail=tb[-500:],
        )
        print(f"\nCRITICAL ERROR: {type(e).__name__}: {e}")
        print(f"Execution time before crash: {duration:.2f}s")
        traceback.print_exc()
        sys.exit(1)
    duration = time.time() - start_time

    if result is None:
        print("CRITICAL ERROR: orchestrator.execute_run() returned None")
        sys.exit(1)

    run_id = "unknown"
    if hasattr(result, "data") and result.data and hasattr(result.data, "metadata"):
        run_id = result.data.metadata.run_id

    print_summary(result, run_id)
    print(f"Total execution time: {duration:.2f}s")

    if result.status not in (ResultStatus.SUCCESS, ResultStatus.DEGRADED):
        notifier.send_p0_alert(
            run_id=run_id, stage="result",
            error=str(result.error) if result.error else result.reason_code or "unknown",
        )

    sys.exit(0 if result.status in (ResultStatus.SUCCESS, ResultStatus.DEGRADED) else 1)


if __name__ == "__main__":
    main()
