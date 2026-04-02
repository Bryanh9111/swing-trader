#!/usr/bin/env python3
"""AST Daemon — long-lived process with persistent IBKR connection.

Replaces 4 independent cron jobs with a single daemon that keeps the IBKR
adapter connected all day.  Partial-fill callbacks fire in real-time (no more
4-hour blind windows between runs).

Schedule (US Eastern):
    09:00  PRE_MARKET_FULL_SCAN   — full pipeline
    10:30  INTRADAY_CHECK_1030    — execution-only
    14:30  INTRADAY_CHECK_1430    — execution-only
    16:40  AFTER_MARKET_RECON     — cancel open entries + daily report

Usage:
    python scripts/ast_daemon.py --paper
    python scripts/ast_daemon.py --paper --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / "config" / "secrets.env")

from common.config import SystemConfig
from common.events import InMemoryEventBus
from common.interface import Result, ResultStatus
from common.logging import StructlogLoggerFactory, init_logging
from execution.ibkr_adapter import IBKRAdapter
from execution.interface import BrokerConnectionConfig
from journal import ArtifactManager, JournalWriter
from journal.interface import OperatingMode, RunType
from notifier.telegram import TelegramNotifier
from order_state_machine import IDGenerator
from orchestrator import EODScanOrchestrator, register_real_plugins
from plugins.registry import PluginRegistry
from reporting.run_report import RunReportGenerator
from run_paper_eod_scan import build_run_config, check_trading_day, load_config, update_daily_cache

import structlog

logger = structlog.get_logger("ast_daemon")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = timezone(timedelta(hours=-5))

RUN_SCHEDULE: list[tuple[str, RunType]] = [
    ("09:00", RunType.PRE_MARKET_FULL_SCAN),
    ("10:30", RunType.INTRADAY_CHECK_1030),
    ("12:30", RunType.INTRADAY_CHECK_1230),
    ("14:30", RunType.INTRADAY_CHECK_1430),
    ("15:55", RunType.PRE_CLOSE_CLEANUP),
    ("16:40", RunType.AFTER_MARKET_RECON),
]

_shutdown = threading.Event()

# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------

def _build_adapter(config: dict) -> IBKRAdapter:
    """Create and connect an IBKRAdapter from the execution config block."""

    broker_cfg = config.get("execution", {}).get("broker", {})
    connection = BrokerConnectionConfig(
        host=str(broker_cfg.get("host", "127.0.0.1")),
        port=int(broker_cfg.get("port", 4002)),
        client_id=int(broker_cfg.get("client_id", 1)),
        readonly=bool(broker_cfg.get("readonly", False)),
        account=str(broker_cfg.get("account", "")).strip() or None,
        timeout=int(broker_cfg.get("timeout", 20)),
        max_reconnect_attempts=int(broker_cfg.get("max_reconnect_attempts", 5)),
        enable_dynamic_client_id=bool(broker_cfg.get("enable_dynamic_client_id", False)),
        client_id_range=(
            int(broker_cfg.get("client_id_range", [1, 32])[0]),
            int(broker_cfg.get("client_id_range", [1, 32])[1]),
        ),
    )
    adapter = IBKRAdapter(connection, IDGenerator())
    return adapter


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------

def _execute_single_run(
    config: dict,
    mode: OperatingMode,
    run_type: RunType,
    adapter: IBKRAdapter,
    notifier: TelegramNotifier,
) -> Result[Any]:
    """Build fresh orchestrator infrastructure and execute one run.

    The adapter is shared (daemon-owned), but all other objects
    (StateMachine, Reconciler, Persistence, etc.) are rebuilt per-run
    to avoid cross-run state pollution.
    """

    artifact_mgr = ArtifactManager(base_path=_PROJECT_ROOT / "journal")
    journal_writer = JournalWriter(artifact_mgr)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_real_plugins(registry)

    report_gen = RunReportGenerator(base_dir=_PROJECT_ROOT / "runtime" / "reports")

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

    run_config = build_run_config(config, mode)
    run_config["run_type"] = run_type.value

    # Inject shared adapter into execution config.
    run_config.setdefault("execution_real", {})["_shared_adapter"] = adapter
    # Also inject into the plugins → signals → execution_real config path
    # that the orchestrator uses for plugin configuration.
    for plugin_entry in run_config.get("plugins", {}).get("signals", []):
        if plugin_entry.get("name") == "execution_real":
            plugin_entry.setdefault("config", {})["_shared_adapter"] = adapter

    return orchestrator.execute_run(run_config)


# ---------------------------------------------------------------------------
# Health monitor thread
# ---------------------------------------------------------------------------

def _health_monitor(
    adapter: IBKRAdapter,
    broker_cfg: dict,
    notifier: TelegramNotifier,
    interval: int = 60,
) -> None:
    """Background thread: poll adapter health and attempt recovery on disconnect."""

    while not _shutdown.is_set():
        _shutdown.wait(timeout=interval)
        if _shutdown.is_set():
            break

        if adapter.is_healthy():
            logger.debug("health_check.ok")
            continue

        # 1) Try adapter built-in reconnect.
        logger.warning("health_check.disconnected")
        result = adapter.connect(force=True)
        if result.status is ResultStatus.SUCCESS:
            logger.info("health_check.reconnected")
            continue

        # 2) Attempt IB Gateway restart via watchdog.
        logger.warning("health_check.gateway_restart")
        if notifier.enabled:
            notifier.send("[AST] IBKR disconnected, restarting gateway...")

        try:
            from ibkr_watchdog import ensure_ibkr_ready

            ok = ensure_ibkr_ready(
                host=broker_cfg.get("host", "127.0.0.1"),
                port=broker_cfg.get("port", 4002),
                notifier=notifier,
            )
        except Exception:  # noqa: BLE001
            ok = False

        if ok:
            adapter.reset_reconnect_counter()
            result = adapter.connect(force=True)
            if result.status is ResultStatus.SUCCESS:
                logger.info("health_check.recovered")
                if notifier.enabled:
                    notifier.send("[AST] IBKR connection recovered")
                continue

        logger.error("health_check.recovery_failed")
        if notifier.enabled:
            notifier.send("[AST] IBKR recovery FAILED — manual intervention required")


# ---------------------------------------------------------------------------
# Scheduling loop
# ---------------------------------------------------------------------------

def _run_loop(
    schedule: list[tuple[str, RunType]],
    execute_fn,
) -> None:
    """Walk through today's schedule, sleeping between runs.

    Runs whose scheduled time has already passed are skipped.
    The loop exits after the last scheduled run or on shutdown signal.
    """

    remaining = list(schedule)

    while remaining and not _shutdown.is_set():
        time_str, run_type = remaining[0]
        h, m = map(int, time_str.split(":"))

        now = datetime.now(ET)
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)

        if now >= target:
            # Time already passed — execute immediately or skip.
            delta = (now - target).total_seconds()
            if delta > 300:
                # More than 5 minutes late — skip this run.
                logger.info("schedule.skipped", run_type=run_type.value, reason="past_due")
                remaining.pop(0)
                continue
            # Within 5-minute grace — execute now.
            remaining.pop(0)
            execute_fn(run_type)
            continue

        # Sleep until target time (interruptible).
        wait_seconds = (target - now).total_seconds()
        logger.info(
            "schedule.waiting",
            next_run=run_type.value,
            scheduled=time_str,
            wait_seconds=int(wait_seconds),
        )
        _shutdown.wait(timeout=wait_seconds)
        if _shutdown.is_set():
            break

        remaining.pop(0)
        execute_fn(run_type)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(signum, _frame):
    logger.info("daemon.shutdown_signal", signal=signum)
    _shutdown.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AST Daemon — persistent IBKR connection")
    parser.add_argument("--paper", action="store_true", help="Run in PAPER mode")
    parser.add_argument("--dry-run", action="store_true", help="Run in DRY_RUN mode")
    parser.add_argument("--equity", type=float, default=None, help="Override account equity (USD)")
    args = parser.parse_args()

    if args.paper:
        mode = OperatingMode.PAPER
        mode_str = "paper"
    elif args.dry_run:
        mode = OperatingMode.DRY_RUN
        mode_str = "dry_run"
    else:
        print("Error: Must specify either --paper or --dry-run")
        parser.print_help()
        sys.exit(1)

    # Trading day check.
    if not check_trading_day():
        print("Market closed today. Exiting.")
        sys.exit(0)

    # Load config.
    config = load_config(mode_str)
    if args.equity is not None:
        config.setdefault("strategy", {})["account_equity"] = args.equity
        config.setdefault("risk_gate", {})["account_equity"] = args.equity
        config["account_equity"] = args.equity

    # Initialise logging.
    sys_cfg = SystemConfig(
        version="daemon",
        log_level=config.get("system", {}).get("log_level", "INFO"),
        log_output=config.get("system", {}).get("log_output", "console"),
    )
    init_logging(sys_cfg)

    logger.info("daemon.starting", mode=mode.value)

    # Update daily cache (best-effort).
    try:
        update_daily_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("daemon.cache_update_failed", error=str(exc))

    # Build + connect adapter (one-time).
    notifier = TelegramNotifier()
    adapter = _build_adapter(config)

    # Pre-flight: ensure IB Gateway is reachable.
    broker_cfg = config.get("execution", {}).get("broker", {})
    if mode is OperatingMode.PAPER:
        try:
            from ibkr_watchdog import ensure_ibkr_ready

            ensure_ibkr_ready(
                host=broker_cfg.get("host", "127.0.0.1"),
                port=broker_cfg.get("port", 4002),
                notifier=notifier,
            )
        except Exception:  # noqa: BLE001
            pass

    connect_result = adapter.connect(force=True)
    if connect_result.status is ResultStatus.SUCCESS:
        logger.info("daemon.adapter_connected")
    else:
        logger.error("daemon.adapter_connect_failed", error=str(connect_result.error))
        if notifier.enabled:
            notifier.send(f"[AST] Daemon failed to connect to IBKR: {connect_result.error}")
        sys.exit(1)

    # Register signal handlers.
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Start health monitor thread.
    monitor_thread = threading.Thread(
        target=_health_monitor,
        args=(adapter, broker_cfg, notifier),
        kwargs={"interval": 60},
        daemon=True,
        name="health-monitor",
    )
    monitor_thread.start()
    logger.info("daemon.health_monitor_started")

    # Build the per-run executor closure.
    def execute_run(run_type: RunType) -> None:
        logger.info("run.starting", run_type=run_type.value)
        start = time.time()
        try:
            result = _execute_single_run(config, mode, run_type, adapter, notifier)
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start
            tb = traceback.format_exc()
            logger.error("run.crashed", run_type=run_type.value, error=str(exc), duration_s=round(duration, 1))
            if notifier.enabled:
                notifier.send_p0_alert(
                    run_id="daemon", stage=run_type.value,
                    error=str(exc), log_tail=tb[-500:],
                )
            return

        duration = time.time() - start
        if result is None:
            logger.error("run.returned_none", run_type=run_type.value)
            return

        run_id = "unknown"
        if hasattr(result, "data") and result.data and hasattr(result.data, "metadata"):
            run_id = result.data.metadata.run_id

        logger.info(
            "run.completed",
            run_type=run_type.value,
            status=result.status.value,
            run_id=run_id,
            duration_s=round(duration, 1),
        )

        if result.status is ResultStatus.FAILED and notifier.enabled:
            notifier.send_p0_alert(
                run_id=run_id, stage=run_type.value,
                error=str(result.error) if result.error else result.reason_code or "unknown",
            )

    # Run the schedule.
    logger.info("daemon.schedule_start", runs=len(RUN_SCHEDULE))
    _run_loop(RUN_SCHEDULE, execute_run)

    # Shutdown.
    logger.info("daemon.shutting_down")
    adapter.disconnect()
    logger.info("daemon.exited")


if __name__ == "__main__":
    main()
