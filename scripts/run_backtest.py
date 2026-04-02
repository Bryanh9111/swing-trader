#!/usr/bin/env python3
"""Backtest runner for 2020-2024 historical validation.

Prerequisites:
    1. Ensure valid universe cache exists:
       - Run EOD scan once to build cache, OR
       - Build manually: python -m universe_builder.core
    2. S3 credentials configured in config/secrets.env
    3. **Data Availability**: Polygon/S3 data available from 2020-01-01 onwards (5-year limit)
       - Lookback window reduced to 90 days to avoid 2019 data requests
       - For testing 2020-01-02, fetches data from 2019-10-04 onwards (still might hit S3 limits)
       - **Recommended**: Start backtest from 2020-07-01 to ensure full 90-day lookback coverage

Usage:
    python scripts/run_backtest.py --start 2020-07-01 --end 2024-12-31
    python scripts/run_backtest.py --start 2024-01-01 --end 2024-03-31 --dry-run

Troubleshooting (init stalls):
    - Skip git subprocess calls (system version): set `AST_SYSTEM_VERSION=unknown`
    - Skip Polygon market-calendar HTTP checks: set `market_calendar.enabled: false` in `config/config.yaml`
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess

import yaml
from dotenv import load_dotenv

from calendar_data import MarketCalendar
from common.events import InMemoryEventBus
from common.interface import ResultStatus
from common.logging import StructlogLoggerFactory
from data.s3_adapter import S3DataAdapter
from journal import ArtifactManager, JournalWriter
from journal.interface import OperatingMode, RunType
from orchestrator import EODScanOrchestrator, register_real_plugins
from plugins.registry import PluginRegistry


def _get_git_commit_hash() -> str:
    """Return short git commit hash, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _clear_python_cache() -> int:
    """Clear Python bytecode cache to ensure fresh code execution.

    Removes all __pycache__ directories and .pyc files from the project.
    This prevents stale compiled code from affecting backtest results.

    Returns:
        Number of cache directories/files removed.
    """
    import shutil

    project_root = Path(__file__).parent.parent
    removed_count = 0

    # Remove __pycache__ directories
    for cache_dir in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(cache_dir)
            removed_count += 1
        except (PermissionError, OSError):
            pass  # Skip if cannot remove

    # Remove .pyc files
    for pyc_file in project_root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            removed_count += 1
        except (PermissionError, OSError):
            pass

    return removed_count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run historical backtest for Scanner parameter validation")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--mode",
        default="scan-only",
        choices=["scan-only", "full-backtest"],
        help="Run mode: scan-only (candidate counting) or full-backtest (positions + PnL)",
    )
    parser.add_argument("--initial-capital", type=float, default=2000.0, help="Initial capital for full-backtest mode")
    parser.add_argument("--max-hold-days", type=int, default=30, help="Max holding period (days) for full-backtest mode")
    parser.add_argument("--output", type=str, default="trades.csv", help="Output filename for trades CSV (default: trades.csv)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (validate setup only)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode: only show one line per day in terminal")
    parser.add_argument(
        "--regime-mode",
        default="none",
        choices=["none", "auto", "bull", "choppy", "bear"],
        help="Market regime mode: none (default), auto (detect daily), or force bull/choppy/bear",
    )
    return parser.parse_args(argv)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file and ensure it is a mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must be a mapping, got {type(loaded).__name__}")
    return loaded


def _force_s3_primary_source(config: dict[str, Any]) -> None:
    """Force S3 as the primary data source for historical backtests."""

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        data_cfg = {}
        config["data"] = data_cfg
    data_cfg["primary_source"] = "s3"


def _get_s3_config(config: dict[str, Any]) -> tuple[str, str, str, str]:
    """Resolve S3 endpoint/bucket/prefix/cache_dir from config.

    Preference order:
      1) config.data_sources.s3.{endpoint_url,bucket_name,path_prefix,local_cache_dir}
      2) config.data.{s3_endpoint_url,s3_bucket_name,s3_path_prefix}

    Returns:
        tuple: (endpoint_url, bucket_name, path_prefix, local_cache_dir)
    """

    data_sources_cfg = config.get("data_sources")
    if isinstance(data_sources_cfg, dict):
        s3_node = data_sources_cfg.get("s3")
        if isinstance(s3_node, dict):
            endpoint_url = str(s3_node.get("endpoint_url") or "").strip()
            bucket_name = str(s3_node.get("bucket_name") or "").strip()
            path_prefix = str(s3_node.get("path_prefix") or "").strip()
            local_cache_dir = str(s3_node.get("local_cache_dir") or ".cache/historical").strip()
            if endpoint_url and bucket_name and path_prefix:
                return endpoint_url, bucket_name, path_prefix, local_cache_dir

    data_cfg = config.get("data")
    if isinstance(data_cfg, dict):
        endpoint_url = str(data_cfg.get("s3_endpoint_url") or "").strip()
        bucket_name = str(data_cfg.get("s3_bucket_name") or "").strip()
        path_prefix = str(data_cfg.get("s3_path_prefix") or "").strip()
        local_cache_dir = ".cache/historical"
        return endpoint_url, bucket_name, path_prefix, local_cache_dir

    return "", "", "", ".cache/historical"


def _validate_s3_setup(config: dict[str, Any]) -> tuple[bool, str, str, str, str, str]:
    """Validate S3 credentials and required configuration fields."""

    access_key_id = os.getenv("S3_ACCESS_KEY_ID", "").strip()
    secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
    if not access_key_id or not secret_access_key:
        missing = []
        if not access_key_id:
            missing.append("S3_ACCESS_KEY_ID")
        if not secret_access_key:
            missing.append("S3_SECRET_ACCESS_KEY")
        return False, f"Missing required S3 credentials: {', '.join(missing)}", "", "", "", ".cache/historical"

    endpoint_url, bucket_name, path_prefix, local_cache_dir = _get_s3_config(config)
    missing_cfg = []
    if not endpoint_url:
        missing_cfg.append("endpoint_url")
    if not bucket_name:
        missing_cfg.append("bucket_name")
    if not path_prefix:
        missing_cfg.append("path_prefix")
    if missing_cfg:
        return (
            False,
            f"Missing required S3 configuration fields in config/config.yaml: {', '.join(missing_cfg)}",
            endpoint_url,
            bucket_name,
            path_prefix,
            local_cache_dir,
        )

    return True, "OK", endpoint_url, bucket_name, path_prefix, local_cache_dir


def _preflight_s3_connection(endpoint_url: str, bucket_name: str, path_prefix: str, local_cache_dir: str) -> None:
    """Test S3 connectivity (client init + head_bucket)."""

    adapter = S3DataAdapter(
        endpoint_url=endpoint_url,
        bucket_name=bucket_name,
        path_prefix=path_prefix,
        local_cache_dir=local_cache_dir,
    )
    s3 = adapter._get_s3_client()
    s3.head_bucket(Bucket=bucket_name)


def _get_trading_days(start_date: str, end_date: str, calendar: MarketCalendar) -> list[date]:
    """Get all trading days between start_date and end_date (inclusive).

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        calendar: MarketCalendar instance

    Returns:
        List of trading days (datetime.date objects) sorted in ascending order

    Raises:
        ValueError: If date format is invalid or start > end
    """

    try:
        start = datetime.strptime(str(start_date).strip(), "%Y-%m-%d").date()
        end = datetime.strptime(str(end_date).strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Invalid date format; expected YYYY-MM-DD for --start/--end") from exc

    if start > end:
        raise ValueError("Invalid date range: start_date must be <= end_date")

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


def _print_trading_days_stats(trading_days: list[date]) -> None:
    print("Trading days (US equities)")
    print(f"   Total trading days in range: {len(trading_days)}")
    if not trading_days:
        return
    sample_first = ", ".join(d.isoformat() for d in trading_days[:5])
    sample_last = ", ".join(d.isoformat() for d in trading_days[-5:])
    print(f"   First trading day: {trading_days[0].isoformat()}")
    print(f"   Last trading day:  {trading_days[-1].isoformat()}")
    print(f"   Sample (first 5):  {sample_first}")
    print(f"   Sample (last 5):   {sample_last}")


def _build_backtest_run_config(
    config: dict[str, Any],
    target_date: date,
    account_equity: float | None = None,
) -> dict[str, Any]:
    """Build run configuration for a single backtest date.

    Args:
        config: Loaded config.yaml content
        target_date: Trading date to backtest
        account_equity: Account equity for position sizing (defaults to 100_000.0 if not provided)

    Returns:
        Run configuration dict for EODScanOrchestrator

    Notes:
        - Forces S3 as primary data source
        - Sets start_date/end_date around target_date for data fetching
        - Enables full pipeline: Universe → Data → Scanner → Event Guard → Strategy → Risk Gate
        - account_equity is passed to Strategy plugin for correct position sizing
    """

    polygon_api_key = os.getenv("POLYGON_API_KEY")

    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    data_sources_cfg = config.get("data_sources") if isinstance(config.get("data_sources"), dict) else {}

    lookback_days_raw = data_cfg.get("lookback_days", 90) if isinstance(data_cfg, dict) else 90
    try:
        lookback_days = max(2, int(lookback_days_raw))
    except (TypeError, ValueError):
        lookback_days = 90

    start_date = target_date - timedelta(days=lookback_days)

    run_config: dict[str, Any] = {
        "mode": OperatingMode.DRY_RUN.value,
        "run_type": RunType.PRE_MARKET_FULL_SCAN.value,
        "market_calendar": config.get("market_calendar", {}),
        "universe_plugin": "universe_cached",
        "plugins": {
            "data_sources": [
                {
                    "name": "universe_cached",
                    "config": {
                        # Prefer a user-provided cached UniverseSnapshot if present, otherwise
                        # fall back to the yfinance cache produced by `universe.data_sources`.
                        "cache_path": ".cache/universe/universe.json",
                    },
                },
                {
                    "name": "data_real",
                    "config": {
                        **(data_cfg if isinstance(data_cfg, dict) else {}),
                        "data_sources": data_sources_cfg,
                        "primary_source": "s3",
                        "secondary_source": None,
                        "s3_enabled": True,
                        "polygon_api_key": polygon_api_key,
                        "polygon_unlimited": True,
                        "start_date": start_date.isoformat(),
                        "end_date": target_date.isoformat(),
                        # Disable incremental cache for backtests: the symbol-level
                        # cache (AAPL_1D.json) is shared across all processes and
                        # causes non-deterministic results when running multiple
                        # backtest windows in parallel.  The per-request cache
                        # (AAPL_1D_start_end.json) is date-scoped and safe.
                        "enable_incremental_cache": False,
                        "cache_dir": None,
                    },
                },
            ],
            "scanners": [
                {
                    "name": "scanner_real",
                    "config": config.get("scanner", {}) if isinstance(config.get("scanner"), dict) else {},
                },
            ],
            "strategies": [
                {
                    "name": "strategy_real",
                    "config": {
                        **(config.get("strategy", {}) if isinstance(config.get("strategy"), dict) else {}),
                        # Pass account_equity to Strategy plugin for correct position sizing
                        **({"account_equity": account_equity} if account_equity is not None else {}),
                    },
                },
            ],
            "risk_policies": [
                {
                    "name": "event_guard_real",
                    "config": config.get("event_guard", {}) if isinstance(config.get("event_guard"), dict) else {},
                },
                {
                    "name": "risk_gate_real",
                    "config": config.get("risk_gate", {}) if isinstance(config.get("risk_gate"), dict) else {},
                },
            ],
        },
        "event_guard_plugin": "event_guard_real",
        "strategy_plugin": "strategy_real",
        "risk_gate_plugin": "risk_gate_real",
    }

    # Pass through market_regime config for regime-aware scanning
    market_regime_cfg = config.get("market_regime")
    if not isinstance(market_regime_cfg, dict):
        scanner_cfg = config.get("scanner")
        if isinstance(scanner_cfg, dict):
            market_regime_cfg = scanner_cfg.get("market_regime")
    if isinstance(market_regime_cfg, dict):
        run_config["market_regime"] = dict(market_regime_cfg)

    return run_config


def _extract_candidates_count_from_journal(journal_path: Path) -> int:
    candidates_path = journal_path / "outputs" / "candidates.json"
    if candidates_path.exists():
        try:
            import json

            payload = json.loads(candidates_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                total_detected = payload.get("total_detected")
                if isinstance(total_detected, int):
                    return total_detected
                candidates = payload.get("candidates")
                if isinstance(candidates, list):
                    return len(candidates)
        except Exception:  # noqa: BLE001
            return 0

    events_path = journal_path / "events" / "events.jsonl"
    if events_path.exists():
        try:
            import json

            candidates = 0
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    continue
                if event.get("event_type") != "module.executed":
                    continue
                if str(event.get("module") or "").strip().lower() != "scanner":
                    continue
                data = event.get("data")
                if isinstance(data, dict) and isinstance(data.get("candidates"), int):
                    candidates = int(data["candidates"])
            return candidates
        except Exception:  # noqa: BLE001
            return 0

    return 0


def _run_single_day_backtest(
    target_date: date,
    config: dict[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run backtest for a single trading day.

    Args:
        target_date: Trading date to backtest
        config: Loaded config.yaml content
        verbose: Enable verbose logging

    Returns:
        Result dict with keys:
            - date: target_date
            - candidates_count: Number of detected candidates
            - success: bool (True if run completed)
            - error: Optional error message
            - journal_path: Path to journal output
    """

    artifact_base = project_root / "journal"
    existing_runs = {
        p.name
        for p in artifact_base.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    } if artifact_base.exists() else set()

    artifact_manager = ArtifactManager(base_path=artifact_base)
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_real_plugins(registry)

    run_config = _build_backtest_run_config(config, target_date)
    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )

    try:
        result = orchestrator.execute_run(run_config, run_type=RunType.PRE_MARKET_FULL_SCAN)
    except Exception as exc:  # noqa: BLE001
        return {
            "date": target_date,
            "candidates_count": 0,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "journal_path": None,
        }

    candidates_count = 0
    if result.data is not None:
        candidates_count = int(getattr(result.data, "candidates_count", 0) or 0)

    run_failed = result.status is ResultStatus.FAILED
    if run_failed and verbose and result.error is not None:
        traceback.print_exception(result.error)

    run_dir: Path | None = None
    if artifact_base.exists():
        after_runs = {
            p.name
            for p in artifact_base.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        }
        created = after_runs - existing_runs
        if created:
            run_dir = max((artifact_base / name for name in created), key=lambda p: p.stat().st_mtime)

    intents_count = 0
    risk_allows = 0
    risk_blocks = 0
    if run_dir is not None:
        candidates_count = _extract_candidates_count_from_journal(run_dir) or candidates_count

        # Extract additional outputs for full pipeline
        intents_path = run_dir / "outputs" / "intents.json"
        risk_decisions_path = run_dir / "outputs" / "risk_decisions.json"

        intents_count = 0
        if intents_path.exists():
            try:
                import json

                payload = json.loads(intents_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    # intents.json structure: {"intents": {"intent_groups": [...]}, ...}
                    intents_data = payload.get("intents", {})
                    if isinstance(intents_data, dict):
                        intent_groups = intents_data.get("intent_groups", [])
                    else:
                        intent_groups = payload.get("intent_groups", [])
                    if isinstance(intent_groups, list):
                        intents_count = len(intent_groups)
            except Exception:
                pass

        risk_allows = 0
        risk_blocks = 0
        if risk_decisions_path.exists():
            try:
                import json

                payload = json.loads(risk_decisions_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    decisions = payload.get("decisions", [])
                    if isinstance(decisions, list):
                        for decision in decisions:
                            if isinstance(decision, dict):
                                dtype = decision.get("decision_type", "")
                                if dtype == "ALLOW":
                                    risk_allows += 1
                                elif dtype == "BLOCK":
                                    risk_blocks += 1
            except Exception:
                pass

    return {
        "date": target_date,
        "candidates_count": candidates_count,
        "intents_count": intents_count,
        "risk_allows": risk_allows,
        "risk_blocks": risk_blocks,
        "success": not run_failed,
        "error": str(result.error) if run_failed and result.error is not None else None,
        "journal_path": str(run_dir) if run_dir is not None else None,
    }


def _run_full_backtest(
    trading_days: list[date],
    config: dict[str, Any],
    initial_capital: float,
    max_hold_days: int,
    verbose: bool,
    quiet: bool = False,
    trades_filename: str = "trades.csv",
    regime_mode: str = "none",
) -> None:
    execution_buffer_days = 5

    if not trading_days:
        print("No trading days provided; nothing to backtest.")
        return

    scan_start = trading_days[0]
    scan_end = trading_days[-1]
    scan_len = len(trading_days)

    if execution_buffer_days > 0:
        calendar = MarketCalendar()
        horizon_days = max(14, execution_buffer_days * 4)
        extra: list[date] = []
        attempts = 0
        while len(extra) < execution_buffer_days and attempts < 5:
            candidate_end = scan_end + timedelta(days=horizon_days)
            candidates = _get_trading_days(scan_end.isoformat(), candidate_end.isoformat(), calendar=calendar)
            extra = [d for d in candidates if d > scan_end][:execution_buffer_days]
            horizon_days *= 2
            attempts += 1

        if extra:
            trading_days = list(trading_days) + extra

    actual_end = trading_days[-1]
    print(f"Backtest period: {scan_start.isoformat()} to {scan_end.isoformat()} (scan)")
    added_days = max(0, len(trading_days) - scan_len)
    print(f"Extended to: {actual_end.isoformat()} (+{added_days} days for order execution)")

    # Initialize components
    artifact_manager = ArtifactManager(base_path=project_root / "journal")
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_real_plugins(registry)

    eod_orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )

    # Create BacktestOrchestrator
    from backtest.orchestrator import BacktestOrchestrator
    from portfolio.interface import CapitalAllocationConfig, DEFAULT_SIZING_TIERS, DynamicSizingTier
    from portfolio.position_rotator import RotationConfig

    # Find applicable sizing tier based on initial capital
    def _find_sizing_tier(equity: float, tiers: tuple[DynamicSizingTier, ...]) -> DynamicSizingTier:
        """Find the applicable sizing tier for the given total equity."""
        applicable = tiers[0]
        for tier in tiers:
            if tier.equity_threshold <= equity:
                applicable = tier
            else:
                break
        return applicable

    current_tier = _find_sizing_tier(initial_capital, DEFAULT_SIZING_TIERS)
    print(f"Dynamic sizing tier: ${current_tier.equity_threshold:,.0f}+ "
          f"(positions: {current_tier.max_positions}, "
          f"per_position: ${current_tier.min_per_position}-${current_tier.max_per_position}, "
          f"new/day: {current_tier.max_new_positions_per_day}, "
          f"rotate/day: {current_tier.max_positions_to_rotate}, "
          f"min_score: {current_tier.min_score_threshold:.0%})")

    # Load capital allocation config if enabled
    capital_allocation_config = None
    cap_alloc_cfg = config.get("capital_allocation", {})
    if isinstance(cap_alloc_cfg, dict) and cap_alloc_cfg.get("enabled", False):
        def _optional_float(value: object) -> float | None:
            if value is None:
                return None
            return float(value)

        capital_allocation_config = CapitalAllocationConfig(
            max_positions=current_tier.max_positions,  # Use tier value
            max_exposure_pct=float(cap_alloc_cfg.get("max_exposure_pct", 0.85)),
            min_position_pct=float(cap_alloc_cfg.get("min_position_pct", 0.025)),
            max_position_pct=float(cap_alloc_cfg.get("max_position_pct", 0.10)),
            base_position_pct=float(cap_alloc_cfg.get("base_position_pct", 0.05)),
            cash_reserve_pct=float(cap_alloc_cfg.get("cash_reserve_pct", 0.10)),
            opportunity_reserve_pct=float(cap_alloc_cfg.get("opportunity_reserve_pct", 0.05)),
            min_position_value=float(cap_alloc_cfg.get("min_position_value", 50.0)),
            max_commission_ratio=float(cap_alloc_cfg.get("max_commission_ratio", 0.02)),
            commission_per_trade=float(cap_alloc_cfg.get("commission_per_trade", 1.0)),
            sizing_policy=str(cap_alloc_cfg.get("sizing_policy", "tier_dollar")),
            target_position_pct=float(cap_alloc_cfg.get("target_position_pct", 0.06)),
            hard_max_position_value=_optional_float(cap_alloc_cfg.get("hard_max_position_value")),
            hard_min_position_value=_optional_float(cap_alloc_cfg.get("hard_min_position_value")),
        )
        print(f"Capital allocation enabled: max_exposure={capital_allocation_config.max_exposure_pct:.0%}, "
              f"reserves={capital_allocation_config.cash_reserve_pct + capital_allocation_config.opportunity_reserve_pct:.0%}")

    # Load rotation config if enabled (use tier values for dynamic params)
    rotation_config = None
    rotation_cfg = config.get("position_rotation", {})
    if isinstance(rotation_cfg, dict) and rotation_cfg.get("enabled", False):
        rotation_config = RotationConfig(
            enabled=True,
            min_new_opportunity_score=float(rotation_cfg.get("min_new_opportunity_score", 0.95)),
            max_positions_to_rotate=current_tier.max_positions_to_rotate,  # Use tier value
            min_health_score_to_keep=float(rotation_cfg.get("min_health_score_to_keep", 0.30)),
            require_pnl_loss=bool(rotation_cfg.get("require_pnl_loss", True)),
            min_loss_pct_to_replace=float(rotation_cfg.get("min_loss_pct_to_replace", -0.03)),
        )
        print(
            f"Position rotation enabled: min_opportunity_score={rotation_config.min_new_opportunity_score:.0%}, "
            f"max_rotate={rotation_config.max_positions_to_rotate}"
        )

    # V27.5: Load candidate_filter config from regime YAML
    _cf_config: dict[str, Any] | None = None
    if regime_mode not in ("none", ""):
        _regime_map = {"auto": "bull_market", "bull": "bull_market", "choppy": "choppy_market", "bear": "bear_market"}
        _regime_yaml_name = _regime_map.get(regime_mode, "bull_market")
        _regime_yaml_path = project_root / "config" / "regimes" / f"{_regime_yaml_name}.yaml"
        if _regime_yaml_path.exists():
            try:
                _regime_raw = _load_yaml_mapping(_regime_yaml_path)
                _cf_config = _regime_raw.get("candidate_filter")
                if isinstance(_cf_config, dict) and _cf_config:
                    print(f"Candidate filter config: ATR% range "
                          f"[{_cf_config.get('trend_atr_pct_min', 'N/A')}, "
                          f"{_cf_config.get('trend_atr_pct_max', 'N/A')}]")
                else:
                    _cf_config = None
            except Exception:  # noqa: BLE001
                _cf_config = None

    backtest_orch = BacktestOrchestrator(
        eod_orchestrator=eod_orchestrator,
        initial_capital=initial_capital,
        max_hold_days=max_hold_days,
        output_dir="backtest_results",
        quiet=quiet,
        regime_mode=regime_mode,
        capital_allocation_config=capital_allocation_config,
        rotation_config=rotation_config,
        candidate_filter_config=_cf_config,
    )

    # Prepare run_config (reuses _build_backtest_run_config logic)
    # Pass initial_capital as account_equity for correct position sizing
    run_config = _build_backtest_run_config(config, trading_days[0], account_equity=initial_capital)

    # Run backtest
    stats = backtest_orch.run_backtest(trading_days, run_config, trades_filename=trades_filename)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Trades: {stats.total_trades}")
    print(f"Win Rate: {stats.win_rate * 100:.2f}%")
    print(f"Total PnL: ${stats.total_pnl:.2f} ({stats.total_pnl_pct * 100:.2f}%)")
    print(f"Avg Win: ${stats.avg_win:.2f}")
    print(f"Avg Loss: ${stats.avg_loss:.2f}")
    print(f"Max Win: ${stats.max_win:.2f}")
    print(f"Max Loss: ${stats.max_loss:.2f}")
    print(f"Avg Hold Days: {stats.avg_hold_days:.1f}")
    print("=" * 60)
    print(f"Trades exported to: backtest_results/{trades_filename}")
    print("=" * 60)


def _run_scan_only_backtest(
    trading_days: list[date],
    config: dict[str, Any],
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run backtest for all trading days in the range.

    Args:
        trading_days: List of trading days to backtest
        config: Loaded config.yaml content
        verbose: Enable verbose logging

    Returns:
        List of backtest results (one per trading day)
    """

    results: list[dict[str, Any]] = []
    if not trading_days:
        print("No trading days in range; nothing to backtest.")
        return results

    total_days = len(trading_days)
    start_day = trading_days[0].isoformat()
    end_day = trading_days[-1].isoformat()

    print(f"Historical Backtest ({start_day} to {end_day})")
    print(f"Total trading days: {total_days}")
    print("=" * 72)

    failed_count = 0
    error_types: dict[str, int] = {}

    def classify_error(error: str | None) -> str:
        if not error:
            return "UNKNOWN"
        if "NO_DATA" in error:
            return "NO_DATA"
        if ":" in error:
            return error.split(":", 1)[0].strip() or "UNKNOWN"
        return error.strip().split()[0] if error.strip() else "UNKNOWN"

    for idx, trading_day in enumerate(trading_days, start=1):
        try:
            if verbose:
                result = _run_single_day_backtest(trading_day, config, verbose=True)
            else:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    result = _run_single_day_backtest(trading_day, config, verbose=False)
        except Exception as exc:  # noqa: BLE001
            result = {
                "date": trading_day,
                "candidates_count": 0,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "journal_path": None,
            }

        results.append(result)

        candidates_count = int(result.get("candidates_count") or 0)
        if result.get("success"):
            print(f"[{idx}/{total_days}] {trading_day.isoformat()} ... ✓ {candidates_count} candidates")
        else:
            failed_count += 1
            error_type = classify_error(result.get("error"))
            error_types[error_type] = error_types.get(error_type, 0) + 1
            error_msg = result.get("error") or "UNKNOWN"
            print(f"[{idx}/{total_days}] {trading_day.isoformat()} ... ✗ Error: {error_msg}")

    if failed_count and verbose:
        print("\nError summary (verbose)")
        print(f"Failed runs: {failed_count}/{total_days}")
        for error_type, count in sorted(error_types.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {error_type}: {count}")
        print("\nFailed days:")
        for result in results:
            if result.get("success"):
                continue
            day = result["date"].isoformat() if isinstance(result.get("date"), date) else str(result.get("date"))
            print(f"  - {day}: {result.get('error') or 'UNKNOWN'}")

    return results


def _generate_backtest_report(
    results: list[dict[str, Any]],
    start_date: str,
    end_date: str,
    output_path: Path | None = None,
) -> str:
    """Generate backtest summary report with statistics.

    Args:
        results: List of backtest results from _run_scan_only_backtest()
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        output_path: Optional output path for saving report

    Returns:
        Report content as string
    """

    def normalize_date(value: Any) -> date | None:
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return datetime.strptime(value.strip(), "%Y-%m-%d").date()
            except ValueError:
                return None
        return None

    def normalize_bool(value: Any) -> bool:
        return bool(value is True)

    def normalize_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def classify_error(error: Any) -> str:
        err = str(error or "").strip()
        if not err or err.lower() == "none":
            return "UNKNOWN"
        if "NO_DATA" in err:
            return "NO_DATA"
        if ":" in err:
            head = err.split(":", 1)[0].strip()
            return head.upper() if head else "UNKNOWN"
        head = err.split()[0].strip()
        return head.upper() if head else "UNKNOWN"

    def percentile(values: list[int], p: float) -> float | None:
        if not values:
            return None
        if p <= 0:
            return float(min(values))
        if p >= 100:
            return float(max(values))
        xs = sorted(values)
        k = (len(xs) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(xs) - 1)
        if f == c:
            return float(xs[f])
        d0 = xs[f] * (c - k)
        d1 = xs[c] * (k - f)
        return float(d0 + d1)

    normalized: list[dict[str, Any]] = []
    for r in results:
        normalized.append(
            {
                "date": normalize_date(r.get("date")),
                "date_raw": r.get("date"),
                "candidates_count": normalize_int(r.get("candidates_count")),
                "success": normalize_bool(r.get("success")),
                "error": r.get("error"),
                "journal_path": r.get("journal_path"),
            }
        )

    total_trading_days = len(normalized)
    successful_runs = sum(1 for r in normalized if r["success"])
    failed_runs = total_trading_days - successful_runs
    total_candidates = sum(r["candidates_count"] for r in normalized if r["success"])
    avg_candidates_per_day = (total_candidates / total_trading_days) if total_trading_days else 0.0
    avg_candidates_per_successful_day = (total_candidates / successful_runs) if successful_runs else 0.0

    successful_candidate_counts = [r["candidates_count"] for r in normalized if r["success"]]
    max_day: dict[str, Any] | None = None
    min_day: dict[str, Any] | None = None
    if successful_candidate_counts:
        max_day = max((r for r in normalized if r["success"]), key=lambda x: x["candidates_count"])
        min_day = min((r for r in normalized if r["success"]), key=lambda x: x["candidates_count"])

    zero_candidate_days = [r for r in normalized if r["success"] and r["candidates_count"] == 0]
    days_with_10_plus = [r for r in normalized if r["success"] and r["candidates_count"] >= 10]

    error_types: dict[str, int] = {}
    for r in normalized:
        if r["success"]:
            continue
        error_type = classify_error(r.get("error"))
        error_types[error_type] = error_types.get(error_type, 0) + 1

    def date_label(r: dict[str, Any]) -> str:
        d = r.get("date")
        if isinstance(d, date):
            return d.isoformat()
        raw = r.get("date_raw")
        return str(raw) if raw is not None else "UNKNOWN_DATE"

    top_days = sorted(
        (r for r in normalized if r["success"]),
        key=lambda x: (x["candidates_count"], date_label(x)),
        reverse=True,
    )[:10]

    # Temporal distributions
    by_year: dict[int, dict[str, int]] = {}
    by_month: dict[str, dict[str, int]] = {}
    for r in normalized:
        d = r.get("date")
        if not isinstance(d, date):
            continue
        year = d.year
        month = f"{d.year:04d}-{d.month:02d}"

        year_bucket = by_year.setdefault(year, {"days": 0, "successful": 0, "candidates": 0})
        year_bucket["days"] += 1
        if r["success"]:
            year_bucket["successful"] += 1
            year_bucket["candidates"] += r["candidates_count"]

        month_bucket = by_month.setdefault(month, {"days": 0, "successful": 0, "candidates": 0})
        month_bucket["days"] += 1
        if r["success"]:
            month_bucket["successful"] += 1
            month_bucket["candidates"] += r["candidates_count"]

    p50 = percentile(successful_candidate_counts, 50)
    p75 = percentile(successful_candidate_counts, 75)
    p90 = percentile(successful_candidate_counts, 90)
    p95 = percentile(successful_candidate_counts, 95)

    def fmt_pct(n: int, d: int) -> str:
        if d <= 0:
            return "0.0%"
        return f"{(100.0 * n / d):.1f}%"

    lines: list[str] = []
    lines.append("# Backtest Summary Report")
    lines.append(f"Date Range: {start_date} to {end_date}")
    lines.append(f"Total Trading Days: {total_trading_days}")
    lines.append("")

    lines.append("## Overall Statistics")
    lines.append(f"- Successful Runs: {successful_runs}/{total_trading_days} ({fmt_pct(successful_runs, total_trading_days)})")
    lines.append(f"- Failed Runs: {failed_runs}/{total_trading_days} ({fmt_pct(failed_runs, total_trading_days)})")
    lines.append(f"- Total Candidates Detected (successful days only): {total_candidates:,}")
    lines.append(f"- Average Candidates/Trading Day: {avg_candidates_per_day:.2f}")
    lines.append(f"- Average Candidates/Successful Day: {avg_candidates_per_successful_day:.2f}")
    lines.append("")

    lines.append("## Candidate Statistics")
    if max_day is not None and min_day is not None:
        lines.append(f"- Max Candidates (Single Day): {max_day['candidates_count']} ({date_label(max_day)})")
        lines.append(f"- Min Candidates (Single Day): {min_day['candidates_count']} ({date_label(min_day)})")
    else:
        lines.append("- Max Candidates (Single Day): N/A")
        lines.append("- Min Candidates (Single Day): N/A")
    lines.append(
        f"- Days with 0 Candidates (successful runs): {len(zero_candidate_days)} ({fmt_pct(len(zero_candidate_days), total_trading_days)})"
    )
    lines.append(
        f"- Days with 10+ Candidates (successful runs): {len(days_with_10_plus)} ({fmt_pct(len(days_with_10_plus), total_trading_days)})"
    )
    lines.append("")

    lines.append("## Candidate Count Percentiles (Successful Runs)")
    lines.append(f"- p50: {p50:.2f}" if p50 is not None else "- p50: N/A")
    lines.append(f"- p75: {p75:.2f}" if p75 is not None else "- p75: N/A")
    lines.append(f"- p90: {p90:.2f}" if p90 is not None else "- p90: N/A")
    lines.append(f"- p95: {p95:.2f}" if p95 is not None else "- p95: N/A")
    lines.append("")

    lines.append("## Temporal Distribution (Yearly)")
    lines.append("| Year | Trading Days | Successful Runs | Candidates | Avg Candidates/Trading Day |")
    lines.append("|---:|---:|---:|---:|---:|")
    for year in sorted(by_year.keys()):
        bucket = by_year[year]
        avg = (bucket["candidates"] / bucket["days"]) if bucket["days"] else 0.0
        lines.append(
            f"| {year} | {bucket['days']} | {bucket['successful']} | {bucket['candidates']:,} | {avg:.2f} |"
        )
    lines.append("")

    lines.append("## Temporal Distribution (Monthly)")
    lines.append("| Month | Trading Days | Successful Runs | Candidates | Avg Candidates/Trading Day |")
    lines.append("|---:|---:|---:|---:|---:|")
    for month in sorted(by_month.keys()):
        bucket = by_month[month]
        avg = (bucket["candidates"] / bucket["days"]) if bucket["days"] else 0.0
        lines.append(
            f"| {month} | {bucket['days']} | {bucket['successful']} | {bucket['candidates']:,} | {avg:.2f} |"
        )
    lines.append("")

    lines.append("## Error Analysis (Failed Runs)")
    if failed_runs == 0:
        lines.append("- No failed runs.")
    else:
        for error_type, count in sorted(error_types.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {error_type}: {count} occurrences ({fmt_pct(count, failed_runs)})")
    lines.append("")

    lines.append("## Top 10 Days by Candidates (Successful Runs)")
    if not top_days:
        lines.append("- N/A")
    else:
        lines.append("| Rank | Date | Candidates | Journal Path |")
        lines.append("|---:|---:|---:|---|")
        for idx, r in enumerate(top_days, start=1):
            jp = str(r.get("journal_path") or "").strip()
            lines.append(f"| {idx} | {date_label(r)} | {r['candidates_count']} | {jp} |")
    lines.append("")

    lines.append("## Zero-Candidate Days (Successful Runs)")
    if not zero_candidate_days:
        lines.append("- None")
    else:
        for r in zero_candidate_days:
            lines.append(f"- {date_label(r)}")
    lines.append("")

    report_content = "\n".join(lines).rstrip() + "\n"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_content, encoding="utf-8")

        csv_path = output_path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["date", "candidates_count", "success", "error", "journal_path"],
            )
            writer.writeheader()
            for r in normalized:
                writer.writerow(
                    {
                        "date": date_label(r),
                        "candidates_count": r["candidates_count"],
                        "success": r["success"],
                        "error": str(r.get("error") or ""),
                        "journal_path": str(r.get("journal_path") or ""),
                    }
                )

    return report_content


def main(argv: list[str] | None = None) -> int:
    """Script entrypoint."""

    os.chdir(project_root)

    # Clear Python bytecode cache to ensure fresh code execution
    cache_cleared = _clear_python_cache()
    if cache_cleared > 0:
        print(f"🧹 Cleared {cache_cleared} Python cache entries")

    args = _parse_args(argv)

    config_dir = project_root / "config"
    config_path = config_dir / "config.yaml"
    dotenv_path = config_dir / "secrets.env"

    if not dotenv_path.exists():
        print(f"❌ Missing dotenv file: {dotenv_path}")
        print("   Create it from `config/secrets.env.example` and fill credentials.")
        return 2

    load_dotenv(dotenv_path=dotenv_path, override=False)

    try:
        config = _load_yaml_mapping(config_path)
    except Exception as exc:  # noqa: BLE001
        print("❌ Failed to load config/config.yaml")
        print(f"   Error: {exc}")
        if args.verbose:
            print(traceback.format_exc().rstrip())
        return 2

    _force_s3_primary_source(config)

    ok, reason, endpoint_url, bucket_name, path_prefix, local_cache_dir = _validate_s3_setup(config)
    if not ok:
        print("❌ S3 setup validation failed")
        print(f"   Error: {reason}")
        return 2

    if args.dry_run:
        print("Dry run: validating S3 connectivity...")
        print(f"   endpoint_url: {endpoint_url}")
        print(f"   bucket_name:  {bucket_name}")
        print(f"   path_prefix:  {path_prefix}")
        try:
            _preflight_s3_connection(
                endpoint_url=endpoint_url,
                bucket_name=bucket_name,
                path_prefix=path_prefix,
                local_cache_dir=local_cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print("❌ S3 connection preflight failed")
            print(f"   Error type: {type(exc).__name__}")
            print(f"   Error: {exc}")
            if args.verbose:
                print(traceback.format_exc().rstrip())
            return 1

        print("✅ S3 connection preflight OK")
        try:
            trading_days = _get_trading_days(args.start, args.end, calendar=MarketCalendar())
        except Exception as exc:  # noqa: BLE001
            print("❌ Failed to compute trading days for date range")
            print(f"   Error: {exc}")
            if args.verbose:
                print(traceback.format_exc().rstrip())
            return 2

        print()
        _print_trading_days_stats(trading_days)
        return 0

    try:
        trading_days = _get_trading_days(args.start, args.end, calendar=MarketCalendar())
    except Exception as exc:  # noqa: BLE001
        print("❌ Failed to compute trading days for date range")
        print(f"   Error: {exc}")
        if args.verbose:
            print(traceback.format_exc().rstrip())
        return 2

    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    primary_source = data_cfg.get("primary_source") if isinstance(data_cfg, dict) else None

    git_hash = _get_git_commit_hash()

    print("=" * 72)
    print(f"Historical Backtest (Bootstrap)  commit: {git_hash}")
    print("=" * 72)
    _print_trading_days_stats(trading_days)
    print("=" * 72)
    print(f"date_range.start: {args.start}")
    print(f"date_range.end:   {args.end}")
    print(f"data.primary_source (forced): {primary_source}")
    print(f"s3.endpoint_url: {endpoint_url}")
    print(f"s3.bucket_name:  {bucket_name}")
    print(f"s3.path_prefix:  {path_prefix}")
    print(f"s3.local_cache_dir: {local_cache_dir}")

    if args.verbose:
        print("\nLoaded config keys:")
        for key in sorted(config.keys()):
            print(f"  - {key}")

    if not trading_days:
        print("\nNo trading days in range; nothing to backtest.")
        return 0

    from scanner.filters.stats import reset_global_filter_stats

    reset_global_filter_stats()

    # Inject market_regime settings based on --regime-mode argument
    # This must be done BEFORE calling _run_full_backtest or _run_scan_only_backtest
    # so that _build_backtest_run_config can include the market_regime config
    if args.regime_mode != "none":
        market_regime = config.setdefault("market_regime", {})
        if isinstance(market_regime, dict):
            market_regime["enabled"] = True
            market_regime["mode"] = args.regime_mode

    if args.mode == "full-backtest":
        regime_info = f", regime={args.regime_mode}" if args.regime_mode != "none" else ""
        print(f"\nRunning FULL BACKTEST mode (initial capital: ${args.initial_capital}{regime_info})")
        _run_full_backtest(
            trading_days=trading_days,
            config=config,
            initial_capital=args.initial_capital,
            max_hold_days=args.max_hold_days,
            verbose=args.verbose,
            quiet=args.quiet,
            trades_filename=args.output,
            regime_mode=args.regime_mode,
        )
        from scanner.filters.stats import get_global_filter_stats

        print()
        print(get_global_filter_stats().summary_text())
        return 0

    regime_info = f", regime={args.regime_mode}" if args.regime_mode != "none" else ""
    print(f"\nRunning SCAN-ONLY mode (candidate counting{regime_info})")

    print("\n" + "=" * 72)
    print(f"Starting historical backtest ({trading_days[0].isoformat()} to {trading_days[-1].isoformat()})")
    print("=" * 72)

    results = _run_scan_only_backtest(trading_days, config, verbose=args.verbose)

    print("\n" + "=" * 72)
    print("Backtest Complete")
    print("=" * 72)
    print(f"Total days processed: {len(results)}")
    print(f"Successful runs: {sum(1 for r in results if r.get('success'))}")
    print(f"Failed runs: {sum(1 for r in results if not r.get('success'))}")
    print(f"Total candidates detected: {sum(int(r.get('candidates_count') or 0) for r in results)}")

    # Step 5: Generate and save backtest report
    report_output = project_root / "backtest_results" / f"backtest_{args.start}_to_{args.end}.md"
    report_output.parent.mkdir(parents=True, exist_ok=True)

    report_content = _generate_backtest_report(
        results=results,
        start_date=args.start,
        end_date=args.end,
        output_path=report_output,
    )

    print(f"\n📊 Backtest report saved to: {report_output}")
    print(f"📄 Backtest CSV saved to: {report_output.with_suffix('.csv')}")
    if args.verbose:
        print("\n" + "=" * 72)
        print("Report Preview")
        print("=" * 72)
        print(report_content)

    from scanner.filters.stats import get_global_filter_stats

    print()
    print(get_global_filter_stats().summary_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
