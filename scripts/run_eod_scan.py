"""CLI runner for PRE_MARKET_FULL_SCAN orchestrator (Phase 1.5).

This script wires up the common infrastructure components (Journal, EventBus,
LoggerFactory, PluginRegistry), registers the real plugins, and executes a
single PRE_MARKET_FULL_SCAN run via `EODScanOrchestrator.execute_run()`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.events import InMemoryEventBus  # noqa: E402
from common.interface import DomainEvent, ResultStatus  # noqa: E402
from common.logging import StructlogLoggerFactory, init_logging  # noqa: E402
from journal import ArtifactManager, JournalWriter  # noqa: E402
from orchestrator import EODScanOrchestrator, register_real_plugins  # noqa: E402
from plugins.registry import PluginRegistry  # noqa: E402


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except Exception:
        return {}
    if not isinstance(loaded, Mapping):
        return {}
    return dict(loaded)


def _build_scanner_windows(scanner_cfg: Mapping[str, Any]) -> list[int]:
    min_days = scanner_cfg.get("min_platform_days")
    max_days = scanner_cfg.get("max_platform_days")
    try:
        min_val = int(min_days)
        max_val = int(max_days)
    except Exception:
        return [20, 30, 60]

    if min_val <= 0 or max_val <= 0:
        return [20, 30, 60]
    if max_val < min_val:
        min_val, max_val = max_val, min_val

    mid = int((min_val + max_val) // 2)
    windows = sorted({min_val, mid, max_val})
    return windows or [20, 30, 60]


def _read_candidates(candidate_path: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    if not candidate_path.exists():
        return []
    try:
        import msgspec.json

        raw = msgspec.json.decode(candidate_path.read_bytes())
    except Exception:
        return []

    if not isinstance(raw, Mapping):
        return []
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        return []

    parsed: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        symbol = item.get("symbol")
        score = item.get("score")
        window = item.get("window")
        if not isinstance(symbol, str) or not symbol.strip():
            continue
        try:
            score_val = float(score)
        except Exception:
            continue
        try:
            window_val = int(window)
        except Exception:
            window_val = 0
        parsed.append({"symbol": symbol.strip().upper(), "score": score_val, "window": window_val})

    parsed.sort(key=lambda row: (-row["score"], row["symbol"], row["window"]))
    return parsed[: max(0, int(limit))]


def _print_summary(
    *,
    run_id: str | None,
    mode: str,
    status: ResultStatus,
    duration_ms: int | None,
    universe_symbols: int | None,
    data_fetched: int | None,
    data_failed: int | None,
    candidates_detected: int | None,
    top_candidates: list[dict[str, Any]],
    run_dir: Path | None,
) -> None:
    status_label = status.value.upper()
    duration_s = (duration_ms or 0) / 1000.0

    print("========================================")
    print("PRE_MARKET_FULL_SCAN Run Summary")
    print("========================================")
    print(f"Run ID: {run_id or 'UNKNOWN'}")
    print(f"Mode: {mode}")
    print(f"Status: {status_label}")
    print(f"Duration: {duration_s:.1f}s")
    print("")

    if universe_symbols is not None:
        print(f"Universe: {universe_symbols} symbols")
    else:
        print("Universe: (unknown)")

    if data_fetched is not None and data_failed is not None:
        print(f"Data: {data_fetched} symbols fetched ({data_failed} failed)")
    elif data_fetched is not None:
        print(f"Data: {data_fetched} symbols fetched")
    else:
        print("Data: (unknown)")

    if candidates_detected is not None:
        print(f"Scanner: {candidates_detected} candidates detected")
    else:
        print("Scanner: (unknown)")
    print("")

    if top_candidates:
        print("Top Candidates:")
        for idx, row in enumerate(top_candidates, start=1):
            symbol = row.get("symbol", "UNKNOWN")
            score = float(row.get("score", 0.0))
            window = int(row.get("window", 0))
            window_label = f"{window} day window" if window > 0 else "unknown window"
            print(f"{idx}. {symbol} - Score: {score:.2f} (platform detected, {window_label})")
        print("")

    if run_dir is not None:
        print(f"Artifacts saved to: {run_dir.as_posix()}/")
    print("========================================")


def _print_error(error: BaseException) -> None:
    print("========================================", file=sys.stderr)
    print("PRE_MARKET_FULL_SCAN Run Error", file=sys.stderr)
    print("========================================", file=sys.stderr)
    print(str(error), file=sys.stderr)
    print("========================================", file=sys.stderr)


def _build_orchestrator_config(base_config: Mapping[str, Any], mode: str) -> dict[str, Any]:
    universe_builder = base_config.get("universe_builder")
    universe_filters: Mapping[str, Any] = {}
    if isinstance(universe_builder, Mapping):
        filters = universe_builder.get("filters")
        if isinstance(filters, Mapping):
            universe_filters = filters

    data_sources = base_config.get("data_sources")
    data_sources_map: Mapping[str, Any] = data_sources if isinstance(data_sources, Mapping) else {}

    scanner_cfg = base_config.get("scanner")
    scanner_map: Mapping[str, Any] = scanner_cfg if isinstance(scanner_cfg, Mapping) else {}

    strategy_cfg = base_config.get("strategy")
    strategy_map: Mapping[str, Any] = strategy_cfg if isinstance(strategy_cfg, Mapping) else {}

    risk_gate_cfg = base_config.get("risk_gate")
    risk_gate_map: Mapping[str, Any] = risk_gate_cfg if isinstance(risk_gate_cfg, Mapping) else {}

    lookback_days = scanner_map.get("lookback_days", 120)
    try:
        lookback_days_int = int(lookback_days)
    except Exception:
        lookback_days_int = 120

    windows = _build_scanner_windows(scanner_map)
    weight = 1.0 / max(1, len(windows))
    window_weights = {int(w): weight for w in windows}

    scanner_config: dict[str, Any] = {"windows": windows, "window_weights": window_weights}
    if "consolidation_threshold" in scanner_map:
        scanner_config["box_threshold"] = scanner_map.get("consolidation_threshold")
    if "volume_spike_threshold" in scanner_map:
        scanner_config["volume_increase_threshold"] = scanner_map.get("volume_spike_threshold")

    polygon_api_key = os.getenv("POLYGON_API_KEY") or os.getenv("AST_POLYGON_API_KEY")
    data_config: dict[str, Any] = {
        "data_sources": dict(data_sources_map),
        "lookback_days": lookback_days_int,
    }
    if polygon_api_key:
        data_config["polygon_api_key"] = polygon_api_key

    orchestrator_config: dict[str, Any] = {
        "mode": mode,
        "universe_plugin": "universe_real",
        "data_plugin": "data_real",
        "scanner_plugin": "scanner_real",
        "strategy_plugin": "strategy_real",
        "risk_gate_plugin": "risk_gate_real",
        "plugins": {
            "data_sources": [
                {"name": "universe_real", "config": {"filters": dict(universe_filters)}},
                {"name": "data_real", "config": data_config},
            ],
            "scanners": [
                {"name": "scanner_real", "config": scanner_config},
            ],
            "strategies": [
                {"name": "strategy_real", "config": dict(strategy_map)},
            ],
            "risk_policies": [
                {"name": "risk_gate_real", "config": dict(risk_gate_map)},
            ],
        },
    }

    market_regime_cfg = base_config.get("market_regime")
    if not isinstance(market_regime_cfg, Mapping):
        market_regime_cfg = scanner_map.get("market_regime")
    if isinstance(market_regime_cfg, Mapping):
        orchestrator_config["market_regime"] = dict(market_regime_cfg)

    return orchestrator_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PRE_MARKET_FULL_SCAN orchestrator with real plugins.")
    parser.add_argument(
        "--mode",
        choices=("DRY_RUN", "PAPER", "LIVE"),
        default="DRY_RUN",
        help="Operating mode for the run.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Optional YAML config path (defaults to config/config.yaml).",
    )
    args = parser.parse_args(argv)

    config_path = Path(str(args.config))
    base_config = _load_yaml_mapping(config_path)

    system_cfg = base_config.get("system")
    if isinstance(system_cfg, Mapping):
        try:
            from common.config import SystemConfig
            import msgspec

            system = msgspec.convert(dict(system_cfg), type=SystemConfig)
            init_logging(system)
        except Exception:
            pass

    logger_factory = StructlogLoggerFactory()
    event_bus = InMemoryEventBus()

    captured: dict[str, Any] = {"run_id": None, "data_meta": None, "scanner_meta": None}

    def handle_run_start(event: DomainEvent) -> None:
        captured["run_id"] = event.run_id

    def handle_data_executed(event: DomainEvent) -> None:
        if isinstance(event.data, Mapping):
            captured["data_meta"] = dict(event.data)

    def handle_scanner_executed(event: DomainEvent) -> None:
        if isinstance(event.data, Mapping):
            captured["scanner_meta"] = dict(event.data)

    event_bus.subscribe("runs.*.orchestrator.run.start", handle_run_start)
    event_bus.subscribe("runs.*.data.module.executed", handle_data_executed)
    event_bus.subscribe("runs.*.scanner.module.executed", handle_scanner_executed)

    journal_cfg = base_config.get("journal")
    journal_map: Mapping[str, Any] = journal_cfg if isinstance(journal_cfg, Mapping) else {}
    storage_path = str(journal_map.get("storage_path") or "journal")
    artifacts_root = Path(storage_path) / "runs"
    artifact_manager = ArtifactManager(base_path=artifacts_root)
    journal_writer = JournalWriter(artifact_manager)

    registry = PluginRegistry()
    register_real_plugins(registry)

    orchestrator = EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )

    orchestrator_config = _build_orchestrator_config(base_config, str(args.mode))

    try:
        run_result = orchestrator.execute_run(orchestrator_config)
    except BaseException as exc:
        _print_error(exc)
        return 1

    if run_result.status is ResultStatus.FAILED:
        _print_error(run_result.error or RuntimeError("PRE_MARKET_FULL_SCAN failed without error context."))
    elif run_result.status is ResultStatus.DEGRADED and run_result.error is not None:
        print(f"WARNING: {run_result.error}", file=sys.stderr)

    run_id = captured.get("run_id")
    run_dir = (artifacts_root / run_id) if isinstance(run_id, str) and run_id else None

    data_meta = captured.get("data_meta") if isinstance(captured.get("data_meta"), Mapping) else {}
    scanner_meta = captured.get("scanner_meta") if isinstance(captured.get("scanner_meta"), Mapping) else {}

    universe_symbols = None
    if isinstance(data_meta.get("symbols_requested"), int):
        universe_symbols = int(data_meta["symbols_requested"])

    data_fetched = int(data_meta["symbols_fetched"]) if isinstance(data_meta.get("symbols_fetched"), int) else None
    data_failed = int(data_meta["symbols_failed"]) if isinstance(data_meta.get("symbols_failed"), int) else None

    candidates_detected = None
    if isinstance(scanner_meta.get("candidates"), int):
        candidates_detected = int(scanner_meta["candidates"])

    top_candidates: list[dict[str, Any]] = []
    if run_dir is not None:
        candidate_path = run_dir / "outputs" / "candidates.json"
        top_candidates = _read_candidates(candidate_path, limit=10)
        if candidates_detected is None and candidate_path.exists():
            try:
                import msgspec.json

                raw = msgspec.json.decode(candidate_path.read_bytes())
                if isinstance(raw, Mapping) and isinstance(raw.get("total_detected"), int):
                    candidates_detected = int(raw["total_detected"])
            except Exception:
                pass

    summary = run_result.data
    duration_ms = getattr(summary, "duration_ms", None) if summary is not None else None

    _print_summary(
        run_id=run_id if isinstance(run_id, str) else None,
        mode=str(args.mode),
        status=run_result.status,
        duration_ms=int(duration_ms) if isinstance(duration_ms, int) else None,
        universe_symbols=universe_symbols,
        data_fetched=data_fetched,
        data_failed=data_failed,
        candidates_detected=candidates_detected,
        top_candidates=top_candidates,
        run_dir=run_dir,
    )

    if run_result.status is ResultStatus.SUCCESS:
        return 0
    if run_result.status is ResultStatus.DEGRADED:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
