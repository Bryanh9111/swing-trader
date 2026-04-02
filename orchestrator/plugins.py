"""Real plugin adapters for the Phase 1.5 orchestrator pipeline.

The Phase 1.5 orchestrator (`orchestrator/eod_scan.py`) runs a simple
dataflow pipeline where each module is a `plugins.interface.PluginBase`
instance that accepts/returns plain `Mapping[str, Any]` payloads.

Phase 2+ domain modules (Universe/Data/Scanner) already expose typed
`msgspec.Struct` snapshots and `Result[T]` semantics, but they do not match the
Phase 1.5 mapping payload shapes.

This module bridges that gap by wrapping:

- `universe.builder.UniverseBuilder` (builds `UniverseSnapshot`)
- `data.orchestrator.DataOrchestrator` (fetches `PriceSeriesSnapshot`)
- `scanner.detector.detect_platform_candidate` (emits `CandidateSet`)

All adapters:
- Read parameters from `self.config` (a mapping passed at construction time).
- Use `context["logger"]` / `context["run_context"].logger` when available.
- Use `context["emit_event"]` when available.
- Return `Result[dict]` payloads (builtins) that are safe to serialize into the
  journal.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, cast

import msgspec

from common.exceptions import ConfigurationError, PartialDataError, RecoverableError, ValidationError
from common.interface import BoundLogger, Result, ResultStatus
from journal.run_id import RunIDGenerator
from plugins.interface import PluginBase, PluginCategory, PluginContext, PluginMetadata

from data import S3DataAdapter
from data.cache import FileCache
from data.interface import PriceSeriesSnapshot, normalize_date, DataLayerStats
from data.orchestrator import DataOrchestrator
from data.polygon_adapter import PolygonDataAdapter
from data.yahoo_adapter import YahooDataAdapter
from event_guard.guard import apply_event_guard
from event_guard.interface import (
    EventGuardConfig,
    EventGuardOutput,
    EventSnapshot,
    RiskLevel,
    TradeConstraints,
)
from event_guard.sources import fetch_events_with_fallback
from execution.fills_persistence import FillsPersistence
from execution.ibkr_adapter import IBKRAdapter
from execution.interface import (
    BrokerConnectionConfig,
    ExecutionOutput,
    ExecutionReport,
)
from execution.order_manager import OrderManager
from order_state_machine import IDGenerator, Persistence, Reconciler, StateMachine
from order_state_machine.interface import OrderState
from risk_gate import checks as risk_checks
from risk_gate.capital_usage_check import CapitalUsageCheck
from risk_gate.engine import evaluate_intents
from risk_gate.interface import (
    CheckStatus,
    DecisionType,
    RiskCheckContext,
    RiskDecisionSet,
    RiskGateConfig,
    RiskGateOutput,
    SafeModeState,
)
from scanner.detector import detect_platform_candidate
from scanner.interface import CandidateSet, ScannerConfig
from strategy.engine import generate_intents
from strategy.interface import (
    IntentType,
    OrderIntentSet,
    StrategyEngineConfig,
    StrategyOutput,
    TradeIntent,
)
from strategy.pricing import create_price_policy
from strategy.sizing import create_position_sizer
from universe.builder import UniverseBuilder
from universe.interface import UniverseFilterCriteria, UniverseSnapshot

__all__ = [
    "UniversePlugin",
    "DataPlugin",
    "ScannerPlugin",
    "EventGuardPlugin",
    "StrategyPlugin",
    "RiskGatePlugin",
    "ExecutionPlugin",
    "register_real_plugins",
]

EventEmitter = Callable[[str, str, Mapping[str, Any] | None], None]


def _intent_get(obj: Any, key: str, default: Any = None) -> Any:
    """Access a field on an intent that may be a dict or a msgspec Struct.

    At runtime ``IntentOrderMapping.intent_snapshot`` is typed as ``Any`` so
    msgspec deserialises it as a plain ``dict``.  Code that needs to read
    fields must handle both dict and struct access patterns.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_logger(context: PluginContext | None, module_name: str) -> logging.Logger | BoundLogger:
    """Return the best-available logger from `context`, falling back to stdlib logging."""

    if isinstance(context, Mapping):
        logger = context.get("logger")
        if logger is None:
            run_context = context.get("run_context")
            if run_context is not None:
                logger = getattr(run_context, "logger", None)
        if logger is not None:
            bound = cast(logging.Logger | BoundLogger, logger)
            if hasattr(bound, "bind"):
                return bound.bind(module=module_name)
            return bound
    return logging.getLogger(module_name)


def _get_emitter(context: PluginContext | None) -> EventEmitter | None:
    """Extract an `emit_event` callback from `context` when present."""

    if not isinstance(context, Mapping):
        return None
    emitter = context.get("emit_event")
    if callable(emitter):
        return cast(EventEmitter, emitter)
    return None


def _emit(emitter: EventEmitter | None, event_type: str, module_name: str, data: Mapping[str, Any] | None = None) -> None:
    """Best-effort event emission (must never raise)."""

    if emitter is None:
        return
    try:
        emitter(event_type, module_name, data)
    except Exception:  # noqa: BLE001 - eventing should never break module execution.
        return


def _as_utc_date_from_ns(timestamp_ns: int) -> date:
    """Convert a nanosecond epoch timestamp into a UTC date."""

    dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC)
    return dt.date()


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return `value` as a mapping when possible."""

    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return None


def _safe_int(value: Any, *, default: int) -> int:
    """Best-effort int coercion with fallback."""

    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def _safe_float(value: Any, *, default: float) -> float:
    """Best-effort float coercion with fallback."""

    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _safe_bool(value: Any, *, default: bool) -> bool:
    """Best-effort bool coercion with fallback."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def _safe_tuple_int(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    """Convert value to tuple[int, int] safely."""

    if isinstance(value, tuple) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (ValueError, TypeError):
            return default
    if isinstance(value, list) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (ValueError, TypeError):
            return default
    return default


class UniversePluginConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration schema for `UniversePlugin`.

    Notes:
        This configuration is intentionally tolerant of multiple shapes so it
        can be driven by different higher-level configs. The plugin will look
        for either:

        - `criteria`: mapping matching `UniverseFilterCriteria`, or
        - `filters`: mapping similar to `config/config.yaml` under `universe_builder.filters`.
    """

    enabled: bool = True

    criteria: Mapping[str, Any] | None = None
    filters: Mapping[str, Any] | None = None
    polygon_api_key: str | None = None


class UniversePlugin(PluginBase[UniversePluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Orchestrator-facing adapter for `universe.builder.UniverseBuilder`.

    Input:
        Phase 1.5 orchestrator passes a mapping payload (unused by this plugin).

    Output:
        A builtins `dict` representation of `UniverseSnapshot` (safe to journal).
    """

    metadata: ClassVar[PluginMetadata] = PluginMetadata(
        name="universe",
        version="2.1.0",
        schema_version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
        description="UniverseBuilder adapter producing a serializable UniverseSnapshot payload.",
    )

    _MODULE_NAME = "orchestrator.plugins.universe"
    _CONFIG_INVALID_REASON = "UNIVERSE_PLUGIN_CONFIG_INVALID"

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._validated: UniversePluginConfig = UniversePluginConfig()
        self._criteria: UniverseFilterCriteria | None = None
        self._builder: UniverseBuilder | None = None
        self._polygon_api_key: str | None = None
        self._module_name: str = self.metadata.name
        self._logger: logging.Logger | BoundLogger = logging.getLogger(self._MODULE_NAME)
        self._emit_event: EventEmitter | None = None

    def init(self, context: PluginContext | None = None) -> Result[None]:
        """Initialise the underlying `UniverseBuilder` and bind logging/event handles."""

        self._logger = _get_logger(context, self._MODULE_NAME)
        self._emit_event = _get_emitter(context)

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str) and module_name:
                self._module_name = module_name

        if self._criteria is None:
            error = ValidationError(
                "UniversePlugin criteria was not validated before init.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(self.config)},
            )
            return Result.failed(error, error.reason_code)

        try:
            self._builder = UniverseBuilder(self._criteria, polygon_api_key=self._polygon_api_key)  # type: ignore[call-arg]
        except TypeError as exc:
            if "polygon_api_key" not in str(exc):
                raise
            self._builder = UniverseBuilder(self._criteria)
        builder_context: dict[str, Any] = {
            "module_name": self._module_name,
            "logger": self._logger,
            "emit_event": self._emit_event,
        }
        if self._polygon_api_key:
            builder_context["polygon_api_key"] = self._polygon_api_key

        _emit(self._emit_event, "module.initialised", self._module_name, {"module": "universe"})
        init_result = self._builder.init(builder_context)
        if init_result.status is ResultStatus.FAILED:
            return Result.failed(init_result.error or RuntimeError("UniverseBuilder init failed."), init_result.reason_code or "INIT_FAILED")
        if init_result.status is ResultStatus.DEGRADED:
            return Result.degraded(None, init_result.error or RuntimeError("UniverseBuilder init degraded."), init_result.reason_code or "INIT_DEGRADED")
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[UniversePluginConfig]:
        """Validate config and prepare `UniverseFilterCriteria` for the builder."""

        if not isinstance(config, Mapping):
            return Result.success(UniversePluginConfig())

        enabled = _safe_bool(config.get("enabled", True), default=True)
        criteria_map = _coerce_mapping(config.get("criteria"))
        filters_map = _coerce_mapping(config.get("filters"))
        import os

        polygon_api_key = config.get("polygon_api_key") or os.environ.get("POLYGON_API_KEY")
        if polygon_api_key and isinstance(polygon_api_key, str):
            polygon_api_key = polygon_api_key.strip() or None
        else:
            polygon_api_key = None

        validated = UniversePluginConfig(
            enabled=enabled,
            criteria=criteria_map,
            filters=filters_map,
            polygon_api_key=polygon_api_key,
        )
        self._validated = validated
        self._polygon_api_key = polygon_api_key

        if not validated.enabled:
            self._criteria = UniverseFilterCriteria(
                exchanges=["NYSE", "NASDAQ"],
                min_price=0.0,
                max_price=None,
                min_avg_dollar_volume_20d=0.0,
                min_market_cap=None,
                exclude_otc=False,
                exclude_halted=False,
                max_results=0,
            )
            return Result.success(validated)

        criteria_payload: dict[str, Any] = {}

        if validated.criteria is not None:
            criteria_payload.update(dict(validated.criteria))
        elif validated.filters is not None:
            filters = dict(validated.filters)
            criteria_payload.update(
                {
                    "exchanges": filters.get("exchanges"),
                    "exclude_otc": filters.get("exclude_otc"),
                    "exclude_halted": filters.get("exclude_halted"),
                    "min_price": filters.get("min_price"),
                    "max_price": filters.get("max_price"),
                    "min_market_cap": filters.get("min_market_cap"),
                    "max_results": filters.get("max_symbols") or filters.get("max_results"),
                    "enable_details_enrich": filters.get("enable_details_enrich"),
                    "details_enrich_top_k": filters.get("details_enrich_top_k"),
                    "details_enrich_multiplier": filters.get("details_enrich_multiplier"),
                }
            )
            if "min_avg_dollar_volume_20d" in filters:
                criteria_payload["min_avg_dollar_volume_20d"] = filters.get("min_avg_dollar_volume_20d")
            elif "min_avg_dollar_volume" in filters:
                criteria_payload["min_avg_dollar_volume_20d"] = filters.get("min_avg_dollar_volume")
            elif "min_avg_volume" in filters:
                criteria_payload["min_avg_dollar_volume_20d"] = filters.get("min_avg_volume")

        # Provide sane defaults so an empty config can still run.
        criteria_payload.setdefault("exchanges", ["NYSE", "NASDAQ"])
        criteria_payload.setdefault("min_price", 0.0)
        criteria_payload.setdefault("max_price", 2000.0)
        criteria_payload.setdefault("min_avg_dollar_volume_20d", 0.0)
        criteria_payload.setdefault("min_market_cap", None)
        criteria_payload.setdefault("exclude_otc", True)
        criteria_payload.setdefault("exclude_halted", True)
        criteria_payload.setdefault("max_results", None)
        criteria_payload.setdefault("enable_details_enrich", True)
        criteria_payload.setdefault("details_enrich_top_k", 500)
        criteria_payload.setdefault("details_enrich_multiplier", 1.5)

        try:
            criteria = msgspec.convert(criteria_payload, type=UniverseFilterCriteria)
        except Exception as exc:  # noqa: BLE001 - surface schema issues as validation failure.
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"criteria": criteria_payload},
            )
            return Result.failed(error, error.reason_code)

        self._criteria = criteria
        return Result.success(validated)

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Build the universe snapshot and return it as a serializable mapping."""

        if not self._validated.enabled:
            empty = UniverseSnapshot(
                schema_version="1.0.0",
                system_version=RunIDGenerator.get_system_version(),
                asof_timestamp=time.time_ns(),
                source="disabled",
                equities=[],
                filter_criteria={},
                total_candidates=0,
                total_filtered=0,
            )
            return Result.success(msgspec.to_builtins(empty))

        if self._builder is None:
            error = RuntimeError("UniversePlugin was not initialised.")
            return Result.failed(error, "NOT_INITIALISED")

        started_ns = time.time_ns()
        result = self._builder.execute(None)
        duration_ms = int((time.time_ns() - started_ns) // 1_000_000)

        snapshot = result.data
        payload_dict: Mapping[str, Any] | None = msgspec.to_builtins(snapshot) if snapshot is not None else None
        meta = {
            "duration_ms": duration_ms,
            "status": result.status.value,
            "reason_code": result.reason_code,
        }
        _emit(self._emit_event, "module.executed", self._module_name, meta)

        if result.status is ResultStatus.FAILED:
            return Result.failed(result.error or RuntimeError("Universe build failed."), result.reason_code or "UNIVERSE_FAILED")
        if result.status is ResultStatus.DEGRADED:
            return Result.degraded(payload_dict, result.error or RuntimeError("Universe build degraded."), result.reason_code or "UNIVERSE_DEGRADED")
        return Result.success(payload_dict or {})

    def cleanup(self) -> Result[None]:
        """Cleanup the underlying builder."""

        _emit(self._emit_event, "module.cleaned_up", self._module_name, {"module": "universe"})
        if self._builder is None:
            return Result.success(data=None)
        result = self._builder.cleanup()
        if result.status is ResultStatus.FAILED:
            return Result.failed(result.error or RuntimeError("UniverseBuilder cleanup failed."), result.reason_code or "CLEANUP_FAILED")
        if result.status is ResultStatus.DEGRADED:
            return Result.degraded(None, result.error or RuntimeError("UniverseBuilder cleanup degraded."), result.reason_code or "CLEANUP_DEGRADED")
        return Result.success(data=None)


class UniverseCachedPluginConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration schema for `UniverseCachedPlugin`.

    Expects a local JSON file containing either:
    - a full `UniverseSnapshot` mapping (e.g. from `journal/.../outputs/universe.json`), or
    - a cache entry mapping with `equities` (e.g. `.cache/universe/yfinance_universe.json`).
    """

    enabled: bool = True
    cache_path: str | None = None


class UniverseCachedPlugin(PluginBase[UniverseCachedPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Universe plugin that loads a pre-cached universe snapshot from disk.

    This avoids calling Polygon during backtests where the universe has already
    been precomputed/cached locally.
    """

    metadata: ClassVar[PluginMetadata] = PluginMetadata(
        name="universe_cached",
        version="1.0.0",
        schema_version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
        description="Load a cached UniverseSnapshot/universe equities list from disk.",
    )

    _MODULE_NAME = "orchestrator.plugins.universe_cached"
    _CONFIG_INVALID_REASON = "UNIVERSE_CACHED_PLUGIN_CONFIG_INVALID"

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._validated: UniverseCachedPluginConfig = UniverseCachedPluginConfig()
        self._module_name: str = self.metadata.name
        self._logger: logging.Logger | BoundLogger = logging.getLogger(self._MODULE_NAME)
        self._emit_event: EventEmitter | None = None

    def init(self, context: PluginContext | None = None) -> Result[None]:
        self._logger = _get_logger(context, self._MODULE_NAME)
        self._emit_event = _get_emitter(context)

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str) and module_name:
                self._module_name = module_name

        _emit(self._emit_event, "module.initialised", self._module_name, {"module": "universe"})
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[UniverseCachedPluginConfig]:
        if not isinstance(config, Mapping):
            self._validated = UniverseCachedPluginConfig()
            return Result.success(self._validated)

        enabled = _safe_bool(config.get("enabled", True), default=True)
        cache_path = config.get("cache_path")
        if cache_path is not None and not isinstance(cache_path, str):
            error = ValidationError(
                "cache_path must be a string when provided.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"cache_path": cache_path},
            )
            return Result.failed(error, error.reason_code)

        self._validated = UniverseCachedPluginConfig(enabled=enabled, cache_path=(cache_path.strip() if cache_path else None))
        return Result.success(self._validated)

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        from universe.interface import EquityInfo, UniverseSnapshot

        if not self._validated.enabled:
            empty = UniverseSnapshot(
                schema_version="1.0.0",
                system_version=RunIDGenerator.get_system_version(),
                asof_timestamp=time.time_ns(),
                source="disabled",
                equities=[],
                filter_criteria={},
                total_candidates=0,
                total_filtered=0,
            )
            return Result.success(msgspec.to_builtins(empty))

        candidate_paths: list[Path] = []
        if self._validated.cache_path:
            candidate_paths.append(Path(self._validated.cache_path))
        candidate_paths.extend(
            [
                Path(".cache/universe/universe.json"),
                Path(".cache/universe/yfinance_universe.json"),
            ]
        )

        cache_file: Path | None = next((path for path in candidate_paths if path.exists()), None)
        if cache_file is None:
            error = ConfigurationError(
                "No cached universe file found.",
                module=self._MODULE_NAME,
                reason_code="UNIVERSE_CACHE_MISSING",
                details={"candidates": [str(p) for p in candidate_paths]},
            )
            return Result.failed(error, error.reason_code)

        try:
            raw = cache_file.read_bytes()
        except Exception as exc:  # noqa: BLE001
            error = ConfigurationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code="UNIVERSE_CACHE_READ_FAILED",
                details={"path": str(cache_file)},
            )
            return Result.failed(error, error.reason_code)

        try:
            decoded = msgspec.json.decode(raw)
        except Exception:  # noqa: BLE001
            import json

            try:
                decoded = json.loads(raw.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                error = ValidationError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code="UNIVERSE_CACHE_INVALID_JSON",
                    details={"path": str(cache_file)},
                )
                return Result.failed(error, error.reason_code)

        snapshot: UniverseSnapshot | None = None

        if isinstance(decoded, Mapping):
            if "schema_version" in decoded and "asof_timestamp" in decoded and "equities" in decoded:
                try:
                    snapshot = msgspec.convert(dict(decoded), type=UniverseSnapshot)
                except Exception:  # noqa: BLE001
                    snapshot = None

            if snapshot is None and "equities" in decoded:
                asof_timestamp = decoded.get("asof_timestamp") or decoded.get("saved_at_ns") or time.time_ns()
                try:
                    equities = msgspec.convert(decoded.get("equities") or [], type=list[EquityInfo])
                except Exception as exc:  # noqa: BLE001
                    error = ValidationError.from_error(
                        exc,
                        module=self._MODULE_NAME,
                        reason_code="UNIVERSE_CACHE_INVALID_EQUITIES",
                        details={"path": str(cache_file)},
                    )
                    return Result.failed(error, error.reason_code)

                snapshot = UniverseSnapshot(
                    schema_version="1.0.0",
                    system_version=RunIDGenerator.get_system_version(),
                    asof_timestamp=int(asof_timestamp),
                    source="cache_file",
                    equities=equities,
                    filter_criteria={},
                    total_candidates=len(equities),
                    total_filtered=len(equities),
                )

        if snapshot is None and isinstance(decoded, list):
            try:
                equities = msgspec.convert(decoded, type=list[EquityInfo])
            except Exception as exc:  # noqa: BLE001
                error = ValidationError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code="UNIVERSE_CACHE_INVALID_EQUITIES",
                    details={"path": str(cache_file)},
                )
                return Result.failed(error, error.reason_code)

            snapshot = UniverseSnapshot(
                schema_version="1.0.0",
                system_version=RunIDGenerator.get_system_version(),
                asof_timestamp=time.time_ns(),
                source="cache_file",
                equities=equities,
                filter_criteria={},
                total_candidates=len(equities),
                total_filtered=len(equities),
            )

        if snapshot is None:
            error = ValidationError(
                "Cached universe JSON did not match a supported shape.",
                module=self._MODULE_NAME,
                reason_code="UNIVERSE_CACHE_UNSUPPORTED_SHAPE",
                details={"path": str(cache_file)},
            )
            return Result.failed(error, error.reason_code)

        meta = {"path": str(cache_file), "equities": len(snapshot.equities)}
        _emit(self._emit_event, "module.executed", self._module_name, meta)
        return Result.success(msgspec.to_builtins(snapshot))

    def cleanup(self) -> Result[None]:
        _emit(self._emit_event, "module.cleaned_up", self._module_name, {"module": "universe"})
        return Result.success(data=None)


class DataPluginConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration schema for `DataPlugin`.

    Fields are intentionally minimal and focused on adapter selection + date
    range derivation.

    Supported keys (all optional):
        enabled: bool
        primary_source: "polygon" | "yahoo"
        secondary_source: "polygon" | "yahoo" | None
        cache_dir: filesystem path
        cache_ttl_seconds: int
        enable_incremental_cache: bool (use symbol-level incremental caching)
        lookback_days: int (calendar days; ensure ~0.7× yields enough trading days)
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        requests_timeout_seconds: float
        polygon_api_key: optional Polygon API key override
        polygon_unlimited: bool (use higher Polygon RPM budget)
        calls_per_minute: int (Polygon RPM budget when not unlimited)
    """

    enabled: bool = True

    primary_source: str = "yahoo"
    secondary_source: str | None = None

    cache_dir: str | None = ".cache/data"
    cache_ttl_seconds: int = 86400
    enable_incremental_cache: bool = True

    lookback_days: int = 180  # Calendar days; ~127 trading days to support 2×60-day windows
    start_date: str | None = None
    end_date: str | None = None

    requests_timeout_seconds: float = 15.0
    polygon_api_key: str | None = None
    polygon_unlimited: bool = False
    calls_per_minute: int = 5
    s3_enabled: bool = False
    s3_endpoint_url: str | None = None
    s3_bucket_name: str | None = None
    s3_path_prefix: str = "us_stocks_sip/day_aggs_v1"
    s3_local_cache_dir: str | None = ".cache/historical"
    s3_max_workers: int = 5


@dataclass(frozen=True, slots=True)
class _DataRunWindow:
    start: date
    end: date


class DataPlugin(PluginBase[DataPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Orchestrator-facing adapter for `data.orchestrator.DataOrchestrator`.

    Input:
        A builtins `dict` representation of `UniverseSnapshot` (produced by
        `UniversePlugin`).

    Output:
        A serializable mapping:
            {
              "universe": <UniverseSnapshot builtins dict>,
              "series_by_symbol": { "AAPL": <PriceSeriesSnapshot dict>, ... },
              "window": { "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" }
            }
    """

    metadata: ClassVar[PluginMetadata] = PluginMetadata(
        name="data",
        version="2.2.0",
        schema_version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
        description="DataOrchestrator adapter fetching per-symbol price series snapshots.",
    )

    _MODULE_NAME = "orchestrator.plugins.data"
    _CONFIG_INVALID_REASON = "DATA_PLUGIN_CONFIG_INVALID"

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._validated: DataPluginConfig = DataPluginConfig()
        self._logger: logging.Logger | BoundLogger = logging.getLogger(self._MODULE_NAME)
        self._emit_event: EventEmitter | None = None
        self._module_name: str = self.metadata.name

        self._orchestrator: DataOrchestrator | None = None
        self._window: _DataRunWindow | None = None
        self._market_calendar: MarketCalendar | None = None

    def init(self, context: PluginContext | None = None) -> Result[None]:
        """Initialise the data orchestrator using configured adapters and cache."""

        self._logger = _get_logger(context, self._MODULE_NAME)
        self._emit_event = _get_emitter(context)

        # Diagnostic: Check market_calendar config availability
        market_calendar_config_payload: Any | None = None
        if isinstance(context, Mapping):
            market_calendar_config_payload = context.get("market_calendar")
        _emit(
            self._emit_event,
            "data.init.diagnostic",
            self._module_name,
            {
                "context_is_none": context is None,
                "context_keys": list(context.keys()) if isinstance(context, Mapping) else None,
                "market_calendar_value": market_calendar_config_payload,
                "market_calendar_type": str(type(market_calendar_config_payload)),
            },
        )
        _emit(
            self._emit_event,
            "data.init.market_calendar_config",
            self._module_name,
            {
                "config_type": str(type(market_calendar_config_payload)),
                "config_is_mapping": isinstance(market_calendar_config_payload, Mapping),
                "enabled": market_calendar_config_payload.get("enabled")
                if isinstance(market_calendar_config_payload, Mapping)
                else None,
            },
        )

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str) and module_name:
                self._module_name = module_name

        # Create MarketCalendar if enabled (for backfill optimization)
        if context is not None and isinstance(context, Mapping):
            market_calendar_config = context.get("market_calendar")
            if isinstance(market_calendar_config, Mapping) and market_calendar_config.get("enabled", False):
                # Try to get polygon_api_key from config hierarchy
                polygon_api_key = None

                # First try: run_context.config (full config_map with data_sources)
                if context is not None and isinstance(context, Mapping):
                    run_context = context.get("run_context")
                    if run_context is not None and hasattr(run_context, "config"):
                        full_config = run_context.config
                        if isinstance(full_config, Mapping):
                            # Check data_sources section (mirrors validate_config logic)
                            data_sources = full_config.get("data_sources")
                            if isinstance(data_sources, Mapping):
                                primary = data_sources.get("primary")
                                if isinstance(primary, Mapping):
                                    polygon_api_key = primary.get("polygon_api_key")

                            # Also check top-level polygon_api_key
                            if not polygon_api_key:
                                polygon_api_key = full_config.get("polygon_api_key")

                # Second try: plugin-local config
                if not polygon_api_key and context is not None and isinstance(context, Mapping):
                    config_payload = context.get("config")
                    if isinstance(config_payload, Mapping):
                        polygon_api_key = config_payload.get("polygon_api_key")

                # Third try: environment variable
                if not polygon_api_key:
                    polygon_api_key = os.getenv("POLYGON_API_KEY")

                # Normalize string
                if polygon_api_key and isinstance(polygon_api_key, str):
                    polygon_api_key = polygon_api_key.strip() or None

                # Diagnostic: Log polygon_api_key resolution result
                _emit(
                    self._emit_event,
                    "data.init.polygon_api_key_resolution",
                    self._module_name,
                    {
                        "api_key_found": polygon_api_key is not None,
                        "api_key_length": len(polygon_api_key) if polygon_api_key else 0,
                    },
                )

                if polygon_api_key:
                    try:
                        from common.market_calendar import MarketCalendar

                        self._market_calendar = MarketCalendar(
                            api_key=polygon_api_key,
                            cache_ttl_hours=market_calendar_config.get("cache_ttl_hours", 24),
                            requests_timeout_seconds=market_calendar_config.get("requests_timeout_seconds", 30),
                        )
                        # Diagnostic: Log successful creation
                        _emit(
                            self._emit_event,
                            "data.init.market_calendar_created",
                            self._module_name,
                            {"status": "success"},
                        )
                    except Exception as exc:
                        # Log warning and emit event
                        self._logger.warning("market_calendar.init_failed", error=str(exc))
                        _emit(
                            self._emit_event,
                            "data.init.market_calendar_failed",
                            self._module_name,
                            {"error": str(exc), "error_type": type(exc).__name__},
                        )
                else:
                    # Diagnostic: Log API key not found
                    _emit(
                        self._emit_event,
                        "data.init.market_calendar_skipped",
                        self._module_name,
                        {"reason": "polygon_api_key_not_found"},
                    )

        if self._validated.enabled and self._orchestrator is None:
            error = ValidationError(
                "DataPlugin configuration was not validated before init.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(self.config)},
            )
            return Result.failed(error, error.reason_code)

        # Debug: Check if orchestrator will be rebuilt with market_calendar
        if self._market_calendar is not None:
            self._logger.info(
                "data.init.rebuilding_orchestrator_with_calendar",
                has_existing_orchestrator=self._orchestrator is not None,
                market_calendar_type=str(type(self._market_calendar)),
            )

        if self._validated.enabled and self._market_calendar is not None:
            orchestrator_result = self._build_orchestrator(self._validated)
            if orchestrator_result.status is ResultStatus.SUCCESS and orchestrator_result.data is not None:
                self._orchestrator = orchestrator_result.data
                self._logger.info(
                    "data.init.orchestrator_rebuilt",
                    has_market_calendar=hasattr(self._orchestrator, "_cache")
                    and hasattr(self._orchestrator._cache, "_market_calendar")
                    and self._orchestrator._cache._market_calendar is not None,
                )

        _emit(self._emit_event, "module.initialised", self._module_name, {"module": "data"})
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[DataPluginConfig]:
        """Validate config and construct a `DataOrchestrator` instance."""

        if not isinstance(config, Mapping):
            validated = DataPluginConfig()
            self._validated = validated
            self._orchestrator = self._build_orchestrator(validated).data
            return Result.success(validated)

        merged: MutableMapping[str, Any] = dict(config)

        # Allow passing "data_sources" blocks (e.g. from config/config.yaml).
        if "data_sources" in merged and isinstance(merged["data_sources"], Mapping):
            data_sources = cast(Mapping[str, Any], merged["data_sources"])
            primary = _coerce_mapping(data_sources.get("primary")) or {}
            merged.setdefault("primary_source", primary.get("eod_historical"))
            s3 = _coerce_mapping(data_sources.get("s3")) or {}
            merged.setdefault("s3_enabled", s3.get("enabled"))
            merged.setdefault("s3_endpoint_url", s3.get("endpoint_url"))
            merged.setdefault("s3_bucket_name", s3.get("bucket_name"))
            merged.setdefault("s3_path_prefix", s3.get("path_prefix"))
            merged.setdefault("s3_local_cache_dir", s3.get("local_cache_dir"))
            merged.setdefault("s3_max_workers", s3.get("max_workers"))

        enabled = _safe_bool(merged.get("enabled", True), default=True)
        primary_source = str(merged.get("primary_source") or "yahoo").strip().lower()
        secondary_source_raw = merged.get("secondary_source")
        secondary_source = str(secondary_source_raw).strip().lower() if secondary_source_raw else None

        validated = DataPluginConfig(
            enabled=enabled,
            primary_source=primary_source or "yahoo",
            secondary_source=secondary_source or None,
            cache_dir=str(merged["cache_dir"]) if merged.get("cache_dir") else None,
            cache_ttl_seconds=_safe_int(merged.get("cache_ttl_seconds"), default=86400),
            enable_incremental_cache=_safe_bool(merged.get("enable_incremental_cache", True), default=True),
            lookback_days=_safe_int(merged.get("lookback_days"), default=120),
            start_date=str(merged["start_date"]) if merged.get("start_date") else None,
            end_date=str(merged["end_date"]) if merged.get("end_date") else None,
            requests_timeout_seconds=_safe_float(merged.get("requests_timeout_seconds"), default=15.0),
            polygon_api_key=str(merged["polygon_api_key"]) if merged.get("polygon_api_key") else None,
            polygon_unlimited=_safe_bool(merged.get("polygon_unlimited", False), default=False),
            calls_per_minute=max(1, _safe_int(merged.get("calls_per_minute"), default=5)),
            s3_enabled=_safe_bool(merged.get("s3_enabled", False), default=False),
            s3_endpoint_url=str(merged["s3_endpoint_url"]) if merged.get("s3_endpoint_url") else None,
            s3_bucket_name=str(merged["s3_bucket_name"]) if merged.get("s3_bucket_name") else None,
            s3_path_prefix=str(merged.get("s3_path_prefix") or "us_stocks_sip/day_aggs_v1"),
            s3_local_cache_dir=str(merged.get("s3_local_cache_dir") or ".cache/historical"),
            s3_max_workers=max(1, _safe_int(merged.get("s3_max_workers"), default=5)),
        )
        self._validated = validated

        if not validated.enabled:
            self._orchestrator = None
            self._window = None
            return Result.success(validated)

        orchestrator_result = self._build_orchestrator(validated)
        if orchestrator_result.status is ResultStatus.FAILED or orchestrator_result.data is None:
            return Result.failed(
                orchestrator_result.error or ConfigurationError(
                    "Failed to configure data orchestrator.",
                    module=self._MODULE_NAME,
                    reason_code=self._CONFIG_INVALID_REASON,
                    details={"config": dict(config)},
                ),
                orchestrator_result.reason_code or self._CONFIG_INVALID_REASON,
            )

        self._orchestrator = orchestrator_result.data
        return Result.success(validated)

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Fetch price series for symbols in the universe snapshot payload."""

        if not self._validated.enabled:
            return Result.success({"universe": dict(payload), "series_by_symbol": {}, "window": None})

        if self._orchestrator is None:
            return Result.failed(RuntimeError("DataPlugin was not initialised."), "NOT_INITIALISED")

        universe_result = self._parse_universe(payload)
        if universe_result.status is ResultStatus.FAILED or universe_result.data is None:
            return Result.failed(universe_result.error or RuntimeError("Universe payload invalid."), universe_result.reason_code or "INVALID_UNIVERSE_PAYLOAD")
        universe = universe_result.data

        window = self._resolve_window(universe)
        self._window = window

        symbols = [equity.symbol for equity in universe.equities if getattr(equity, "symbol", None)]
        total = len(symbols)
        if hasattr(self._logger, "info"):
            try:
                self._logger.info("data.fetch.start", symbols=total, start=window.start.isoformat(), end=window.end.isoformat())
            except Exception:  # noqa: BLE001
                pass

        series_by_symbol: dict[str, Mapping[str, Any]] = {}
        failures: list[tuple[str, BaseException, str | None]] = []
        degraded_seen = False

        for symbol in symbols:
            if self._validated.enable_incremental_cache:
                result = self._orchestrator.fetch_with_incremental_cache(symbol, window.start, window.end)
            else:
                result = self._orchestrator.fetch_with_cache(symbol, window.start, window.end)
            if result.status is ResultStatus.FAILED or result.data is None:
                failures.append((symbol, result.error or RuntimeError("Fetch failed."), result.reason_code))
                continue
            if result.status is ResultStatus.DEGRADED:
                degraded_seen = True
            series_by_symbol[symbol] = msgspec.to_builtins(result.data)

        output: Mapping[str, Any] = {
            "universe": msgspec.to_builtins(universe),
            "series_by_symbol": series_by_symbol,
            "window": {"start": window.start.isoformat(), "end": window.end.isoformat()},
        }

        meta = {
            "symbols_requested": total,
            "symbols_fetched": len(series_by_symbol),
            "symbols_failed": len(failures),
        }
        _emit(self._emit_event, "module.executed", self._module_name, meta)

        # Emit Data Layer statistics
        if self._orchestrator is not None:
            stats: DataLayerStats = self._orchestrator.get_run_stats()
            stats_payload = {
                "total_symbols": stats.total_symbols,
                "freshness_pass_count": stats.freshness_pass_count,
                "freshness_fail_count": stats.freshness_fail_count,
                "freshness_rate": stats.freshness_rate,
                "backfill_attempt_count": stats.backfill_attempt_count,
                "backfill_success_count": stats.backfill_success_count,
                "backfill_fail_count": stats.backfill_fail_count,
                "backfill_success_rate": stats.backfill_success_rate,
                "cache_hit_count": stats.cache_hit_count,
                "cache_miss_count": stats.cache_miss_count,
                "cache_hit_rate": stats.cache_hit_rate,
            }
            _emit(self._emit_event, "data.stats", self._module_name, stats_payload)

        if failures and not series_by_symbol:
            symbol, error, reason = failures[0]
            wrapped = PartialDataError.from_error(
                error,
                module=self._MODULE_NAME,
                reason_code="DATA_ALL_FAILED",
                details={"failed_symbols": [s for s, _, _ in failures], "first_failed": symbol, "first_reason": reason},
            )
            return Result.failed(wrapped, wrapped.reason_code)

        if failures or degraded_seen:
            wrapped = PartialDataError(
                "Some symbols failed to fetch price data.",
                module=self._MODULE_NAME,
                reason_code="DATA_PARTIAL",
                details={"failures": [{"symbol": s, "reason_code": r, "error": str(e)} for s, e, r in failures]},
            )
            return Result.degraded(output, wrapped, wrapped.reason_code)

        return Result.success(output)

    def cleanup(self) -> Result[None]:
        """Cleanup hook for the plugin (no-op for the orchestrator wrapper)."""

        _emit(self._emit_event, "module.cleaned_up", self._module_name, {"module": "data"})
        return Result.success(data=None)

    def _build_orchestrator(self, config: DataPluginConfig) -> Result[DataOrchestrator]:
        """Create a `DataOrchestrator` instance based on adapter configuration."""

        def build_adapter(source: str) -> Result[Any]:
            source_norm = (source or "").strip().lower()
            if source_norm == "polygon":
                calls_per_minute = 1000 if config.polygon_unlimited else int(config.calls_per_minute)
                return Result.success(
                    PolygonDataAdapter(
                        api_key=config.polygon_api_key,
                        requests_timeout_seconds=config.requests_timeout_seconds,
                        calls_per_minute=calls_per_minute,
                    )
                )
            if source_norm == "yahoo":
                return Result.success(YahooDataAdapter(requests_timeout_seconds=config.requests_timeout_seconds))
            if source_norm == "s3":
                if not config.s3_enabled:
                    error = ConfigurationError(
                        "S3 data source requested but data_sources.s3.enabled is false.",
                        module=self._MODULE_NAME,
                        reason_code=self._CONFIG_INVALID_REASON,
                        details={"source": source, "s3_enabled": config.s3_enabled},
                    )
                    return Result.failed(error, error.reason_code)
                return Result.success(
                    S3DataAdapter(
                        endpoint_url=config.s3_endpoint_url or "",
                        bucket_name=config.s3_bucket_name or "",
                        path_prefix=config.s3_path_prefix,
                        local_cache_dir=config.s3_local_cache_dir,
                        max_workers=config.s3_max_workers,
                    )
                )
            error = ConfigurationError(
                f"Unsupported data source: {source!r}",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"source": source},
            )
            return Result.failed(error, error.reason_code)

        primary_result = build_adapter(config.primary_source)
        if primary_result.status is ResultStatus.FAILED or primary_result.data is None:
            return Result.failed(primary_result.error or RuntimeError("Primary adapter failed."), primary_result.reason_code or self._CONFIG_INVALID_REASON)

        secondary_adapter = None
        if config.secondary_source:
            secondary_result = build_adapter(config.secondary_source)
            if secondary_result.status is ResultStatus.FAILED:
                return Result.failed(
                    secondary_result.error or RuntimeError("Secondary adapter failed."),
                    secondary_result.reason_code or self._CONFIG_INVALID_REASON,
                )
            secondary_adapter = secondary_result.data

        cache: FileCache | None = None
        if config.cache_dir is not None:
            cache = FileCache(
                cache_dir=config.cache_dir,
                ttl_seconds=config.cache_ttl_seconds,
                market_calendar=self._market_calendar,
            )
        orchestrator = DataOrchestrator(
            primary_adapter=primary_result.data,
            secondary_adapter=secondary_adapter,
            cache=cache,
            emit_event=self._emit_event,
        )
        return Result.success(orchestrator)

    def _parse_universe(self, payload: Mapping[str, Any]) -> Result[UniverseSnapshot]:
        """Decode the universe payload into a `UniverseSnapshot`."""

        raw = payload.get("universe") if isinstance(payload, Mapping) else None
        universe_dict = _coerce_mapping(raw) or payload
        try:
            return Result.success(msgspec.convert(dict(universe_dict), type=UniverseSnapshot))
        except Exception as exc:  # noqa: BLE001
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code="INVALID_UNIVERSE_PAYLOAD",
                details={"payload_keys": list(universe_dict.keys())},
            )
            return Result.failed(error, error.reason_code)

    def _resolve_window(self, universe: UniverseSnapshot) -> _DataRunWindow:
        """Resolve the data fetch window from config and universe `asof_timestamp`."""

        if self._validated.start_date and self._validated.end_date:
            start = normalize_date(self._validated.start_date)
            end = normalize_date(self._validated.end_date)
            return _DataRunWindow(start=start, end=end)

        end_date = _as_utc_date_from_ns(int(getattr(universe, "asof_timestamp", time.time_ns())))
        lookback = max(2, int(self._validated.lookback_days))
        start_date = end_date - timedelta(days=lookback)
        return _DataRunWindow(start=start_date, end=end_date)


def _attach_platform_shadow_score(
    candidate: Any,
    bars: list[Any],
    window: int,
    benchmark_closes: list[float] | None,
) -> Any:
    """Assemble shadow_score dict and attach to platform candidate meta.

    Pure post-processing — does NOT alter score, detection logic, or trading decisions.
    """
    import msgspec as _msgspec
    from scanner.gates.relative_strength import compute_rs_slope

    features = getattr(candidate, "features", None)
    meta = dict(getattr(candidate, "meta", None) or {})

    # Extract features (all available in PlatformFeatures)
    atr_pct = getattr(features, "atr_pct", None) if features else None
    box_quality = getattr(features, "box_quality", None) if features else None
    volatility = getattr(features, "volatility", None) if features else None
    vol_chg_ratio = getattr(features, "volume_change_ratio", None) if features else None
    support = getattr(features, "support_level", None) if features else None

    # RS slope: compute from stock closes vs benchmark (QQQ)
    rs_slope: float | None = None
    if benchmark_closes and bars:
        try:
            asset_closes = [float(b.close) for b in bars]
            rs_slope = compute_rs_slope(asset_closes, benchmark_closes)
        except Exception:  # noqa: BLE001
            pass

    # Distance to support: (last_close - support) / last_close
    dist_to_support: float | None = None
    if bars and support is not None and float(support) > 0:
        last_close = float(bars[-1].close)
        if last_close > 0:
            dist_to_support = (last_close - float(support)) / last_close

    meta["shadow_score"] = {
        "ss_version": 1,
        "ss_atr_pct": float(atr_pct) if atr_pct is not None else None,
        "ss_box_quality": float(box_quality) if box_quality is not None else None,
        "ss_volatility": float(volatility) if volatility is not None else None,
        "ss_volume_chg_ratio": float(vol_chg_ratio) if vol_chg_ratio is not None else None,
        "ss_rs_slope": float(rs_slope) if rs_slope is not None else None,
        "ss_rs_lookback": 20,
        "ss_consolidation_days": int(window),
        "ss_dist_to_support_pct": float(dist_to_support) if dist_to_support is not None else None,
        "ss_pattern_type": "platform",
    }

    return _msgspec.structs.replace(candidate, meta=meta)


class ScannerPlugin(PluginBase[ScannerConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Orchestrator-facing adapter for the Phase 2.3 scanner detector.

    Input:
        A serializable mapping produced by `DataPlugin`, containing:
            - "universe": UniverseSnapshot builtins dict
            - "series_by_symbol": {symbol: PriceSeriesSnapshot builtins dict}

    Output:
        A builtins `dict` representation of `CandidateSet`.
    """

    metadata: ClassVar[PluginMetadata] = PluginMetadata(
        name="scanner",
        version="2.3.0",
        schema_version="1.0.0",
        category=PluginCategory.SCANNER,
        enabled=True,
        description="Scanner detector adapter producing a serializable CandidateSet snapshot.",
    )

    _MODULE_NAME = "orchestrator.plugins.scanner"
    _CONFIG_INVALID_REASON = "SCANNER_PLUGIN_CONFIG_INVALID"

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._scanner_config: ScannerConfig = ScannerConfig()
        self._logger: logging.Logger | BoundLogger = logging.getLogger(self._MODULE_NAME)
        self._emit_event: EventEmitter | None = None
        self._module_name: str = self.metadata.name
        self.regime_breakthrough_config: dict[str, Any] | None = None
        self.current_regime: str | None = None

    def init(self, context: PluginContext | None = None) -> Result[None]:
        """Initialise the scanner plugin (bind logger + event emitter)."""

        self._logger = _get_logger(context, self._MODULE_NAME)
        self._emit_event = _get_emitter(context)

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str) and module_name:
                self._module_name = module_name

        _emit(self._emit_event, "module.initialised", self._module_name, {"module": "scanner"})
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[ScannerConfig]:
        """Validate scanner config mapping and materialize a `ScannerConfig` instance."""

        if not isinstance(config, Mapping):
            self._scanner_config = ScannerConfig()
            return Result.success(self._scanner_config)

        try:
            coerced = msgspec.convert(dict(config), type=ScannerConfig)
        except Exception as exc:  # noqa: BLE001
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(config)},
            )
            return Result.failed(error, error.reason_code)

        self._scanner_config = coerced
        return Result.success(coerced)

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Detect platform candidates across all symbols and configured windows."""

        parsed = self._parse_input(payload)
        if parsed.status is ResultStatus.FAILED or parsed.data is None:
            return Result.failed(parsed.error or RuntimeError("Scanner input invalid."), parsed.reason_code or "INVALID_INPUT")
        universe, series_by_symbol = parsed.data

        event_constraints_dict = _coerce_mapping(payload.get("constraints")) or {}
        equity_meta_by_symbol: dict[str, dict[str, Any]] = {str(s).strip().upper(): {} for s in series_by_symbol.keys() if str(s).strip()}
        for equity in getattr(universe, "equities", []) or []:
            symbol = getattr(equity, "symbol", None)
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            market_cap = getattr(equity, "market_cap", None)
            if market_cap is None:
                continue
            equity_meta_by_symbol.setdefault(symbol.strip().upper(), {})["market_cap"] = market_cap

        detected_at = time.time_ns()
        candidates = []
        total_scanned = 0
        current_regime = (self.current_regime or "").strip().lower() or None

        errors: list[tuple[str, BaseException, str | None]] = []
        skipped: list[tuple[str, str]] = []  # (symbol, reason_code)
        degraded_seen = False

        # Apply seasonal adaptation if enabled
        active_config = self._scanner_config
        if getattr(self._scanner_config, "enable_seasonal_adaptation", False):
            from datetime import datetime
            current_month = datetime.now().month
            low_volume_months = getattr(self._scanner_config, "seasonal_low_volume_months", [])
            if current_month in low_volume_months:
                seasonal_adjustments = getattr(self._scanner_config, "seasonal_adjustments", {})
                if seasonal_adjustments:
                    config_dict = msgspec.to_builtins(self._scanner_config)
                    if isinstance(config_dict, dict):
                        for key, value in seasonal_adjustments.items():
                            if key in config_dict:
                                config_dict[key] = value
                        try:
                            active_config = msgspec.convert(config_dict, type=ScannerConfig)
                        except Exception:  # noqa: BLE001
                            active_config = self._scanner_config

        active_breakthrough_config = self.regime_breakthrough_config

        # Pre-extract benchmark closes for shadow_score RS slope calculation.
        # Uses QQQ (or SPY fallback) already loaded in series_by_symbol — no extra fetch.
        _benchmark_series_ss = series_by_symbol.get("QQQ") or series_by_symbol.get("SPY")
        _benchmark_closes_ss: list[float] | None = None
        if _benchmark_series_ss is not None:
            _bm_bars = list(getattr(_benchmark_series_ss, "bars", []))
            if _bm_bars:
                _benchmark_closes_ss = [float(b.close) for b in _bm_bars]

        for symbol, series in series_by_symbol.items():
            total_scanned += 1
            bars = list(getattr(series, "bars", []))
            for window in cast(Sequence[int], active_config.windows):
                try:
                    result = detect_platform_candidate(
                        symbol,
                        bars,
                        int(window),
                        active_config,
                        detected_at,
                        regime=current_regime,
                        meta=equity_meta_by_symbol.get(symbol),
                        event_constraints=event_constraints_dict,
                        breakthrough_config=active_breakthrough_config,
                    )
                except TypeError as exc:
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    result = detect_platform_candidate(symbol, bars, int(window), active_config, detected_at)
                if result.status is ResultStatus.FAILED:
                    errors.append((symbol, result.error or RuntimeError("Candidate detection failed."), result.reason_code))
                    continue
                if result.status is ResultStatus.DEGRADED:
                    degraded_seen = True
                # Track skipped stocks (SUCCESS with None data + specific reason codes)
                if result.data is None and result.reason_code in ("INSUFFICIENT_BARS", "INSUFFICIENT_BARS_SKIPPED"):
                    skipped.append((symbol, result.reason_code or "UNKNOWN"))
                if result.data is not None:
                    # Attach shadow_score to platform candidate meta
                    cand = result.data
                    cand = _attach_platform_shadow_score(cand, bars, window, _benchmark_closes_ss)
                    candidates.append(cand)

        # Trend pattern detection (Growth/AI stocks channel)
        if getattr(active_config, "use_trend_pattern_detector", False):
            from datetime import date as date_type
            from scanner.trend_pattern_router import TrendPatternRouter, TrendPatternRouterConfig

            trend_config_dict = getattr(active_config, "trend_pattern_config", None) or {}
            try:
                trend_router_config = msgspec.convert(trend_config_dict, type=TrendPatternRouterConfig)
            except Exception:  # noqa: BLE001
                trend_router_config = TrendPatternRouterConfig()

            trend_router = TrendPatternRouter(trend_router_config)

            # Get benchmark bars for gates
            # XLK/SOXX for sector regime, QQQ for relative strength
            sector_series = series_by_symbol.get("XLK") or series_by_symbol.get("SOXX")
            benchmark_series = series_by_symbol.get("QQQ") or series_by_symbol.get("SPY")

            sector_bars = list(getattr(sector_series, "bars", [])) if sector_series else None
            benchmark_bars = list(getattr(benchmark_series, "bars", [])) if benchmark_series else None

            # Determine current date from latest bar timestamp
            current_date = date_type.today()
            for series in series_by_symbol.values():
                bars_list = list(getattr(series, "bars", []))
                if bars_list:
                    last_bar = bars_list[-1]
                    ts = getattr(last_bar, "timestamp", None)
                    if ts and isinstance(ts, (int, float)):
                        try:
                            current_date = date_type.fromtimestamp(ts / 1_000_000_000)  # ns to s
                        except (ValueError, OSError):
                            pass
                    break

            trend_candidates_count = 0
            for symbol, series in series_by_symbol.items():
                # Skip benchmark symbols themselves
                if symbol.upper() in ("XLK", "SOXX", "QQQ", "SPY", "VIXY", "VXX", "UVXY"):
                    continue

                bars = list(getattr(series, "bars", []))
                if len(bars) < 50:  # Minimum bars for EMA50
                    continue

                try:
                    result = trend_router.detect(
                        symbol=symbol,
                        bars=bars,
                        current_date=current_date,
                        sector_bars=sector_bars,
                        benchmark_bars=benchmark_bars,
                        detected_at=detected_at,
                        meta=equity_meta_by_symbol.get(symbol),
                    )
                    if result.status is ResultStatus.SUCCESS and result.data is not None:
                        candidates.append(result.data)
                        trend_candidates_count += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append((symbol, exc, "TREND_PATTERN_FAILED"))

            if hasattr(self._logger, "info"):
                try:
                    self._logger.info(
                        "scanner.trend_pattern_detection_completed",
                        trend_candidates=trend_candidates_count,
                        sector_bars_available=sector_bars is not None,
                        benchmark_bars_available=benchmark_bars is not None,
                    )
                except Exception:  # noqa: BLE001
                    pass

        if candidates and getattr(self._scanner_config, "enrich_with_fundamentals", False):
            from data.fundamental_adapter import fetch_fundamentals

            symbols = sorted({str(getattr(cand, "symbol", "")).strip().upper() for cand in candidates if getattr(cand, "symbol", None)})
            if symbols:
                fundamentals = fetch_fundamentals(symbols, timeout=5.0)
                if fundamentals.data:
                    for candidate in candidates:
                        candidate_symbol = str(getattr(candidate, "symbol", "")).strip().upper()
                        if not candidate_symbol:
                            continue
                        metrics = fundamentals.data.get(candidate_symbol)
                        if not isinstance(metrics, dict) or not metrics:
                            continue
                        meta = getattr(candidate, "meta", None)
                        if not isinstance(meta, dict):
                            continue
                        meta["fundamental_metrics"] = metrics
                        meta["fundamental_source"] = "yfinance"

        data_source = "unknown"
        for series in series_by_symbol.values():
            src = getattr(series, "source", None)
            if isinstance(src, str) and src:
                data_source = src
                break

        candidate_set = CandidateSet(
            schema_version="1.0.0",
            system_version=RunIDGenerator.get_system_version(),
            asof_timestamp=detected_at,
            candidates=candidates,
            total_scanned=total_scanned,
            total_detected=len(candidates),
            config_snapshot=msgspec.to_builtins(self._scanner_config),
            data_source=data_source,
            universe_source=str(getattr(universe, "source", "unknown")),
        )

        output = msgspec.to_builtins(candidate_set)
        _emit(
            self._emit_event,
            "module.executed",
            self._module_name,
            {
                "symbols_scanned": total_scanned,
                "candidates": len(candidates),
                "errors": len(errors),
                "skipped": len(skipped),
            },
        )

        self.regime_breakthrough_config = None
        self.current_regime = None

        if errors and total_scanned > 0 and len(errors) >= total_scanned:
            symbol, error, reason = errors[0]
            wrapped = PartialDataError.from_error(
                error,
                module=self._MODULE_NAME,
                reason_code="SCANNER_ALL_FAILED",
                details={"first_failed": symbol, "first_reason": reason, "failed_symbols": sorted({s for s, _, _ in errors})},
            )
            return Result.failed(wrapped, wrapped.reason_code)

        if errors or degraded_seen or skipped:
            details = {"errors": [{"symbol": s, "reason_code": r, "error": str(e)} for s, e, r in errors]}
            if skipped:
                # Include skipped stocks in details for backtest analysis
                details["skipped"] = [{"symbol": s, "reason_code": r} for s, r in skipped]
                details["skipped_count"] = len(skipped)
            wrapped = PartialDataError(
                "Scanner degraded: some symbols/windows failed or skipped.",
                module=self._MODULE_NAME,
                reason_code="SCANNER_DEGRADED",
                details=details,
            )
            return Result.degraded(output, wrapped, wrapped.reason_code)

        return Result.success(output)

    def cleanup(self) -> Result[None]:
        """Cleanup hook for the scanner adapter (no-op)."""

        _emit(self._emit_event, "module.cleaned_up", self._module_name, {"module": "scanner"})
        return Result.success(data=None)

    def _parse_input(self, payload: Mapping[str, Any]) -> Result[tuple[UniverseSnapshot, dict[str, PriceSeriesSnapshot]]]:
        """Parse orchestrator payload into typed universe + per-symbol series snapshots."""

        universe_raw = payload.get("universe")
        if not isinstance(universe_raw, Mapping):
            return Result.failed(ValidationError("Missing universe payload.", module=self._MODULE_NAME, reason_code="MISSING_UNIVERSE"), "MISSING_UNIVERSE")

        series_raw = payload.get("series_by_symbol")
        if not isinstance(series_raw, Mapping):
            return Result.failed(ValidationError("Missing series_by_symbol payload.", module=self._MODULE_NAME, reason_code="MISSING_SERIES"), "MISSING_SERIES")

        try:
            universe = msgspec.convert(dict(universe_raw), type=UniverseSnapshot)
        except Exception as exc:  # noqa: BLE001
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code="INVALID_UNIVERSE_PAYLOAD",
                details={"payload_keys": list(universe_raw.keys())},
            )
            return Result.failed(error, error.reason_code)

        series_by_symbol: dict[str, PriceSeriesSnapshot] = {}
        for symbol, raw in series_raw.items():
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            if not isinstance(raw, Mapping):
                continue
            try:
                series_by_symbol[symbol.strip().upper()] = msgspec.convert(dict(raw), type=PriceSeriesSnapshot)
            except Exception as exc:  # noqa: BLE001
                if hasattr(self._logger, "warning"):
                    try:
                        self._logger.warning("scanner.series_decode_failed", symbol=symbol, error=str(exc))
                    except Exception:  # noqa: BLE001
                        pass
                continue

        return Result.success((universe, series_by_symbol))


class EventGuardPluginConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration schema for `EventGuardPlugin`.

    Notes:
        The underlying `event_guard.interface.EventGuardConfig` requires
        `primary_source`. This adapter defaults it to `"polygon"` when omitted
        so the plugin can be enabled with minimal config.
    """

    enabled: bool = True
    lookahead_days: int | None = None

    # EventGuardConfig fields (kept optional here; we apply defaults).
    primary_source: str | None = None
    fallback_source: str | None = None
    manual_events_file: str | None = None
    earnings_blackout_days_before: int | None = None
    earnings_blackout_days_after: int | None = None
    split_blackout_days: int | None = None
    lockup_blackout_days: int | None = None
    use_conservative_defaults: bool | None = None
    sources: dict[str, Any] | None = None


class EventGuardPlugin(PluginBase[EventGuardPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Orchestrator-facing adapter for Event Guard (Phase 3.1).

    Input:
        A builtins `dict` representation of `scanner.interface.CandidateSet`
        (as emitted by `ScannerPlugin`).

    Output:
        A mapping:
            - "candidates": CandidateSet builtins dict (pass-through)
            - "constraints": {symbol: TradeConstraints builtins dict}
    """

    metadata: ClassVar[PluginMetadata] = PluginMetadata(
        name="event_guard",
        version="3.1.0",
        schema_version="1.0.0",
        category=PluginCategory.RISK_POLICY,
        enabled=True,
        description="Event Guard adapter producing per-symbol TradeConstraints for candidates.",
    )

    _MODULE_NAME = "orchestrator.plugins.event_guard"
    _CONFIG_INVALID_REASON = "EVENT_GUARD_PLUGIN_CONFIG_INVALID"

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._validated: EventGuardPluginConfig = EventGuardPluginConfig()
        self._event_guard_config: EventGuardConfig | None = None

        self._logger: logging.Logger | BoundLogger = logging.getLogger(self._MODULE_NAME)
        self._emit_event: EventEmitter | None = None
        self._module_name: str = self.metadata.name

    def init(self, context: PluginContext | None = None) -> Result[None]:
        """Initialise the Event Guard plugin (bind logger + event emitter)."""

        self._logger = _get_logger(context, self._MODULE_NAME)
        self._emit_event = _get_emitter(context)

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str) and module_name:
                self._module_name = module_name

        if self._validated.enabled and self._event_guard_config is None:
            error = ValidationError(
                "EventGuardPlugin configuration was not validated before init.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(self.config)},
            )
            return Result.failed(error, error.reason_code)

        _emit(self._emit_event, "module.initialised", self._module_name, {"module": "event_guard"})
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[EventGuardPluginConfig]:
        """Validate config and materialize an `EventGuardConfig` instance."""

        if not isinstance(config, Mapping):
            validated = EventGuardPluginConfig()
            self._validated = validated
            self._event_guard_config = self._build_event_guard_config(validated).data
            return Result.success(validated)

        config_map = dict(config)
        # Allow nested blocks: {"event_guard": {...}} (common in unified configs).
        nested = _coerce_mapping(config_map.get("event_guard"))
        merged = dict(nested or config_map)

        validated = EventGuardPluginConfig(
            enabled=_safe_bool(merged.get("enabled", True), default=True),
            lookahead_days=_safe_int(merged.get("lookahead_days"), default=0) or None,
            primary_source=str(merged.get("primary_source") or "").strip() or None,
            fallback_source=str(merged.get("fallback_source") or "").strip() or None,
            manual_events_file=str(merged.get("manual_events_file") or "").strip() or None,
            earnings_blackout_days_before=_safe_int(merged.get("earnings_blackout_days_before"), default=0) or None,
            earnings_blackout_days_after=_safe_int(merged.get("earnings_blackout_days_after"), default=0) or None,
            split_blackout_days=_safe_int(merged.get("split_blackout_days"), default=0) or None,
            lockup_blackout_days=_safe_int(merged.get("lockup_blackout_days"), default=0) or None,
            use_conservative_defaults=_safe_bool(merged.get("use_conservative_defaults"), default=True),
            sources=_coerce_mapping(merged.get("sources")),
        )
        self._validated = validated

        if not validated.enabled:
            self._event_guard_config = None
            return Result.success(validated)

        cfg_result = self._build_event_guard_config(validated)
        if cfg_result.status is ResultStatus.FAILED or cfg_result.data is None:
            err = cfg_result.error or ConfigurationError(
                "Failed to configure Event Guard.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(config)},
            )
            return Result.failed(err, cfg_result.reason_code or self._CONFIG_INVALID_REASON)

        self._event_guard_config = cfg_result.data
        return Result.success(validated)

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Fetch events and compute trading constraints for candidate symbols."""

        parsed = self._parse_candidate_set(payload)
        if parsed.status is ResultStatus.FAILED or parsed.data is None:
            return Result.failed(parsed.error or RuntimeError("CandidateSet payload invalid."), parsed.reason_code or "INVALID_CANDIDATES_PAYLOAD")
        candidate_set = parsed.data

        candidate_builtins = msgspec.to_builtins(candidate_set)

        if not self._validated.enabled or self._event_guard_config is None:
            output = EventGuardOutput(
                schema_version="1.1.0",
                system_version=str(getattr(candidate_set, "system_version", RunIDGenerator.get_system_version())),
                asof_timestamp=int(time.time_ns()),
                candidates=candidate_builtins,
                constraints={},
            )
            output_builtins = msgspec.to_builtins(output)
            _emit(self._emit_event, "module.executed", self._module_name, {"symbols": 0, "constraints": 0, "disabled": True})
            return Result.success(output_builtins)

        symbols = self._extract_symbols(candidate_set)
        if not symbols:
            output = EventGuardOutput(
                schema_version="1.1.0",
                system_version=str(getattr(candidate_set, "system_version", RunIDGenerator.get_system_version())),
                asof_timestamp=int(time.time_ns()),
                candidates=candidate_builtins,
                constraints={},
            )
            output_builtins = msgspec.to_builtins(output)
            _emit(self._emit_event, "module.executed", self._module_name, {"symbols": 0, "constraints": 0})
            return Result.success(output_builtins)

        _override_ts = payload.get("scan_timestamp_ns")
        now_ns = int(_override_ts) if _override_ts is not None else int(time.time_ns())
        lookahead_days = int(self._validated.lookahead_days or self._compute_lookahead_days(self._event_guard_config))
        end_ns = now_ns + max(1, lookahead_days) * 86_400 * 1_000_000_000

        fetch_result = fetch_events_with_fallback(self._event_guard_config, symbols, now_ns, end_ns)
        snapshot: EventSnapshot
        fetch_degraded = fetch_result.status is ResultStatus.DEGRADED
        if fetch_result.status is ResultStatus.FAILED or fetch_result.data is None:
            # Degrade into an empty snapshot; `apply_event_guard` will enforce conservative defaults when configured.
            snapshot = EventSnapshot(
                schema_version="1.0.0",
                system_version=RunIDGenerator.get_system_version(),
                asof_timestamp=now_ns,
                events=[],
                source="degraded",
                symbols_covered=list(symbols),
            )
            fetch_degraded = True
        else:
            snapshot = fetch_result.data

        try:
            constraints_by_symbol = apply_event_guard(symbols, snapshot, self._event_guard_config, current_time_ns=now_ns)
        except Exception as exc:  # noqa: BLE001 - must not crash orchestrator
            err = PartialDataError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code="EVENT_GUARD_APPLY_FAILED",
                details={"symbols": symbols},
            )
            return Result.failed(err, err.reason_code)

        constraints_builtins: dict[str, Any] = {}
        for symbol, constraints in constraints_by_symbol.items():
            try:
                constraints_builtins[str(symbol)] = msgspec.to_builtins(constraints)
            except Exception:  # noqa: BLE001
                continue

        output = EventGuardOutput(
            schema_version="1.1.0",
            system_version=str(getattr(candidate_set, "system_version", RunIDGenerator.get_system_version())),
            asof_timestamp=int(time.time_ns()),
            candidates=candidate_builtins,
            constraints=constraints_builtins,
        )

        meta = {
            "symbols": len(symbols),
            "constraints": len(constraints_builtins),
            "events_source": getattr(snapshot, "source", "unknown"),
        }
        _emit(self._emit_event, "module.executed", self._module_name, meta)

        output_builtins = msgspec.to_builtins(output)

        if fetch_degraded:
            err = fetch_result.error or RuntimeError("Event Guard degraded.")
            reason = fetch_result.reason_code or "EVENT_GUARD_DEGRADED"
            return Result.degraded(output_builtins, err, reason)

        return Result.success(output_builtins)

    def cleanup(self) -> Result[None]:
        """Cleanup hook for the Event Guard adapter (no-op)."""

        _emit(self._emit_event, "module.cleaned_up", self._module_name, {"module": "event_guard"})
        return Result.success(data=None)

    def _build_event_guard_config(self, validated: EventGuardPluginConfig) -> Result[EventGuardConfig]:
        payload: dict[str, Any] = {
            "primary_source": (validated.primary_source or "polygon").strip() or "polygon",
            "fallback_source": validated.fallback_source,
            "manual_events_file": validated.manual_events_file,
            "earnings_blackout_days_before": validated.earnings_blackout_days_before,
            "earnings_blackout_days_after": validated.earnings_blackout_days_after,
            "split_blackout_days": validated.split_blackout_days,
            "lockup_blackout_days": validated.lockup_blackout_days,
            "use_conservative_defaults": validated.use_conservative_defaults,
            "sources": validated.sources,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            return Result.success(msgspec.convert(payload, type=EventGuardConfig))
        except Exception as exc:  # noqa: BLE001
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": payload},
            )
            return Result.failed(error, error.reason_code)

    def _compute_lookahead_days(self, config: EventGuardConfig) -> int:
        earnings_span = int(config.earnings_blackout_days_before) + int(config.earnings_blackout_days_after) + 1
        lockup_span = int(config.lockup_blackout_days) * 2 + 1
        split_span = int(config.split_blackout_days) + 1
        return max(7, earnings_span, lockup_span, split_span)

    def _parse_candidate_set(self, payload: Mapping[str, Any]) -> Result[CandidateSet]:
        try:
            return Result.success(msgspec.convert(dict(payload), type=CandidateSet))
        except Exception as exc:  # noqa: BLE001
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code="INVALID_CANDIDATES_PAYLOAD",
                details={"payload_keys": list(payload.keys()) if isinstance(payload, Mapping) else []},
            )
            return Result.failed(error, error.reason_code)

    def _extract_symbols(self, candidate_set: CandidateSet) -> list[str]:
        symbols: list[str] = []
        seen: set[str] = set()
        for candidate in list(candidate_set.candidates or []):
            symbol = str(getattr(candidate, "symbol", "")).strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
        return symbols


class StrategyPluginConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration schema for `StrategyPlugin`."""

    enabled: bool = True
    account_equity: float = 100_000.0
    engine: Mapping[str, Any] | None = None


class StrategyPlugin(PluginBase[StrategyPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Orchestrator-facing adapter for Strategy Engine (Phase 3.2).

    Input:
        A mapping produced by `EventGuardPlugin`, containing:
            - "candidates": CandidateSet builtins dict
            - "constraints": {symbol: TradeConstraints builtins dict}

        The adapter also tolerates optional market data fields:
            - "market_data" or "series_by_symbol": {symbol: PriceSeriesSnapshot builtins dict}

    Output:
        A mapping:
            - "intents": OrderIntentSet builtins dict
            - "candidates": pass-through CandidateSet builtins dict
            - "constraints": pass-through constraints builtins dict
    """

    metadata: ClassVar[PluginMetadata] = PluginMetadata(
        name="strategy",
        version="3.2.0",
        schema_version="1.0.0",
        category=PluginCategory.STRATEGY,
        enabled=True,
        description="Strategy Engine adapter producing a serializable OrderIntentSet payload.",
    )

    _MODULE_NAME = "orchestrator.plugins.strategy"
    _CONFIG_INVALID_REASON = "STRATEGY_PLUGIN_CONFIG_INVALID"

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._validated: StrategyPluginConfig = StrategyPluginConfig()
        self._engine_config: StrategyEngineConfig = StrategyEngineConfig()
        self._position_sizer: Any | None = None
        self._price_policy: Any | None = None

        self._logger: logging.Logger | BoundLogger = logging.getLogger(self._MODULE_NAME)
        self._emit_event: EventEmitter | None = None
        self._module_name: str = self.metadata.name

    def init(self, context: PluginContext | None = None) -> Result[None]:
        """Initialise the strategy plugin (bind logger + event emitter)."""

        self._logger = _get_logger(context, self._MODULE_NAME)
        self._emit_event = _get_emitter(context)

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str) and module_name:
                self._module_name = module_name

        if self._validated.enabled and (self._position_sizer is None or self._price_policy is None):
            error = ValidationError(
                "StrategyPlugin configuration was not validated before init.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(self.config)},
            )
            return Result.failed(error, error.reason_code)

        _emit(self._emit_event, "module.initialised", self._module_name, {"module": "strategy"})
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[StrategyPluginConfig]:
        """Validate config and materialize Strategy Engine dependencies."""

        if not isinstance(config, Mapping):
            validated = StrategyPluginConfig()
            self._validated = validated
            self._engine_config = StrategyEngineConfig()
            try:
                self._position_sizer = create_position_sizer(self._engine_config)
                self._price_policy = create_price_policy(self._engine_config)
            except Exception as exc:  # noqa: BLE001
                error = ConfigurationError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code=self._CONFIG_INVALID_REASON,
                    details={"config": dict(config) if isinstance(config, Mapping) else {}},
                )
                return Result.failed(error, error.reason_code)
            return Result.success(validated)

        config_map = dict(config)
        nested = _coerce_mapping(config_map.get("strategy"))
        merged = dict(nested or config_map)

        engine_map = _coerce_mapping(merged.get("engine")) or merged
        validated = StrategyPluginConfig(
            enabled=_safe_bool(merged.get("enabled", True), default=True),
            account_equity=_safe_float(merged.get("account_equity"), default=100_000.0),
            engine=engine_map,
        )
        self._validated = validated

        if not validated.enabled:
            self._position_sizer = None
            self._price_policy = None
            return Result.success(validated)

        try:
            self._engine_config = msgspec.convert(dict(validated.engine or {}), type=StrategyEngineConfig)
        except Exception as exc:  # noqa: BLE001
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(validated.engine or {})},
            )
            return Result.failed(error, error.reason_code)

        try:
            self._position_sizer = create_position_sizer(self._engine_config)
            self._price_policy = create_price_policy(self._engine_config)
        except Exception as exc:  # noqa: BLE001
            error = ConfigurationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"engine": msgspec.to_builtins(self._engine_config)},
            )
            return Result.failed(error, error.reason_code)

        return Result.success(validated)

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Convert candidates + constraints into an OrderIntentSet snapshot."""

        parsed_candidates = self._parse_candidate_set(payload)
        if parsed_candidates.status is ResultStatus.FAILED or parsed_candidates.data is None:
            return Result.failed(parsed_candidates.error or RuntimeError("Missing candidates."), parsed_candidates.reason_code or "MISSING_CANDIDATES")
        candidates = parsed_candidates.data
        candidates_builtins = msgspec.to_builtins(candidates)

        constraints_result = self._parse_constraints(payload)
        if constraints_result.status is ResultStatus.FAILED:
            return Result.failed(constraints_result.error or RuntimeError("Invalid constraints payload."), constraints_result.reason_code or "INVALID_CONSTRAINTS")
        constraints = constraints_result.data or {}
        constraints_builtins = {sym: msgspec.to_builtins(c) for sym, c in constraints.items()}

        market_data, market_decode_errors = self._parse_market_data(payload)

        if not self._validated.enabled:
            now_ns = int(time.time_ns())
            empty = OrderIntentSet(
                schema_version="1.2.0",
                system_version=candidates.system_version,
                asof_timestamp=now_ns,
                intent_groups=[],
                constraints_applied={},
                source_candidates=[],
            )
            output: Mapping[str, Any] = {"intents": msgspec.to_builtins(empty), "candidates": candidates_builtins, "constraints": constraints_builtins}
            _emit(self._emit_event, "module.executed", self._module_name, {"disabled": True, "intent_groups": 0})
            return Result.success(output)

        if self._position_sizer is None or self._price_policy is None:
            return Result.failed(RuntimeError("StrategyPlugin was not initialised."), "NOT_INITIALISED")

        account_equity = self._validated.account_equity
        if "account_equity" in payload:
            account_equity = _safe_float(payload.get("account_equity"), default=account_equity)

        _scan_ts = payload.get("scan_timestamp_ns")
        engine_result = generate_intents(
            candidates=candidates,
            constraints=constraints,
            market_data=market_data,
            account_equity=account_equity,
            config=self._engine_config,
            position_sizer=self._position_sizer,
            price_policy=self._price_policy,
            current_time_ns=int(_scan_ts) if _scan_ts is not None else None,
        )
        if engine_result.status is ResultStatus.FAILED or engine_result.data is None:
            return Result.failed(engine_result.error or RuntimeError("Strategy engine failed."), engine_result.reason_code or "STRATEGY_FAILED")

        intents_builtins = msgspec.to_builtins(engine_result.data)
        output = StrategyOutput(
            schema_version="1.2.0",
            system_version=str(getattr(engine_result.data, "system_version", RunIDGenerator.get_system_version())),
            asof_timestamp=int(time.time_ns()),
            intents=intents_builtins,
            candidates=candidates_builtins,
            constraints=constraints_builtins,
        )

        meta = {
            "candidates": int(getattr(candidates, "total_detected", len(getattr(candidates, "candidates", []) or []))),
            "intent_groups": len(getattr(engine_result.data, "intent_groups", []) or []),
            "market_data_decoding_errors": market_decode_errors,
        }
        _emit(self._emit_event, "module.executed", self._module_name, meta)

        output_builtins = msgspec.to_builtins(output)
        if engine_result.status is ResultStatus.DEGRADED or market_decode_errors:
            err = engine_result.error or RuntimeError("Strategy degraded.")
            reason = engine_result.reason_code or ("MARKET_DATA_DECODE_PARTIAL" if market_decode_errors else "STRATEGY_DEGRADED")
            return Result.degraded(output_builtins, err, reason)

        return Result.success(output_builtins)

    def cleanup(self) -> Result[None]:
        """Cleanup hook for the strategy adapter (no-op)."""

        _emit(self._emit_event, "module.cleaned_up", self._module_name, {"module": "strategy"})
        return Result.success(data=None)

    def _parse_candidate_set(self, payload: Mapping[str, Any]) -> Result[CandidateSet]:
        raw = payload.get("candidates")
        if isinstance(raw, Mapping):
            try:
                return Result.success(msgspec.convert(dict(raw), type=CandidateSet))
            except Exception as exc:  # noqa: BLE001
                error = ValidationError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code="INVALID_CANDIDATES_PAYLOAD",
                    details={"payload_keys": list(raw.keys())},
                )
                return Result.failed(error, error.reason_code)

        return Result.failed(
            ValidationError("Missing candidates payload.", module=self._MODULE_NAME, reason_code="MISSING_CANDIDATES"),
            "MISSING_CANDIDATES",
        )

    def _parse_constraints(self, payload: Mapping[str, Any]) -> Result[dict[str, TradeConstraints]]:
        constraints_raw = payload.get("constraints")
        if constraints_raw is None:
            return Result.success({})
        if not isinstance(constraints_raw, Mapping):
            return Result.failed(
                ValidationError("constraints must be a mapping.", module=self._MODULE_NAME, reason_code="INVALID_CONSTRAINTS"),
                "INVALID_CONSTRAINTS",
            )

        parsed: dict[str, TradeConstraints] = {}
        decode_errors = 0
        for symbol, raw in constraints_raw.items():
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            raw_map = _coerce_mapping(raw)
            if raw_map is None:
                decode_errors += 1
                continue
            try:
                parsed[symbol.strip().upper()] = msgspec.convert(dict(raw_map), type=TradeConstraints)
            except Exception:  # noqa: BLE001
                decode_errors += 1
                continue

        if decode_errors:
            err = PartialDataError(
                "Some TradeConstraints entries failed to decode.",
                module=self._MODULE_NAME,
                reason_code="CONSTRAINTS_DECODE_PARTIAL",
                details={"decode_errors": decode_errors},
            )
            return Result.degraded(parsed, err, err.reason_code)

        return Result.success(parsed)

    def _parse_market_data(self, payload: Mapping[str, Any]) -> tuple[dict[str, PriceSeriesSnapshot], int]:
        series_raw = payload.get("market_data") or payload.get("series_by_symbol") or {}
        if not isinstance(series_raw, Mapping):
            return {}, 0

        decoded: dict[str, PriceSeriesSnapshot] = {}
        errors = 0
        for symbol, raw in series_raw.items():
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            raw_map = _coerce_mapping(raw)
            if raw_map is None:
                errors += 1
                continue
            try:
                decoded[symbol.strip().upper()] = msgspec.convert(dict(raw_map), type=PriceSeriesSnapshot)
            except Exception:  # noqa: BLE001
                errors += 1
                continue
        return decoded, errors


class RiskGatePluginConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration schema for `RiskGatePlugin`."""

    enabled: bool = True
    account_equity: float = 100_000.0
    safe_mode_state: str = "ACTIVE"
    state_path: str = "artifacts/system_state/safe_mode.json"
    checks: list[str] = msgspec.field(default_factory=lambda: [
        "leverage", "concentration", "capital_usage",
        "reduce_only", "order_count", "rate_limit", "price_band",
    ])
    gate: Mapping[str, Any] | None = None


class RiskGatePlugin(PluginBase[RiskGatePluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Orchestrator-facing adapter for Risk Gate (Phase 3.3).

    Input:
        A mapping containing:
            - "intents": OrderIntentSet builtins dict (required)
        Optional:
            - "market_data" / "series_by_symbol": {symbol: PriceSeriesSnapshot builtins dict}
            - "positions": {symbol: Position} (shape is Any in early phases)
            - "portfolio_snapshot": Any

    Output:
        A mapping:
            - "intents": OrderIntentSet builtins dict (pass-through)
            - "risk_decisions": RiskDecisionSet builtins dict
    """

    metadata: ClassVar[PluginMetadata] = PluginMetadata(
        name="risk_gate",
        version="3.3.0",
        schema_version="1.0.0",
        category=PluginCategory.RISK_POLICY,
        enabled=True,
        description="Risk Gate adapter producing a serializable RiskDecisionSet payload.",
    )

    _MODULE_NAME = "orchestrator.plugins.risk_gate"
    _CONFIG_INVALID_REASON = "RISK_GATE_PLUGIN_CONFIG_INVALID"

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._validated: RiskGatePluginConfig = RiskGatePluginConfig()
        self._gate_config: RiskGateConfig = RiskGateConfig()
        self._checks: list[Any] = []
        self._safe_mode_manager: Any | None = None

        self._logger: logging.Logger | BoundLogger = logging.getLogger(self._MODULE_NAME)
        self._emit_event: EventEmitter | None = None
        self._module_name: str = self.metadata.name

    def init(self, context: PluginContext | None = None) -> Result[None]:
        """Initialise the risk gate plugin (bind logger + event emitter)."""

        self._logger = _get_logger(context, self._MODULE_NAME)
        self._emit_event = _get_emitter(context)

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str) and module_name:
                self._module_name = module_name

        if self._validated.enabled and not self._checks:
            error = ValidationError(
                "RiskGatePlugin configuration was not validated before init.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(self.config)},
            )
            return Result.failed(error, error.reason_code)

        _emit(self._emit_event, "module.initialised", self._module_name, {"module": "risk_gate"})
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[RiskGatePluginConfig]:
        """Validate config and materialize checks + RiskGateConfig."""

        if not isinstance(config, Mapping):
            validated = RiskGatePluginConfig()
            self._validated = validated
            self._gate_config = RiskGateConfig()
            self._checks = self._build_checks(validated.checks)
            return Result.success(validated)

        config_map = dict(config)
        nested = _coerce_mapping(config_map.get("risk_gate"))
        merged = dict(nested or config_map)

        checks_raw = merged.get("checks")
        checks_list: list[str]
        if isinstance(checks_raw, Sequence) and not isinstance(checks_raw, (str, bytes)):
            checks_list = [str(item) for item in checks_raw if item is not None]
        else:
            checks_list = list(RiskGatePluginConfig().checks)

        validated = RiskGatePluginConfig(
            enabled=_safe_bool(merged.get("enabled", True), default=True),
            account_equity=_safe_float(merged.get("account_equity"), default=100_000.0),
            safe_mode_state=str(merged.get("safe_mode_state") or "ACTIVE").strip().upper() or "ACTIVE",
            state_path=str(merged.get("state_path") or RiskGatePluginConfig().state_path),
            checks=checks_list,
            gate=_coerce_mapping(merged.get("gate")) or _coerce_mapping(merged.get("config")) or merged,
        )
        self._validated = validated

        if not validated.enabled:
            self._checks = []
            self._safe_mode_manager = None
            return Result.success(validated)

        try:
            from risk_gate.safe_mode import SafeModeManager

            self._safe_mode_manager = SafeModeManager(state_path=validated.state_path)
        except Exception:  # noqa: BLE001
            self._safe_mode_manager = None

        try:
            self._gate_config = msgspec.convert(dict(validated.gate or {}), type=RiskGateConfig)
        except Exception as exc:  # noqa: BLE001
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(validated.gate or {})},
            )
            return Result.failed(error, error.reason_code)

        self._checks = self._build_checks(validated.checks)
        return Result.success(validated)

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Evaluate intents and produce a RiskDecisionSet snapshot."""

        intents_result = self._parse_intents(payload)
        if intents_result.status is ResultStatus.FAILED or intents_result.data is None:
            return Result.failed(intents_result.error or RuntimeError("Invalid intents payload."), intents_result.reason_code or "INVALID_INTENTS")
        intents = intents_result.data
        intents_builtins = msgspec.to_builtins(intents)

        if not self._validated.enabled:
            now_ns = int(time.time_ns())
            empty = RiskDecisionSet(
                schema_version="1.3.0",
                system_version=intents.system_version,
                asof_timestamp=now_ns,
                decisions=[],
                safe_mode_active=False,
                safe_mode_reason=None,
                constraints_snapshot={},
            )
            output = RiskGateOutput(
                schema_version="1.3.0",
                system_version=str(getattr(empty, "system_version", RunIDGenerator.get_system_version())),
                asof_timestamp=now_ns,
                intents=intents,
                decisions=empty,
                constraints={},
            )
            output_builtins = msgspec.to_builtins(output)
            _emit(self._emit_event, "module.executed", self._module_name, {"disabled": True, "decisions": 0})
            return Result.success(output_builtins)

        market_data, market_decode_errors = self._parse_market_data(payload)

        load_error: BaseException | None = None
        safe_mode_state_value = payload.get("safe_mode_state")
        if safe_mode_state_value is None and self._safe_mode_manager is not None:
            try:
                load_result = self._safe_mode_manager.load()
                if load_result.status is ResultStatus.DEGRADED:
                    load_error = load_result.error
                if load_result.data is not None:
                    safe_mode_state_value = load_result.data.state
            except Exception as exc:  # noqa: BLE001
                load_error = exc

        safe_mode = self._parse_safe_mode(safe_mode_state_value or self._validated.safe_mode_state)
        if safe_mode.status is ResultStatus.FAILED or safe_mode.data is None:
            return Result.failed(safe_mode.error or RuntimeError("Invalid safe_mode_state"), safe_mode.reason_code or "INVALID_SAFE_MODE")

        account_equity = self._validated.account_equity
        if "account_equity" in payload:
            account_equity = _safe_float(payload.get("account_equity"), default=account_equity)

        positions_raw = payload.get("positions") if isinstance(payload, Mapping) else None
        positions = dict(positions_raw) if isinstance(positions_raw, Mapping) else {}

        context = RiskCheckContext(
            portfolio_snapshot=payload.get("portfolio_snapshot"),
            account_equity=float(account_equity),
            positions=cast(dict[str, Any], positions),
            market_data=market_data,
            safe_mode_state=safe_mode.data,
        )

        flattened_intents: list[TradeIntent] = []
        for group in list(getattr(intents, "intent_groups", []) or []):
            flattened_intents.extend(list(getattr(group, "intents", []) or []))

        daily_limit_check = risk_checks.check_daily_new_position_limit(
            flattened_intents,
            self._gate_config,
            market_data=market_data,
        )
        if daily_limit_check.status == CheckStatus.FAIL:
            kept_symbols = set(daily_limit_check.details.get("kept_symbols", []) or [])
            filtered_groups = []
            for group in list(getattr(intents, "intent_groups", []) or []):
                group_intents = list(getattr(group, "intents", []) or [])
                is_new_position_group = any(
                    getattr(intent, "intent_type", None) in (IntentType.OPEN_LONG, IntentType.OPEN_SHORT)
                    and not bool(getattr(intent, "reduce_only", False))
                    for intent in group_intents
                )
                if (not is_new_position_group) or str(getattr(group, "symbol", "")).strip() in kept_symbols:
                    filtered_groups.append(group)
            intents = msgspec.structs.replace(intents, intent_groups=filtered_groups)

            self._logger.warning(
                "daily_new_position_limit_exceeded rejected_count=%s rejected_symbols=%s kept_symbols=%s",
                daily_limit_check.details.get("rejected_count"),
                daily_limit_check.details.get("rejected_symbols"),
                sorted(kept_symbols),
            )

        engine_result = evaluate_intents(
            intents=intents,
            context=context,
            config=self._gate_config,
            checks=self._checks,
        )
        if engine_result.status is ResultStatus.FAILED or engine_result.data is None:
            return Result.failed(engine_result.error or RuntimeError("Risk gate failed."), engine_result.reason_code or "RISK_GATE_FAILED")

        output = RiskGateOutput(
            schema_version="1.3.0",
            system_version=str(getattr(engine_result.data, "system_version", RunIDGenerator.get_system_version())),
            asof_timestamp=int(time.time_ns()),
            intents=intents,
            decisions=engine_result.data,
            constraints=msgspec.to_builtins(self._gate_config),
        )
        output_builtins = msgspec.to_builtins(output)
        _emit(self._emit_event, "module.executed", self._module_name, {"decisions": len(getattr(engine_result.data, "decisions", []) or []), "market_data_decoding_errors": market_decode_errors})

        if engine_result.status is ResultStatus.DEGRADED or market_decode_errors or load_error is not None:
            err = engine_result.error or load_error or RuntimeError("Risk gate degraded.")
            reason = engine_result.reason_code or ("MARKET_DATA_DECODE_PARTIAL" if market_decode_errors else "RISK_GATE_DEGRADED")
            return Result.degraded(output_builtins, err, reason)

        return Result.success(output_builtins)

    def cleanup(self) -> Result[None]:
        """Cleanup hook for the risk gate adapter (no-op)."""

        _emit(self._emit_event, "module.cleaned_up", self._module_name, {"module": "risk_gate"})
        return Result.success(data=None)

    def _build_checks(self, names: Sequence[str]) -> list[Any]:
        mapping: dict[str, Any] = {
            "leverage": risk_checks.LeverageCheck,
            "drawdown": risk_checks.DrawdownCheck,
            "daily_loss": risk_checks.DailyLossCheck,
            "concentration": risk_checks.ConcentrationCheck,
            "max_position": risk_checks.MaxPositionCheck,
            "price_band": risk_checks.PriceBandCheck,
            "volatility_halt": risk_checks.VolatilityHaltCheck,
            "reduce_only": risk_checks.ReduceOnlyCheck,
            "order_count": risk_checks.OrderCountCheck,
            "rate_limit": risk_checks.RateLimitCheck,
            "capital_usage": CapitalUsageCheck,
        }
        out: list[Any] = []
        for raw in list(names or []):
            key = str(raw).strip().lower()
            if not key:
                continue
            cls = mapping.get(key)
            if cls is None:
                continue
            try:
                out.append(cls())
            except Exception:  # noqa: BLE001
                continue
        return out

    def _parse_intents(self, payload: Mapping[str, Any]) -> Result[OrderIntentSet]:
        raw = payload.get("intents")
        if isinstance(raw, Mapping):
            try:
                return Result.success(msgspec.convert(dict(raw), type=OrderIntentSet))
            except Exception as exc:  # noqa: BLE001
                error = ValidationError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code="INVALID_INTENTS",
                    details={"payload_keys": list(raw.keys())},
                )
                return Result.failed(error, error.reason_code)

        raw_intent_sets = payload.get("intent_sets")
        if isinstance(raw_intent_sets, list):
            first_intent_set = raw_intent_sets[0] if raw_intent_sets else None
            if isinstance(first_intent_set, Mapping):
                try:
                    return Result.success(msgspec.convert(dict(first_intent_set), type=OrderIntentSet))
                except Exception as exc:  # noqa: BLE001
                    error = ValidationError.from_error(
                        exc,
                        module=self._MODULE_NAME,
                        reason_code="INVALID_INTENTS",
                        details={"payload_keys": list(first_intent_set.keys())},
                    )
                    return Result.failed(error, error.reason_code)
            if all(key in payload for key in ("schema_version", "system_version", "asof_timestamp")):
                return Result.success(
                    OrderIntentSet(
                        schema_version=str(payload["schema_version"]),
                        system_version=str(payload["system_version"]),
                        asof_timestamp=int(payload["asof_timestamp"]),
                    )
                )

        if isinstance(payload, Mapping) and "schema_version" in payload and "intent_groups" in payload:
            try:
                return Result.success(msgspec.convert(dict(payload), type=OrderIntentSet))
            except Exception as exc:  # noqa: BLE001
                error = ValidationError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code="INVALID_INTENTS",
                    details={"payload_keys": list(payload.keys())},
                )
                return Result.failed(error, error.reason_code)

        return Result.failed(
            ValidationError("Missing intents payload.", module=self._MODULE_NAME, reason_code="MISSING_INTENTS"),
            "MISSING_INTENTS",
        )

    def _parse_market_data(self, payload: Mapping[str, Any]) -> tuple[dict[str, PriceSeriesSnapshot], int]:
        series_raw = payload.get("market_data") or payload.get("series_by_symbol") or {}
        if not isinstance(series_raw, Mapping):
            return {}, 0
        decoded: dict[str, PriceSeriesSnapshot] = {}
        errors = 0
        for symbol, raw in series_raw.items():
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            raw_map = _coerce_mapping(raw)
            if raw_map is None:
                errors += 1
                continue
            try:
                decoded[symbol.strip().upper()] = msgspec.convert(dict(raw_map), type=PriceSeriesSnapshot)
            except Exception:  # noqa: BLE001
                errors += 1
                continue
        return decoded, errors

    def _parse_safe_mode(self, value: Any) -> Result[SafeModeState]:
        if isinstance(value, SafeModeState):
            return Result.success(value)
        if isinstance(value, str):
            candidate = value.strip().upper()
            try:
                return Result.success(SafeModeState[candidate])
            except KeyError:
                try:
                    return Result.success(SafeModeState(candidate))
                except ValueError:
                    pass
        error = ValidationError(
            "safe_mode_state must be a SafeModeState or string literal.",
            module=self._MODULE_NAME,
            reason_code="INVALID_SAFE_MODE",
            details={"safe_mode_state": value},
        )
        return Result.failed(error, error.reason_code)


class ExecutionPluginConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration schema for `ExecutionPlugin`.

    Supported keys (all optional):
        enabled: bool
        adapter: str
        broker: mapping
        persistence_dir: filesystem path
        connect_on_init: bool
        enforce_risk_decisions: bool
        dry_run: bool | None
        cleanup_strategy: str
            - "keep_all": do not cancel any open orders on cleanup (default; backwards compatible)
            - "cancel_open_only": cancel only OPEN_LONG/OPEN_SHORT orders on cleanup
            - "cancel_all": cancel all open/pending orders on cleanup
    """

    enabled: bool = True
    adapter: str = "ibkr"
    broker: Mapping[str, Any] | None = None
    persistence_dir: str = ".cache/execution"
    connect_on_init: bool = False
    enforce_risk_decisions: bool = True
    dry_run: bool | None = None
    # Strategy for `ExecutionPlugin.cleanup()` cancellation pass (best-effort).
    # Options:
    #   - "keep_all": keep all open orders (default; backwards compatible)
    #   - "cancel_open_only": cancel only OPEN_LONG/OPEN_SHORT orders
    #   - "cancel_all": cancel all open/pending orders
    cleanup_strategy: str = "keep_all"


class ExecutionPlugin(PluginBase[ExecutionPluginConfig, Mapping[str, Any], Mapping[str, Any]]):
    """Orchestrator-facing adapter for Execution (Phase 3.5).

    Input:
        A mapping containing:
            - "intents": OrderIntentSet builtins dict
            - "risk_decisions": RiskDecisionSet builtins dict

    Output:
        A mapping:
            - "reports": list[ExecutionReport builtins dict]
            - "intents": pass-through OrderIntentSet builtins dict
            - "risk_decisions": pass-through RiskDecisionSet builtins dict
    """

    metadata: ClassVar[PluginMetadata] = PluginMetadata(
        name="execution",
        version="3.5.0",
        schema_version="1.0.0",
        category=PluginCategory.SIGNAL,
        enabled=True,
        description="Execution adapter producing a serializable list of ExecutionReport payloads.",
    )

    _MODULE_NAME = "orchestrator.plugins.execution"
    _CONFIG_INVALID_REASON = "EXECUTION_PLUGIN_CONFIG_INVALID"

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._validated: ExecutionPluginConfig = ExecutionPluginConfig()
        self._connection_config: BrokerConnectionConfig | None = None
        self._manager: OrderManager | None = None
        self._adapter: Any | None = None
        self._run_id: str = "unknown"
        # PRE_MARKET_CHECK integration point:
        # ExecutionPlugin is executed standalone for RunType.PRE_MARKET_CHECK (it does not
        # run the EventGuard module). When available, the orchestrator can inject an
        # "event_guard_snapshot" payload (e.g., from a previous run artifact) which we
        # temporarily store here for `_scan_overnight_events`.
        self._event_guard_snapshot: Any | None = None

        self._logger: logging.Logger | BoundLogger = logging.getLogger(self._MODULE_NAME)
        self._emit_event: EventEmitter | None = None
        self._module_name: str = self.metadata.name

    def _scan_overnight_events(self) -> list[dict[str, Any]]:
        """Scan for overnight events affecting current positions/orders.

        This routine is intentionally "best effort" and must not block the rest of the
        pre-market checks. The EventGuard module is not executed in the PRE_MARKET_CHECK
        run type; instead we accept an optional `event_guard_snapshot` injected via the
        payload (stored temporarily in `self._event_guard_snapshot`).

        Returns:
            List of overnight events with symbol, event_type, risk_level, event_date
        """

        if self._manager is None:
            self._logger.error("pre_market.overnight_events_scanned", status="FAILED", reason="NOT_INITIALISED")
            return []

        # 1) Extract the set of symbols we care about (open positions + pending orders).
        try:
            orders_result = self._manager.get_all_orders()
        except Exception as exc:  # noqa: BLE001 - pre-market checks should not crash
            self._logger.exception("pre_market.overnight_events_scanned", status="FAILED", reason="ORDER_SNAPSHOT_EXCEPTION", error=str(exc))
            return []

        if orders_result.status is not ResultStatus.SUCCESS:
            self._logger.error(
                "pre_market.overnight_events_scanned",
                status="FAILED",
                reason=orders_result.reason_code or "ORDER_SNAPSHOT_FAILED",
                error=str(orders_result.error or "failed to load orders"),
            )
            return []

        mappings = list(orders_result.data or [])
        symbols: set[str] = set()
        for mapping in mappings:
            intent_snapshot = getattr(mapping, "intent_snapshot", None)
            symbol = str(_intent_get(intent_snapshot, "symbol", "") or "").strip().upper()
            if symbol:
                symbols.add(symbol)

        if not symbols:
            self._logger.info("pre_market.overnight_events_scanned", status="SUCCESS", symbols=0, events=0, filtered=0)
            return []

        # 2) Define scan window: yesterday 20:00 UTC (PRE_MARKET_FULL_SCAN) to now.
        now = datetime.now(UTC)
        yesterday_eod = now.replace(hour=20, minute=0, second=0, microsecond=0) - timedelta(days=1)
        start_ns = int(yesterday_eod.timestamp() * 1e9)
        end_ns = int(now.timestamp() * 1e9)

        # 3) Extract events from an injected Event Guard snapshot.
        snapshot = self._event_guard_snapshot
        if snapshot is None:
            self._logger.warning(
                "pre_market.overnight_events_scanned",
                status="SKIPPED",
                reason="EVENT_GUARD_SNAPSHOT_MISSING",
            )
            return []

        # Snapshot may be:
        # - an `event_guard.interface.EventSnapshot`
        # - a builtins dict containing {"events": [...]}
        # - a raw list of events (already extracted by the orchestrator/tests)
        raw_events: list[Any]
        if isinstance(snapshot, EventSnapshot):
            raw_events = list(snapshot.events)
        elif isinstance(snapshot, Mapping) and isinstance(snapshot.get("events"), Sequence):
            raw_events = list(cast(Sequence[Any], snapshot.get("events")))
        elif isinstance(snapshot, Sequence) and not isinstance(snapshot, (str, bytes, bytearray)):
            raw_events = list(snapshot)
        else:
            self._logger.warning(
                "pre_market.overnight_events_scanned",
                status="SKIPPED",
                reason="EVENT_GUARD_SNAPSHOT_UNSUPPORTED",
                snapshot_type=type(snapshot).__name__,
            )
            return []

        # 4) Filter HIGH/CRITICAL events within the scan window and matching symbols.
        high_risk: set[str] = {RiskLevel.HIGH.value, RiskLevel.CRITICAL.value, "HIGH", "CRITICAL"}
        filtered_events: list[dict[str, Any]] = []
        for event in raw_events:
            if isinstance(event, Mapping):
                symbol = str(event.get("symbol", "") or "").strip().upper()
                event_type = str(event.get("event_type", "") or "").strip().upper()
                risk_level = event.get("risk_level")
                risk_level_value = getattr(risk_level, "value", risk_level)
                risk_level_str = str(risk_level_value or "").strip().upper()
                event_date = event.get("event_date")
                source = str(event.get("source", "") or "").strip() or "unknown"
            else:
                symbol = str(getattr(event, "symbol", "") or "").strip().upper()
                event_type = str(getattr(getattr(event, "event_type", None), "value", getattr(event, "event_type", "")) or "").strip().upper()
                risk_level = getattr(event, "risk_level", None)
                risk_level_str = str(getattr(risk_level, "value", risk_level) or "").strip().upper()
                event_date = getattr(event, "event_date", None)
                source = str(getattr(event, "source", "") or "").strip() or "unknown"

            if not symbol or symbol not in symbols:
                continue
            if risk_level_str not in high_risk:
                continue
            if not isinstance(event_date, int):
                continue
            if event_date < start_ns or event_date > end_ns:
                continue

            filtered_events.append(
                {
                    "symbol": symbol,
                    "event_type": event_type or "OTHER",
                    "risk_level": risk_level_str,
                    "event_date": int(event_date),
                    "source": source,
                }
            )

        self._logger.info(
            "pre_market.overnight_events_scanned",
            status="SUCCESS",
            symbols=len(symbols),
            events=len(raw_events),
            filtered=len(filtered_events),
            start_ns=start_ns,
            end_ns=end_ns,
        )
        return filtered_events

    def _validate_overnight_orders(
        self, overnight_events: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """Validate overnight orders against overnight events.

        Args:
            overnight_events: List of overnight events from _scan_overnight_events

        Returns:
            (orders_validated, conflicting_orders_cancelled)
        """

        if self._manager is None:
            self._logger.error("pre_market.overnight_order_validation", status="FAILED", reason="NOT_INITIALISED")
            return (0, 0)

        try:
            orders_result = self._manager.get_all_orders()
        except Exception as exc:  # noqa: BLE001 - best effort validation
            self._logger.exception("pre_market.overnight_order_validation", status="FAILED", reason="ORDER_SNAPSHOT_EXCEPTION", error=str(exc))
            return (0, 0)

        if orders_result.status is not ResultStatus.SUCCESS:
            self._logger.error(
                "pre_market.overnight_order_validation",
                status="FAILED",
                reason=orders_result.reason_code or "ORDER_SNAPSHOT_FAILED",
                error=str(orders_result.error or "failed to load orders"),
            )
            return (0, 0)

        mappings = list(orders_result.data or [])
        pending_states = {OrderState.SUBMITTED, OrderState.ACCEPTED, OrderState.PENDING}
        entry_intent_types = {IntentType.OPEN_LONG, IntentType.OPEN_SHORT}
        pending_orders = [
            m
            for m in mappings
            if getattr(m, "state", None) in pending_states
            and getattr(getattr(m, "intent_snapshot", None), "intent_type", None) in entry_intent_types
        ]

        if not pending_orders:
            return (0, 0)

        events_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for event in overnight_events:
            symbol = str(event.get("symbol", "") or "").strip().upper()
            if not symbol:
                continue
            events_by_symbol.setdefault(symbol, []).append(dict(event))

        high_risk_levels = {"HIGH", "CRITICAL"}
        orders_validated = 0
        cancelled = 0

        for mapping in pending_orders:
            intent_snapshot = getattr(mapping, "intent_snapshot", None)
            symbol = str(_intent_get(intent_snapshot, "symbol", "") or "").strip().upper()
            if not symbol:
                continue

            orders_validated += 1
            events = events_by_symbol.get(symbol, [])
            has_critical_event = any(str(e.get("risk_level", "") or "").strip().upper() in high_risk_levels for e in events)
            if not has_critical_event:
                continue

            client_order_id = str(getattr(mapping, "client_order_id", "") or "").strip()
            intent_id = str(getattr(mapping, "intent_id", "") or "").strip()
            if not client_order_id:
                self._logger.error(
                    "pre_market.order_cancelled",
                    status="FAILED",
                    intent_id=intent_id,
                    symbol=symbol,
                    reason="MISSING_CLIENT_ORDER_ID",
                )
                continue

            cancel_result = self._manager.cancel_order(client_order_id)
            if cancel_result.status is ResultStatus.SUCCESS:
                cancelled += 1
                self._logger.info(
                    "pre_market.order_cancelled",
                    intent_id=intent_id,
                    client_order_id=client_order_id,
                    symbol=symbol,
                    reason="OVERNIGHT_CRITICAL_EVENT",
                    events=[str(e.get("event_type", "") or "") for e in events],
                )
            else:
                self._logger.error(
                    "pre_market.order_cancelled",
                    status="FAILED",
                    intent_id=intent_id,
                    client_order_id=client_order_id,
                    symbol=symbol,
                    reason=cancel_result.reason_code or "CANCEL_FAILED",
                    error=str(cancel_result.error or "cancel failed"),
                )

        return (orders_validated, cancelled)

    def init(self, context: PluginContext | None = None) -> Result[None]:
        """Initialise the execution plugin (construct OrderManager + adapter)."""

        self._logger = _get_logger(context, self._MODULE_NAME)
        self._emit_event = _get_emitter(context)

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str) and module_name:
                self._module_name = module_name
            run_context = context.get("run_context")
            if run_context is not None:
                run_id = getattr(run_context, "run_id", None)
                if isinstance(run_id, str) and run_id.strip():
                    self._run_id = run_id.strip()

        if self._validated.enabled and (self._manager is None or self._adapter is None):
            error = ValidationError(
                "ExecutionPlugin configuration was not validated before init.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(self.config)},
            )
            return Result.failed(error, error.reason_code)

        _emit(self._emit_event, "module.initialised", self._module_name, {"module": "execution"})
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[ExecutionPluginConfig]:
        """Validate config and build the underlying OrderManager."""

        if not isinstance(config, Mapping):
            self._shared_adapter = None
            validated = ExecutionPluginConfig()
            self._validated = validated
            built = self._build_manager(validated)
            if built.status is ResultStatus.FAILED:
                return Result.failed(built.error or RuntimeError("Execution manager build failed."), built.reason_code or self._CONFIG_INVALID_REASON)
            return Result.success(validated)

        config_map = dict(config)
        # Daemon mode: extract pre-connected shared adapter (if any).
        self._shared_adapter = config_map.get("_shared_adapter")
        nested = _coerce_mapping(config_map.get("execution"))
        merged = dict(nested or config_map)

        cleanup_strategy = str(merged.get("cleanup_strategy") or "keep_all").strip().lower() or "keep_all"
        valid_cleanup_strategies = {"keep_all", "cancel_open_only", "cancel_all"}
        if cleanup_strategy not in valid_cleanup_strategies:
            error = ValidationError(
                "cleanup_strategy must be one of: keep_all, cancel_open_only, cancel_all",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"cleanup_strategy": merged.get("cleanup_strategy")},
            )
            return Result.failed(error, error.reason_code)

        validated = ExecutionPluginConfig(
            enabled=_safe_bool(merged.get("enabled", True), default=True),
            adapter=str(merged.get("adapter") or "ibkr").strip().lower() or "ibkr",
            broker=_coerce_mapping(merged.get("broker")) or _coerce_mapping(merged.get("connection")) or {},
            persistence_dir=str(merged.get("persistence_dir") or ".cache/execution"),
            connect_on_init=_safe_bool(merged.get("connect_on_init", False), default=False),
            enforce_risk_decisions=_safe_bool(merged.get("enforce_risk_decisions", True), default=True),
            dry_run=(None if merged.get("dry_run") is None else _safe_bool(merged.get("dry_run"), default=False)),
            cleanup_strategy=cleanup_strategy,
        )
        self._validated = validated

        if not validated.enabled:
            self._manager = None
            self._adapter = None
            self._connection_config = None
            return Result.success(validated)

        built = self._build_manager(validated)
        if built.status is ResultStatus.FAILED or built.data is None:
            err = built.error or ConfigurationError(
                "Failed to configure execution manager.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": dict(config)},
            )
            return Result.failed(err, built.reason_code or self._CONFIG_INVALID_REASON)

        return Result.success(validated)

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Submit allowed intents and emit ExecutionReport entries."""

        if bool(payload.get("pre_market_check")):
            return self._execute_pre_market_check(payload)

        if bool(payload.get("intraday_check")):
            return self._execute_intraday_check(payload)

        if bool(payload.get("post_close_recon")):
            return self._execute_post_close_recon(payload)

        if bool(payload.get("pre_close_recon")):
            return self._execute_pre_close_recon(payload)

        intents_result = self._parse_intents(payload)
        if intents_result.status is ResultStatus.FAILED or intents_result.data is None:
            return Result.failed(intents_result.error or RuntimeError("Invalid intents payload."), intents_result.reason_code or "INVALID_INTENTS")
        intents = intents_result.data
        intents_builtins = msgspec.to_builtins(intents)

        decisions_result = self._parse_risk_decisions(payload)
        if decisions_result.status is ResultStatus.FAILED or decisions_result.data is None:
            return Result.failed(decisions_result.error or RuntimeError("Invalid risk_decisions payload."), decisions_result.reason_code or "INVALID_RISK_DECISIONS")
        risk_decisions = decisions_result.data
        risk_decisions_builtins = msgspec.to_builtins(risk_decisions)

        if not self._validated.enabled:
            output = ExecutionOutput(
                schema_version="1.4.0",
                system_version=str(getattr(risk_decisions, "system_version", RunIDGenerator.get_system_version())),
                asof_timestamp=int(time.time_ns()),
                reports=[],
                intents=intents_builtins,
                risk_decisions=risk_decisions_builtins,
            )
            output_builtins = msgspec.to_builtins(output)
            _emit(self._emit_event, "module.executed", self._module_name, {"disabled": True, "reports": 0})
            return Result.success(output_builtins)

        if self._manager is None:
            return Result.failed(RuntimeError("ExecutionPlugin was not initialised."), "NOT_INITIALISED")

        effective_dry_run = bool(self._validated.dry_run) if self._validated.dry_run is not None else False

        # Ensure IBKR connectivity before submitting orders (connection may have
        # dropped during the long pipeline).  Mirrors _execute_pre_market_check.
        if not effective_dry_run and self._adapter is not None and hasattr(self._adapter, "is_connected"):
            try:
                if not self._adapter.is_connected():
                    connect_result = self._adapter.connect()
                    if connect_result.status is ResultStatus.FAILED:
                        return Result.failed(
                            connect_result.error or RuntimeError("IBKR reconnect failed before order submission"),
                            connect_result.reason_code or "CONNECT_FAILED",
                        )
            except Exception as exc:  # noqa: BLE001
                return Result.failed(exc, "CONNECT_EXCEPTION")

        reconcile_issue: tuple[BaseException, str] | None = None
        reconciliation_meta: dict[str, Any] | None = None
        if bool(payload.get("reconcile_first")):
            if effective_dry_run:
                reconciliation_meta = {"requested": True, "skipped": True, "reason": "DRY_RUN"}
            else:
                try:
                    reconcile_result = self._manager.reconcile_periodic()
                except Exception as exc:  # noqa: BLE001 - reconciliation must not crash execution
                    reconcile_issue = (exc, "RECONCILE_EXCEPTION")
                    reconciliation_meta = {"requested": True, "status": "FAILED", "reason_code": "RECONCILE_EXCEPTION"}
                else:
                    reconciliation_meta = {
                        "requested": True,
                        "status": reconcile_result.status.value,
                        "reason_code": reconcile_result.reason_code,
                    }
                    if reconcile_result.status is not ResultStatus.SUCCESS:
                        reconcile_issue = (
                            reconcile_result.error or RuntimeError("reconciliation degraded"),
                            reconcile_result.reason_code or "RECONCILE_DEGRADED",
                        )

        decision_by_intent: dict[str, Any] = {}
        for decision in list(getattr(risk_decisions, "decisions", []) or []):
            intent_id = str(getattr(decision, "intent_id", "")).strip()
            if intent_id:
                decision_by_intent[intent_id] = decision

        reports: list[ExecutionReport] = []
        failures: list[tuple[str, BaseException, str]] = []

        for group in list(getattr(intents, "intent_groups", []) or []):
            for intent in list(getattr(group, "intents", []) or []):
                intent_id = str(getattr(intent, "intent_id", "")).strip()
                symbol = str(getattr(intent, "symbol", "")).strip().upper() or "UNKNOWN"

                decision = decision_by_intent.get(intent_id)
                decision_type = getattr(decision, "decision_type", DecisionType.ALLOW)

                if self._validated.enforce_risk_decisions and decision is None:
                    report = self._skipped_report(
                        run_id=self._run_id,
                        intent_id=intent_id,
                        symbol=symbol,
                        reason="MISSING_RISK_DECISION",
                    )
                    reports.append(report)
                    failures.append((intent_id, RuntimeError("missing risk decision"), "MISSING_RISK_DECISION"))
                    continue

                if self._validated.enforce_risk_decisions and decision_type != DecisionType.ALLOW:
                    report = self._skipped_report(
                        run_id=self._run_id,
                        intent_id=intent_id,
                        symbol=symbol,
                        reason=f"RISK_{getattr(decision_type, 'value', str(decision_type))}",
                    )
                    reports.append(report)
                    continue

                if effective_dry_run:
                    report = self._skipped_report(run_id=self._run_id, intent_id=intent_id, symbol=symbol, reason="DRY_RUN")
                    reports.append(report)
                    continue

                # Ensure run_id is present for downstream id generation.
                intent_with_run_id = self._attach_run_id(intent, self._run_id)
                submit_result = self._manager.submit_order(intent_with_run_id)
                if submit_result.status is ResultStatus.FAILED or submit_result.data is None:
                    err = submit_result.error or RuntimeError("order submission failed")
                    failures.append((intent_id, err, submit_result.reason_code or "SUBMIT_FAILED"))
                    reports.append(
                        self._rejected_report(
                            run_id=self._run_id,
                            intent_id=intent_id,
                            symbol=symbol,
                            reason_code=submit_result.reason_code or "SUBMIT_FAILED",
                            error=err,
                        )
                    )
                    continue

                mapping = submit_result.data
                if submit_result.status is ResultStatus.DEGRADED:
                    failures.append(
                        (
                            intent_id,
                            submit_result.error or RuntimeError("order submission degraded"),
                            submit_result.reason_code or "SUBMIT_DEGRADED",
                        )
                    )
                # Best-effort immediate refresh; ignore failures.
                try:
                    self._manager.refresh_order_status(mapping.client_order_id)
                except Exception:  # noqa: BLE001
                    pass

                reports.append(
                    ExecutionReport(
                        run_id=self._run_id,
                        intent_id=mapping.intent_id,
                        client_order_id=mapping.client_order_id,
                        broker_order_id=str(mapping.broker_order_id or "unknown"),
                        symbol=symbol,
                        final_state=mapping.state,
                        filled_quantity=0.0,
                        remaining_quantity=float(getattr(intent, "quantity", 0.0) or 0.0),
                        avg_fill_price=None,
                        commissions=None,
                        fills=[],
                        state_transitions=[],
                        executed_at_ns=int(time.time_ns()),
                        metadata={"submit_status": submit_result.status.value, "submit_reason_code": submit_result.reason_code},
                    )
                )

        reports_builtins = [msgspec.to_builtins(report) for report in reports]
        output = ExecutionOutput(
            schema_version="1.4.0",
            system_version=str(getattr(risk_decisions, "system_version", RunIDGenerator.get_system_version())),
            asof_timestamp=int(time.time_ns()),
            reports=reports_builtins,
            intents=intents_builtins,
            risk_decisions=risk_decisions_builtins,
        )
        output_builtins = msgspec.to_builtins(output)

        # Best-effort: attach account summary so RunReport can show Cash/Exposure.
        if self._adapter is not None and hasattr(self._adapter, "get_account_summary"):
            try:
                acct_result = self._adapter.get_account_summary()
                if acct_result.status is ResultStatus.SUCCESS and acct_result.data:
                    summary = dict(acct_result.data)
                    acct_data: dict[str, Any] = {
                        "cash": summary.get("total_cash_value"),
                        "gross_exposure": summary.get("gross_position_value"),
                        "net_liquidation": summary.get("net_liquidation"),
                        "open_positions_count": summary.get("open_positions_count"),
                        "buying_power": summary.get("buying_power"),
                        "currency_balances": summary.get("currency_balances"),
                    }
                    # Compute cost basis from portfolio positions for accurate BP calculation.
                    if hasattr(self._adapter, "get_portfolio_positions"):
                        try:
                            pos_result = self._adapter.get_portfolio_positions()
                            if pos_result.status is ResultStatus.SUCCESS and pos_result.data:
                                cost_basis = 0.0
                                for pos in pos_result.data:
                                    avg_cost = pos.get("avg_cost") or 0.0
                                    qty = abs(pos.get("position") or 0.0)
                                    cost_basis += avg_cost * qty
                                acct_data["cost_basis"] = cost_basis
                                acct_data["open_positions_count"] = len(pos_result.data)
                                acct_data["positions"] = [
                                    {
                                        "symbol": p.get("symbol", ""),
                                        "qty": abs(p.get("position") or 0.0),
                                        "avg_cost": p.get("avg_cost") or 0.0,
                                        "market_value": p.get("market_value") or 0.0,
                                        "unrealized_pnl": p.get("unrealized_pnl") or 0.0,
                                    }
                                    for p in pos_result.data
                                    if p.get("position") and abs(p.get("position") or 0) > 0
                                ]
                        except Exception:  # noqa: BLE001
                            pass
                    # Attach active bracket orders (SL/TP) for position reporting.
                    if self._manager is not None:
                        try:
                            orders_result = self._manager.get_all_orders()
                            if orders_result.status is ResultStatus.SUCCESS and orders_result.data:
                                pos_syms = [p.get("symbol", "") for p in acct_data.get("positions", []) if p.get("symbol")]
                                acct_data["bracket_orders"] = self._build_bracket_summary_with_broker_fallback(
                                    list(orders_result.data), symbols=pos_syms,
                                )
                        except Exception:  # noqa: BLE001
                            pass
                    output_builtins["account_summary"] = acct_data
            except Exception:  # noqa: BLE001 - must not block execution output
                pass

        _emit(self._emit_event, "module.executed", self._module_name, {"reports": len(reports), "failures": len(failures)})

        if failures:
            first_intent, first_error, first_reason = failures[0]
            wrapped = PartialDataError.from_error(
                first_error,
                module=self._MODULE_NAME,
                reason_code="EXECUTION_DEGRADED",
                details={"first_failed_intent": first_intent, "first_reason": first_reason, "failures": [{"intent_id": i, "reason_code": r, "error": str(e)} for i, e, r in failures]},
            )
            return Result.degraded(output_builtins, wrapped, wrapped.reason_code)

        if reconcile_issue is not None:
            error, reason = reconcile_issue
            return Result.degraded(output_builtins, error, reason)

        return Result.success(output_builtins)

    # ------------------------------------------------------------------
    # Bracket health-check helpers (shared by pre_market / intraday / pre_close)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_intent_type(mapping: Any) -> IntentType | None:
        intent_snapshot = getattr(mapping, "intent_snapshot", None)
        return _intent_get(intent_snapshot, "intent_type")

    @staticmethod
    def _get_symbol(mapping: Any) -> str:
        intent_snapshot = getattr(mapping, "intent_snapshot", None)
        symbol = _intent_get(intent_snapshot, "symbol", "")
        return str(symbol).strip().upper()

    @staticmethod
    def _get_parent(mapping: Any) -> str | None:
        intent_snapshot = getattr(mapping, "intent_snapshot", None)
        parent = _intent_get(intent_snapshot, "parent_intent_id")
        return str(parent).strip() if parent else None

    @staticmethod
    def _get_intent_id(mapping: Any) -> str:
        return str(getattr(mapping, "intent_id", "")).strip()

    def _repair_missing_stops(
        self,
        mappings: list[Any],
        *,
        intent_prefix: str,
        generated_by: str,
        dry_run: bool,
    ) -> dict[str, Any]:
        """Detect filled entries missing active SL and submit repair stops.

        Returns dict with keys:
            stops_verified: int          — entries with existing active SL
            protected_intent_ids: list[str] — intent IDs confirmed protected
            generated_stops: list[dict]  — newly submitted SL orders
            positions_at_risk: list[dict] — entries that couldn't be protected
            failures: list[dict]         — submission errors
        """
        stop_mappings: list[Any] = []
        for mapping in mappings:
            intent_type = self._get_intent_type(mapping)
            if intent_type != IntentType.STOP_LOSS:
                continue
            state = getattr(mapping, "state", None)
            if isinstance(state, OrderState) and state.is_terminal():
                continue
            stop_mappings.append(mapping)

        stops_by_parent: dict[str, list[Any]] = {}
        stops_by_symbol: dict[str, list[Any]] = {}
        for stop_mapping in stop_mappings:
            parent = self._get_parent(stop_mapping)
            if parent:
                stops_by_parent.setdefault(parent, []).append(stop_mapping)
            stops_by_symbol.setdefault(self._get_symbol(stop_mapping), []).append(stop_mapping)

        entry_intent_types = {IntentType.OPEN_LONG, IntentType.OPEN_SHORT}
        open_position_states = {OrderState.FILLED, OrderState.PARTIALLY_FILLED}
        open_positions: list[Any] = []
        for mapping in mappings:
            intent_type = self._get_intent_type(mapping)
            if intent_type not in entry_intent_types:
                continue
            state = getattr(mapping, "state", None)
            if state in open_position_states:
                open_positions.append(mapping)

        protected_positions: set[str] = set()
        missing_stop_positions: list[Any] = []
        for position in open_positions:
            parent_intent_id = self._get_intent_id(position)
            symbol = self._get_symbol(position)
            if parent_intent_id and stops_by_parent.get(parent_intent_id):
                protected_positions.add(parent_intent_id)
                continue
            if symbol and stops_by_symbol.get(symbol):
                if parent_intent_id:
                    protected_positions.add(parent_intent_id)
                continue
            missing_stop_positions.append(position)

        generated_stops: list[dict[str, Any]] = []
        positions_at_risk: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for position in missing_stop_positions:
            intent_snapshot = getattr(position, "intent_snapshot", None)
            parent_intent_id = self._get_intent_id(position)
            symbol = self._get_symbol(position)
            stop_price = _intent_get(intent_snapshot, "stop_loss_price")

            # Prefer actual filled qty from broker metadata; fall back to planned qty.
            pos_metadata = getattr(position, "metadata", {}) or {}
            ibkr_meta = pos_metadata.get("ibkr", {}) or {}
            filled_qty = abs(float(ibkr_meta.get("filled", 0) or 0))
            planned_qty = float(_intent_get(intent_snapshot, "quantity", 0.0) or 0.0)
            quantity = filled_qty if filled_qty > 0 else planned_qty

            # Guardrail: no position → no stop needed
            if quantity <= 0:
                continue

            if stop_price is None:
                positions_at_risk.append({
                    "intent_id": parent_intent_id,
                    "symbol": symbol,
                    "reason": "MISSING_STOP_LOSS_PRICE",
                })
                continue

            if dry_run:
                positions_at_risk.append({
                    "intent_id": parent_intent_id,
                    "symbol": symbol,
                    "reason": "DRY_RUN_STOP_NOT_SUBMITTED",
                    "stop_loss_price": float(stop_price),
                })
                continue

            stop_intent = TradeIntent(
                intent_id=f"{intent_prefix}{parent_intent_id}",
                symbol=symbol,
                intent_type=IntentType.STOP_LOSS,
                quantity=quantity,
                created_at_ns=int(time.time_ns()),
                stop_loss_price=float(stop_price),
                parent_intent_id=parent_intent_id,
                reduce_only=True,
                metadata={
                    "order_type": "STOP_MARKET",
                    "generated_by": generated_by,
                },
            )
            stop_with_run_id = self._attach_run_id(stop_intent, self._run_id)
            submit_result = self._manager.submit_order(stop_with_run_id)
            if submit_result.status is ResultStatus.FAILED or submit_result.data is None:
                failures.append({
                    "category": "generate_stop",
                    "intent_id": parent_intent_id,
                    "symbol": symbol,
                    "reason_code": submit_result.reason_code or "STOP_SUBMIT_FAILED",
                    "error": str(submit_result.error or "stop submission failed"),
                })
                positions_at_risk.append({
                    "intent_id": parent_intent_id,
                    "symbol": symbol,
                    "reason": "STOP_SUBMIT_FAILED",
                })
                continue

            result_mapping = submit_result.data
            generated_stops.append({
                "intent_id": getattr(result_mapping, "intent_id", ""),
                "client_order_id": getattr(result_mapping, "client_order_id", ""),
                "symbol": symbol,
                "submit_status": submit_result.status.value,
                "submit_reason_code": submit_result.reason_code,
            })
            if parent_intent_id:
                protected_positions.add(parent_intent_id)
            if submit_result.status is ResultStatus.DEGRADED:
                failures.append({
                    "category": "generate_stop",
                    "intent_id": parent_intent_id,
                    "symbol": symbol,
                    "reason_code": submit_result.reason_code or "STOP_SUBMIT_DEGRADED",
                    "error": str(submit_result.error or "stop submission degraded"),
                })

        return {
            "stops_verified": len(protected_positions),
            "protected_intent_ids": sorted(protected_positions),
            "generated_stops": generated_stops,
            "positions_at_risk": positions_at_risk,
            "failures": failures,
            "open_positions": open_positions,
        }

    @staticmethod
    def _build_bracket_summary(mappings: list[Any]) -> dict[str, dict[str, Any]]:
        """Extract active SL/TP orders grouped by symbol for reporting.

        Returns ``{symbol: {"sl": {...}, "tp": {...}}}`` where each sub-dict
        contains ``price``, ``qty``, and ``status`` (order state string).
        """

        bracket_types = {IntentType.STOP_LOSS: "sl", IntentType.TAKE_PROFIT: "tp"}
        result: dict[str, dict[str, Any]] = {}

        for mapping in mappings:
            intent_snapshot = getattr(mapping, "intent_snapshot", None)
            if intent_snapshot is None:
                continue
            intent_type = _intent_get(intent_snapshot, "intent_type")
            label = bracket_types.get(intent_type)  # type: ignore[arg-type]
            if label is None:
                continue
            state = getattr(mapping, "state", None)
            if isinstance(state, OrderState) and state.is_terminal():
                continue
            symbol = _intent_get(intent_snapshot, "symbol", "") or ""
            if not symbol:
                continue

            qty = float(_intent_get(intent_snapshot, "quantity", 0) or 0)
            if label == "sl":
                price = float(_intent_get(intent_snapshot, "stop_loss_price", 0) or 0)
            else:
                price = float(_intent_get(intent_snapshot, "take_profit_price", 0) or 0)
            state_str = state.value if isinstance(state, OrderState) else str(state or "UNKNOWN")

            entry = result.setdefault(symbol, {})
            entry[label] = {"price": price, "qty": qty, "status": state_str}

        return result

    def _build_bracket_summary_with_broker_fallback(
        self,
        mappings: list[Any],
        symbols: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Build bracket summary from AST mappings, falling back to IBKR open orders.

        For each symbol in *symbols*, if the AST mapping-based summary has no
        SL or TP, query the broker adapter for live open orders and fill in
        the gaps.  This ensures that manually placed or repair-script orders
        are visible in Telegram reports.
        """
        summary = self._build_bracket_summary(mappings)

        # Determine which symbols are missing any bracket leg (SL or TP).
        if symbols is None:
            missing = set()
        else:
            missing = set()
            for s in symbols:
                sym_data = summary.get(s, {})
                if "sl" not in sym_data or "tp" not in sym_data:
                    missing.add(s)

        if not missing:
            return summary

        adapter = self._adapter
        if adapter is None or not hasattr(adapter, "get_open_bracket_orders"):
            return summary

        try:
            broker_result = adapter.get_open_bracket_orders()
            if broker_result.status.name != "SUCCESS" or broker_result.data is None:
                return summary
            broker_brackets: dict[str, dict[str, Any]] = broker_result.data
        except Exception:  # noqa: BLE001
            return summary

        for sym in missing:
            broker_entry = broker_brackets.get(sym)
            if not broker_entry:
                continue
            existing = summary.setdefault(sym, {})
            if "sl" not in existing and "sl" in broker_entry:
                existing["sl"] = broker_entry["sl"]
            if "tp" not in existing and "tp" in broker_entry:
                existing["tp"] = broker_entry["tp"]

        return summary

    @staticmethod
    def _derive_order_stats(
        mappings: list[Any],
        position_symbols: set[str] | None = None,
    ) -> dict[str, Any]:
        """Derive cumulative order/intent/fill counts and details from mappings.

        Used by execution-only runs (intraday, pre_close, post_close) so Telegram
        reports show the day's cumulative numbers instead of zeros.

        Args:
            position_symbols: Symbols currently held in IBKR.  When an entry
                mapping's symbol appears here, the entry is counted as filled
                even if the mapping state was not updated (fills between runs).
        """
        entry_types = {IntentType.OPEN_LONG, IntentType.OPEN_SHORT}
        bracket_types = {IntentType.STOP_LOSS, IntentType.TAKE_PROFIT}
        held = position_symbols or set()

        entry_symbols: set[str] = set()
        orders_total = 0
        orders_cancelled = 0
        fills_entry = 0
        fills_exit = 0
        order_details: list[dict[str, Any]] = []

        for mapping in mappings:
            intent = getattr(mapping, "intent_snapshot", None)
            if intent is None:
                continue
            itype = _intent_get(intent, "intent_type")
            state = getattr(mapping, "state", None)

            orders_total += 1
            if isinstance(state, OrderState) and state in (OrderState.CANCELLED, OrderState.EXPIRED):
                orders_cancelled += 1

            if itype in entry_types:
                symbol = _intent_get(intent, "symbol", "") or ""
                if symbol:
                    entry_symbols.add(symbol)
                state_filled = isinstance(state, OrderState) and state in (OrderState.FILLED, OrderState.PARTIALLY_FILLED)
                inferred_filled = bool(symbol and symbol in held)
                if state_filled or inferred_filled:
                    fills_entry += 1
            elif itype in bracket_types:
                if isinstance(state, OrderState) and state is OrderState.FILLED:
                    fills_exit += 1

            # Collect per-order detail.
            meta = getattr(mapping, "metadata", None) or {}
            ibkr_meta = meta.get("ibkr", {}) if isinstance(meta, dict) else {}
            ordered_qty = float(_intent_get(intent, "quantity", 0) or 0)
            filled_qty = float(ibkr_meta.get("filled", 0) or 0)
            avg_fill = float(ibkr_meta.get("avgFillPrice", 0) or 0)
            # Determine the "order price" based on intent type.
            if itype in entry_types:
                price = float(_intent_get(intent, "entry_price", 0) or 0)
            elif itype == IntentType.STOP_LOSS:
                price = float(_intent_get(intent, "stop_loss_price", 0) or 0)
            elif itype == IntentType.TAKE_PROFIT:
                price = float(_intent_get(intent, "take_profit_price", 0) or 0)
            else:
                price = 0.0
            state_str = state.value if isinstance(state, OrderState) else str(state or "UNKNOWN")
            itype_str = itype.value if isinstance(itype, IntentType) else str(itype or "")
            order_details.append({
                "symbol": _intent_get(intent, "symbol", "") or "",
                "intent_type": itype_str,
                "ordered_qty": ordered_qty,
                "filled_qty": filled_qty,
                "price": price,
                "avg_fill_price": avg_fill,
                "state": state_str,
            })

        return {
            "intents_generated": len(entry_symbols),
            "orders_submitted": orders_total,
            "orders_cancelled": orders_cancelled,
            "fills_entry_count": fills_entry,
            "fills_exit_count": fills_exit,
            "order_details": order_details,
        }

    def _get_position_symbols(self) -> set[str]:
        """Return set of symbols currently held in IBKR (best-effort)."""
        return set(self._get_broker_position_quantities().keys())

    def _get_broker_position_quantities(self) -> dict[str, float]:
        """Return {symbol: abs_qty} for all positions currently held in IBKR."""
        if self._adapter is None or not hasattr(self._adapter, "get_portfolio_positions"):
            return {}
        try:
            res = self._adapter.get_portfolio_positions()
            if res.status is ResultStatus.SUCCESS and res.data:
                return {
                    p.get("symbol", ""): abs(float(p.get("position") or 0))
                    for p in res.data
                    if p.get("position") and abs(p.get("position") or 0) > 0
                }
        except Exception:  # noqa: BLE001
            pass
        return {}

    def _repair_bracket_quantities_from_broker(self) -> dict[str, Any] | None:
        """Broker-direct bracket quantity repair — bypasses mapping state entirely.

        Calls ``adapter.repair_bracket_quantities()`` which compares IBKR
        positions against open SL/TP orders and cancel+resubmits any with
        mismatched quantities.  This is the safety net for partial fills that
        occurred between runs (when mapping state is stale/EXPIRED).
        """
        if self._adapter is None or not hasattr(self._adapter, "repair_bracket_quantities"):
            return None
        try:
            res = self._adapter.repair_bracket_quantities()
            if res.status is ResultStatus.SUCCESS and res.data:
                data = res.data
                if data.get("repaired") and hasattr(self._logger, "info"):
                    try:
                        self._logger.info(
                            "broker_qty_repair.completed",
                            aligned=data.get("aligned", 0),
                            repaired_count=len(data["repaired"]),
                            repaired=data["repaired"],
                        )
                    except Exception:  # noqa: BLE001
                        pass
                return data
            if res.status is ResultStatus.FAILED and hasattr(self._logger, "warning"):
                try:
                    self._logger.warning(
                        "broker_qty_repair.failed",
                        error=str(res.error),
                        reason_code=res.reason_code,
                    )
                except Exception:  # noqa: BLE001
                    pass
            return None
        except Exception:  # noqa: BLE001
            return None

    def _repair_quantity_mismatch(
        self,
        mappings: list[Any],
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        """Detect and fix SL/TP quantity mismatches after partial fills.

        For each filled/partially-filled entry, checks that sibling SL/TP orders
        have quantity == abs(entry filled_qty). Mismatches are fixed via
        cancel + resubmit with corrected quantity.

        When mapping state is stale (e.g. EXPIRED due to fills between runs),
        falls back to IBKR actual positions to determine filled quantities.

        Returns dict with keys:
            aligned: int            — siblings already correct
            repaired: list[dict]    — cancel+resubmit actions taken
            positions_at_risk: list[dict] — could not be repaired
            failures: list[dict]    — errors encountered
        """
        child_intent_types = {IntentType.STOP_LOSS, IntentType.TAKE_PROFIT}
        entry_intent_types = {IntentType.OPEN_LONG, IntentType.OPEN_SHORT}
        open_position_states = {OrderState.FILLED, OrderState.PARTIALLY_FILLED}

        # Get actual IBKR positions for fallback fill detection.
        broker_positions = self._get_broker_position_quantities()

        # Index: parent_intent_id → list of child (SL/TP) mappings
        children_by_parent: dict[str, list[Any]] = {}
        for m in mappings:
            intent_type = self._get_intent_type(m)
            if intent_type not in child_intent_types:
                continue
            state = getattr(m, "state", None)
            if isinstance(state, OrderState) and state.is_terminal():
                continue
            parent = self._get_parent(m)
            if parent:
                children_by_parent.setdefault(parent, []).append(m)

        # Collect entries that are filled or have actual IBKR positions.
        entries: list[Any] = []
        for m in mappings:
            intent_type = self._get_intent_type(m)
            if intent_type not in entry_intent_types:
                continue
            state = getattr(m, "state", None)
            symbol = self._get_symbol(m)
            if state in open_position_states:
                entries.append(m)
            elif symbol and symbol in broker_positions:
                # Mapping state is stale but IBKR shows a position — treat as filled.
                entries.append(m)

        aligned = 0
        repaired: list[dict[str, Any]] = []
        positions_at_risk: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for entry in entries:
            parent_intent_id = self._get_intent_id(entry)
            symbol = self._get_symbol(entry)

            # Guardrail A: read actual filled qty from broker metadata,
            # falling back to IBKR actual position if mapping metadata is stale.
            metadata = getattr(entry, "metadata", {}) or {}
            ibkr_meta = metadata.get("ibkr", {}) or {}
            filled_qty = abs(float(ibkr_meta.get("filled", 0) or 0))

            if filled_qty <= 0 and symbol and symbol in broker_positions:
                filled_qty = broker_positions[symbol]

            if filled_qty <= 0:
                # No position — nothing to align; cancel orphan children
                continue

            siblings = children_by_parent.get(parent_intent_id, [])
            if not siblings:
                continue

            for child in siblings:
                child_intent_snapshot = getattr(child, "intent_snapshot", None)
                child_qty = float(getattr(child_intent_snapshot, "quantity", 0) or 0)
                child_order_id = str(getattr(child, "client_order_id", "")).strip()
                child_intent_type = self._get_intent_type(child)

                # Guardrail B: use abs for comparison
                if abs(child_qty - filled_qty) < 0.001:
                    aligned += 1
                    continue

                # Mismatch detected
                oca_group = ""
                if hasattr(IBKRAdapter, "_derive_oca_group"):
                    oca_group = IBKRAdapter._derive_oca_group(parent_intent_id)

                if hasattr(self._logger, "warning"):
                    try:
                        self._logger.warning(
                            "qty_mismatch.detected",
                            symbol=symbol,
                            parent_intent_id=parent_intent_id,
                            planned_qty=child_qty,
                            filled_qty=filled_qty,
                            child_order_id=child_order_id,
                            child_type=child_intent_type.value if child_intent_type else "UNKNOWN",
                            oca_group=oca_group,
                        )
                    except Exception:  # noqa: BLE001
                        pass

                if dry_run:
                    positions_at_risk.append({
                        "intent_id": parent_intent_id,
                        "symbol": symbol,
                        "child_order_id": child_order_id,
                        "child_type": child_intent_type.value if child_intent_type else "UNKNOWN",
                        "planned_qty": child_qty,
                        "filled_qty": filled_qty,
                        "reason": "DRY_RUN_QTY_MISMATCH",
                    })
                    continue

                # Cancel the mismatched child
                cancel_result = self._manager.cancel_order(child_order_id)
                if cancel_result.status is ResultStatus.FAILED:
                    # Guardrail C: cancel failed → do NOT resubmit (avoid double-active)
                    failures.append({
                        "category": "qty_align_cancel",
                        "intent_id": parent_intent_id,
                        "symbol": symbol,
                        "child_order_id": child_order_id,
                        "reason_code": cancel_result.reason_code or "CANCEL_FAILED",
                        "error": str(cancel_result.error or "cancel failed"),
                    })
                    positions_at_risk.append({
                        "intent_id": parent_intent_id,
                        "symbol": symbol,
                        "child_order_id": child_order_id,
                        "reason": "CANCEL_FAILED_NO_RESUBMIT",
                    })
                    if hasattr(self._logger, "error"):
                        try:
                            self._logger.error(
                                "qty_mismatch.cancel_failed",
                                symbol=symbol,
                                child_order_id=child_order_id,
                                error=str(cancel_result.error),
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    continue

                # Resubmit with corrected quantity
                new_intent = TradeIntent(
                    intent_id=f"QA-{child_order_id}",
                    symbol=symbol,
                    intent_type=child_intent_type,
                    quantity=filled_qty,
                    created_at_ns=int(time.time_ns()),
                    stop_loss_price=getattr(child_intent_snapshot, "stop_loss_price", None),
                    take_profit_price=getattr(child_intent_snapshot, "take_profit_price", None),
                    parent_intent_id=parent_intent_id,
                    reduce_only=True,
                    metadata=dict(getattr(child_intent_snapshot, "metadata", {}) or {}),
                )
                new_with_run_id = self._attach_run_id(new_intent, self._run_id)
                submit_result = self._manager.submit_order(new_with_run_id)

                new_order_id = ""
                if submit_result.data is not None:
                    new_order_id = str(getattr(submit_result.data, "client_order_id", ""))

                if hasattr(self._logger, "info"):
                    try:
                        self._logger.info(
                            "qty_mismatch.repaired",
                            symbol=symbol,
                            parent_intent_id=parent_intent_id,
                            old_qty=child_qty,
                            new_qty=filled_qty,
                            cancelled_order_id=child_order_id,
                            new_order_id=new_order_id,
                            submit_status=submit_result.status.value,
                        )
                    except Exception:  # noqa: BLE001
                        pass

                repaired.append({
                    "intent_id": parent_intent_id,
                    "symbol": symbol,
                    "child_type": child_intent_type.value if child_intent_type else "UNKNOWN",
                    "old_qty": child_qty,
                    "new_qty": filled_qty,
                    "cancelled_order_id": child_order_id,
                    "new_order_id": new_order_id,
                    "submit_status": submit_result.status.value,
                })

                if submit_result.status is ResultStatus.FAILED:
                    failures.append({
                        "category": "qty_align_resubmit",
                        "intent_id": parent_intent_id,
                        "symbol": symbol,
                        "child_order_id": child_order_id,
                        "reason_code": submit_result.reason_code or "RESUBMIT_FAILED",
                        "error": str(submit_result.error or "resubmit failed"),
                    })
                    positions_at_risk.append({
                        "intent_id": parent_intent_id,
                        "symbol": symbol,
                        "child_order_id": child_order_id,
                        "reason": "RESUBMIT_FAILED",
                    })

        return {
            "aligned": aligned,
            "repaired": repaired,
            "positions_at_risk": positions_at_risk,
            "failures": failures,
        }

    def _execute_pre_market_check(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Execute pre-market lightweight checks.

        Design constraints (Phase 3.8):
            - Lightweight: do not pull fresh market data, do not scan new candidates,
              and do not generate new entry orders.
            - Safe: only verifies and protects; the only allowed order submission is
              generating missing protective STOP_LOSS orders for existing positions.

        Checks performed:
            1) Connection health: validate IBKR connectivity + basic account/funds.
            2) Position protection: ensure each open position has an active STOP_LOSS.
            3) Overnight events: placeholder hook (no data fetch in v1).
            4) Overnight order validation: placeholder hook (no broker-side search in v1).
        """

        # Store the snapshot (if supplied) for `_scan_overnight_events`; always cleared before return.
        self._event_guard_snapshot = payload.get("event_guard_snapshot")

        if not self._validated.enabled:
            output: dict[str, Any] = {
                "schema_version": "1.0.0",
                "pre_market_check": True,
                "connection": {"connected": True, "account_summary": None},
                "reconciliation": {"requested": False},
                "stops_verified": 0,
                "generated_stops": [],
                "positions_at_risk": [],
                "overnight_events": [],
                "orders_validated": 0,
                "conflicting_orders_cancelled": 0,
            }
            _emit(self._emit_event, "module.executed", self._module_name, {"disabled": True, "pre_market_check": True})
            self._event_guard_snapshot = None
            return Result.success(output)

        if self._manager is None:
            self._event_guard_snapshot = None
            return Result.failed(RuntimeError("ExecutionPlugin was not initialised."), "NOT_INITIALISED")

        failures: list[dict[str, Any]] = []
        effective_dry_run = bool(self._validated.dry_run) if self._validated.dry_run is not None else False

        connection_details: dict[str, Any] = {"connected": None, "account_summary": None, "funds_available": None}
        adapter = self._adapter
        is_connected = None
        if adapter is not None and hasattr(adapter, "is_connected"):
            try:
                is_connected = bool(adapter.is_connected())
            except Exception as exc:  # noqa: BLE001
                failures.append({"category": "connection_check", "reason_code": "CONNECTION_CHECK_FAILED", "error": str(exc)})
                is_connected = False
        else:
            is_connected = False

        if not is_connected and adapter is not None and hasattr(adapter, "connect"):
            try:
                connect_result = adapter.connect()
            except Exception as exc:  # noqa: BLE001
                failures.append({"category": "connection_check", "reason_code": "CONNECT_EXCEPTION", "error": str(exc)})
                is_connected = False
            else:
                if connect_result.status is ResultStatus.FAILED:
                    failures.append(
                        {
                            "category": "connection_check",
                            "reason_code": getattr(connect_result, "reason_code", None) or "CONNECT_FAILED",
                            "error": str(getattr(connect_result, "error", None) or "connect failed"),
                        }
                    )
                    is_connected = False
                elif connect_result.status is ResultStatus.DEGRADED:
                    failures.append(
                        {
                            "category": "connection_check",
                            "reason_code": getattr(connect_result, "reason_code", None) or "CONNECT_DEGRADED",
                            "error": str(getattr(connect_result, "error", None) or "connect degraded"),
                        }
                    )
                    is_connected = True
                else:
                    is_connected = True

        connection_details["connected"] = bool(is_connected)

        if adapter is not None and hasattr(adapter, "get_account_summary") and is_connected:
            try:
                account_result = adapter.get_account_summary()
            except Exception as exc:  # noqa: BLE001
                failures.append({"category": "account_summary", "reason_code": "ACCOUNT_QUERY_EXCEPTION", "error": str(exc)})
            else:
                if account_result.status is ResultStatus.SUCCESS:
                    summary = dict(account_result.data or {})
                    connection_details["account_summary"] = summary
                    buying_power = summary.get("buying_power")
                    cash_value = summary.get("total_cash_value")
                    if isinstance(buying_power, (int, float)) and isinstance(cash_value, (int, float)):
                        connection_details["funds_available"] = bool(buying_power > 0.0 or cash_value > 0.0)
                else:
                    failures.append(
                        {
                            "category": "account_summary",
                            "reason_code": account_result.reason_code or "ACCOUNT_QUERY_FAILED",
                            "error": str(account_result.error or "account summary failed"),
                        }
                    )

        reconcile_issue: tuple[BaseException, str] | None = None
        reconciliation_meta: dict[str, Any] | None = None
        if bool(payload.get("reconcile_first")):
            if effective_dry_run:
                reconciliation_meta = {"requested": True, "skipped": True, "reason": "DRY_RUN"}
            else:
                try:
                    reconcile_result = self._manager.reconcile_periodic()
                except Exception as exc:  # noqa: BLE001 - reconciliation must not crash checks
                    reconcile_issue = (exc, "RECONCILE_EXCEPTION")
                    reconciliation_meta = {"requested": True, "status": "FAILED", "reason_code": "RECONCILE_EXCEPTION"}
                else:
                    reconciliation_meta = {
                        "requested": True,
                        "status": reconcile_result.status.value,
                        "reason_code": reconcile_result.reason_code,
                    }
                    if reconcile_result.status is not ResultStatus.SUCCESS:
                        reconcile_issue = (
                            reconcile_result.error or RuntimeError("reconciliation degraded"),
                            reconcile_result.reason_code or "RECONCILE_DEGRADED",
                        )

        if reconcile_issue is not None:
            error, reason = reconcile_issue
            failures.append({"category": "reconcile", "reason_code": reason, "error": str(error)})

        orders_result = self._manager.get_all_orders()
        if orders_result.status is ResultStatus.FAILED:
            return Result.failed(
                orders_result.error or RuntimeError("failed to load orders"),
                orders_result.reason_code or "ORDER_SNAPSHOT_FAILED",
            )

        mappings = list(orders_result.data or [])

        generated_stops: list[dict[str, Any]] = []
        positions_at_risk: list[dict[str, Any]] = []
        protected_positions: set[str] = set()

        if bool(payload.get("check_stops")):
            repair_result = self._repair_missing_stops(
                mappings, intent_prefix="PM-STOP-", generated_by="PRE_MARKET_CHECK", dry_run=effective_dry_run,
            )
            generated_stops = repair_result["generated_stops"]
            positions_at_risk = repair_result["positions_at_risk"]
            for entry_id in repair_result.get("protected_intent_ids", []):
                protected_positions.add(entry_id)
            if repair_result["failures"]:
                failures.extend(repair_result["failures"])

        overnight_events: list[dict[str, Any]] = []
        if bool(payload.get("scan_overnight_events")):
            overnight_events = self._scan_overnight_events()

        orders_validated = 0
        conflicting_orders_cancelled = 0
        if bool(payload.get("validate_orders")):
            orders_validated, conflicting_orders_cancelled = self._validate_overnight_orders(overnight_events)

        output: dict[str, Any] = {
            "schema_version": "1.0.0",
            "pre_market_check": True,
            "connection": dict(connection_details),
            "stops_verified": len(protected_positions),
            "generated_stops": list(generated_stops),
            "positions_at_risk": list(positions_at_risk),
            "overnight_events": list(overnight_events),
            "orders_validated": orders_validated,
            "conflicting_orders_cancelled": conflicting_orders_cancelled,
        }
        if reconciliation_meta is not None:
            output["reconciliation"] = dict(reconciliation_meta)

        _emit(
            self._emit_event,
            "module.executed",
            self._module_name,
            {
                "pre_market_check": True,
                "connected": bool(connection_details.get("connected")),
                "stops_verified": len(protected_positions),
                "generated_stops": len(generated_stops),
                "positions_at_risk": len(positions_at_risk),
                "failures": len(failures),
            },
        )

        if failures:
            wrapped = PartialDataError(
                "Pre-market check degraded.",
                module=self._MODULE_NAME,
                reason_code="PRE_MARKET_CHECK_DEGRADED",
                details={"failures": failures},
            )
            self._event_guard_snapshot = None
            return Result.degraded(output, wrapped, wrapped.reason_code)

        self._event_guard_snapshot = None
        return Result.success(output)

    def _execute_intraday_check(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Lightweight intraday path: reconcile + check stops, no new intents."""

        if not self._validated.enabled:
            output: dict[str, Any] = {
                "schema_version": "1.0.0",
                "intraday_check": True,
                "reconciliation": {"requested": False},
                "stops_verified": 0,
                "pending_fills_checked": 0,
            }
            _emit(self._emit_event, "module.executed", self._module_name, {"disabled": True, "intraday_check": True})
            return Result.success(output)

        if self._manager is None:
            return Result.failed(RuntimeError("ExecutionPlugin was not initialised."), "NOT_INITIALISED")

        effective_dry_run = bool(self._validated.dry_run) if self._validated.dry_run is not None else False
        failures: list[dict[str, Any]] = []

        reconciliation_meta: dict[str, Any] | None = None
        if bool(payload.get("reconcile_first")):
            if effective_dry_run:
                reconciliation_meta = {"requested": True, "skipped": True, "reason": "DRY_RUN"}
            else:
                try:
                    reconcile_result = self._manager.reconcile_periodic()
                except Exception as exc:  # noqa: BLE001 - reconciliation must not crash
                    reconciliation_meta = {"requested": True, "status": "FAILED", "reason_code": "RECONCILE_EXCEPTION"}
                    failures.append({"category": "reconcile", "reason_code": "RECONCILE_EXCEPTION", "error": str(exc)})
                else:
                    reconciliation_meta = {
                        "requested": True,
                        "status": reconcile_result.status.value,
                        "reason_code": reconcile_result.reason_code,
                    }
                    if reconcile_result.status is not ResultStatus.SUCCESS:
                        failures.append({
                            "category": "reconcile",
                            "reason_code": reconcile_result.reason_code or "RECONCILE_DEGRADED",
                            "error": str(reconcile_result.error or "reconciliation degraded"),
                        })

        orders_result = self._manager.get_all_orders()
        if orders_result.status is ResultStatus.FAILED:
            return Result.failed(
                orders_result.error or RuntimeError("failed to load orders"),
                orders_result.reason_code or "ORDER_SNAPSHOT_FAILED",
            )

        mappings = list(orders_result.data or [])

        stops_verified = 0
        pending_fills_checked = 0
        for mapping in mappings:
            intent_snapshot = getattr(mapping, "intent_snapshot", None)
            intent_type = _intent_get(intent_snapshot, "intent_type")
            state = getattr(mapping, "state", None)
            if isinstance(state, OrderState) and state.is_terminal():
                continue
            if intent_type == IntentType.STOP_LOSS:
                stops_verified += 1
            elif intent_type in (IntentType.OPEN_LONG, IntentType.OPEN_SHORT):
                pending_fills_checked += 1

        repair_result: dict[str, Any] | None = None
        if bool(payload.get("repair_brackets")):
            repair_result = self._repair_missing_stops(
                mappings, intent_prefix="ID-STOP-", generated_by="INTRADAY_CHECK", dry_run=effective_dry_run,
            )
            stops_verified = repair_result["stops_verified"]
            if repair_result["failures"]:
                failures.extend(repair_result["failures"])

        qty_align_result: dict[str, Any] | None = None
        if bool(payload.get("align_quantities")):
            qty_align_result = self._repair_quantity_mismatch(
                mappings, dry_run=effective_dry_run,
            )
            if qty_align_result["failures"]:
                failures.extend(qty_align_result["failures"])

        # Broker-direct bracket quantity repair: bypass stale mappings entirely.
        # This catches mismatches that _repair_quantity_mismatch misses when
        # mapping states are EXPIRED (e.g. partial fills between runs).
        broker_qty_repair: dict[str, Any] | None = None
        if bool(payload.get("align_quantities")) and not effective_dry_run:
            broker_qty_repair = self._repair_bracket_quantities_from_broker()
            if broker_qty_repair and broker_qty_repair.get("failures"):
                failures.extend(broker_qty_repair["failures"])

        output_data: dict[str, Any] = {
            "schema_version": "1.0.0",
            "intraday_check": True,
            "reconciliation": reconciliation_meta or {"requested": False},
            "stops_verified": stops_verified,
            "pending_fills_checked": pending_fills_checked,
            "active_orders": len([m for m in mappings if not (isinstance(getattr(m, "state", None), OrderState) and getattr(m, "state").is_terminal())]),
            "order_stats": self._derive_order_stats(mappings, position_symbols=self._get_position_symbols()),
        }
        if repair_result is not None:
            output_data["generated_stops"] = repair_result["generated_stops"]
            output_data["positions_at_risk"] = repair_result["positions_at_risk"]
        if qty_align_result is not None:
            output_data["qty_aligned"] = qty_align_result["aligned"]
            output_data["qty_repaired"] = qty_align_result["repaired"]
            if qty_align_result["positions_at_risk"]:
                existing = output_data.get("positions_at_risk", [])
                output_data["positions_at_risk"] = list(existing) + qty_align_result["positions_at_risk"]
        if broker_qty_repair is not None:
            output_data["broker_qty_repair"] = broker_qty_repair

        if payload.get("safe_mode") == "SAFE_REDUCING":
            output_data["safe_mode"] = "SAFE_REDUCING"

        # Best-effort: attach account summary for RunReport Cash/Exposure display.
        if self._adapter is not None and hasattr(self._adapter, "get_account_summary"):
            try:
                acct_result = self._adapter.get_account_summary()
                if acct_result.status is ResultStatus.SUCCESS and acct_result.data:
                    summary = dict(acct_result.data)
                    acct_data: dict[str, Any] = {
                        "cash": summary.get("total_cash_value"),
                        "gross_exposure": summary.get("gross_position_value"),
                        "net_liquidation": summary.get("net_liquidation"),
                        "open_positions_count": summary.get("open_positions_count"),
                        "buying_power": summary.get("buying_power"),
                        "currency_balances": summary.get("currency_balances"),
                    }
                    if hasattr(self._adapter, "get_portfolio_positions"):
                        try:
                            pos_result = self._adapter.get_portfolio_positions()
                            if pos_result.status is ResultStatus.SUCCESS and pos_result.data:
                                cost_basis = 0.0
                                for pos in pos_result.data:
                                    avg_cost = pos.get("avg_cost") or 0.0
                                    qty = abs(pos.get("position") or 0.0)
                                    cost_basis += avg_cost * qty
                                acct_data["cost_basis"] = cost_basis
                                acct_data["open_positions_count"] = len(pos_result.data)
                                acct_data["positions"] = [
                                    {
                                        "symbol": p.get("symbol", ""),
                                        "qty": abs(p.get("position") or 0.0),
                                        "avg_cost": p.get("avg_cost") or 0.0,
                                        "market_value": p.get("market_value") or 0.0,
                                        "unrealized_pnl": p.get("unrealized_pnl") or 0.0,
                                    }
                                    for p in pos_result.data
                                    if p.get("position") and abs(p.get("position") or 0) > 0
                                ]
                        except Exception:  # noqa: BLE001
                            pass
                    pos_syms = [p.get("symbol", "") for p in acct_data.get("positions", []) if p.get("symbol")]
                    acct_data["bracket_orders"] = self._build_bracket_summary_with_broker_fallback(mappings, symbols=pos_syms)
                    output_data["account_summary"] = acct_data
            except Exception:  # noqa: BLE001 - must not block output
                pass

        _emit(self._emit_event, "module.executed", self._module_name, {
            "intraday_check": True,
            "stops_verified": stops_verified,
            "pending_fills_checked": pending_fills_checked,
        })

        if failures:
            first = failures[0]
            error = RecoverableError(
                str(first.get("error", "intraday check degraded")),
                module=self._MODULE_NAME,
                reason_code=str(first.get("reason_code", "INTRADAY_DEGRADED")),
            )
            return Result.degraded(output_data, error, error.reason_code)

        return Result.success(output_data)

    def _execute_pre_close_recon(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        if not self._validated.enabled:
            output: dict[str, Any] = {
                "schema_version": "1.0.0",
                "pre_close_recon": True,
                "orders_cancelled": 0,
                "stops_verified": 0,
                "generated_stops": [],
                "positions_at_risk": [],
                "risk_summary": {"gross_exposure_shares": 0.0, "overnight_risk_shares": 0.0},
            }
            _emit(self._emit_event, "module.executed", self._module_name, {"disabled": True, "pre_close_recon": True})
            return Result.success(output)

        if self._manager is None:
            return Result.failed(RuntimeError("ExecutionPlugin was not initialised."), "NOT_INITIALISED")

        effective_dry_run = bool(self._validated.dry_run) if self._validated.dry_run is not None else False

        reconcile_issue: tuple[BaseException, str] | None = None
        reconciliation_meta: dict[str, Any] | None = None
        if bool(payload.get("reconcile_first")):
            if effective_dry_run:
                reconciliation_meta = {"requested": True, "skipped": True, "reason": "DRY_RUN"}
            else:
                try:
                    reconcile_result = self._manager.reconcile_periodic()
                except Exception as exc:  # noqa: BLE001 - reconciliation must not crash execution
                    reconcile_issue = (exc, "RECONCILE_EXCEPTION")
                    reconciliation_meta = {"requested": True, "status": "FAILED", "reason_code": "RECONCILE_EXCEPTION"}
                else:
                    reconciliation_meta = {
                        "requested": True,
                        "status": reconcile_result.status.value,
                        "reason_code": reconcile_result.reason_code,
                    }
                    if reconcile_result.status is not ResultStatus.SUCCESS:
                        reconcile_issue = (
                            reconcile_result.error or RuntimeError("reconciliation degraded"),
                            reconcile_result.reason_code or "RECONCILE_DEGRADED",
                        )

        orders_result = self._manager.get_all_orders()
        if orders_result.status is ResultStatus.FAILED:
            return Result.failed(orders_result.error or RuntimeError("failed to load orders"), orders_result.reason_code or "ORDER_SNAPSHOT_FAILED")

        mappings = list(orders_result.data or [])

        failures: list[dict[str, Any]] = []

        # Cancel stale entry orders (pre_close exclusive).
        entry_intent_types = {IntentType.OPEN_LONG, IntentType.OPEN_SHORT}
        cancelled: list[str] = []
        would_cancel: list[str] = []
        for mapping in mappings:
            intent_type = self._get_intent_type(mapping)
            if intent_type not in entry_intent_types:
                continue
            state = getattr(mapping, "state", None)
            if isinstance(state, OrderState) and state.is_terminal():
                continue
            client_order_id = str(getattr(mapping, "client_order_id", "")).strip()
            if not client_order_id:
                continue

            if effective_dry_run:
                would_cancel.append(client_order_id)
                continue

            cancel_result = self._manager.cancel_order(client_order_id)
            if cancel_result.status is ResultStatus.FAILED:
                failures.append({
                    "category": "cancel_entry_order",
                    "client_order_id": client_order_id,
                    "reason_code": cancel_result.reason_code or "CANCEL_FAILED",
                    "error": str(cancel_result.error or "cancel failed"),
                })
                continue
            cancelled.append(client_order_id)

        # Repair missing stops via shared method.
        repair_result = self._repair_missing_stops(
            mappings, intent_prefix="PC-STOP-", generated_by="AFTER_MARKET_RECON", dry_run=effective_dry_run,
        )
        generated_stops = repair_result["generated_stops"]
        positions_at_risk = repair_result["positions_at_risk"]
        protected_positions = set(repair_result["protected_intent_ids"])
        open_positions = repair_result["open_positions"]
        if repair_result["failures"]:
            failures.extend(repair_result["failures"])

        # Align SL/TP quantities to actual filled qty.
        qty_align_result = self._repair_quantity_mismatch(
            mappings, dry_run=effective_dry_run,
        )
        if qty_align_result["failures"]:
            failures.extend(qty_align_result["failures"])
        if qty_align_result["positions_at_risk"]:
            positions_at_risk = list(positions_at_risk) + qty_align_result["positions_at_risk"]

        # Broker-direct bracket repair (safety net for stale mappings).
        broker_qty_repair: dict[str, Any] | None = None
        if not effective_dry_run:
            broker_qty_repair = self._repair_bracket_quantities_from_broker()
            if broker_qty_repair and broker_qty_repair.get("failures"):
                failures.extend(broker_qty_repair["failures"])

        gross_exposure_shares = 0.0
        overnight_risk_shares = 0.0
        for position in open_positions:
            intent_snapshot = getattr(position, "intent_snapshot", None)
            qty = float(_intent_get(intent_snapshot, "quantity", 0.0) or 0.0)
            gross_exposure_shares += abs(qty)

        risk_intent_ids = {entry.get("intent_id") for entry in positions_at_risk if isinstance(entry.get("intent_id"), str)}
        for position in open_positions:
            parent_intent_id = self._get_intent_id(position)
            if parent_intent_id not in risk_intent_ids:
                continue
            intent_snapshot = getattr(position, "intent_snapshot", None)
            qty = float(_intent_get(intent_snapshot, "quantity", 0.0) or 0.0)
            overnight_risk_shares += abs(qty)

        output = {
            "schema_version": "1.0.0",
            "pre_close_recon": True,
            "orders_cancelled": len(cancelled) if not effective_dry_run else len(would_cancel),
            "stops_verified": len(protected_positions),
            "generated_stops": list(generated_stops),
            "positions_at_risk": list(positions_at_risk),
            "risk_summary": {
                "gross_exposure_shares": gross_exposure_shares,
                "overnight_risk_shares": overnight_risk_shares,
            },
            "order_stats": self._derive_order_stats(mappings, position_symbols=self._get_position_symbols()),
        }
        if reconciliation_meta is not None:
            output["reconciliation"] = dict(reconciliation_meta)

        # Best-effort: attach account summary for RunReport Cash/Exposure display.
        if self._adapter is not None and hasattr(self._adapter, "get_account_summary"):
            try:
                acct_result = self._adapter.get_account_summary()
                if acct_result.status is ResultStatus.SUCCESS and acct_result.data:
                    summary = dict(acct_result.data)
                    acct_data: dict[str, Any] = {
                        "cash": summary.get("total_cash_value"),
                        "gross_exposure": summary.get("gross_position_value"),
                        "net_liquidation": summary.get("net_liquidation"),
                        "open_positions_count": summary.get("open_positions_count"),
                        "buying_power": summary.get("buying_power"),
                        "currency_balances": summary.get("currency_balances"),
                    }
                    if hasattr(self._adapter, "get_portfolio_positions"):
                        try:
                            pos_result = self._adapter.get_portfolio_positions()
                            if pos_result.status is ResultStatus.SUCCESS and pos_result.data:
                                cost_basis = 0.0
                                for pos in pos_result.data:
                                    avg_cost = pos.get("avg_cost") or 0.0
                                    qty = abs(pos.get("position") or 0.0)
                                    cost_basis += avg_cost * qty
                                acct_data["cost_basis"] = cost_basis
                                acct_data["open_positions_count"] = len(pos_result.data)
                                acct_data["positions"] = [
                                    {
                                        "symbol": p.get("symbol", ""),
                                        "qty": abs(p.get("position") or 0.0),
                                        "avg_cost": p.get("avg_cost") or 0.0,
                                        "market_value": p.get("market_value") or 0.0,
                                        "unrealized_pnl": p.get("unrealized_pnl") or 0.0,
                                    }
                                    for p in pos_result.data
                                    if p.get("position") and abs(p.get("position") or 0) > 0
                                ]
                        except Exception:  # noqa: BLE001
                            pass
                    pos_syms = [p.get("symbol", "") for p in acct_data.get("positions", []) if p.get("symbol")]
                    acct_data["bracket_orders"] = self._build_bracket_summary_with_broker_fallback(mappings, symbols=pos_syms)
                    output["account_summary"] = acct_data
            except Exception:  # noqa: BLE001 - must not block pre-close output
                pass

        _emit(
            self._emit_event,
            "module.executed",
            self._module_name,
            {
                "pre_close_recon": True,
                "cancelled": len(cancelled),
                "generated_stops": len(generated_stops),
                "positions_at_risk": len(positions_at_risk),
                "failures": len(failures),
            },
        )

        if failures:
            wrapped = PartialDataError(
                "Pre-close reconciliation degraded.",
                module=self._MODULE_NAME,
                reason_code="AFTER_MARKET_RECON_DEGRADED",
                details={"failures": failures},
            )
            return Result.degraded(output, wrapped, wrapped.reason_code)

        if reconcile_issue is not None:
            error, reason = reconcile_issue
            return Result.degraded(output, error, reason)

        return Result.success(output)

    def _execute_post_close_recon(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        """Post-close reconciliation: reconcile + snapshot only, no cancel/stop actions."""
        if not self._validated.enabled:
            output: dict[str, Any] = {
                "schema_version": "1.0.0",
                "post_close_recon": True,
                "open_positions_count": 0,
                "open_orders_count": 0,
            }
            _emit(self._emit_event, "module.executed", self._module_name, {"disabled": True, "post_close_recon": True})
            return Result.success(output)

        if self._manager is None:
            return Result.failed(RuntimeError("ExecutionPlugin was not initialised."), "NOT_INITIALISED")

        effective_dry_run = bool(self._validated.dry_run) if self._validated.dry_run is not None else False

        reconcile_issue: tuple[BaseException, str] | None = None
        reconciliation_meta: dict[str, Any] | None = None
        if bool(payload.get("reconcile_first")):
            if effective_dry_run:
                reconciliation_meta = {"requested": True, "skipped": True, "reason": "DRY_RUN"}
            else:
                try:
                    reconcile_result = self._manager.reconcile_periodic()
                except Exception as exc:  # noqa: BLE001 - reconciliation must not crash execution
                    reconcile_issue = (exc, "RECONCILE_EXCEPTION")
                    reconciliation_meta = {"requested": True, "status": "FAILED", "reason_code": "RECONCILE_EXCEPTION"}
                else:
                    reconciliation_meta = {
                        "requested": True,
                        "status": reconcile_result.status.value,
                        "reason_code": reconcile_result.reason_code,
                    }
                    if reconcile_result.status is not ResultStatus.SUCCESS:
                        reconcile_issue = (
                            reconcile_result.error or RuntimeError("reconciliation degraded"),
                            reconcile_result.reason_code or "RECONCILE_DEGRADED",
                        )

        orders_result = self._manager.get_all_orders()
        if orders_result.status is ResultStatus.FAILED:
            return Result.failed(orders_result.error or RuntimeError("failed to load orders"), orders_result.reason_code or "ORDER_SNAPSHOT_FAILED")

        mappings = list(orders_result.data or [])

        failures: list[dict[str, Any]] = []
        if reconcile_issue is not None:
            error_r, reason_r = reconcile_issue
            failures.append({"category": "reconcile", "reason_code": reason_r, "error": str(error_r)})

        open_positions_count = 0
        open_orders_count = 0
        terminal_no_position = {OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED}
        for mapping in mappings:
            state = getattr(mapping, "state", None)
            if isinstance(state, OrderState) and state in terminal_no_position:
                continue
            if state in (OrderState.FILLED, OrderState.PARTIALLY_FILLED):
                open_positions_count += 1
            else:
                open_orders_count += 1

        # Repair missing stops and align SL/TP quantities (safety net for overnight positions).
        repair_result = self._repair_missing_stops(
            mappings, intent_prefix="AM-STOP-", generated_by="AFTER_MARKET_RECON", dry_run=effective_dry_run,
        )
        generated_stops = repair_result["generated_stops"]
        positions_at_risk = repair_result["positions_at_risk"]
        if repair_result["failures"]:
            failures.extend(repair_result["failures"])

        qty_align_result = self._repair_quantity_mismatch(
            mappings, dry_run=effective_dry_run,
        )
        if qty_align_result["failures"]:
            failures.extend(qty_align_result["failures"])
        if qty_align_result["positions_at_risk"]:
            positions_at_risk = list(positions_at_risk) + qty_align_result["positions_at_risk"]

        # Broker-direct bracket repair (safety net for stale mappings).
        if not effective_dry_run:
            broker_qty_repair = self._repair_bracket_quantities_from_broker()
            if broker_qty_repair and broker_qty_repair.get("failures"):
                failures.extend(broker_qty_repair["failures"])

        output: dict[str, Any] = {
            "schema_version": "1.0.0",
            "post_close_recon": True,
            "open_positions_count": open_positions_count,
            "open_orders_count": open_orders_count,
            "generated_stops": list(generated_stops),
            "positions_at_risk": list(positions_at_risk),
            "order_stats": self._derive_order_stats(mappings, position_symbols=self._get_position_symbols()),
        }
        if reconciliation_meta is not None:
            output["reconciliation"] = dict(reconciliation_meta)

        if self._adapter is not None and hasattr(self._adapter, "get_account_summary"):
            try:
                acct_result = self._adapter.get_account_summary()
                if acct_result.status is ResultStatus.SUCCESS and acct_result.data:
                    summary = dict(acct_result.data)
                    acct_data_post: dict[str, Any] = {
                        "cash": summary.get("total_cash_value"),
                        "gross_exposure": summary.get("gross_position_value"),
                        "net_liquidation": summary.get("net_liquidation"),
                        "open_positions_count": summary.get("open_positions_count"),
                        "buying_power": summary.get("buying_power"),
                        "currency_balances": summary.get("currency_balances"),
                    }
                    if hasattr(self._adapter, "get_portfolio_positions"):
                        try:
                            pos_result = self._adapter.get_portfolio_positions()
                            if pos_result.status is ResultStatus.SUCCESS and pos_result.data:
                                cost_basis = 0.0
                                for pos in pos_result.data:
                                    avg_cost = pos.get("avg_cost") or 0.0
                                    qty = abs(pos.get("position") or 0.0)
                                    cost_basis += avg_cost * qty
                                acct_data_post["cost_basis"] = cost_basis
                                acct_data_post["open_positions_count"] = len(pos_result.data)
                                acct_data_post["positions"] = [
                                    {
                                        "symbol": p.get("symbol", ""),
                                        "qty": abs(p.get("position") or 0.0),
                                        "avg_cost": p.get("avg_cost") or 0.0,
                                        "market_value": p.get("market_value") or 0.0,
                                        "unrealized_pnl": p.get("unrealized_pnl") or 0.0,
                                    }
                                    for p in pos_result.data
                                    if p.get("position") and abs(p.get("position") or 0) > 0
                                ]
                        except Exception:  # noqa: BLE001
                            pass
                    pos_syms = [p.get("symbol", "") for p in acct_data_post.get("positions", []) if p.get("symbol")]
                    acct_data_post["bracket_orders"] = self._build_bracket_summary_with_broker_fallback(mappings, symbols=pos_syms)
                    output["account_summary"] = acct_data_post
            except Exception:  # noqa: BLE001 - must not block post-close output
                pass

        _emit(
            self._emit_event,
            "module.executed",
            self._module_name,
            {
                "post_close_recon": True,
                "open_positions": open_positions_count,
                "open_orders": open_orders_count,
                "generated_stops": len(generated_stops),
                "positions_at_risk": len(positions_at_risk),
                "failures": len(failures),
            },
        )

        if failures:
            wrapped = PartialDataError(
                "After-market reconciliation degraded.",
                module=self._MODULE_NAME,
                reason_code="AFTER_MARKET_RECON_DEGRADED",
                details={"failures": failures},
            )
            return Result.degraded(output, wrapped, "AFTER_MARKET_RECON_DEGRADED")

        return Result.success(output)

    def cleanup(self) -> Result[None]:
        """Disconnect broker adapter and cancel open orders (best-effort).

        Cleanup must never raise. Any issues (cancellation/disconnect) are recorded
        as degraded status but must not prevent the remaining cleanup steps.
        """

        degraded_error: BaseException | None = None
        degraded_reason: str | None = None

        # Step 1: Cancel open orders (best-effort) before disconnecting.
        try:
            if self._manager is not None:
                cleanup_strategy = str(getattr(self._validated, "cleanup_strategy", "keep_all") or "keep_all").strip().lower() or "keep_all"

                cancel_result: Result[dict[str, Any]] | None = None
                if cleanup_strategy == "cancel_all":
                    cancel_result = self._manager.cancel_all_open_orders()
                elif cleanup_strategy == "cancel_open_only":
                    cancel_result = self._manager.cancel_all_open_orders(intent_types=["OPEN_LONG", "OPEN_SHORT"])

                if cancel_result is not None:
                    summary = dict(cancel_result.data or {})

                    if cancel_result.status is ResultStatus.FAILED:
                        degraded_error = cancel_result.error or RuntimeError("Execution cleanup cancellation failed.")
                        degraded_reason = cancel_result.reason_code or "CLEANUP_CANCEL_FAILED"
                    elif cancel_result.status is ResultStatus.DEGRADED:
                        degraded_error = cancel_result.error or RuntimeError("Execution cleanup cancellation degraded.")
                        degraded_reason = cancel_result.reason_code or "CLEANUP_CANCEL_DEGRADED"

                    if hasattr(self._logger, "info"):
                        try:
                            self._logger.info(
                                "execution.cleanup.cancel_orders",
                                strategy=cleanup_strategy,
                                status=cancel_result.status.value,
                                total_cancelled=int(summary.get("total_cancelled", 0) or 0),
                                failed_cancellations=len(summary.get("failed_cancellations", []) or []),
                                reason_code=cancel_result.reason_code,
                            )
                        except Exception:  # noqa: BLE001
                            pass

                    _emit(
                        self._emit_event,
                        "execution.cleanup",
                        self._module_name,
                        {
                            "strategy": cleanup_strategy,
                            "status": cancel_result.status.value,
                            "reason_code": cancel_result.reason_code,
                            "total_cancelled": int(summary.get("total_cancelled", 0) or 0),
                            "failed_cancellations": len(summary.get("failed_cancellations", []) or []),
                        },
                    )
        except Exception as exc:  # noqa: BLE001 - cleanup must never raise
            degraded_error = degraded_error or exc
            degraded_reason = degraded_reason or "CLEANUP_CANCEL_EXCEPTION"
            _emit(
                self._emit_event,
                "execution.cleanup",
                self._module_name,
                {
                    "strategy": str(getattr(self._validated, "cleanup_strategy", "keep_all") or "keep_all").strip().lower() or "keep_all",
                    "status": "exception",
                    "reason_code": degraded_reason,
                    "total_cancelled": 0,
                    "failed_cancellations": 0,
                },
            )

        # Step 2: Disconnect adapter (best-effort).
        # Skip disconnect when using a daemon-shared adapter — the daemon owns
        # the connection lifecycle and will disconnect on shutdown.
        try:
            if self._adapter is not None and getattr(self, "_shared_adapter", None) is None:
                self._adapter.disconnect()
        except Exception as exc:  # noqa: BLE001
            degraded_error = degraded_error or exc
            degraded_reason = degraded_reason or "CLEANUP_DISCONNECT_EXCEPTION"

        _emit(self._emit_event, "module.cleaned_up", self._module_name, {"module": "execution"})
        if degraded_error is not None:
            return Result.degraded(None, degraded_error, degraded_reason or "CLEANUP_DEGRADED")
        return Result.success(data=None)

    def _build_manager(self, validated: ExecutionPluginConfig) -> Result[OrderManager]:
        # Daemon mode: reuse pre-connected shared adapter.
        if getattr(self, "_shared_adapter", None) is not None:
            adapter = self._shared_adapter
            self._adapter = adapter
            id_gen = IDGenerator()
            sm = StateMachine()
            persistence = Persistence(storage_dir=Path(validated.persistence_dir))
            reconciler = Reconciler(state_machine=sm, fetch_broker_order=adapter.fetch_broker_order)
            manager = OrderManager(
                adapter=adapter,
                id_generator=id_gen,
                state_machine=sm,
                persistence=persistence,
                reconciler=reconciler,
            )
            self._manager = manager
            return Result.success(manager)

        broker_cfg = dict(validated.broker or {})
        broker_payload = {
            "host": str(broker_cfg.get("host") or "127.0.0.1"),
            "port": _safe_int(broker_cfg.get("port"), default=7497),
            "client_id": _safe_int(broker_cfg.get("client_id"), default=1),
            "readonly": _safe_bool(broker_cfg.get("readonly", False), default=False),
            "account": str(broker_cfg.get("account") or "").strip() or None,
            "timeout": _safe_int(broker_cfg.get("timeout"), default=20),
            "max_reconnect_attempts": _safe_int(broker_cfg.get("max_reconnect_attempts"), default=5),
            "enable_dynamic_client_id": _safe_bool(broker_cfg.get("enable_dynamic_client_id", False), default=False),
            "client_id_range": _safe_tuple_int(broker_cfg.get("client_id_range"), default=(1, 32)),
        }
        try:
            connection = msgspec.convert(broker_payload, type=BrokerConnectionConfig)
        except Exception as exc:  # noqa: BLE001
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": broker_payload},
            )
            return Result.failed(error, error.reason_code)
        self._connection_config = connection

        adapter_type = (validated.adapter or "").strip().lower()
        if adapter_type != "ibkr":
            error = ConfigurationError(
                f"Unsupported execution adapter: {adapter_type!r}",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"adapter": adapter_type},
            )
            return Result.failed(error, error.reason_code)

        id_gen = IDGenerator()
        fills_store = FillsPersistence(storage_dir=Path(validated.persistence_dir))
        adapter = IBKRAdapter(connection, id_gen, fills_persistence=fills_store)

        # Restore persisted fills into adapter memory (best-effort).
        try:
            restore_result = fills_store.load_all()
            if restore_result.status is not ResultStatus.FAILED and restore_result.data:
                adapter.restore_fills(restore_result.data)
        except Exception:  # noqa: BLE001 - must not block manager construction
            pass

        sm = StateMachine()
        persistence = Persistence(storage_dir=Path(validated.persistence_dir))
        reconciler = Reconciler(state_machine=sm, fetch_broker_order=adapter.fetch_broker_order)
        manager = OrderManager(
            adapter=adapter,
            id_generator=id_gen,
            state_machine=sm,
            persistence=persistence,
            reconciler=reconciler,
        )
        self._adapter = adapter
        self._manager = manager

        if validated.connect_on_init:
            connect_result = adapter.connect(force=True)
            if connect_result.status is ResultStatus.FAILED:
                return Result.degraded(manager, connect_result.error or RuntimeError("connect failed"), connect_result.reason_code or "CONNECT_FAILED")
            if connect_result.status is ResultStatus.DEGRADED:
                return Result.degraded(manager, connect_result.error or RuntimeError("connect degraded"), connect_result.reason_code or "CONNECT_DEGRADED")

        return Result.success(manager)

    def _parse_intents(self, payload: Mapping[str, Any]) -> Result[OrderIntentSet]:
        raw = payload.get("intents")
        if isinstance(raw, Mapping):
            try:
                return Result.success(msgspec.convert(dict(raw), type=OrderIntentSet))
            except Exception as exc:  # noqa: BLE001
                error = ValidationError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code="INVALID_INTENTS",
                    details={"payload_keys": list(raw.keys())},
                )
                return Result.failed(error, error.reason_code)
        return Result.failed(
            ValidationError("Missing intents payload.", module=self._MODULE_NAME, reason_code="MISSING_INTENTS"),
            "MISSING_INTENTS",
        )

    def _parse_risk_decisions(self, payload: Mapping[str, Any]) -> Result[RiskDecisionSet]:
        raw = payload.get("risk_decisions")
        if raw is None:
            raw = payload.get("decisions")
        if isinstance(raw, Mapping):
            try:
                return Result.success(msgspec.convert(dict(raw), type=RiskDecisionSet))
            except Exception as exc:  # noqa: BLE001
                error = ValidationError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code="INVALID_RISK_DECISIONS",
                    details={"payload_keys": list(raw.keys())},
                )
                return Result.failed(error, error.reason_code)
        return Result.failed(
            ValidationError("Missing risk_decisions payload.", module=self._MODULE_NAME, reason_code="MISSING_RISK_DECISIONS"),
            "MISSING_RISK_DECISIONS",
        )

    def _attach_run_id(self, intent: Any, run_id: str) -> Any:
        try:
            metadata = dict(getattr(intent, "metadata", {}) or {})
        except Exception:  # noqa: BLE001
            metadata = {}
        metadata.setdefault("run_id", run_id)
        try:
            return msgspec.structs.replace(intent, metadata=metadata)
        except Exception:  # noqa: BLE001
            return intent

    def _skipped_report(self, *, run_id: str, intent_id: str, symbol: str, reason: str) -> ExecutionReport:
        return ExecutionReport(
            run_id=run_id,
            intent_id=intent_id,
            client_order_id=f"SKIPPED-{intent_id}",
            broker_order_id="SKIPPED",
            symbol=symbol,
            final_state=OrderState.REJECTED,
            filled_quantity=0.0,
            remaining_quantity=0.0,
            avg_fill_price=None,
            commissions=None,
            fills=[],
            state_transitions=[],
            executed_at_ns=int(time.time_ns()),
            metadata={"skipped": True, "reason": reason},
        )

    def _rejected_report(
        self,
        *,
        run_id: str,
        intent_id: str,
        symbol: str,
        reason_code: str,
        error: BaseException,
    ) -> ExecutionReport:
        return ExecutionReport(
            run_id=run_id,
            intent_id=intent_id,
            client_order_id=f"REJECTED-{intent_id}",
            broker_order_id="REJECTED",
            symbol=symbol,
            final_state=OrderState.REJECTED,
            filled_quantity=0.0,
            remaining_quantity=0.0,
            avg_fill_price=None,
            commissions=None,
            fills=[],
            state_transitions=[],
            executed_at_ns=int(time.time_ns()),
            metadata={"error": str(error), "reason_code": reason_code},
        )


def register_real_plugins(registry: "PluginRegistry") -> None:
    """Register orchestrator-facing real plugins (Universe/Data/Scanner).

    This mirrors :func:`orchestrator.stubs.register_stub_plugins` but registers
    production adapters under distinct ``*_real`` names so stub plugins remain
    available for tests and dry-runs.
    """

    from common.exceptions import ConfigurationError
    from plugins.registry import PluginRegistry  # Local import to avoid cycles.

    if not isinstance(registry, PluginRegistry):
        raise TypeError("registry must be an instance of PluginRegistry.")

    registrations: list[tuple[str, type[PluginBase[Any, Any, Any]], PluginCategory, str, str]] = [
        (
            "universe_real",
            UniversePlugin,
            PluginCategory.DATA_SOURCE,
            "UniverseBuilder adapter producing a serializable UniverseSnapshot payload.",
            "1.0.0",
        ),
        (
            "universe_cached",
            UniverseCachedPlugin,
            PluginCategory.DATA_SOURCE,
            "Load a cached universe snapshot from disk (no live API calls).",
            "1.0.0",
        ),
        (
            "data_real",
            DataPlugin,
            PluginCategory.DATA_SOURCE,
            "DataOrchestrator adapter fetching per-symbol price series snapshots.",
            "1.0.0",
        ),
        (
            "scanner_real",
            ScannerPlugin,
            PluginCategory.SCANNER,
            "Scanner detector adapter producing a serializable CandidateSet snapshot.",
            "1.0.0",
        ),
        (
            "event_guard_real",
            EventGuardPlugin,
            PluginCategory.RISK_POLICY,
            "Event Guard adapter producing per-symbol TradeConstraints for candidates.",
            "1.0.0",
        ),
        (
            "strategy_real",
            StrategyPlugin,
            PluginCategory.STRATEGY,
            "Strategy Engine adapter producing a serializable OrderIntentSet payload.",
            "1.0.0",
        ),
        (
            "risk_gate_real",
            RiskGatePlugin,
            PluginCategory.RISK_POLICY,
            "Risk Gate adapter producing a serializable RiskDecisionSet payload.",
            "1.0.0",
        ),
        (
            "execution_real",
            ExecutionPlugin,
            PluginCategory.SIGNAL,
            "Execution adapter producing a serializable list of ExecutionReport payloads.",
            "1.0.0",
        ),
    ]

    for name, plugin_class, category, description, schema_version in registrations:
        version = getattr(getattr(plugin_class, "metadata", None), "version", "1.0.0")
        enabled = bool(getattr(getattr(plugin_class, "metadata", None), "enabled", True))
        metadata = PluginMetadata(
            name=name,
            version=str(version),
            schema_version=schema_version,
            category=category,
            enabled=enabled,
            description=description,
        )
        try:
            registry.register(name, plugin_class, metadata=metadata)
        except ConfigurationError:
            continue
