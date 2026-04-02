"""PRE_MARKET_FULL_SCAN orchestrator wiring stub modules into an executable pipeline."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

import msgspec

from common.config import _deep_merge
from common.exceptions import ConfigurationError, OperationalError, RecoverableError
from common.interface import DomainEvent, EventBus, LoggerFactory, Result, ResultStatus
from common.market_calendar import MarketCalendar
from data.interface import PriceSeriesSnapshot
from universe.interface import EquityInfo
from journal import JournalWriter, RunIDGenerator
from journal.interface import OperatingMode, RunMetadata, RunType
from plugins.lifecycle import PluginLifecycleManager
from plugins.registry import PluginRegistry
from scanner.market_regime.config_loader import RegimeConfigLoader
from scanner.market_regime.detector import MarketRegimeDetector
from scanner.market_regime.interface import MarketRegime, RegimeConfig, RegimeDetectionResult

from .interface import ModuleResult, OrchestratorContext, RunSummary
from .summary import generate_summary

__all__ = ["EODScanOrchestrator"]


class EODScanOrchestrator:
    """Coordinate the Phase 1.5 PRE_MARKET_FULL_SCAN stub pipeline end-to-end."""

    _MODULE_NAME = "orchestrator.eod_scan"
    _INVALID_CONFIG_REASON = "EOD_INVALID_CONFIG"
    _RUN_FAILED_REASON = "EOD_RUN_FAILED"
    _RUN_DEGRADED_REASON = "EOD_RUN_DEGRADED"
    _EVENT_FAILED_REASON = "EOD_EVENT_FAILED"
    _PERSIST_FAILED_REASON = "EOD_PERSIST_FAILED"
    _PERSIST_DEGRADED_REASON = "EOD_PERSIST_DEGRADED"

    _DEFAULT_UNIVERSE_PLUGIN = "universe_real"  # was "universe_stub"
    _DEFAULT_DATA_PLUGIN = "data_real"  # was "data_stub"
    _DEFAULT_SCANNER_PLUGIN = "scanner_real"  # was "scanner_stub"
    _DEFAULT_EVENT_GUARD_PLUGIN = "event_guard_real"
    _DEFAULT_STRATEGY_PLUGIN = "strategy_real"
    _DEFAULT_RISK_GATE_PLUGIN = "risk_gate_real"
    _DEFAULT_EXECUTION_PLUGIN = "execution_real"

    def __init__(
        self,
        *,
        journal_writer: JournalWriter,
        event_bus: EventBus,
        logger_factory: LoggerFactory,
        plugin_registry: PluginRegistry,
        lifecycle_manager: PluginLifecycleManager | None = None,
        clock: Callable[[], int] | None = None,
        system_version_getter: Callable[[], str] | None = None,
        report_hook: Callable[..., None] | None = None,
    ) -> None:
        self._journal_writer = journal_writer
        self._event_bus = event_bus
        self._logger_factory = logger_factory
        self._plugin_registry = plugin_registry
        self._lifecycle_manager = lifecycle_manager or PluginLifecycleManager()
        self._clock = clock or time.time_ns
        self._system_version_getter = system_version_getter or RunIDGenerator.get_system_version
        self._report_hook = report_hook
        self._last_run_id: str | None = None
        self._last_module_results: list[ModuleResult] = []
        self._last_outputs_by_module: dict[str, Any] = {}
        self._pending_regime_output: dict[str, Any] | None = None
        # V27.4: Optional candidate-level pattern filter applied before strategy dedup
        self.candidate_filter_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        # Wall-clock override: when set, plugins use this instead of time.time_ns()
        self.scan_timestamp_ns: int | None = None

    @property
    def last_run_id(self) -> str | None:
        """Return the most recently executed run id (best-effort)."""

        return self._last_run_id

    @property
    def last_module_results(self) -> list[ModuleResult]:
        """Return module results captured from the most recent run (best-effort)."""

        return list(self._last_module_results)

    @property
    def last_outputs_by_module(self) -> dict[str, Any]:
        """Return non-failed module outputs from the most recent run (best-effort)."""

        return dict(self._last_outputs_by_module)

    def _capture_last_run(self, run_id: str, module_results: list[ModuleResult]) -> None:
        """Capture module outputs for downstream orchestration (best-effort)."""

        self._last_run_id = run_id
        self._last_module_results = list(module_results)
        outputs = {
            result.module_name: result.output_data
            for result in module_results
            if result.status is not ResultStatus.FAILED
        }
        if self._pending_regime_output is not None:
            outputs["market_regime"] = dict(self._pending_regime_output)
        self._last_outputs_by_module = outputs
        self._pending_regime_output = None

    def execute_run(self, config: Mapping[str, Any], *, run_type: RunType | str | None = None) -> Result[RunSummary]:
        """Execute the run using the registered orchestrator modules."""

        if not isinstance(config, Mapping):
            error = ConfigurationError(
                "Orchestrator configuration must be a mapping.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
            )
            return Result.failed(error, error.reason_code)

        config_map = dict(config)
        run_type_raw = run_type if run_type is not None else config_map.get("run_type", RunType.PRE_MARKET_FULL_SCAN)
        run_type_result = self._parse_run_type(run_type_raw)
        if run_type_result.status is ResultStatus.FAILED:
            error = run_type_result.error or ConfigurationError(
                "Invalid run type specified.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
                details={"run_type": run_type_raw},
            )
            reason = run_type_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
            return Result.failed(error, reason)
        resolved_run_type = run_type_result.data

        mode_result = self._parse_mode(config_map.get("mode", OperatingMode.DRY_RUN))
        if mode_result.status is ResultStatus.FAILED:
            error = mode_result.error or ConfigurationError(
                "Invalid operating mode specified.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
                details={"mode": config_map.get("mode")},
            )
            reason = mode_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
            return Result.failed(error, reason)
        mode = mode_result.data

        run_id = RunIDGenerator.generate_run_id(resolved_run_type)
        logger = self._logger_factory.get_logger(self._MODULE_NAME, run_id).bind(
            run_id=run_id,
            run_type=resolved_run_type.value,
        )
        self._pending_regime_output = None

        orchestrator_context = OrchestratorContext(
            run_id=run_id,
            run_type=resolved_run_type,
            mode=mode,
            config=config_map,
            logger=logger,
            event_bus=self._event_bus,
        )

        events: list[DomainEvent] = []
        event_status = ResultStatus.SUCCESS
        event_failure: BaseException | None = None
        event_degraded: RecoverableError | None = None

        def emit_event(event_type: str, module_name: str, data: Mapping[str, Any] | None = None) -> None:
            nonlocal event_status, event_failure, event_degraded
            event_result = self._emit_event(run_id, event_type, module_name, data)
            if event_result.status is ResultStatus.FAILED:
                event_status = ResultStatus.FAILED
                event_failure = event_result.error
            elif event_result.status is ResultStatus.DEGRADED and event_status is ResultStatus.SUCCESS:
                event_status = ResultStatus.DEGRADED
                if isinstance(event_result.error, RecoverableError):
                    event_degraded = event_result.error
                else:
                    event_degraded = RecoverableError.from_error(
                        event_result.error or RuntimeError("Event degraded."),
                        module=self._MODULE_NAME,
                        reason_code=self._EVENT_FAILED_REASON,
                        details={"run_id": run_id, "module": module_name},
                    )
            if event_result.status is ResultStatus.SUCCESS and event_result.data is not None:
                events.append(event_result.data)

        start_time_ns = self._clock()
        run_metadata = RunMetadata(
            run_id=run_id,
            run_type=resolved_run_type,
            mode=mode,
            system_version=self._system_version_getter(),
            start_time=start_time_ns,
            end_time=None,
            status="running",
        )

        emit_event("run.start", "orchestrator", {"mode": mode.value})

        # Market calendar holiday check (Phase 3.8.4)
        market_calendar_config = config_map.get("market_calendar", {})
        holiday_behavior_config = config_map.get("holiday_behavior", {})
        is_holiday = False
        holiday_name = None

        # Backtests/dry-runs do not require live holiday status checks and these
        # HTTP calls can become an init-time stall point in restricted-network
        # environments.
        if mode is not OperatingMode.DRY_RUN and isinstance(market_calendar_config, Mapping) and market_calendar_config.get("enabled", False):
            polygon_api_key = os.getenv("POLYGON_API_KEY")
            if polygon_api_key:
                try:
                    market_calendar = MarketCalendar(
                        api_key=polygon_api_key,
                        cache_ttl_hours=market_calendar_config.get("cache_ttl_hours", 24),
                        requests_timeout_seconds=market_calendar_config.get("requests_timeout_seconds", 30),
                    )
                    status_result = market_calendar.get_current_status()
                    if status_result.data:
                        is_holiday = status_result.data.is_holiday
                        if is_holiday:
                            # Find holiday name
                            holidays_result = market_calendar.fetch_upcoming_holidays(days_ahead=1)
                            if holidays_result.data:
                                today_str = datetime.now(tz=UTC).date().isoformat()
                                for holiday in holidays_result.data:
                                    if holiday.date == today_str:
                                        holiday_name = holiday.name
                                        break

                            logger.info(
                                "holiday_mode_activated",
                                is_holiday=True,
                                holiday_name=holiday_name or "Unknown",
                                behavior="scan_only",
                            )
                except Exception as exc:
                    logger.warning(
                        "market_calendar_check_failed",
                        error=str(exc)[:200],
                        default_behavior="assume_trading_day",
                    )

        plugins_result = self._plugin_registry.load_from_config(config_map)
        if plugins_result.status is ResultStatus.FAILED:
            error = plugins_result.error or ConfigurationError(
                "Failed to load orchestrator plugins.",
                module=self._MODULE_NAME,
                reason_code=plugins_result.reason_code or self._INVALID_CONFIG_REASON,
            )
            reason = plugins_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
            return Result.failed(error, reason)
        plugin_instances = plugins_result.data or {}

        run_status = ResultStatus.SUCCESS
        failure_error: BaseException | None = None
        degrade_error: RecoverableError | None = None

        if plugins_result.status is ResultStatus.DEGRADED:
            reason = plugins_result.reason_code or self._RUN_DEGRADED_REASON
            degrade_error = self._ensure_recoverable(plugins_result.error, reason)
            run_status = ResultStatus.DEGRADED

        module_results: list[ModuleResult] = []

        def update_status_from_module(result: ModuleResult) -> None:
            nonlocal run_status, failure_error, degrade_error
            if result.status is ResultStatus.FAILED:
                run_status = ResultStatus.FAILED
                failure_error = failure_error or result.error
            elif result.status is ResultStatus.DEGRADED and run_status is ResultStatus.SUCCESS:
                run_status = ResultStatus.DEGRADED
                degrade_error = degrade_error or self._ensure_recoverable(
                    result.error,
                    self._RUN_DEGRADED_REASON,
                )

        def append_optional_skip(module_name: str, error: RecoverableError) -> None:
            nonlocal run_status, degrade_error
            module_results.append(
                ModuleResult(
                    module_name=module_name,
                    status=ResultStatus.DEGRADED,
                    output_data=None,
                    error=error,
                )
            )
            if run_status is ResultStatus.SUCCESS:
                run_status = ResultStatus.DEGRADED
            degrade_error = degrade_error or error

        def apply_event_status() -> None:
            nonlocal run_status, failure_error, degrade_error
            if event_status is ResultStatus.FAILED:
                if failure_error is None:
                    failure_error = event_failure or OperationalError(
                        "Event emission failed.",
                        module=self._MODULE_NAME,
                        reason_code=self._EVENT_FAILED_REASON,
                        details={"run_id": run_id},
                    )
                run_status = ResultStatus.FAILED
            elif event_status is ResultStatus.DEGRADED and run_status is ResultStatus.SUCCESS:
                degrade_error = degrade_error or self._ensure_recoverable(
                    event_degraded,
                    self._EVENT_FAILED_REASON,
                )
                run_status = ResultStatus.DEGRADED

        def finalise_run() -> Result[RunSummary]:
            nonlocal run_status, failure_error, degrade_error

            apply_event_status()

            end_time_ns = self._clock()
            status_label = self._status_string(run_status)
            emit_event("run.completed", "orchestrator", {"status": status_label})
            apply_event_status()
            status_label = self._status_string(run_status)

            final_metadata = RunMetadata(
                run_id=run_metadata.run_id,
                run_type=run_metadata.run_type,
                mode=run_metadata.mode,
                system_version=run_metadata.system_version,
                start_time=run_metadata.start_time,
                end_time=end_time_ns,
                status=status_label,
            )

            outputs: dict[str, Any] = {}
            for result in module_results:
                if result.status is ResultStatus.FAILED:
                    continue
                payload = result.output_data
                if not isinstance(payload, Mapping):
                    continue
                module_key = result.module_name.lower().strip()
                # Trust plugin outputs - no validation needed.
                if module_key == "universe":
                    outputs["universe.json"] = dict(payload)
                elif module_key == "scanner":
                    outputs["candidates.json"] = dict(payload)
                elif module_key == "event_guard":
                    outputs["constraints.json"] = dict(payload)
                elif module_key == "strategy":
                    outputs["intents.json"] = dict(payload)
                elif module_key == "risk_gate":
                    outputs["risk_decisions.json"] = dict(payload)
                elif module_key == "execution":
                    outputs["execution_reports.json"] = dict(payload)

            # Save market regime output if available.
            # _pending_regime_output is cleared by _capture_last_run(), so fall back
            # to the cached copy in _last_outputs_by_module.
            regime_out = self._pending_regime_output or self._last_outputs_by_module.get("market_regime")
            if regime_out is not None:
                outputs["market_regime.json"] = dict(regime_out)

            persist_result = self._journal_writer.persist_complete_run(
                run_id,
                final_metadata,
                inputs={},
                outputs=outputs,
                events=events,
            )
            if persist_result.status is ResultStatus.FAILED:
                reason = persist_result.reason_code or self._PERSIST_FAILED_REASON
                failure_error = persist_result.error or OperationalError(
                    "Failed to persist run artifacts.",
                    module=self._MODULE_NAME,
                    reason_code=reason,
                    details={"run_id": run_id},
                )
                summary = generate_summary(final_metadata, module_results)
                normalised_error = self._normalise_failure_error(failure_error, reason, run_id, summary)
                return Result.failed(normalised_error, reason)
            if persist_result.status is ResultStatus.DEGRADED and run_status is ResultStatus.SUCCESS:
                run_status = ResultStatus.DEGRADED
                degrade_reason = persist_result.reason_code or self._PERSIST_DEGRADED_REASON
                degrade_error = self._ensure_recoverable(persist_result.error, degrade_reason)
                status_label = self._status_string(run_status)

            effective_metadata = RunMetadata(
                run_id=final_metadata.run_id,
                run_type=final_metadata.run_type,
                mode=final_metadata.mode,
                system_version=final_metadata.system_version,
                start_time=final_metadata.start_time,
                end_time=final_metadata.end_time,
                status=status_label,
            )
            summary = generate_summary(effective_metadata, module_results)

            # Report hook — must never crash the run
            if self._report_hook is not None:
                try:
                    self._report_hook(
                        run_id=run_id,
                        metadata=effective_metadata,
                        module_results=module_results,
                        outputs=outputs,
                        events=events,
                    )
                except Exception:  # noqa: BLE001
                    pass  # report failure must not affect run outcome

            if run_status is ResultStatus.FAILED:
                reason = getattr(failure_error, "reason_code", None) or self._RUN_FAILED_REASON
                error = self._normalise_failure_error(failure_error, reason, run_id, summary)
                return Result.failed(error, reason)

            if run_status is ResultStatus.DEGRADED:
                reason = getattr(degrade_error, "reason_code", None) or self._RUN_DEGRADED_REASON
                error = degrade_error or RecoverableError(
                    "Orchestrator degraded.",
                    module=self._MODULE_NAME,
                    reason_code=reason,
                    details={"run_id": run_id, "run_summary": summary},
                )
                if isinstance(error, RecoverableError):
                    error.details.setdefault("run_summary", summary)
                return Result.degraded(summary, error, reason)

            return Result.success(summary)

        if resolved_run_type in (RunType.PRE_CLOSE_CLEANUP, RunType.AFTER_MARKET_RECON):
            execution_name_result = self._coerce_plugin_name(
                config_map.get("execution_plugin"),
                default=self._DEFAULT_EXECUTION_PLUGIN,
                role="execution",
            )
            if execution_name_result.status is ResultStatus.FAILED:
                error = execution_name_result.error or ConfigurationError(
                    "Execution plugin selection failed.",
                    module=self._MODULE_NAME,
                    reason_code=self._INVALID_CONFIG_REASON,
                )
                reason = execution_name_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
                return Result.failed(error, reason)

            execution_plugin_name = execution_name_result.data
            execution_plugin_result = self._get_plugin_instance(execution_plugin_name, plugin_instances)
            if execution_plugin_result.status is ResultStatus.FAILED:
                error = execution_plugin_result.error or ConfigurationError(
                    "Execution plugin is not available.",
                    module=self._MODULE_NAME,
                    reason_code=self._INVALID_CONFIG_REASON,
                    details={"plugin": execution_plugin_name},
                )
                reason = execution_plugin_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
                return Result.failed(error, reason)
            execution_plugin = execution_plugin_result.data

            if resolved_run_type == RunType.PRE_CLOSE_CLEANUP:
                execution_input = {
                    "pre_close_recon": True,
                    "reconcile_first": True,
                }
            else:
                execution_input = {
                    "post_close_recon": True,
                    "reconcile_first": True,
                }

            execution_config = dict(self._lookup_plugin_config(config_map, execution_plugin_name))
            if mode is OperatingMode.DRY_RUN:
                nested = execution_config.get("execution")
                if isinstance(nested, Mapping):
                    nested_config = dict(nested)
                    nested_config.setdefault("dry_run", True)
                    execution_config["execution"] = nested_config
                else:
                    execution_config.setdefault("dry_run", True)

            execution_result = self._run_module(
                module_name="execution",
                plugin=execution_plugin,
                input_payload=execution_input,
                context=orchestrator_context,
                emit_event=emit_event,
                module_config=execution_config,
            )
            module_results.append(execution_result)
            update_status_from_module(execution_result)
            self._capture_last_run(run_id, module_results)
            return finalise_run()

        if resolved_run_type == RunType.PRE_MARKET_CHECK:
            execution_name_result = self._coerce_plugin_name(
                config_map.get("execution_plugin"),
                default=self._DEFAULT_EXECUTION_PLUGIN,
                role="execution",
            )
            if execution_name_result.status is ResultStatus.FAILED:
                error = execution_name_result.error or ConfigurationError(
                    "Execution plugin selection failed.",
                    module=self._MODULE_NAME,
                    reason_code=self._INVALID_CONFIG_REASON,
                )
                reason = execution_name_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
                return Result.failed(error, reason)

            execution_plugin_name = execution_name_result.data
            execution_plugin_result = self._get_plugin_instance(execution_plugin_name, plugin_instances)
            if execution_plugin_result.status is ResultStatus.FAILED:
                error = execution_plugin_result.error or ConfigurationError(
                    "Execution plugin is not available.",
                    module=self._MODULE_NAME,
                    reason_code=self._INVALID_CONFIG_REASON,
                    details={"plugin": execution_plugin_name},
                )
                reason = execution_plugin_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
                return Result.failed(error, reason)
            execution_plugin = execution_plugin_result.data

            execution_input = {
                "pre_market_check": True,
                "reconcile_first": True,
                "check_stops": True,
                "scan_overnight_events": True,
                "validate_orders": True,
            }

            execution_config = dict(self._lookup_plugin_config(config_map, execution_plugin_name))
            if mode is OperatingMode.DRY_RUN:
                nested = execution_config.get("execution")
                if isinstance(nested, Mapping):
                    nested_config = dict(nested)
                    nested_config.setdefault("dry_run", True)
                    execution_config["execution"] = nested_config
                else:
                    execution_config.setdefault("dry_run", True)

            execution_result = self._run_module(
                module_name="execution",
                plugin=execution_plugin,
                input_payload=execution_input,
                context=orchestrator_context,
                emit_event=emit_event,
                module_config=execution_config,
            )
            module_results.append(execution_result)
            update_status_from_module(execution_result)
            self._capture_last_run(run_id, module_results)
            return finalise_run()

        if resolved_run_type in (RunType.INTRADAY_CHECK_1030, RunType.INTRADAY_CHECK_1230, RunType.INTRADAY_CHECK_1430):
            execution_name_result = self._coerce_plugin_name(
                config_map.get("execution_plugin"),
                default=self._DEFAULT_EXECUTION_PLUGIN,
                role="execution",
            )
            if execution_name_result.status is ResultStatus.FAILED:
                error = execution_name_result.error or ConfigurationError(
                    "Execution plugin selection failed.",
                    module=self._MODULE_NAME,
                    reason_code=self._INVALID_CONFIG_REASON,
                )
                reason = execution_name_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
                return Result.failed(error, reason)

            execution_plugin_name = execution_name_result.data
            execution_plugin_result = self._get_plugin_instance(execution_plugin_name, plugin_instances)
            if execution_plugin_result.status is ResultStatus.FAILED:
                error = execution_plugin_result.error or ConfigurationError(
                    "Execution plugin is not available.",
                    module=self._MODULE_NAME,
                    reason_code=self._INVALID_CONFIG_REASON,
                    details={"plugin": execution_plugin_name},
                )
                reason = execution_plugin_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
                return Result.failed(error, reason)
            execution_plugin = execution_plugin_result.data

            execution_input: dict[str, Any] = {
                "intraday_check": True,
                "reconcile_first": True,
                "check_stops": True,
                "repair_brackets": True,
                "align_quantities": True,
            }
            if resolved_run_type == RunType.INTRADAY_CHECK_1430:
                execution_input["safe_mode"] = "SAFE_REDUCING"

            execution_config = dict(self._lookup_plugin_config(config_map, execution_plugin_name))
            if mode is OperatingMode.DRY_RUN:
                nested = execution_config.get("execution")
                if isinstance(nested, Mapping):
                    nested_config = dict(nested)
                    nested_config.setdefault("dry_run", True)
                    execution_config["execution"] = nested_config
                else:
                    execution_config.setdefault("dry_run", True)

            execution_result = self._run_module(
                module_name="execution",
                plugin=execution_plugin,
                input_payload=execution_input,
                context=orchestrator_context,
                emit_event=emit_event,
                module_config=execution_config,
            )
            module_results.append(execution_result)
            update_status_from_module(execution_result)
            self._capture_last_run(run_id, module_results)
            return finalise_run()

        universe_name_result = self._coerce_plugin_name(
            config_map.get("universe_plugin"),
            default=self._DEFAULT_UNIVERSE_PLUGIN,
            role="universe",
        )
        if universe_name_result.status is ResultStatus.FAILED:
            error = universe_name_result.error or ConfigurationError(
                "Universe plugin selection failed.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
            )
            reason = universe_name_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
            return Result.failed(error, reason)
        universe_plugin_result = self._get_plugin_instance(universe_name_result.data, plugin_instances)
        if universe_plugin_result.status is ResultStatus.FAILED:
            error = universe_plugin_result.error or ConfigurationError(
                "Universe plugin is not available.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
                details={"plugin": universe_name_result.data},
            )
            reason = universe_plugin_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
            return Result.failed(error, reason)
        universe_plugin = universe_plugin_result.data

        data_name_result = self._coerce_plugin_name(
            config_map.get("data_plugin"),
            default=self._DEFAULT_DATA_PLUGIN,
            role="data",
        )
        if data_name_result.status is ResultStatus.FAILED:
            error = data_name_result.error or ConfigurationError(
                "Data plugin selection failed.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
            )
            reason = data_name_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
            return Result.failed(error, reason)
        data_plugin_result = self._get_plugin_instance(data_name_result.data, plugin_instances)
        if data_plugin_result.status is ResultStatus.FAILED:
            error = data_plugin_result.error or ConfigurationError(
                "Data plugin is not available.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
                details={"plugin": data_name_result.data},
            )
            reason = data_plugin_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
            return Result.failed(error, reason)
        data_plugin = data_plugin_result.data

        scanner_name_result = self._coerce_plugin_name(
            config_map.get("scanner_plugin"),
            default=self._DEFAULT_SCANNER_PLUGIN,
            role="scanner",
        )
        if scanner_name_result.status is ResultStatus.FAILED:
            error = scanner_name_result.error or ConfigurationError(
                "Scanner plugin selection failed.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
            )
            reason = scanner_name_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
            return Result.failed(error, reason)
        scanner_plugin_result = self._get_plugin_instance(scanner_name_result.data, plugin_instances)
        if scanner_plugin_result.status is ResultStatus.FAILED:
            error = scanner_plugin_result.error or ConfigurationError(
                "Scanner plugin is not available.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
                details={"plugin": scanner_name_result.data},
            )
            reason = scanner_plugin_result.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
            return Result.failed(error, reason)
        scanner_plugin = scanner_plugin_result.data

        def has_plugin_entry(plugin_name: str) -> bool:
            plugins_section = config_map.get("plugins")
            if not isinstance(plugins_section, Mapping):
                return False
            for category_entries in plugins_section.values():
                if not isinstance(category_entries, list):
                    continue
                for entry in category_entries:
                    if not isinstance(entry, Mapping):
                        continue
                    if entry.get("name") == plugin_name:
                        return True
            return False

        intraday_run = resolved_run_type in (RunType.INTRADAY_CHECK_1030, RunType.INTRADAY_CHECK_1230, RunType.INTRADAY_CHECK_1430)
        # Check if holiday behavior should skip modules
        skip_event_guard = False
        skip_order_modules = False
        if is_holiday and isinstance(holiday_behavior_config, Mapping):
            skip_event_guard = not holiday_behavior_config.get("event_guard", False)
            skip_strategy = not holiday_behavior_config.get("strategy", False)
            skip_risk_gate = not holiday_behavior_config.get("risk_gate", False)
            skip_execution = not holiday_behavior_config.get("execution", False)
            skip_order_modules = skip_strategy and skip_risk_gate and skip_execution

        # Event guard is independent from order modules
        event_guard_requested = (intraday_run or any(
            key in config_map for key in ("event_guard_plugin",)
        ) or has_plugin_entry(self._DEFAULT_EVENT_GUARD_PLUGIN)) and not skip_event_guard

        # Order modules (strategy/risk_gate/execution) depend on skip_order_modules
        order_modules_requested = (intraday_run or any(
            key in config_map
            for key in (
                "strategy_plugin",
                "risk_gate_plugin",
                "execution_plugin",
            )
        ) or any(
            has_plugin_entry(plugin_name)
            for plugin_name in (
                self._DEFAULT_STRATEGY_PLUGIN,
                self._DEFAULT_RISK_GATE_PLUGIN,
                self._DEFAULT_EXECUTION_PLUGIN,
            )
        )) and not skip_order_modules

        full_pipeline_requested = event_guard_requested or order_modules_requested

        event_guard_plugin_name = self._DEFAULT_EVENT_GUARD_PLUGIN
        strategy_plugin_name = self._DEFAULT_STRATEGY_PLUGIN
        risk_gate_plugin_name = self._DEFAULT_RISK_GATE_PLUGIN
        execution_plugin_name = self._DEFAULT_EXECUTION_PLUGIN

        event_guard_plugin: Any | None = None
        strategy_plugin: Any | None = None
        risk_gate_plugin: Any | None = None
        execution_plugin: Any | None = None

        optional_plugin_failures: dict[str, RecoverableError] = {}

        if full_pipeline_requested:
            def resolve_optional_plugin(*, config_key: str, default: str, role: str) -> Result[tuple[str, Any | None]]:
                name_result = self._coerce_plugin_name(
                    config_map.get(config_key),
                    default=default,
                    role=role,
                )
                if name_result.status is ResultStatus.FAILED:
                    return Result.failed(
                        name_result.error
                        or ConfigurationError(
                            "Invalid plugin name.",
                            module=self._MODULE_NAME,
                            reason_code=self._INVALID_CONFIG_REASON,
                        ),
                        name_result.reason_code or self._INVALID_CONFIG_REASON,
                    )

                plugin_name = name_result.data
                plugin_result = self._get_plugin_instance(plugin_name, plugin_instances)
                if plugin_result.status is ResultStatus.FAILED:
                    reason = plugin_result.reason_code or self._RUN_DEGRADED_REASON
                    optional_plugin_failures[role] = self._ensure_recoverable(plugin_result.error, reason)
                    return Result.success((plugin_name, None))

                return Result.success((plugin_name, plugin_result.data))

            if event_guard_requested:
                event_guard_resolved = resolve_optional_plugin(
                    config_key="event_guard_plugin",
                    default=self._DEFAULT_EVENT_GUARD_PLUGIN,
                    role="event_guard",
                )
                if event_guard_resolved.status is ResultStatus.FAILED:
                    error = event_guard_resolved.error or ConfigurationError(
                        "Event Guard plugin selection failed.",
                        module=self._MODULE_NAME,
                        reason_code=self._INVALID_CONFIG_REASON,
                    )
                    reason = (
                        event_guard_resolved.reason_code
                        or getattr(error, "reason_code", None)
                        or self._INVALID_CONFIG_REASON
                    )
                    return Result.failed(error, reason)

                event_guard_plugin_name, event_guard_plugin = event_guard_resolved.data

            if order_modules_requested:
                strategy_resolved = resolve_optional_plugin(
                    config_key="strategy_plugin",
                    default=self._DEFAULT_STRATEGY_PLUGIN,
                    role="strategy",
                )
                if strategy_resolved.status is ResultStatus.FAILED:
                    error = strategy_resolved.error or ConfigurationError(
                        "Strategy plugin selection failed.",
                        module=self._MODULE_NAME,
                        reason_code=self._INVALID_CONFIG_REASON,
                    )
                    reason = strategy_resolved.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
                    return Result.failed(error, reason)

                risk_gate_resolved = resolve_optional_plugin(
                    config_key="risk_gate_plugin",
                    default=self._DEFAULT_RISK_GATE_PLUGIN,
                    role="risk_gate",
                )
                if risk_gate_resolved.status is ResultStatus.FAILED:
                    error = risk_gate_resolved.error or ConfigurationError(
                        "Risk Gate plugin selection failed.",
                        module=self._MODULE_NAME,
                        reason_code=self._INVALID_CONFIG_REASON,
                    )
                    reason = risk_gate_resolved.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
                    return Result.failed(error, reason)

                execution_resolved = resolve_optional_plugin(
                    config_key="execution_plugin",
                    default=self._DEFAULT_EXECUTION_PLUGIN,
                    role="execution",
                )
                if execution_resolved.status is ResultStatus.FAILED:
                    error = execution_resolved.error or ConfigurationError(
                        "Execution plugin selection failed.",
                        module=self._MODULE_NAME,
                        reason_code=self._INVALID_CONFIG_REASON,
                    )
                    reason = execution_resolved.reason_code or getattr(error, "reason_code", None) or self._INVALID_CONFIG_REASON
                    return Result.failed(error, reason)

                strategy_plugin_name, strategy_plugin = strategy_resolved.data
                risk_gate_plugin_name, risk_gate_plugin = risk_gate_resolved.data
                execution_plugin_name, execution_plugin = execution_resolved.data

        universe_result = self._run_module(
            module_name="universe",
            plugin=universe_plugin,
            input_payload={},
            context=orchestrator_context,
            emit_event=emit_event,
            module_config=self._lookup_plugin_config(config_map, universe_name_result.data),
        )
        module_results.append(universe_result)
        update_status_from_module(universe_result)
        if universe_result.status is ResultStatus.FAILED:
            emit_event("run.abort", "orchestrator", {"stage": "universe"})
            apply_event_status()
            reason = getattr(universe_result.error, "reason_code", None) or self._RUN_FAILED_REASON
            self._capture_last_run(run_id, module_results)
            return self._finalise_failure(
                metadata=run_metadata,
                events=events,
                module_results=module_results,
                failure_error=failure_error,
                reason=reason,
            )

        data_input = universe_result.output_data if isinstance(universe_result.output_data, Mapping) else {}

        # Inject market regime symbols (SPY/QQQ) into data request if regime detection is enabled
        data_input = self._inject_regime_symbols(data_input, config_map)

        data_result = self._run_module(
            module_name="data",
            plugin=data_plugin,
            input_payload=data_input,
            context=orchestrator_context,
            emit_event=emit_event,
            module_config=self._lookup_plugin_config(config_map, data_name_result.data),
        )
        module_results.append(data_result)
        update_status_from_module(data_result)
        if data_result.status is ResultStatus.FAILED:
            emit_event("run.abort", "orchestrator", {"stage": "data"})
            apply_event_status()
            reason = getattr(data_result.error, "reason_code", None) or self._RUN_FAILED_REASON
            self._capture_last_run(run_id, module_results)
            return self._finalise_failure(
                metadata=run_metadata,
                events=events,
                module_results=module_results,
                failure_error=failure_error,
                reason=reason,
            )

        data_output = data_result.output_data if isinstance(data_result.output_data, Mapping) else {}

        current_regime: str | None = None
        regime_detection = self._apply_regime_detection(
            data_output=data_output,
            config_map=config_map,
            logger=logger,
        )
        if regime_detection is not None:
            regime_result_payload, regime_config_obj = regime_detection
            self._pending_regime_output = regime_result_payload
            detected_regime_value = regime_result_payload.get("detected_regime") or regime_result_payload.get("regime_name")
            if isinstance(detected_regime_value, str) and detected_regime_value.strip():
                current_regime = detected_regime_value.strip().lower()
        else:
            regime_result_payload = None
            regime_config_obj = None
            if hasattr(scanner_plugin, "regime_breakthrough_config"):
                try:
                    setattr(scanner_plugin, "regime_breakthrough_config", None)
                except Exception:
                    pass

        scanner_module_config = self._lookup_plugin_config(config_map, scanner_name_result.data)
        if regime_config_obj is not None and regime_result_payload is not None:
            scanner_module_config = self._apply_regime_overrides(
                base_config=scanner_module_config,
                regime_config=regime_config_obj,
                scanner_plugin=scanner_plugin,
                logger=logger,
                regime_result=regime_result_payload,
            )

        try:
            setattr(scanner_plugin, "current_regime", current_regime)
        except Exception:
            logger.warning(
                "scanner_regime_assignment_failed",
                regime=current_regime,
            )

        scanner_input = data_output
        scanner_result = self._run_module(
            module_name="scanner",
            plugin=scanner_plugin,
            input_payload=scanner_input,
            context=orchestrator_context,
            emit_event=emit_event,
            module_config=scanner_module_config,
        )
        module_results.append(scanner_result)
        update_status_from_module(scanner_result)
        if scanner_result.status is ResultStatus.FAILED:
            emit_event("run.abort", "orchestrator", {"stage": "scanner"})
            apply_event_status()
            reason = getattr(scanner_result.error, "reason_code", None) or self._RUN_FAILED_REASON
            self._capture_last_run(run_id, module_results)
            return self._finalise_failure(
                metadata=run_metadata,
                events=events,
                module_results=module_results,
                failure_error=failure_error,
                reason=reason,
            )

        if full_pipeline_requested:
            data_payload = data_result.output_data if isinstance(data_result.output_data, Mapping) else {}
            market_data_raw = data_payload.get("series_by_symbol") if isinstance(data_payload, Mapping) else None
            market_data_payload = dict(market_data_raw) if isinstance(market_data_raw, Mapping) else {}
            market_data_override = config_map.get("market_data") or config_map.get("series_by_symbol")
            if isinstance(market_data_override, Mapping) and market_data_override:
                market_data_payload = {**market_data_payload, **dict(market_data_override)}

            def optional_plugin_error(role: str) -> RecoverableError | None:
                return optional_plugin_failures.get(role)

            scanner_payload = scanner_result.output_data if isinstance(scanner_result.output_data, Mapping) else {}

            event_guard_payload: Mapping[str, Any] = scanner_payload
            if event_guard_requested:
                if optional_plugin_error("event_guard") is not None or event_guard_plugin is None:
                    append_optional_skip(
                        "event_guard",
                        optional_plugin_error("event_guard") or self._ensure_recoverable(None, self._RUN_DEGRADED_REASON),
                    )
                else:
                    event_guard_config = self._lookup_plugin_config(config_map, event_guard_plugin_name)
                    eg_input: Mapping[str, Any] = scanner_payload
                    if self.scan_timestamp_ns is not None:
                        eg_input = {**scanner_payload, "scan_timestamp_ns": self.scan_timestamp_ns}
                    event_guard_result = self._run_module(
                        module_name="event_guard",
                        plugin=event_guard_plugin,
                        input_payload=eg_input,
                        context=orchestrator_context,
                        emit_event=emit_event,
                        module_config=event_guard_config,
                    )
                    if event_guard_result.status is ResultStatus.FAILED:
                        append_optional_skip(
                            "event_guard",
                            self._ensure_recoverable(
                                event_guard_result.error,
                                getattr(event_guard_result.error, "reason_code", None) or self._RUN_DEGRADED_REASON,
                            ),
                        )
                    else:
                        module_results.append(event_guard_result)
                        update_status_from_module(event_guard_result)
                        event_guard_payload = (
                            event_guard_result.output_data if isinstance(event_guard_result.output_data, Mapping) else {}
                        )

            if order_modules_requested:
                if optional_plugin_error("strategy") is not None or strategy_plugin is None:
                    append_optional_skip(
                        "strategy",
                        optional_plugin_error("strategy") or self._ensure_recoverable(None, self._RUN_DEGRADED_REASON),
                    )
                else:
                    strategy_input = dict(event_guard_payload)
                    # V27.4: Apply candidate-level pattern filter before strategy dedup
                    if self.candidate_filter_fn is not None:
                        strategy_input = self.candidate_filter_fn(strategy_input)
                    if market_data_payload:
                        strategy_input.setdefault("market_data", dict(market_data_payload))
                    if self.scan_timestamp_ns is not None:
                        strategy_input["scan_timestamp_ns"] = self.scan_timestamp_ns

                    strategy_config = self._lookup_plugin_config(config_map, strategy_plugin_name)
                    # Apply regime strategy overrides if available
                    # NOTE: Regime overrides are merged at both root level AND "engine" subkey (if exists).
                    # - Root level: for config/config.yaml which has params directly under strategy:
                    # - Engine subkey: for configs that use strategy.engine: structure
                    # The Strategy plugin's validate_config() handles both via:
                    #   engine_map = _coerce_mapping(merged.get("engine")) or merged
                    if regime_config_obj is not None:
                        strategy_overrides = dict(regime_config_obj.strategy)
                        if strategy_overrides:
                            strategy_config = dict(strategy_config) if isinstance(strategy_config, Mapping) else {}
                            # Merge at root level for flat config structure
                            strategy_config.update(strategy_overrides)
                            # Also merge into engine subkey if it exists (for nested config structure)
                            if "engine" in strategy_config and isinstance(strategy_config["engine"], Mapping):
                                strategy_config["engine"] = {**dict(strategy_config["engine"]), **strategy_overrides}
                            logger.info(
                                "market_regime_strategy_overrides_applied",
                                regime=regime_config_obj.regime_name,
                                overrides=sorted(strategy_overrides.keys()),
                            )
                    strategy_result = self._run_module(
                        module_name="strategy",
                        plugin=strategy_plugin,
                        input_payload=strategy_input,
                        context=orchestrator_context,
                        emit_event=emit_event,
                        module_config=strategy_config,
                    )
                    if strategy_result.status is ResultStatus.FAILED:
                        append_optional_skip(
                            "strategy",
                            self._ensure_recoverable(
                                strategy_result.error,
                                getattr(strategy_result.error, "reason_code", None) or self._RUN_DEGRADED_REASON,
                            ),
                        )
                    else:
                        module_results.append(strategy_result)
                        update_status_from_module(strategy_result)

                        if optional_plugin_error("risk_gate") is not None or risk_gate_plugin is None:
                            append_optional_skip(
                                "risk_gate",
                                optional_plugin_error("risk_gate") or self._ensure_recoverable(None, self._RUN_DEGRADED_REASON),
                            )
                        else:
                            strategy_payload = (
                                strategy_result.output_data if isinstance(strategy_result.output_data, Mapping) else {}
                            )
                            risk_gate_input = dict(strategy_payload)
                            if market_data_payload:
                                risk_gate_input.setdefault("market_data", dict(market_data_payload))

                            risk_gate_config: Mapping[str, Any] = self._lookup_plugin_config(config_map, risk_gate_plugin_name)
                            if resolved_run_type == RunType.INTRADAY_CHECK_1430:
                                risk_gate_config_map = dict(risk_gate_config)
                                nested = risk_gate_config_map.get("risk_gate")
                                nested_map: dict[str, Any]
                                if isinstance(nested, Mapping):
                                    nested_map = dict(nested)
                                else:
                                    nested_map = {}
                                nested_map["safe_mode"] = "SAFE_REDUCING"
                                nested_map["safe_mode_state"] = "SAFE_REDUCING"
                                risk_gate_config_map["risk_gate"] = nested_map
                                risk_gate_config = risk_gate_config_map

                            # Apply regime-aware exposure limits if available.
                            if regime_config_obj is not None and regime_result_payload is not None:
                                risk_gate_overrides = getattr(regime_config_obj, "risk_gate", None)
                                override_value: Any | None = None
                                if isinstance(risk_gate_overrides, Mapping):
                                    override_value = risk_gate_overrides.get("max_total_exposure")
                                    if override_value is None:
                                        nested_override = risk_gate_overrides.get("portfolio")
                                        if isinstance(nested_override, Mapping):
                                            override_value = nested_override.get("max_total_exposure")

                                max_total_exposure: float | None = None
                                if isinstance(override_value, (int, float)):
                                    max_total_exposure = float(override_value)
                                elif isinstance(override_value, str) and override_value.strip():
                                    try:
                                        max_total_exposure = float(override_value)
                                    except ValueError:
                                        max_total_exposure = None

                                if max_total_exposure is not None:
                                    risk_gate_config_map = dict(risk_gate_config)
                                    portfolio_cfg = risk_gate_config_map.get("portfolio")
                                    portfolio_map = dict(portfolio_cfg) if isinstance(portfolio_cfg, Mapping) else {}
                                    portfolio_map["max_leverage"] = max_total_exposure
                                    risk_gate_config_map["portfolio"] = portfolio_map
                                    risk_gate_config = risk_gate_config_map

                                    applied: dict[str, Any] = regime_result_payload.setdefault("applied_overrides", {})
                                    applied["risk_gate"] = {"max_total_exposure": max_total_exposure}
                                    logger.info(
                                        "market_regime_risk_gate_overrides_applied",
                                        regime=regime_config_obj.regime_name,
                                        overrides=["max_total_exposure"],
                                    )

                            risk_gate_result = self._run_module(
                                module_name="risk_gate",
                                plugin=risk_gate_plugin,
                                input_payload=risk_gate_input,
                                context=orchestrator_context,
                                emit_event=emit_event,
                                module_config=risk_gate_config,
                            )
                            if risk_gate_result.status is ResultStatus.FAILED:
                                append_optional_skip(
                                    "risk_gate",
                                    self._ensure_recoverable(
                                        risk_gate_result.error,
                                        getattr(risk_gate_result.error, "reason_code", None) or self._RUN_DEGRADED_REASON,
                                    ),
                                )
                            else:
                                module_results.append(risk_gate_result)
                                update_status_from_module(risk_gate_result)

                                if optional_plugin_error("execution") is not None or execution_plugin is None:
                                    append_optional_skip(
                                        "execution",
                                        optional_plugin_error("execution")
                                        or self._ensure_recoverable(None, self._RUN_DEGRADED_REASON),
                                    )
                                else:
                                    risk_gate_payload = (
                                        risk_gate_result.output_data
                                        if isinstance(risk_gate_result.output_data, Mapping)
                                        else {}
                                    )
                                    execution_input = dict(risk_gate_payload)
                                    if intraday_run:
                                        execution_input["reconcile_first"] = True

                                    execution_config = dict(self._lookup_plugin_config(config_map, execution_plugin_name))
                                    if mode is OperatingMode.DRY_RUN:
                                        nested = execution_config.get("execution")
                                        if isinstance(nested, Mapping):
                                            nested_config = dict(nested)
                                            nested_config.setdefault("dry_run", True)
                                            execution_config["execution"] = nested_config
                                        else:
                                            execution_config.setdefault("dry_run", True)

                                    execution_result = self._run_module(
                                        module_name="execution",
                                        plugin=execution_plugin,
                                        input_payload=execution_input,
                                        context=orchestrator_context,
                                        emit_event=emit_event,
                                        module_config=execution_config,
                                    )
                                    if execution_result.status is ResultStatus.FAILED:
                                        append_optional_skip(
                                            "execution",
                                            self._ensure_recoverable(
                                                execution_result.error,
                                                getattr(execution_result.error, "reason_code", None) or self._RUN_DEGRADED_REASON,
                                            ),
                                        )
                                    else:
                                        module_results.append(execution_result)
                                        update_status_from_module(execution_result)

            self._capture_last_run(run_id, module_results)
            return finalise_run()

        # Fallback: if full_pipeline_requested is False (all optional modules skipped),
        # still need to finalize the run properly instead of returning None.
        self._capture_last_run(run_id, module_results)
        return finalise_run()

    def _apply_regime_detection(
        self,
        *,
        data_output: Mapping[str, Any],
        config_map: Mapping[str, Any],
        logger: Any,
    ) -> tuple[dict[str, Any], RegimeConfig | None] | None:
        regime_section = config_map.get("market_regime")
        if not isinstance(regime_section, Mapping):
            return None

        enabled = bool(regime_section.get("enabled", True))
        mode_raw = regime_section.get("mode", "none")
        mode = str(mode_raw).strip().lower()
        if not enabled or mode in ("", "none"):
            return None

        regime_result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "mode": mode,
            "enabled": enabled,
        }

        loader_kwargs: dict[str, Any] = {}
        base_dir_raw = regime_section.get("config_dir")
        if isinstance(base_dir_raw, str) and base_dir_raw.strip():
            loader_kwargs["base_dir"] = Path(base_dir_raw).expanduser()
        base_filename_raw = regime_section.get("base_filename")
        if isinstance(base_filename_raw, str) and base_filename_raw.strip():
            loader_kwargs["base_filename"] = base_filename_raw.strip()
        loader = RegimeConfigLoader(**loader_kwargs)

        fallback_raw = regime_section.get("fallback_regime", "base")
        fallback_regime = str(fallback_raw).strip() or "base"

        regime_choice: MarketRegime | str | None = None
        detection_payload: RegimeDetectionResult | None = None

        if mode == "auto":
            symbols_config = regime_section.get("symbols")
            if isinstance(symbols_config, (list, tuple)):
                symbols = [str(sym).strip().upper() for sym in symbols_config if str(sym).strip()]
            else:
                # Include both market index symbols and VIX proxy symbols for detection
                symbols = ["SPY", "QQQ", "VIXY", "VXX", "UVXY"]
            market_series = self._extract_market_regime_series(data_output, symbols, logger)
            if not market_series:
                logger.warning("market_regime_data_missing", symbols=symbols, status="data_missing")
                regime_result["status"] = "data_missing"
            else:
                detector = self._build_regime_detector(regime_section)
                detect_result = detector.detect(market_series)
                if detect_result.status is ResultStatus.FAILED:
                    logger.warning(
                        "market_regime_detection_failed",
                        symbols=list(market_series.keys()),
                        reason=detect_result.reason_code,
                        error=str(detect_result.error)[:200] if detect_result.error else None,
                    )
                    regime_result["status"] = "failed"
                    regime_result["error"] = str(detect_result.error)[:200] if detect_result.error else None
                    regime_result["reason_code"] = detect_result.reason_code
                else:
                    detection_payload = detect_result.data
                    regime_choice = detection_payload.regime
                    regime_result["status"] = "detected"
                    regime_result["detected_regime"] = detection_payload.regime.value
                    regime_result["detection"] = msgspec.to_builtins(detection_payload)
                    # Log successful regime detection
                    logger.info(
                        "market_regime_detected",
                        regime=detection_payload.regime.value,
                        adx=detection_payload.adx,
                        ma50_slope=detection_payload.ma50_slope,
                        is_trending=detection_payload.is_trending,
                        is_stable=detection_payload.is_stable,
                        vix_proxy_symbol=detection_payload.vix_proxy_symbol,
                        vix_spike_detected=detection_payload.vix_spike_detected,
                    )
        else:
            regime_choice = self._map_regime_mode(mode)
            if regime_choice is None:
                logger.warning("market_regime_invalid_mode", mode=mode)
                regime_result["status"] = "invalid_mode"
                return regime_result, None
            regime_result["status"] = "forced"
            regime_result["detected_regime"] = regime_choice.value

        load_target: MarketRegime | str = regime_choice or fallback_regime
        config_result = loader.load(load_target)
        if config_result.status is ResultStatus.FAILED or config_result.data is None:
            logger.warning(
                "market_regime_config_load_failed",
                mode=mode,
                regime=str(load_target),
                reason=config_result.reason_code,
                error=str(config_result.error)[:200] if config_result.error else None,
            )
            regime_result["status"] = "config_failed"
            regime_result["config_error"] = config_result.reason_code
            return regime_result, None

        regime_config = config_result.data
        regime_result["regime_name"] = regime_config.regime_name
        regime_result["config_enabled"] = regime_config.enabled
        regime_result["config"] = regime_config.to_mapping()
        regime_result.setdefault("detected_regime", regime_config.regime_name)

        logger.info(
            "market_regime_configuration_selected",
            regime=regime_config.regime_name,
            mode=mode,
            enabled=regime_config.enabled,
        )
        if not regime_config.enabled:
            logger.warning("market_regime_disabled_trading", regime=regime_config.regime_name, enabled=False)

        return regime_result, regime_config

    def _apply_regime_overrides(
        self,
        *,
        base_config: Mapping[str, Any],
        regime_config: RegimeConfig,
        scanner_plugin: Any | None,
        logger: Any,
        regime_result: dict[str, Any],
    ) -> Mapping[str, Any]:
        merged_config = dict(base_config) if isinstance(base_config, Mapping) else {}
        overrides = dict(regime_config.scanner)
        applied: dict[str, Any] = regime_result.setdefault("applied_overrides", {})

        if overrides:
            # Use deep merge to correctly merge nested config sections like adx_entry_filter
            merged_config = _deep_merge(merged_config, overrides)
            logger.info(
                "market_regime_scanner_overrides_applied",
                regime=regime_config.regime_name,
                overrides=sorted(overrides.keys()),
            )
            applied["scanner"] = dict(overrides)

        breakthrough = dict(regime_config.breakthrough)
        if scanner_plugin is not None:
            try:
                setattr(scanner_plugin, "regime_breakthrough_config", breakthrough or None)
                if breakthrough:
                    applied["breakthrough"] = dict(breakthrough)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "market_regime_breakthrough_override_failed",
                    regime=regime_config.regime_name,
                    error=str(exc)[:200],
                )

        strategy_overrides = dict(regime_config.strategy)
        if strategy_overrides:
            applied["strategy"] = dict(strategy_overrides)
            logger.info(
                "market_regime_strategy_overrides_prepared",
                regime=regime_config.regime_name,
                overrides=sorted(strategy_overrides.keys()),
            )

        return merged_config

    def _inject_regime_symbols(
        self,
        data_input: Mapping[str, Any],
        config_map: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Inject market regime symbols (SPY/QQQ/VIXY) into data request if regime detection is enabled.

        This ensures that market index data and VIX proxy data are available for regime
        detection even when the universe doesn't include these symbols.
        """
        regime_section = config_map.get("market_regime")
        if not isinstance(regime_section, Mapping):
            return dict(data_input) if isinstance(data_input, Mapping) else {}

        enabled = bool(regime_section.get("enabled", True))
        mode = str(regime_section.get("mode", "none")).strip().lower()
        if not enabled or mode in ("", "none"):
            return dict(data_input) if isinstance(data_input, Mapping) else {}

        # Get regime symbols from config or use defaults (SPY/QQQ for market index, VIXY for VIX proxy)
        symbols_config = regime_section.get("symbols")
        if isinstance(symbols_config, (list, tuple)):
            regime_symbols = [str(sym).strip().upper() for sym in symbols_config if str(sym).strip()]
        else:
            # Include sector ETF symbols (XLK/SOXX) for trend pattern detection gates
            regime_symbols = ["SPY", "QQQ", "VIXY", "XLK", "SOXX"]

        # Convert to mutable dict
        result = dict(data_input) if isinstance(data_input, Mapping) else {}

        # Get existing equities list
        equities_raw = result.get("equities", [])
        if not isinstance(equities_raw, list):
            equities_raw = []

        # Find existing symbols
        existing_symbols = set()
        for eq in equities_raw:
            if isinstance(eq, Mapping):
                sym = str(eq.get("symbol", "")).strip().upper()
            elif hasattr(eq, "symbol"):
                sym = str(getattr(eq, "symbol", "")).strip().upper()
            else:
                continue
            if sym:
                existing_symbols.add(sym)

        # Add missing regime symbols as stub EquityInfo entries
        # Exchange mapping for common regime symbols
        exchange_map = {
            "SPY": "NYSE",  # SPDR S&P 500 ETF - NYSE Arca
            "QQQ": "NASDAQ",  # Invesco QQQ Trust - NASDAQ
            "VIXY": "NYSE",  # ProShares VIX Short-Term Futures ETF - NYSE Arca
            "VXX": "NYSE",  # iPath Series B S&P 500 VIX - NYSE Arca
            "UVXY": "NYSE",  # ProShares Ultra VIX Short-Term Futures ETF - NYSE Arca
            "XLK": "NYSE",  # Technology Select Sector SPDR Fund - NYSE Arca
            "SOXX": "NASDAQ",  # iShares Semiconductor ETF - NASDAQ
        }
        new_equities = list(equities_raw)
        for sym in regime_symbols:
            if sym and sym not in existing_symbols:
                stub = EquityInfo(
                    symbol=sym,
                    exchange=exchange_map.get(sym, "NYSE"),
                    price=0.0,  # Will be fetched by data module
                    avg_dollar_volume_20d=0.0,
                    market_cap=0.0,
                    is_otc=False,
                    is_halted=False,
                    sector="ETF",
                )
                new_equities.append(msgspec.to_builtins(stub))
                existing_symbols.add(sym)

        result["equities"] = new_equities
        return result

    @staticmethod
    def _map_regime_mode(mode: str) -> MarketRegime | None:
        mapping = {
            "bull": MarketRegime.BULL,
            "bear": MarketRegime.BEAR,
            "choppy": MarketRegime.CHOPPY,
            "unknown": MarketRegime.UNKNOWN,
        }
        return mapping.get(mode)

    def _extract_market_regime_series(
        self,
        data_output: Mapping[str, Any],
        symbols: list[str],
        logger: Any | None = None,
    ) -> dict[str, PriceSeriesSnapshot]:
        """Extract PriceSeriesSnapshot for specified market regime symbols."""

        series_raw = data_output.get("series_by_symbol")
        if not isinstance(series_raw, Mapping):
            return {}

        symbol_set = {sym.upper() for sym in symbols if isinstance(sym, str) and sym.strip()}

        snapshots: dict[str, PriceSeriesSnapshot] = {}
        for key, raw in series_raw.items():
            symbol = str(key).strip().upper()
            if not symbol:
                continue
            if symbol_set and symbol not in symbol_set:
                continue
            snapshot = self._coerce_price_snapshot(raw)
            if snapshot is None:
                continue
            snapshots[symbol] = snapshot

        if not snapshots:
            for key, raw in series_raw.items():
                symbol = str(key).strip().upper()
                if not symbol:
                    continue
                snapshot = self._coerce_price_snapshot(raw)
                if snapshot is None:
                    continue
                snapshots[symbol] = snapshot
                break

        return snapshots

    @staticmethod
    def _build_regime_detector(regime_section: Mapping[str, Any]) -> MarketRegimeDetector:
        detector_cfg = regime_section.get("detector")
        if not isinstance(detector_cfg, Mapping):
            detector_cfg = regime_section

        kwargs: dict[str, Any] = {}
        allowed = {
            "index_symbols",
            "adx_period",
            "stable_volatility_threshold",
            "adx_trend_threshold",
            "adx_choppy_threshold",
            "ma_slope_days",
            "ma_slope_bull_threshold",
            "ma_slope_bear_threshold",
            "vix_enabled",
            "vix_extreme_threshold",
            "vix_elevated_threshold",
            "vix_proxy_symbols",
            "vix_spike_threshold",
            "vix_spike_forces_choppy",
        }
        for key, value in detector_cfg.items():
            if key not in allowed:
                continue
            if key in {"index_symbols", "vix_proxy_symbols"} and isinstance(value, (list, tuple)):
                symbols = [str(sym).strip().upper() for sym in value if str(sym).strip()]
                if symbols:
                    kwargs[key] = tuple(symbols)
                continue
            kwargs[key] = value
        confirmation_days = detector_cfg.get("confirmation_days", None)
        reset_on_opposite = detector_cfg.get("reset_on_opposite", None)
        if confirmation_days is not None or reset_on_opposite is not None:
            try:
                from scanner.market_regime.interface import RegimeConfirmationConfig
                from scanner.market_regime.regime_tracker import RegimeTransitionTracker

                confirmation_cfg = RegimeConfirmationConfig(
                    confirmation_days=int(confirmation_days) if confirmation_days is not None else 3,
                    reset_on_opposite=bool(reset_on_opposite) if reset_on_opposite is not None else True,
                )
                if int(confirmation_cfg.confirmation_days) > 1:
                    kwargs["tracker"] = RegimeTransitionTracker(config=confirmation_cfg)
            except Exception:  # noqa: BLE001
                pass
        try:
            return MarketRegimeDetector(**kwargs)
        except Exception:  # noqa: BLE001
            return MarketRegimeDetector()

    @staticmethod
    def _coerce_price_snapshot(raw: Any) -> PriceSeriesSnapshot | None:
        if isinstance(raw, PriceSeriesSnapshot):
            return raw
        if isinstance(raw, Mapping):
            try:
                return msgspec.convert(dict(raw), type=PriceSeriesSnapshot)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _parse_run_type(self, value: Any) -> Result[RunType]:
        if isinstance(value, RunType):
            return Result.success(value)
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                try:
                    return Result.success(RunType(candidate))
                except ValueError:
                    try:
                        return Result.success(RunType(candidate.upper()))
                    except ValueError:
                        pass
        if value is None:
            return Result.success(RunType.PRE_MARKET_FULL_SCAN)
        error = ConfigurationError(
            "run_type must be a RunType or string literal.",
            module=self._MODULE_NAME,
            reason_code=self._INVALID_CONFIG_REASON,
            details={"run_type": value},
        )
        return Result.failed(error, error.reason_code)

    def _parse_mode(self, value: Any) -> Result[OperatingMode]:
        if isinstance(value, OperatingMode):
            return Result.success(value)
        if isinstance(value, str):
            candidate = value.strip().upper()
            try:
                return Result.success(OperatingMode[candidate])
            except KeyError:
                try:
                    return Result.success(OperatingMode(candidate))
                except ValueError:
                    pass
        error = ConfigurationError(
            "mode must be an OperatingMode or string literal.",
            module=self._MODULE_NAME,
            reason_code=self._INVALID_CONFIG_REASON,
            details={"mode": value},
        )
        return Result.failed(error, error.reason_code)

    def _coerce_plugin_name(self, value: Any, *, default: str, role: str) -> Result[str]:
        if value is None:
            return Result.success(default)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return Result.success(stripped)
        error = ConfigurationError(
            f"{role.title()} plugin name must be a non-empty string.",
            module=self._MODULE_NAME,
            reason_code=self._INVALID_CONFIG_REASON,
            details={"role": role, "value": value},
        )
        return Result.failed(error, error.reason_code)

    def _get_plugin_instance(
        self,
        plugin_name: str,
        plugin_instances: Mapping[str, Any],
    ) -> Result[Any]:
        plugin = plugin_instances.get(plugin_name)
        if plugin is not None:
            return Result.success(plugin)
        try:
            plugin_class = self._plugin_registry.get_plugin(plugin_name)
        except ConfigurationError as exc:
            stub_name = plugin_name.replace("_real", "_stub") if plugin_name.endswith("_real") else None
            if stub_name:
                stub_plugin = plugin_instances.get(stub_name)
                if stub_plugin is not None:
                    return Result.success(stub_plugin)
                try:
                    plugin_class = self._plugin_registry.get_plugin(stub_name)
                except ConfigurationError as stub_exc:
                    return Result.failed(stub_exc, stub_exc.reason_code)
                plugin_name = stub_name
            else:
                return Result.failed(exc, exc.reason_code)

        try:
            try:
                instance = plugin_class(config={})  # type: ignore[arg-type]
            except TypeError:
                instance = plugin_class()  # type: ignore[call-arg]
        except PermissionError as exc:
            error = ConfigurationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
                details={"plugin": plugin_name},
            )
            return Result.failed(error, error.reason_code)
        except Exception as exc:  # noqa: BLE001 - convert to configuration failure.
            error = ConfigurationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._INVALID_CONFIG_REASON,
                details={"plugin": plugin_name},
            )
            return Result.failed(error, error.reason_code)

        return Result.success(instance)

    def _run_module(
        self,
        *,
        module_name: str,
        plugin: Any,
        input_payload: Mapping[str, Any],
        context: OrchestratorContext,
        emit_event: Callable[[str, str, Mapping[str, Any] | None], None],
        module_config: Mapping[str, Any] | None = None,
    ) -> ModuleResult:
        module_logger = context.logger.bind(module=module_name)
        config_payload: Mapping[str, Any]
        if module_config is not None and isinstance(module_config, Mapping):
            config_payload = dict(module_config)
        else:
            raw_config = getattr(plugin, "config", {})
            config_payload = dict(raw_config) if isinstance(raw_config, Mapping) else {}
        plugin_context = {
            "module_name": module_name,
            "emit_event": emit_event,
            "run_context": context,
            "config": config_payload,
            "market_calendar": context.config_map.get("market_calendar"),
        }

        emit_event("module.start", module_name, {"stage": "init"})
        init_result = self._lifecycle_manager.init_plugin(plugin, plugin_context)
        if init_result.status is ResultStatus.FAILED:
            module_logger.error("module.init_failed", error=init_result.error)
            emit_event("module.failed", module_name, {"stage": "init"})
            cleanup_result = self._lifecycle_manager.cleanup_plugin(plugin)
            failure_error = init_result.error or cleanup_result.error
            return ModuleResult(
                module_name=module_name,
                status=ResultStatus.FAILED,
                output_data=None,
                error=failure_error,
            )

        degraded_errors: list[BaseException] = []
        if init_result.status is ResultStatus.DEGRADED and init_result.error is not None:
            degraded_errors.append(init_result.error)

        emit_event("module.execute", module_name, {"stage": "execute"})
        execute_result = self._lifecycle_manager.execute_plugin(plugin, input_payload)
        if execute_result.status is ResultStatus.FAILED:
            module_logger.error("module.execute_failed", error=execute_result.error)
            emit_event("module.failed", module_name, {"stage": "execute"})
            cleanup_result = self._lifecycle_manager.cleanup_plugin(plugin)
            failure_error = execute_result.error or cleanup_result.error
            return ModuleResult(
                module_name=module_name,
                status=ResultStatus.FAILED,
                output_data=None,
                error=failure_error,
            )

        if execute_result.status is ResultStatus.DEGRADED and execute_result.error is not None:
            degraded_errors.append(execute_result.error)

        cleanup_result = self._lifecycle_manager.cleanup_plugin(plugin)
        if cleanup_result.status is ResultStatus.FAILED:
            module_logger.error("module.cleanup_failed", error=cleanup_result.error)
            emit_event("module.failed", module_name, {"stage": "cleanup"})
            return ModuleResult(
                module_name=module_name,
                status=ResultStatus.FAILED,
                output_data=None,
                error=cleanup_result.error,
            )
        if cleanup_result.status is ResultStatus.DEGRADED and cleanup_result.error is not None:
            degraded_errors.append(cleanup_result.error)

        status = ResultStatus.DEGRADED if degraded_errors else ResultStatus.SUCCESS
        emit_event("module.completed", module_name, {"status": status.value})
        degrade_error = degraded_errors[0] if degraded_errors else None

        return ModuleResult(
            module_name=module_name,
            status=status,
            output_data=execute_result.data,
            error=degrade_error,
        )

    def _lookup_plugin_config(self, config_map: Mapping[str, Any], plugin_name: str) -> Mapping[str, Any]:
        plugins_section = config_map.get("plugins")
        if not isinstance(plugins_section, Mapping):
            return {}

        for category_entries in plugins_section.values():
            if not isinstance(category_entries, list):
                continue
            for entry in category_entries:
                if not isinstance(entry, Mapping):
                    continue
                if entry.get("name") != plugin_name:
                    continue
                cfg = entry.get("config")
                if isinstance(cfg, Mapping):
                    return cfg
        return {}

    def _emit_event(
        self,
        run_id: str,
        event_type: str,
        module_name: str,
        data: Mapping[str, Any] | None,
    ) -> Result[DomainEvent | None]:
        event = DomainEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            run_id=run_id,
            module=module_name,
            timestamp_ns=self._clock(),
            data=dict(data) if data is not None else None,
        )
        topic = f"runs.{run_id}.{module_name}.{event_type}"
        try:
            self._event_bus.publish(topic, event)
        except PermissionError as exc:  # pragma: no cover - defensive.
            error = ConfigurationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._EVENT_FAILED_REASON,
                details={"run_id": run_id, "module": module_name},
            )
            return Result.failed(error, error.reason_code)
        except OSError as exc:  # pragma: no cover - defensive.
            error = RecoverableError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._EVENT_FAILED_REASON,
                details={"run_id": run_id, "module": module_name},
            )
            return Result.degraded(None, error, error.reason_code)
        return Result.success(event)

    def _ensure_recoverable(
        self,
        error: BaseException | None,
        reason: str,
    ) -> RecoverableError:
        if isinstance(error, RecoverableError):
            return error
        if error is not None:
            return RecoverableError.from_error(
                error,
                module=self._MODULE_NAME,
                reason_code=reason,
            )
        return RecoverableError(
            "Operation degraded without error context.",
            module=self._MODULE_NAME,
            reason_code=reason,
        )

    def _finalise_failure(
        self,
        *,
        metadata: RunMetadata,
        events: list[DomainEvent],
        module_results: list[ModuleResult],
        failure_error: BaseException | None,
        reason: str,
    ) -> Result[RunSummary]:
        end_time_ns = self._clock()
        final_metadata = RunMetadata(
            run_id=metadata.run_id,
            run_type=metadata.run_type,
            mode=metadata.mode,
            system_version=metadata.system_version,
            start_time=metadata.start_time,
            end_time=end_time_ns,
            status="failed",
        )
        self._journal_writer.persist_complete_run(
            metadata.run_id,
            final_metadata,
            inputs={},
            outputs={},
            events=events,
        )
        summary = generate_summary(final_metadata, module_results)
        error = self._normalise_failure_error(failure_error, reason, metadata.run_id, summary)
        return Result.failed(error, reason)

    def _normalise_failure_error(
        self,
        error: BaseException | None,
        reason: str,
        run_id: str,
        summary: RunSummary,
    ) -> OperationalError:
        if isinstance(error, OperationalError):
            error.details.setdefault("run_summary", summary)
            return error
        base_error = error or OperationalError(
            "Orchestrator failed.",
            module=self._MODULE_NAME,
            reason_code=reason,
            details={"run_id": run_id},
        )
        return OperationalError.from_error(
            base_error,
            module=self._MODULE_NAME,
            reason_code=reason,
            details={"run_id": run_id, "run_summary": summary},
        )

    @staticmethod
    def _status_string(status: ResultStatus) -> str:
        if status is ResultStatus.SUCCESS:
            return "completed"
        if status is ResultStatus.DEGRADED:
            return "degraded"
        return "failed"
