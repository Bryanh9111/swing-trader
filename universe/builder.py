"""Universe Builder plugin implementation.

The Universe Builder constructs a tradable US equity universe from reference
data and filtering criteria, returning a versioned ``UniverseSnapshot``. Phase
2.1 provides the core structure, filtering logic, event emission hooks, and
run-boundary degradation behaviour.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

import msgspec

from common.exceptions import DataSourceUnavailableError, ValidationError
from common.interface import BoundLogger, DomainEvent, EventBus, Result, ResultStatus
from common.utils import is_etf
from plugins.interface import PluginCategory, PluginMetadata

from . import data_sources
from .interface import EquityInfo, UniverseBuilderPlugin, UniverseFilterCriteria, UniverseSnapshot

__all__ = ["UniverseBuilder"]

# Polygon API ``type`` values that are NOT common stocks.  When
# ``exclude_etfs`` is enabled the builder rejects any equity whose
# ``asset_type`` appears in this set *before* falling back to the
# static ``is_etf()`` heuristic.
_NON_EQUITY_ASSET_TYPES: frozenset[str] = frozenset({
    "ETF", "ETN", "UNIT",
    "PFD",      # Preferred stocks
    "RIGHT",    # Rights
    "SP",       # Special purpose securities
    "WARRANT",  # Warrants
    "FUND",     # Mutual funds / closed-end funds
    "BASKET",   # Index baskets
    "BOND",     # Bonds
    "OS",       # Other securities
    "GDR",      # Global depositary receipts
    "NYRS",     # NY Registry Shares
    "AGEN",     # Agency bonds
    "ETS",      # Exchange-traded structured products
    "ETV",      # Exchange-traded vehicles
    "ADRC",     # ADR certificates
})


class UniverseBuilder(UniverseBuilderPlugin):
    """Universe Builder plugin implementing ``UniverseBuilderPlugin``.

    The plugin fetches US equity reference data (with ordered fallback),
    applies filtering rules defined in ``UniverseFilterCriteria``, emits a
    ``universe.built`` domain event, and returns a versioned snapshot.
    """

    metadata = PluginMetadata(
        name="universe_builder",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
        description="Construct and filter a tradable US equity universe snapshot.",
    )

    _MODULE_NAME = "universe.builder"
    _CONFIG_INVALID_REASON = "UNIVERSE_FILTER_CRITERIA_INVALID"
    _SOURCE_FAILED_REASON = "UNIVERSE_SOURCE_FAILED"
    _SOURCE_FALLBACK_REASON = "UNIVERSE_SOURCE_FALLBACK"
    _ALL_SOURCES_FAILED_REASON = "UNIVERSE_ALL_SOURCES_FAILED"
    _DETAILS_ENRICH_DEGRADED_REASON = "UNIVERSE_DETAILS_ENRICH_DEGRADED"

    def __init__(self, criteria: UniverseFilterCriteria) -> None:
        self._criteria = criteria
        self._logger: logging.Logger | BoundLogger = logging.getLogger(self._MODULE_NAME)
        self._event_bus: EventBus | None = None
        self._run_id: str = "unknown"
        self._module_name: str = self.metadata.name
        self._system_version: str = "dev"
        self._details_enrich_degraded_error: BaseException | None = None
        self._polygon_api_key: str | None = None

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        """Initialise the builder with logger/event bus handles from ``context``."""

        if isinstance(context, Mapping):
            module_name = context.get("module_name")
            if isinstance(module_name, str) and module_name:
                self._module_name = module_name

            logger = context.get("logger")
            if logger is None:
                run_context = context.get("run_context")
                if run_context is not None:
                    logger = getattr(run_context, "logger", None)
            if logger is not None:
                self._logger = logger  # type: ignore[assignment]

            bus = context.get("event_bus")
            if bus is None:
                run_context = context.get("run_context")
                if run_context is not None:
                    bus = getattr(run_context, "event_bus", None)
            if bus is not None:
                self._event_bus = bus  # type: ignore[assignment]

            run_id = context.get("run_id")
            if run_id is None:
                run_context = context.get("run_context")
                if run_context is not None:
                    run_id = getattr(run_context, "run_id", None)
            if isinstance(run_id, str) and run_id:
                self._run_id = run_id

            system_version = context.get("system_version")
            if system_version is None:
                run_context = context.get("run_context")
                if run_context is not None:
                    system_version = getattr(run_context, "system_version", None)
            if isinstance(system_version, str) and system_version:
                self._system_version = system_version

            polygon_api_key = context.get("polygon_api_key")
            config_payload = context.get("config")
            if polygon_api_key is None and isinstance(config_payload, Mapping):
                polygon_api_key = config_payload.get("polygon_api_key")

            run_context = context.get("run_context")
            run_config = getattr(run_context, "config", None) if run_context is not None else None
            if polygon_api_key is None and isinstance(run_config, Mapping):
                polygon_api_key = run_config.get("polygon_api_key")

            if isinstance(polygon_api_key, str) and polygon_api_key.strip():
                self._polygon_api_key = polygon_api_key.strip()
                import sys
                sys.stderr.write(f"🔍 UniverseBuilder.init() - API key received from context: {self._polygon_api_key[:10]}...{self._polygon_api_key[-4:]} (length: {len(self._polygon_api_key)})\n")
                sys.stderr.flush()
            else:
                import sys
                sys.stderr.write(f"🔍 UniverseBuilder.init() - No API key in context (polygon_api_key={polygon_api_key!r})\n")
                sys.stderr.flush()

        if hasattr(self._logger, "bind"):
            self._logger = self._logger.bind(module=self._module_name, run_id=self._run_id)

        validation = self.validate_config(self._criteria)
        if validation.status is ResultStatus.FAILED:
            return Result.failed(validation.error or ValidationError(
                "UniverseFilterCriteria validation failed.",
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
            ), validation.reason_code or self._CONFIG_INVALID_REASON)

        if validation.data is not None:
            self._criteria = validation.data

        self._log_info("universe.initialised", criteria=msgspec.to_builtins(self._criteria))
        return Result.success(data=None)

    def validate_config(self, config: UniverseFilterCriteria) -> Result[UniverseFilterCriteria]:
        """Validate and normalise the supplied ``UniverseFilterCriteria``."""

        try:
            exchanges = [exchange.strip().upper() for exchange in config.exchanges if exchange and exchange.strip()]
            exchanges = list(dict.fromkeys(exchanges))
            if not exchanges:
                raise ValueError("exchanges must be a non-empty list of exchange identifiers")

            if config.min_price < 0:
                raise ValueError("min_price must be >= 0")

            if config.max_price is not None:
                if config.max_price <= 0:
                    raise ValueError("max_price must be > 0 when set")
                if config.max_price < config.min_price:
                    raise ValueError("max_price must be >= min_price when set")

            if config.min_avg_dollar_volume_20d < 0:
                raise ValueError("min_avg_dollar_volume_20d must be >= 0")

            if config.min_market_cap is not None and config.min_market_cap < 0:
                raise ValueError("min_market_cap must be >= 0 when set")

            if config.max_results is not None and config.max_results <= 0:
                raise ValueError("max_results must be > 0 when set")

            if config.details_enrich_top_k < 0:
                raise ValueError("details_enrich_top_k must be >= 0 (0 = enrich all)")
            if config.details_enrich_multiplier <= 0:
                raise ValueError("details_enrich_multiplier must be > 0")
        except ValueError as exc:
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CONFIG_INVALID_REASON,
                details={"config": msgspec.to_builtins(config)},
            )
            return Result.failed(error, error.reason_code)

        # Universe Builder "Universe expansion" defaults (user requirement):
        # - Price range: 0 < price <= 2000
        # - Market cap: no floor / unrestricted
        enforced_min_price = 0.0
        enforced_max_price = 2000.0

        normalised = UniverseFilterCriteria(
            exchanges=exchanges,
            min_price=enforced_min_price,
            max_price=enforced_max_price,
            min_avg_dollar_volume_20d=float(config.min_avg_dollar_volume_20d),
            min_market_cap=None,
            exclude_otc=bool(config.exclude_otc),
            exclude_halted=bool(config.exclude_halted),
            max_results=int(config.max_results) if config.max_results is not None else None,
            enable_details_enrich=bool(config.enable_details_enrich),
            details_enrich_top_k=int(config.details_enrich_top_k),
            details_enrich_multiplier=float(config.details_enrich_multiplier),
        )
        return Result.success(normalised)

    def execute(self, payload: None = None) -> Result[UniverseSnapshot]:
        """Fetch, filter, and snapshot the tradable US equity universe."""

        asof_timestamp = time.time_ns()
        fetch_result, source, degraded_error = self._fetch_universe()

        equities = fetch_result.data or []
        total_candidates = len(equities)
        self._details_enrich_degraded_error = None
        filtered = self._apply_filters(equities, self._criteria)
        total_filtered = len(filtered)

        snapshot = UniverseSnapshot(
            schema_version="1.0.0",
            system_version=self._system_version or "dev",
            asof_timestamp=asof_timestamp,
            source=source,
            equities=filtered,
            filter_criteria=msgspec.to_builtins(self._criteria),
            total_candidates=total_candidates,
            total_filtered=total_filtered,
        )

        self._emit_built_event(snapshot)

        if fetch_result.status is ResultStatus.DEGRADED:
            return Result.degraded(snapshot, fetch_result.error or degraded_error or DataSourceUnavailableError(
                "Universe reference data degraded.",
                module=self._MODULE_NAME,
                reason_code=self._SOURCE_FAILED_REASON,
                details={"source": source},
            ), fetch_result.reason_code or self._SOURCE_FALLBACK_REASON)

        if degraded_error is not None:
            return Result.degraded(snapshot, degraded_error, self._SOURCE_FALLBACK_REASON)

        if self._details_enrich_degraded_error is not None:
            return Result.degraded(snapshot, self._details_enrich_degraded_error, self._DETAILS_ENRICH_DEGRADED_REASON)

        return Result.success(snapshot)

    def cleanup(self) -> Result[None]:
        """Release resources held by the builder (no-op in Phase 2.1)."""

        self._log_info("universe.cleaned_up")
        return Result.success(data=None)

    def _fetch_universe(self) -> tuple[Result[list[EquityInfo]], str, BaseException | None]:
        """Fetch universe reference data with ordered fallback (Polygon → FMP → cache)."""

        sources: Sequence[tuple[str, Callable[[], Result[list[EquityInfo]]]]] = (
            ("polygon", data_sources.fetch_polygon_universe),
            ("fmp", data_sources.fetch_fmp_universe),
            ("cache", data_sources.fetch_cached_universe),
        )

        first_error: BaseException | None = None
        last_error: BaseException | None = None

        for source, fetch in sources:
            try:
                result = fetch()
            except Exception as exc:  # noqa: BLE001 - normalize to recoverable error.
                error = DataSourceUnavailableError.from_error(
                    exc,
                    module=self._MODULE_NAME,
                    reason_code=self._SOURCE_FAILED_REASON,
                    details={"source": source},
                )
                result = Result.failed(error, error.reason_code)

            if result.status is ResultStatus.SUCCESS and result.data is not None:
                self._log_info("universe.source_selected", source=source, candidates=len(result.data))
                if first_error is not None:
                    return result, source, first_error
                return result, source, None

            if result.status is ResultStatus.DEGRADED and result.data is not None:
                self._log_warning(
                    "universe.source_degraded",
                    source=source,
                    reason=result.reason_code,
                    error=repr(result.error),
                    candidates=len(result.data),
                )
                if first_error is None and result.error is not None:
                    first_error = result.error
                return result, source, first_error

            if first_error is None and result.error is not None:
                first_error = result.error
            if result.error is not None:
                last_error = result.error

            self._log_warning(
                "universe.source_failed",
                source=source,
                reason=result.reason_code,
                error=repr(result.error),
            )

        fallback_error = last_error or first_error or DataSourceUnavailableError(
            "All universe reference data sources failed.",
            module=self._MODULE_NAME,
            reason_code=self._ALL_SOURCES_FAILED_REASON,
        )
        degraded = Result.degraded([], fallback_error, self._ALL_SOURCES_FAILED_REASON)
        self._log_warning("universe.all_sources_failed", source="cache_fallback", error=repr(fallback_error))
        return degraded, "cache_fallback", fallback_error

    def _apply_filters(
        self,
        equities: Iterable[EquityInfo],
        criteria: UniverseFilterCriteria,
    ) -> list[EquityInfo]:
        """Apply ``criteria`` to ``equities`` and return the filtered list."""

        # NOTE: Exchange filtering is intentionally relaxed to support a larger
        # tradable universe; we only exclude OTC when configured.
        max_price = criteria.max_price if criteria.max_price is not None else 2000.0
        large_cap_threshold = 10_000_000_000.0
        enable_details_enrich = getattr(criteria, "enable_details_enrich", True)
        details_top_k = int(getattr(criteria, "details_enrich_top_k", 500))
        details_multiplier = float(getattr(criteria, "details_enrich_multiplier", 1.5))

        filtered: list[EquityInfo] = []
        for equity in equities:
            if equity.price is None:
                continue
            if equity.price <= 0:
                continue
            if equity.price > max_price:
                continue

            if equity.avg_dollar_volume_20d is None:
                continue
            if equity.avg_dollar_volume_20d < criteria.min_avg_dollar_volume_20d:
                continue

            if criteria.exclude_otc and equity.is_otc:
                continue

            if criteria.exclude_halted and equity.is_halted:
                continue

            if criteria.exclude_etfs:
                if equity.asset_type in _NON_EQUITY_ASSET_TYPES:
                    continue
                if is_etf(equity.symbol):
                    continue

            filtered.append(equity)

        def sort_key(candidate: EquityInfo) -> tuple[int, float, float, str]:
            market_cap = candidate.market_cap
            avg_dollar_volume = candidate.avg_dollar_volume_20d or 0.0

            # Lower weight for large caps (> $10B), higher weight for small/mid caps.
            if market_cap is None:
                bucket = 1
                cap_value = float("inf")
            elif market_cap < large_cap_threshold:
                bucket = 0
                cap_value = market_cap
            else:
                bucket = 2
                cap_value = market_cap

            return (bucket, -avg_dollar_volume, cap_value, candidate.symbol)

        filtered.sort(key=lambda candidate: (-(candidate.avg_dollar_volume_20d or 0.0), candidate.symbol))

        k = max(0, details_top_k)
        if criteria.max_results is not None:
            k = min(max(1, int(criteria.max_results * details_multiplier)), k)

        api_key = self._polygon_api_key or os.environ.get("POLYGON_API_KEY")

        # 🔍 DEBUG: Force output to stderr (unbuffered) and emit event
        import sys
        sys.stderr.write("\n" + "=" * 80 + "\n")
        sys.stderr.write("🔍 UNIVERSE BUILDER ENRICHMENT DECISION DEBUG\n")
        sys.stderr.write("=" * 80 + "\n")
        sys.stderr.write(f"enable_details_enrich: {enable_details_enrich}\n")
        sys.stderr.write(f"k (top_k count): {k}\n")
        sys.stderr.write(f"filtered_count: {len(filtered)}\n")
        sys.stderr.write(f"api_key_present: {bool(api_key)}\n")
        sys.stderr.write(f"api_key_length: {len(api_key) if api_key else 0}\n")
        sys.stderr.write(f"api_key_source: {'instance' if self._polygon_api_key else 'env' if os.environ.get('POLYGON_API_KEY') else 'none'}\n")
        sys.stderr.write(f"instance_api_key (self._polygon_api_key): {bool(self._polygon_api_key)}\n")
        sys.stderr.write(f"env_api_key (os.environ.get('POLYGON_API_KEY')): {bool(os.environ.get('POLYGON_API_KEY'))}\n")
        sys.stderr.write(f"\nCondition checks:\n")
        sys.stderr.write(f"  enable_details_enrich: {enable_details_enrich} → {'PASS' if enable_details_enrich else 'FAIL'}\n")
        sys.stderr.write(f"  k > 0: {k} > 0 → {'PASS' if k > 0 else 'FAIL'}\n")
        sys.stderr.write(f"  filtered exists: {len(filtered) if filtered else 0} items → {'PASS' if filtered else 'FAIL'}\n")
        sys.stderr.write(f"  api_key exists: {bool(api_key)} → {'PASS' if api_key else 'FAIL'}\n")
        sys.stderr.write(f"\nAll conditions met: {enable_details_enrich and k > 0 and filtered and api_key}\n")
        sys.stderr.write("=" * 80 + "\n")
        sys.stderr.flush()


        if enable_details_enrich and filtered and api_key:
            import sys
            effective_k = len(filtered) if k <= 0 else min(k, len(filtered))
            sys.stderr.write(f"\n✅ ENTERING ENRICHMENT BLOCK - will fetch details for {effective_k} symbols\n")
            sys.stderr.flush()

            top_candidates = filtered[:effective_k]
            rest_candidates = filtered[effective_k:]
            symbols = [equity.symbol for equity in top_candidates]

            sys.stderr.write(f"🔍 Calling fetch_polygon_ticker_details for {len(symbols)} symbols...\n")
            sys.stderr.flush()

            try:
                details = data_sources.fetch_polygon_ticker_details(symbols)
            except Exception as exc:  # noqa: BLE001 - enrichment must not break universe build.
                self._details_enrich_degraded_error = DataSourceUnavailableError(
                    "Universe ticker details enrichment call failed.",
                    module=self._MODULE_NAME,
                    reason_code=self._DETAILS_ENRICH_DEGRADED_REASON,
                    details={"requested": len(symbols), "error": repr(exc)[:200]},
                )
                self._log_warning(
                    "universe.details_enrich_failed",
                    requested=len(symbols),
                    error=repr(exc)[:200],
                )
                filtered = top_candidates + rest_candidates
            else:

                enriched: list[EquityInfo] = []
                succeeded = 0
                for equity in top_candidates:
                    detail = details.get(equity.symbol)
                    market_cap = detail.get("market_cap") if isinstance(detail, dict) else None
                    if market_cap is not None:
                        succeeded += 1
                        payload = msgspec.to_builtins(equity)
                        payload["market_cap"] = market_cap
                        payload["sector"] = detail.get("sector") or equity.sector
                        enriched.append(EquityInfo(**payload))
                    else:
                        enriched.append(equity)

                requested = len(symbols)
                coverage = (succeeded / requested) if requested else 0.0
                self._log_info(
                    "universe.details_enrich_complete",
                    requested=requested,
                    succeeded=succeeded,
                    coverage=coverage,
                )

                if requested and succeeded == 0:
                    self._details_enrich_degraded_error = DataSourceUnavailableError(
                        "Universe ticker details enrichment failed for all requested symbols.",
                        module=self._MODULE_NAME,
                        reason_code=self._DETAILS_ENRICH_DEGRADED_REASON,
                        details={"requested": requested},
                    )
                    self._log_warning("universe.details_enrich_all_failed", requested=requested)
                elif requested and coverage < 0.5:
                    self._details_enrich_degraded_error = DataSourceUnavailableError(
                        "Universe ticker details enrichment coverage below threshold.",
                        module=self._MODULE_NAME,
                        reason_code=self._DETAILS_ENRICH_DEGRADED_REASON,
                        details={"requested": requested, "succeeded": succeeded, "coverage": coverage},
                    )
                    self._log_warning(
                        "universe.details_enrich_low_coverage",
                        requested=requested,
                        succeeded=succeeded,
                        coverage=coverage,
                    )

                if self._details_enrich_degraded_error is not None:
                    filtered = top_candidates + rest_candidates
                else:
                    filtered = enriched + rest_candidates
        else:
            # 🔍 DEBUG: Enrichment was skipped
            skip_reason = (
                "enable_details_enrich=False" if not enable_details_enrich
                else "k=0" if k <= 0
                else "no_filtered_candidates" if not filtered
                else "no_api_key" if not api_key
                else "unknown"
            )
            import sys
            sys.stderr.write(f"\n⚠️  ENRICHMENT SKIPPED: {skip_reason}\n")
            sys.stderr.flush()

            self._log_warning(
                "universe.enrichment_skipped",
                enable_details_enrich=enable_details_enrich,
                k=k,
                filtered_count=len(filtered),
                api_key_present=bool(api_key),
                reason=skip_reason,
            )

        filtered.sort(key=sort_key)

        if criteria.max_results is not None:
            return filtered[: criteria.max_results]

        return filtered

    def _emit_built_event(self, snapshot: UniverseSnapshot) -> None:
        """Emit a ``universe.built`` event onto the configured event bus."""

        if self._event_bus is None:
            return

        event = DomainEvent(
            event_id=str(uuid4()),
            event_type="universe.built",
            run_id=self._run_id,
            module=self._module_name,
            timestamp_ns=time.time_ns(),
            data={
                "schema_version": snapshot.schema_version,
                "system_version": snapshot.system_version,
                "asof_timestamp": snapshot.asof_timestamp,
                "source": snapshot.source,
                "total_candidates": snapshot.total_candidates,
                "total_filtered": snapshot.total_filtered,
            },
        )

        topic = (
            f"runs.{self._run_id}.{self._module_name}.{event.event_type}"
            if self._run_id and self._run_id != "unknown"
            else f"events.{event.event_type}"
        )
        try:
            self._event_bus.publish(topic, event)
        except Exception as exc:  # noqa: BLE001 - eventing must not break execution.
            self._log_warning("universe.event_publish_failed", topic=topic, error=repr(exc))

    def _log_info(self, event: str, **kwargs: Any) -> None:
        if hasattr(self._logger, "bind"):
            self._logger.info(event, **kwargs)
        else:
            self._logger.info(self._format_log(event, kwargs))

    def _log_warning(self, event: str, **kwargs: Any) -> None:
        if hasattr(self._logger, "bind"):
            self._logger.warning(event, **kwargs)
        else:
            self._logger.warning(self._format_log(event, kwargs))

    @staticmethod
    def _format_log(event: str, payload: Mapping[str, Any]) -> str:
        if not payload:
            return event
        formatted = ", ".join(f"{key}={value!r}" for key, value in payload.items())
        return f"{event} | {formatted}"
