"""Layered configuration loader grounded in msgspec structures.

This module implements the ``ConfigLoader`` protocol defined in
``common.interface``. It provides a YAML-backed loader with support for
layered configuration sources (base, environment, local, and environment
variables) and validation hooks tailored to the Automated Swing Trader (AST)
configuration schema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

import yaml
from dotenv import load_dotenv

from .interface import ConfigLoader, Result, ResultStatus

try:  # pragma: no cover - honour optional dependency pattern.
    import msgspec
except ModuleNotFoundError as exc:  # pragma: no cover - fail fast with guidance.
    raise RuntimeError(
        "msgspec is required for configuration loading. "
        "Add 'msgspec' to requirements.txt and install dependencies.",
    ) from exc


class ConfigLoaderError(RuntimeError):
    """Base error for configuration loader failures."""


class ConfigFileNotFoundError(ConfigLoaderError):
    """Raised when a required configuration file is missing."""


class ConfigValidationError(ConfigLoaderError):
    """Raised when the merged configuration fails semantic validation."""


class SystemConfig(msgspec.Struct, kw_only=True, frozen=True):
    """System-level settings shared across environments."""

    version: str
    log_level: str
    log_output: str


class ScheduleEntryConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Configuration for a scheduled job."""

    enabled: bool
    time: str | None = None


class SchedulingConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Scheduling configuration for recurring jobs."""

    eod_scan: ScheduleEntryConfig
    intraday_check_1030: ScheduleEntryConfig
    intraday_check_1430: ScheduleEntryConfig


class DataSourcePrimaryConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Primary data source providers."""

    eod_historical: str
    intraday_delayed: str
    fundamentals: str
    news_sentiment: str


class DataSourceFailoverConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Failover policy for external data providers."""

    enabled: bool
    max_retries: int | None = None
    retry_delay_seconds: int | None = None


class DataSourceHealthCheckConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Health check thresholds for data providers."""

    timeout_seconds: int
    max_error_rate: float
    max_staleness_hours: int


class DataSourcesConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Aggregate data source configuration."""

    primary: DataSourcePrimaryConfig
    failover: DataSourceFailoverConfig
    health_check: DataSourceHealthCheckConfig


class UniverseFiltersConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Universe builder filters applied to the symbol universe."""

    exchanges: list[str]
    exclude_otc: bool
    min_price: float
    max_price: float
    min_avg_volume: int
    min_market_cap: int
    max_symbols: int | None = None


class UniverseBuilderConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Configuration for the universe builder component."""

    enabled: bool
    filters: UniverseFiltersConfig
    snapshot_cache_hours: int


class ScannerConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Scanner settings controlling pattern detection."""

    enabled: bool
    lookback_days: int
    min_platform_days: int
    max_platform_days: int
    consolidation_threshold: float
    volume_spike_threshold: float


class EventGuardConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Event guard settings to avoid trading around market events."""

    enabled: bool
    earnings_window_days: int
    offering_window_days: int
    lockup_window_days: int
    sources: dict[str, Any] = msgspec.field(default_factory=dict)


class MarketCalendarConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Market calendar configuration."""

    enabled: bool = True
    cache_ttl_hours: int = 24
    requests_timeout_seconds: int = 30


class HolidayBehaviorConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Holiday execution behavior configuration."""

    universe: bool = True
    scanner: bool = True
    event_guard: bool = True
    strategy: bool = False
    risk_gate: bool = False
    execution: bool = False


class SignalsModuleConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Generic toggle for optional signal modules."""

    enabled: bool


class SignalsConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Optional signals configuration."""

    sentiment: SignalsModuleConfig
    fear_greed: SignalsModuleConfig


class StrategyEntryConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Entry parameters for strategy execution."""

    max_position_size: float
    ladder_tranches: int
    first_tranche_pct: float


class StrategyExitConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Exit parameters for strategy execution."""

    take_profit_pct: float
    stop_loss_pct: float
    time_stop_days: int


class StrategyConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Aggregate strategy configuration."""

    entry: StrategyEntryConfig
    exit: StrategyExitConfig


class RiskGatePortfolioConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Portfolio-level risk limits."""

    max_daily_exposure: float
    max_total_exposure: float
    max_concentration_per_symbol: float
    max_concentration_per_sector: float


class RiskGateSymbolConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Symbol-level risk filters."""

    min_spread_bps: int
    min_avg_volume: int


class RiskGateOperationalConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Operational guardrails limiting daily activity."""

    max_orders_per_run: int
    max_new_positions_per_day: int
    safe_mode_loss_threshold: float


class RiskGateConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Aggregate risk gate configuration."""

    enabled: bool
    portfolio: RiskGatePortfolioConfig
    symbol: RiskGateSymbolConfig
    operational: RiskGateOperationalConfig


class ExecutionConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Execution adapter configuration."""

    enabled: bool
    order_type: str
    time_in_force: str
    max_price_slippage_pct: float


class JournalConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Trade journal configuration."""

    enabled: bool
    storage_path: str
    retention_days: int


class MonitoringConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Monitoring and alerting configuration."""

    enabled: bool
    alert_channels: list[str]
    metrics: list[str]


class BaseConfig(msgspec.Struct, kw_only=True, frozen=True):
    """AST baseline configuration structure."""

    system: SystemConfig
    mode: str
    scheduling: SchedulingConfig
    data_sources: DataSourcesConfig
    universe_builder: UniverseBuilderConfig
    scanner: ScannerConfig
    event_guard: EventGuardConfig
    market_calendar: MarketCalendarConfig = MarketCalendarConfig()
    holiday_behavior: HolidayBehaviorConfig = HolidayBehaviorConfig()
    signals: SignalsConfig
    strategy: StrategyConfig
    risk_gate: RiskGateConfig
    execution: ExecutionConfig
    journal: JournalConfig
    monitoring: MonitoringConfig


ConfigStructT = TypeVar("ConfigStructT", bound=BaseConfig)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""
    merged: dict[str, Any] = {key: value for key, value in base.items()}
    for key, override_value in override.items():
        if override_value is None:
            continue
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def _normalise_key(token: str) -> str:
    """Normalise environment variable segments to configuration keys."""
    return token.lower()


def _parse_env_value(raw_value: str) -> Any:
    """Parse environment variable values using YAML semantics."""
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def _build_env_override(
    tokens: Iterable[str],
    value: Any,
) -> dict[str, Any]:
    """Construct a nested dictionary from segmented override tokens."""
    tokens_list = list(tokens)
    if not tokens_list:
        return {}

    nested: dict[str, Any] = {}
    current = nested
    for token in tokens_list[:-1]:
        key = _normalise_key(token)
        next_container: dict[str, Any] = {}
        current[key] = next_container
        current = next_container

    current[_normalise_key(tokens_list[-1])] = value
    return nested


def _apply_env_overrides(
    data: Mapping[str, Any],
    env_var_prefix: str,
    env: str,
) -> dict[str, Any]:
    """Merge environment variable overrides on top of the supplied mapping."""
    prefix = f"{env_var_prefix.upper()}__"
    env_specific_prefix = f"{env_var_prefix.upper()}__{env.upper()}__"

    overrides: dict[str, Any] = {}
    for key, raw_value in dict(os.environ).items():
        if key.startswith(env_specific_prefix):
            tokens = key[len(env_specific_prefix) :].split("__")
        elif key.startswith(prefix):
            tokens = key[len(prefix) :].split("__")
        else:
            continue

        value = _parse_env_value(raw_value)
        override = _build_env_override(tokens, value)
        overrides = _deep_merge(overrides, override)

    if not overrides:
        return dict(data)

    return _deep_merge(data, overrides)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    """Load YAML content from ``path`` into a dictionary."""
    with path.open("r", encoding="utf-8") as file:
        try:
            loaded = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ConfigLoaderError(f"Failed to parse YAML file at {path!s}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigLoaderError(f"Configuration file {path!s} must contain a mapping.")
    return loaded


@dataclass(slots=True)
class YAMLConfigLoader(ConfigLoader[ConfigStructT]):
    """Load AST configuration from layered YAML files and environment variables."""

    config_dir: Path
    config_cls: type[ConfigStructT] = BaseConfig
    env_var_prefix: str = "AST"
    dotenv_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.config_dir.exists():
            raise ConfigLoaderError(f"Configuration directory {self.config_dir!s} not found.")
        self._load_dotenv()

    def _load_dotenv(self) -> None:
        """Load environment variables from a dotenv file if present."""
        candidate_paths: list[Path] = []
        if self.dotenv_path:
            candidate_paths.append(self.dotenv_path)
        candidate_paths.append(self.config_dir / "secrets.env")
        candidate_paths.append(self.config_dir / ".env")

        for path in candidate_paths:
            if path.exists():
                load_dotenv(dotenv_path=path, override=False)
                break

    def load_config(self, env: str) -> ConfigStructT:
        """Load and validate the configuration for the supplied environment."""
        result = self.load_config_result(env)
        if result.status is ResultStatus.SUCCESS and result.data is not None:
            return result.data

        raise result.error or ConfigLoaderError("Unknown configuration loading failure.")

    def load_config_result(self, env: str) -> Result[ConfigStructT]:
        """Return a ``Result`` containing the loaded configuration or failure information."""
        try:
            merged = self._load_and_merge_layers(env)
            with_env = _apply_env_overrides(merged, self.env_var_prefix, env)
            config = msgspec.convert(with_env, self.config_cls)
            self._validate_config(config)
        except ConfigLoaderError as exc:
            return Result.failed(exc, reason_code="CONFIG_LOAD_ERROR")
        except msgspec.ValidationError as exc:
            return Result.failed(exc, reason_code="CONFIG_VALIDATION_ERROR")
        except Exception as exc:  # pragma: no cover - guard unknown exceptions.
            return Result.failed(exc, reason_code="CONFIG_UNEXPECTED_ERROR")

        return Result.success(config)

    def _load_and_merge_layers(self, env: str) -> dict[str, Any]:
        """Merge base, environment, and local configuration layers."""
        base_path = self.config_dir / "config.yaml"
        if not base_path.exists():
            raise ConfigFileNotFoundError(f"Base configuration file missing at {base_path!s}")

        merged: dict[str, Any] = dict(_load_yaml(base_path))

        env_path = self.config_dir / f"config.{env.lower()}.yaml"
        if env_path.exists():
            merged = _deep_merge(merged, _load_yaml(env_path))

        local_path = self.config_dir / "config.local.yaml"
        if local_path.exists():
            merged = _deep_merge(merged, _load_yaml(local_path))

        return merged

    def _validate_config(self, config: ConfigStructT) -> None:
        """Validate the semantic correctness of the loaded configuration."""
        allowed_modes = {"DRY_RUN", "PAPER", "LIVE"}
        if config.mode not in allowed_modes:
            raise ConfigValidationError(f"Invalid mode '{config.mode}' (expected {allowed_modes}).")

        entry = config.strategy.entry
        if not 0 < entry.max_position_size <= 1:
            raise ConfigValidationError("strategy.entry.max_position_size must be within (0, 1].")
        if not 0 < entry.first_tranche_pct <= 1:
            raise ConfigValidationError("strategy.entry.first_tranche_pct must be within (0, 1].")
        if entry.first_tranche_pct > entry.max_position_size:
            raise ConfigValidationError(
                "strategy.entry.first_tranche_pct cannot exceed max_position_size.",
            )
        if entry.ladder_tranches <= 0:
            raise ConfigValidationError("strategy.entry.ladder_tranches must be positive.")

        exit_cfg = config.strategy.exit
        if not 0 < exit_cfg.stop_loss_pct < 1:
            raise ConfigValidationError("strategy.exit.stop_loss_pct must be within (0, 1).")
        if not 0 < exit_cfg.take_profit_pct < 1:
            raise ConfigValidationError("strategy.exit.take_profit_pct must be within (0, 1).")
        if exit_cfg.time_stop_days <= 0:
            raise ConfigValidationError("strategy.exit.time_stop_days must be positive.")

        portfolio = config.risk_gate.portfolio
        for field_name in (
            "max_daily_exposure",
            "max_total_exposure",
            "max_concentration_per_symbol",
            "max_concentration_per_sector",
        ):
            value = getattr(portfolio, field_name)
            if not 0 <= value <= 1:
                raise ConfigValidationError(
                    f"risk_gate.portfolio.{field_name} must be within [0, 1].",
                )

        operational = config.risk_gate.operational
        if operational.max_orders_per_run <= 0:
            raise ConfigValidationError("risk_gate.operational.max_orders_per_run must be positive.")
        if operational.max_new_positions_per_day <= 0:
            raise ConfigValidationError("risk_gate.operational.max_new_positions_per_day must be positive.")
        if not 0 < operational.safe_mode_loss_threshold < 1:
            raise ConfigValidationError(
                "risk_gate.operational.safe_mode_loss_threshold must be within (0, 1).",
            )

        if config.execution.max_price_slippage_pct < 0:
            raise ConfigValidationError("execution.max_price_slippage_pct must be non-negative.")

        if config.universe_builder.filters.min_price <= 0:
            raise ConfigValidationError("universe_builder.filters.min_price must be positive.")
        if config.universe_builder.filters.max_price <= 0:
            raise ConfigValidationError("universe_builder.filters.max_price must be positive.")
        if config.universe_builder.filters.min_avg_volume <= 0:
            raise ConfigValidationError("universe_builder.filters.min_avg_volume must be positive.")
        if config.universe_builder.filters.min_market_cap <= 0:
            raise ConfigValidationError("universe_builder.filters.min_market_cap must be positive.")
        if config.universe_builder.filters.max_symbols is not None and (
            config.universe_builder.filters.max_symbols <= 0
        ):
            raise ConfigValidationError("universe_builder.filters.max_symbols must be positive.")

        scanner = config.scanner
        if scanner.lookback_days <= 0 or scanner.max_platform_days <= 0:
            raise ConfigValidationError("scanner lookback_days/max_platform_days must be positive.")
        if scanner.min_platform_days <= 0:
            raise ConfigValidationError("scanner.min_platform_days must be positive.")
        if not 0 < scanner.consolidation_threshold < 1:
            raise ConfigValidationError("scanner.consolidation_threshold must be within (0, 1).")
        if scanner.volume_spike_threshold <= 0:
            raise ConfigValidationError("scanner.volume_spike_threshold must be positive.")

        scheduling_entries = [
            config.scheduling.eod_scan,
            config.scheduling.intraday_check_1030,
            config.scheduling.intraday_check_1430,
        ]
        for entry_cfg in scheduling_entries:
            if entry_cfg.enabled and not entry_cfg.time:
                raise ConfigValidationError("Enabled schedules must provide a run time.")
