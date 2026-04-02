"""Plugin interfaces and metadata definitions for the AST plugin system.

This module defines the structured metadata contract shared by all plugins,
along with runtime-checkable protocol interfaces that downstream components
can rely on for lifecycle orchestration. The interfaces follow the Result[T]
pattern established in ``common.interface`` so that every operation communicates
structured success, degraded, or failure outcomes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Generic, Mapping, Protocol, TypeVar, runtime_checkable

import msgspec

from common.interface import Config, Result

__all__ = [
    "PluginCategory",
    "PluginMetadata",
    "PluginBase",
    "DataSourcePlugin",
    "ScannerPlugin",
    "SignalPlugin",
    "StrategyPlugin",
    "RiskPolicyPlugin",
]


class PluginCategory(str, Enum):
    """Enumerate the supported plugin categories within the trading system."""

    DATA_SOURCE = "data_source"
    SCANNER = "scanner"
    SIGNAL = "signal"
    STRATEGY = "strategy"
    RISK_POLICY = "risk_policy"


class PluginMetadata(msgspec.Struct, kw_only=True, frozen=True):
    """Structured metadata describing a plugin implementation."""

    name: str
    version: str
    schema_version: str | None = None
    category: PluginCategory
    enabled: bool = True
    description: str | None = None


ConfigT = TypeVar("ConfigT", bound=Config)
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


PluginContext = Mapping[str, Any]


@runtime_checkable
class PluginBase(Protocol[ConfigT, InputT, OutputT], Generic[ConfigT, InputT, OutputT]):
    """Runtime-checkable protocol describing the core plugin lifecycle."""

    metadata: ClassVar[PluginMetadata]

    def init(self, context: PluginContext | None = None) -> Result[None]:
        """Initialise the plugin with the supplied ``context``."""

    def validate_config(self, config: ConfigT) -> Result[ConfigT]:
        """Validate and optionally normalise the plugin configuration."""

    def execute(self, payload: InputT) -> Result[OutputT]:
        """Execute the plugin against ``payload`` and return a structured result."""

    def cleanup(self) -> Result[None]:
        """Release resources held by the plugin implementation."""


@runtime_checkable
class DataSourcePlugin(
    PluginBase[ConfigT, Mapping[str, Any], Mapping[str, Any]],
    Protocol[ConfigT],
):
    """Protocol contract for data source plugins returning structured payloads."""


@runtime_checkable
class ScannerPlugin(
    PluginBase[ConfigT, Mapping[str, Any], list[Mapping[str, Any]]],
    Protocol[ConfigT],
):
    """Protocol contract for scanner plugins returning candidate collections."""


@runtime_checkable
class SignalPlugin(
    PluginBase[ConfigT, list[Mapping[str, Any]], Mapping[str, Any]],
    Protocol[ConfigT],
):
    """Protocol contract for signal plugins producing signal maps per symbol."""


@runtime_checkable
class StrategyPlugin(
    PluginBase[ConfigT, Mapping[str, Any], Mapping[str, Any]],
    Protocol[ConfigT],
):
    """Protocol contract for strategy plugins emitting trading instructions."""


@runtime_checkable
class RiskPolicyPlugin(
    PluginBase[ConfigT, Mapping[str, Any], Mapping[str, Any]],
    Protocol[ConfigT],
):
    """Protocol contract for risk policy plugins evaluating risk constraints."""
