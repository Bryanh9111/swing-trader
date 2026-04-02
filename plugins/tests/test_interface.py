from __future__ import annotations

from typing import Any, Mapping

import msgspec

from common.interface import Result
from plugins.interface import (
    DataSourcePlugin,
    PluginCategory,
    PluginMetadata,
    ScannerPlugin,
)


class ExampleConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Sample configuration struct used for protocol compliance tests."""

    endpoint: str


class ExampleDataSourcePlugin:
    """Concrete class satisfying the DataSourcePlugin protocol."""

    metadata = PluginMetadata(
        name="example_data_source",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
        description="Example data source plugin for testing.",
    )

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config = config or {}

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[ExampleConfig]:
        endpoint = config.get("endpoint") if isinstance(config, Mapping) else None
        if not endpoint:
            config_struct = ExampleConfig(endpoint="https://api.example.com")
            return Result.success(config_struct)
        return Result.success(ExampleConfig(endpoint=str(endpoint)))

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        return Result.success({"payload": dict(payload)})

    def cleanup(self) -> Result[None]:
        return Result.success(data=None)


class ExampleScannerPlugin:
    """Concrete class satisfying the ScannerPlugin protocol."""

    metadata = PluginMetadata(
        name="example_scanner",
        version="1.0.0",
        category=PluginCategory.SCANNER,
    )

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config = config or {}

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[ExampleConfig]:
        return Result.success(ExampleConfig(endpoint="https://scanner.example.com"))

    def execute(self, payload: Mapping[str, Any]) -> Result[list[Mapping[str, Any]]]:
        return Result.success([dict(payload)])

    def cleanup(self) -> Result[None]:
        return Result.success(data=None)


def test_plugin_metadata_fields() -> None:
    metadata = ExampleDataSourcePlugin.metadata
    assert metadata.name == "example_data_source"
    assert metadata.category is PluginCategory.DATA_SOURCE
    assert metadata.enabled is True


def test_datasource_plugin_protocol_runtime_checking() -> None:
    plugin = ExampleDataSourcePlugin(config={"endpoint": "https://api.example.com"})

    assert isinstance(plugin, DataSourcePlugin)


def test_scanner_plugin_protocol_runtime_checking() -> None:
    plugin = ExampleScannerPlugin(config={"window": 5})

    assert isinstance(plugin, ScannerPlugin)
