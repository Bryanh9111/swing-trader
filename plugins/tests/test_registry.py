from __future__ import annotations

from typing import Any, Mapping

import msgspec
import pytest

from common.exceptions import ConfigurationError, CriticalError, ValidationError
from common.interface import Result, ResultStatus
from plugins.interface import PluginCategory, PluginMetadata
from plugins.registry import PluginRegistry


class DataSourceConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Sample configuration struct for data source plugin tests."""

    endpoint: str


class ScannerConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Sample configuration struct for scanner plugin tests."""

    lookback_days: int


class DummyDataSourcePlugin:
    """Concrete plugin used to exercise registry behaviours."""

    metadata = PluginMetadata(
        name="dummy_data_source",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
    )

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config = config or {}

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[DataSourceConfig]:
        endpoint = config.get("endpoint") if isinstance(config, Mapping) else None
        if not endpoint:
            return Result.success(DataSourceConfig(endpoint="https://api.example.com"))
        return Result.success(DataSourceConfig(endpoint=str(endpoint)))

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        return Result.success({"payload": dict(payload)})

    def cleanup(self) -> Result[None]:
        return Result.success(data=None)


class DisabledScannerPlugin:
    """Plugin registered but disabled by default to test filtering."""

    metadata = PluginMetadata(
        name="disabled_scanner",
        version="0.1.0",
        category=PluginCategory.SCANNER,
        enabled=False,
    )

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config = config or {}

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[ScannerConfig]:
        lookback = int(config.get("lookback_days", 30))
        return Result.success(ScannerConfig(lookback_days=lookback))

    def execute(self, payload: Mapping[str, Any]) -> Result[list[Mapping[str, Any]]]:
        return Result.success([])

    def cleanup(self) -> Result[None]:
        return Result.success(data=None)


class NoMetadataPlugin:
    """Plugin definition missing metadata for registration validation tests."""


class MismatchedMetadataPlugin:
    """Plugin definition with metadata name mismatch."""

    metadata = PluginMetadata(
        name="mismatched",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
    )


class NoConfigInitPlugin:
    """Plugin whose __init__ does not accept a config keyword."""

    metadata = PluginMetadata(
        name="no_config_init",
        version="1.0.0",
        category=PluginCategory.SIGNAL,
    )

    def __init__(self) -> None:
        self.config: Mapping[str, Any] | ScannerConfig | None = None

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result[ScannerConfig]:
        return Result.success(ScannerConfig(lookback_days=int(config.get("lookback_days", 5))))

    def execute(self, payload: Mapping[str, Any]) -> Result[list[Mapping[str, Any]]]:
        return Result.success([])

    def cleanup(self) -> Result[None]:
        return Result.success(data=None)


class PermissionInitPlugin:
    """Plugin whose construction raises PermissionError."""

    metadata = PluginMetadata(
        name="permission_init",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
    )

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        raise PermissionError("init denied")


class CrashInitPlugin:
    """Plugin whose construction raises a generic exception."""

    metadata = PluginMetadata(
        name="crash_init",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
    )

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        raise RuntimeError("init crashed")


class ValidationPermissionPlugin(DummyDataSourcePlugin):
    """Plugin whose validation raises PermissionError."""

    metadata = PluginMetadata(
        name="validation_permission",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
    )

    def validate_config(self, config: Mapping[str, Any]) -> Result[DataSourceConfig]:
        raise PermissionError("validation denied")


class ValidationErrorPlugin(DummyDataSourcePlugin):
    """Plugin whose validation raises ValidationError."""

    metadata = PluginMetadata(
        name="validation_error",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
    )

    def validate_config(self, config: Mapping[str, Any]) -> Result[DataSourceConfig]:
        raise ValidationError(
            "validation error",
            module="tests.registry",
            reason_code="TEST_VALIDATION_ERROR",
        )


class ValidationGenericExceptionPlugin(DummyDataSourcePlugin):
    """Plugin whose validation raises a generic exception."""

    metadata = PluginMetadata(
        name="validation_generic",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
    )

    def validate_config(self, config: Mapping[str, Any]) -> Result[DataSourceConfig]:
        raise RuntimeError("validation crashed")


class ValidationFailureNoErrorPlugin(DummyDataSourcePlugin):
    """Plugin returning a failed validation result without error payload."""

    metadata = PluginMetadata(
        name="validation_no_error",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
        enabled=True,
    )

    def validate_config(self, config: Mapping[str, Any]) -> Result[DataSourceConfig]:
        return Result(status=ResultStatus.FAILED)

@pytest.fixture
def registry() -> PluginRegistry:
    reg = PluginRegistry()
    reg.register_plugin("dummy_data_source", DummyDataSourcePlugin)
    reg.register_plugin("disabled_scanner", DisabledScannerPlugin)
    return reg


def test_register_plugin_duplicate(registry: PluginRegistry) -> None:
    with pytest.raises(ConfigurationError):
        registry.register_plugin("dummy_data_source", DummyDataSourcePlugin)


def test_list_plugins_filters_enabled(registry: PluginRegistry) -> None:
    all_plugins = registry.list_plugins(enabled_only=False)
    assert len(all_plugins) == 2

    enabled_only = registry.list_plugins()
    assert all(metadata.enabled for metadata in enabled_only)

    scanners = registry.list_plugins(category=PluginCategory.SCANNER, enabled_only=False)
    assert len(scanners) == 1
    assert scanners[0].name == "disabled_scanner"


def test_load_from_config_success(registry: PluginRegistry) -> None:
    config = {
        "plugins": {
            "data_sources": [
                {"name": "dummy_data_source", "enabled": True, "config": {"endpoint": "https://x"}}
            ],
            "scanners": [{"name": "disabled_scanner", "enabled": False, "config": {}}],
            "signals": [],
            "strategies": [],
            "risk_policies": [],
        }
    }

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.SUCCESS
    assert "dummy_data_source" in result.data
    instance = result.data["dummy_data_source"]
    assert isinstance(instance.config, DataSourceConfig)
    assert instance.config.endpoint == "https://x"
    assert "disabled_scanner" not in result.data


def test_load_from_config_unknown_plugin(registry: PluginRegistry) -> None:
    config = {
        "plugins": {
            "data_sources": [{"name": "unregistered", "enabled": True, "config": {}}],
        }
    }

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == registry._UNKNOWN_PLUGIN_REASON  # type: ignore[attr-defined]


def test_register_plugin_missing_metadata() -> None:
    registry = PluginRegistry()

    with pytest.raises(ConfigurationError) as excinfo:
        registry.register_plugin("missing", NoMetadataPlugin)  # type: ignore[arg-type]

    assert registry._INVALID_PLUGIN_REASON in str(excinfo.value)  # type: ignore[attr-defined]


def test_register_plugin_metadata_name_mismatch() -> None:
    registry = PluginRegistry()

    with pytest.raises(ConfigurationError):
        registry.register_plugin("something_else", MismatchedMetadataPlugin)


def test_get_plugin_unknown_raises(registry: PluginRegistry) -> None:
    with pytest.raises(ConfigurationError):
        registry.get_plugin("nonexistent")


def test_load_from_config_missing_plugins_section(registry: PluginRegistry) -> None:
    result = registry.load_from_config({})

    assert result.status is ResultStatus.SUCCESS
    assert result.data == {}


def test_load_from_config_plugins_not_mapping(registry: PluginRegistry) -> None:
    result = registry.load_from_config({"plugins": []})

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == registry._CONFIG_INVALID_REASON  # type: ignore[attr-defined]


def test_load_from_config_unknown_category(registry: PluginRegistry) -> None:
    result = registry.load_from_config({"plugins": {"unknown": []}})

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == registry._CONFIG_INVALID_REASON  # type: ignore[attr-defined]


def test_load_from_config_category_not_iterable(registry: PluginRegistry) -> None:
    result = registry.load_from_config({"plugins": {"data_sources": 123}})

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == registry._CONFIG_INVALID_REASON  # type: ignore[attr-defined]


def test_load_from_config_entry_not_mapping(registry: PluginRegistry) -> None:
    result = registry.load_from_config({"plugins": {"data_sources": ["not-mapping"]}})

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == registry._CONFIG_INVALID_REASON  # type: ignore[attr-defined]


def test_load_from_config_entry_missing_name(registry: PluginRegistry) -> None:
    result = registry.load_from_config({"plugins": {"data_sources": [{"enabled": True}]}})

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == registry._CONFIG_INVALID_REASON  # type: ignore[attr-defined]


def test_load_from_config_category_mismatch(registry: PluginRegistry) -> None:
    config = {"plugins": {"signals": [{"name": "dummy_data_source", "enabled": True, "config": {}}]}}

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == registry._CONFIG_INVALID_REASON  # type: ignore[attr-defined]


def test_load_from_config_typeerror_fallback(registry: PluginRegistry) -> None:
    registry.register_plugin("no_config_init", NoConfigInitPlugin)
    config = {
        "plugins": {
            "signals": [{"name": "no_config_init", "enabled": True, "config": {"lookback_days": 10}}],
        }
    }

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.SUCCESS
    instance = result.data["no_config_init"]
    assert isinstance(instance.config, ScannerConfig)
    assert instance.config.lookback_days == 10


def test_load_from_config_instantiation_permission_error(registry: PluginRegistry) -> None:
    registry.register_plugin("permission_init", PermissionInitPlugin)
    config = {
        "plugins": {
            "data_sources": [{"name": "permission_init", "enabled": True, "config": {}}],
        }
    }

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, CriticalError)


def test_load_from_config_instantiation_failure(registry: PluginRegistry) -> None:
    registry.register_plugin("crash_init", CrashInitPlugin)
    config = {
        "plugins": {
            "data_sources": [{"name": "crash_init", "enabled": True, "config": {}}],
        }
    }

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ConfigurationError)


def test_load_from_config_validation_permission_error(registry: PluginRegistry) -> None:
    registry.register_plugin("validation_permission", ValidationPermissionPlugin)
    config = {
        "plugins": {
            "data_sources": [{"name": "validation_permission", "enabled": True, "config": {"endpoint": "x"}}],
        }
    }

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, CriticalError)


def test_load_from_config_validation_error_wrapped(registry: PluginRegistry) -> None:
    registry.register_plugin("validation_error", ValidationErrorPlugin)
    config = {
        "plugins": {
            "data_sources": [{"name": "validation_error", "enabled": True, "config": {"endpoint": "x"}}],
        }
    }

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ValidationError)


def test_load_from_config_validation_generic_exception(registry: PluginRegistry) -> None:
    registry.register_plugin("validation_generic", ValidationGenericExceptionPlugin)
    config = {
        "plugins": {
            "data_sources": [{"name": "validation_generic", "enabled": True, "config": {"endpoint": "x"}}],
        }
    }

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ValidationError)


def test_load_from_config_validation_failure_without_error(registry: PluginRegistry) -> None:
    registry.register_plugin("validation_no_error", ValidationFailureNoErrorPlugin)
    config = {
        "plugins": {
            "data_sources": [{"name": "validation_no_error", "enabled": True, "config": {"endpoint": "x"}}],
        }
    }

    result = registry.load_from_config(config)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ValidationError)
    assert result.reason_code == registry._CONFIG_INVALID_REASON  # type: ignore[attr-defined]
