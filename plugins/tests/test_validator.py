from __future__ import annotations

from typing import Any, Mapping

import msgspec
import pytest
from pydantic import BaseModel

from common.interface import Result, ResultStatus
from plugins.interface import PluginCategory, PluginMetadata
from plugins.validator import PluginOutputValidator


class ValidatorTestPlugin:
    """Minimal plugin implementation for validator tests."""

    metadata = PluginMetadata(
        name="validator_test_plugin",
        version="1.0.0",
        category=PluginCategory.DATA_SOURCE,
    )

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config = config or {}

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        return Result.success(data=None)

    def validate_config(self, config: Mapping[str, Any]) -> Result["EmptyConfig"]:
        return Result.success(EmptyConfig())

    def execute(self, payload: Mapping[str, Any]) -> Result[Mapping[str, Any]]:
        return Result.success(payload)

    def cleanup(self) -> Result[None]:
        return Result.success(data=None)


class OutputSchema(msgspec.Struct, kw_only=True, frozen=True):
    """Simple msgspec schema used by validation tests."""

    value: int


class PydanticSchema(BaseModel):
    """Pydantic schema used by validation tests."""

    status: str


class EmptyConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Placeholder config struct for validator plugin tests."""


@pytest.fixture
def validator() -> PluginOutputValidator:
    return PluginOutputValidator()


@pytest.fixture
def plugin() -> ValidatorTestPlugin:
    return ValidatorTestPlugin()


def test_validate_output_msgspec_success(
    validator: PluginOutputValidator,
    plugin: ValidatorTestPlugin,
) -> None:
    result = validator.validate_output(plugin, {"value": 42}, OutputSchema)

    assert result.status is ResultStatus.SUCCESS
    assert isinstance(result.data, OutputSchema)
    assert result.data.value == 42


def test_validate_output_pydantic_success(
    validator: PluginOutputValidator,
    plugin: ValidatorTestPlugin,
) -> None:
    result = validator.validate_output(plugin, {"status": "ok"}, PydanticSchema)

    assert result.status is ResultStatus.SUCCESS
    assert isinstance(result.data, PydanticSchema)
    assert result.data.status == "ok"


def test_validate_output_invalid_schema_type(
    validator: PluginOutputValidator,
    plugin: ValidatorTestPlugin,
) -> None:
    result = validator.validate_output(plugin, {"value": 1}, schema_class=int)  # type: ignore[arg-type]

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._INVALID_SCHEMA_REASON  # type: ignore[attr-defined]


def test_validate_output_schema_not_type(
    validator: PluginOutputValidator,
    plugin: ValidatorTestPlugin,
) -> None:
    schema_instance = OutputSchema(value=5)

    result = validator.validate_output(plugin, {"value": 5}, schema_class=schema_instance)  # type: ignore[arg-type]

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._INVALID_SCHEMA_REASON  # type: ignore[attr-defined]


def test_validate_output_failure(
    validator: PluginOutputValidator,
    plugin: ValidatorTestPlugin,
) -> None:
    result = validator.validate_output(plugin, {"value": "not-int"}, OutputSchema)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._VALIDATION_FAILED_REASON  # type: ignore[attr-defined]


def test_validate_output_pydantic_failure(
    validator: PluginOutputValidator,
    plugin: ValidatorTestPlugin,
) -> None:
    result = validator.validate_output(plugin, {"status": 123}, PydanticSchema)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == validator._VALIDATION_FAILED_REASON  # type: ignore[attr-defined]
