"""Plugin output validation utilities enforcing the fail-closed principle."""

from __future__ import annotations

from typing import Any, Type, TypeVar

import msgspec

from common.exceptions import ValidationError
from common.interface import Result, ResultStatus

from .interface import PluginBase

try:  # pragma: no cover - pydantic optional at runtime.
    from pydantic import BaseModel, ValidationError as PydanticValidationError
except Exception:  # noqa: BLE001 - runtime optional dependency.
    BaseModel = None  # type: ignore[assignment]
    PydanticValidationError = None  # type: ignore[assignment]

T = TypeVar("T")

__all__ = ["PluginOutputValidator"]


class PluginOutputValidator:
    """Validate plugin outputs against structured schemas."""

    _MODULE_NAME = "plugins.validator"
    _INVALID_SCHEMA_REASON = "PLUGIN_OUTPUT_SCHEMA_INVALID"
    _VALIDATION_FAILED_REASON = "PLUGIN_OUTPUT_VALIDATION_FAILED"

    def validate_output(
        self,
        plugin: PluginBase[Any, Any, Any],
        output_data: Any,
        schema_class: Type[T],
    ) -> Result[T]:
        """Validate ``output_data`` using ``schema_class`` for ``plugin``."""

        metadata = getattr(plugin, "metadata", None)
        plugin_name = metadata.name if metadata else type(plugin).__name__

        if not isinstance(schema_class, type):
            error = ValidationError(
                "schema_class must be a type.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_SCHEMA_REASON,
                details={"plugin": plugin_name, "schema_class": repr(schema_class)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        if issubclass(schema_class, msgspec.Struct):
            return self._validate_with_msgspec(plugin_name, output_data, schema_class)

        if BaseModel is not None and issubclass(schema_class, BaseModel):
            return self._validate_with_pydantic(plugin_name, output_data, schema_class)

        error = ValidationError(
            "Unsupported schema_class; expected msgspec.Struct or pydantic.BaseModel.",
            module=self._MODULE_NAME,
            reason_code=self._INVALID_SCHEMA_REASON,
            details={"plugin": plugin_name, "schema_class": schema_class.__name__},
        )
        return error.to_result(data=None, status=ResultStatus.FAILED)

    def _validate_with_msgspec(
        self,
        plugin_name: str,
        output_data: Any,
        schema_class: Type[T],
    ) -> Result[T]:
        """Validate ``output_data`` using a msgspec.Struct schema."""

        try:
            validated = msgspec.convert(output_data, type=schema_class)
        except (msgspec.ValidationError, TypeError, ValueError) as exc:
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._VALIDATION_FAILED_REASON,
                details={"plugin": plugin_name, "schema_class": schema_class.__name__},
            )
            return Result.failed(error, error.reason_code)

        return Result.success(validated)

    def _validate_with_pydantic(
        self,
        plugin_name: str,
        output_data: Any,
        schema_class: Type[T],
    ) -> Result[T]:
        """Validate ``output_data`` using a pydantic BaseModel schema."""

        if BaseModel is None or PydanticValidationError is None:  # pragma: no cover - defensive.
            error = ValidationError(
                "Pydantic is not available in this runtime.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_SCHEMA_REASON,
                details={"plugin": plugin_name},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        try:
            validated = schema_class.model_validate(output_data)  # type: ignore[call-arg]
        except PydanticValidationError as exc:
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._VALIDATION_FAILED_REASON,
                details={"plugin": plugin_name, "schema_class": schema_class.__name__},
            )
            return Result.failed(error, error.reason_code)

        return Result.success(validated)

