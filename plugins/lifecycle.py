"""Plugin lifecycle management utilities."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, MutableMapping

from common.exceptions import CriticalError, SystemStateError, ValidationError
from common.interface import Result, ResultStatus

from .interface import PluginBase, PluginContext

__all__ = ["PluginLifecycleManager", "PluginState"]


class PluginState(str, Enum):
    """Discrete lifecycle phases for plugin instances."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    EXECUTING = "executing"
    FAILED = "failed"
    CLEANED_UP = "cleaned_up"


class PluginLifecycleManager:
    """Coordinate plugin lifecycle transitions with structured state tracking."""

    _MODULE_NAME = "plugins.lifecycle"
    _INVALID_STATE_REASON = "PLUGIN_INVALID_STATE"
    _INIT_FAILED_REASON = "PLUGIN_INIT_FAILED"
    _EXECUTE_FAILED_REASON = "PLUGIN_EXECUTE_FAILED"
    _CLEANUP_FAILED_REASON = "PLUGIN_CLEANUP_FAILED"
    _VALIDATION_FAILED_REASON = "PLUGIN_VALIDATION_FAILED"

    def __init__(self) -> None:
        self._states: MutableMapping[int, PluginState] = {}

    def get_state(self, plugin: PluginBase[Any, Any, Any]) -> PluginState:
        """Return the tracked lifecycle state for ``plugin``."""

        return self._states.get(id(plugin), PluginState.UNINITIALIZED)

    def init_plugin(
        self,
        plugin: PluginBase[Any, Any, Any],
        context: PluginContext | None = None,
    ) -> Result[None]:
        """Validate configuration and initialise ``plugin``."""

        plugin_id = id(plugin)
        current_state = self.get_state(plugin)
        if current_state not in {PluginState.UNINITIALIZED, PluginState.CLEANED_UP}:
            error = SystemStateError(
                f"Cannot initialise plugin from state {current_state.value!r}.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_STATE_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        config_payload = None
        if context is not None and isinstance(context, Mapping):
            config_payload = context.get("config")

        if config_payload is not None:
            validation_result = self._run_with_validation(plugin, config_payload)
            if validation_result.status is not ResultStatus.SUCCESS:
                self._states[plugin_id] = PluginState.FAILED
                return validation_result

        try:
            init_result = plugin.init(context)
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._INIT_FAILED_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            self._states[plugin_id] = PluginState.FAILED
            return Result.failed(error, error.reason_code)
        except Exception as exc:  # noqa: BLE001 - convert to CriticalError.
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._INIT_FAILED_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            self._states[plugin_id] = PluginState.FAILED
            return Result.failed(error, error.reason_code)

        if init_result.status is ResultStatus.SUCCESS:
            self._states[plugin_id] = PluginState.READY
            return init_result

        if init_result.status is ResultStatus.DEGRADED:
            self._states[plugin_id] = PluginState.READY
            return init_result

        self._states[plugin_id] = PluginState.FAILED
        return init_result

    def execute_plugin(
        self,
        plugin: PluginBase[Any, Any, Any],
        input_data: Any,
    ) -> Result[Any]:
        """Execute ``plugin`` with ``input_data`` ensuring valid state transitions."""

        plugin_id = id(plugin)
        current_state = self.get_state(plugin)
        if current_state is not PluginState.READY:
            error = SystemStateError(
                f"Cannot execute plugin from state {current_state.value!r}.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_STATE_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        self._states[plugin_id] = PluginState.EXECUTING

        try:
            execute_result = plugin.execute(input_data)
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._EXECUTE_FAILED_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            self._states[plugin_id] = PluginState.FAILED
            return Result.failed(error, error.reason_code)
        except Exception as exc:  # noqa: BLE001 - convert to CriticalError.
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._EXECUTE_FAILED_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            self._states[plugin_id] = PluginState.FAILED
            return Result.failed(error, error.reason_code)

        if execute_result.status is ResultStatus.SUCCESS:
            self._states[plugin_id] = PluginState.READY
            return execute_result

        if execute_result.status is ResultStatus.DEGRADED:
            self._states[plugin_id] = PluginState.READY
            return execute_result

        self._states[plugin_id] = PluginState.FAILED
        return execute_result

    def cleanup_plugin(self, plugin: PluginBase[Any, Any, Any]) -> Result[None]:
        """Trigger ``cleanup`` for ``plugin`` respecting state transitions."""

        plugin_id = id(plugin)
        current_state = self.get_state(plugin)
        if current_state is PluginState.UNINITIALIZED:
            error = SystemStateError(
                "Cannot cleanup plugin that was never initialised.",
                module=self._MODULE_NAME,
                reason_code=self._INVALID_STATE_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            return error.to_result(data=None, status=ResultStatus.FAILED)

        try:
            cleanup_result = plugin.cleanup()
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CLEANUP_FAILED_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            self._states[plugin_id] = PluginState.FAILED
            return Result.failed(error, error.reason_code)
        except Exception as exc:  # noqa: BLE001 - normalise into critical failure.
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._CLEANUP_FAILED_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            self._states[plugin_id] = PluginState.FAILED
            return Result.failed(error, error.reason_code)

        if cleanup_result.status is ResultStatus.SUCCESS:
            self._states[plugin_id] = PluginState.CLEANED_UP
            return cleanup_result

        if cleanup_result.status is ResultStatus.DEGRADED:
            self._states[plugin_id] = PluginState.CLEANED_UP
            return cleanup_result

        self._states[plugin_id] = PluginState.FAILED
        return cleanup_result

    def _run_with_validation(
        self,
        plugin: PluginBase[Any, Any, Any],
        config_payload: Any,
    ) -> Result[None]:
        """Helper applying configuration validation with fail-closed semantics."""

        try:
            validation_result = plugin.validate_config(config_payload)
        except PermissionError as exc:
            error = CriticalError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._VALIDATION_FAILED_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            return Result.failed(error, error.reason_code)
        except ValidationError as exc:
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._VALIDATION_FAILED_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            return Result.failed(error, error.reason_code)
        except Exception as exc:  # noqa: BLE001 - normalise to validation failure.
            error = ValidationError.from_error(
                exc,
                module=self._MODULE_NAME,
                reason_code=self._VALIDATION_FAILED_REASON,
                details={"plugin": getattr(plugin, "metadata", None)},
            )
            return Result.failed(error, error.reason_code)

        if validation_result.status is ResultStatus.SUCCESS:
            return Result.success(data=None)

        error = validation_result.error or ValidationError(
            "Plugin configuration validation failed.",
            module=self._MODULE_NAME,
            reason_code=self._VALIDATION_FAILED_REASON,
            details={"plugin": getattr(plugin, "metadata", None)},
        )
        reason = validation_result.reason_code or self._VALIDATION_FAILED_REASON
        return Result.failed(error, reason)

