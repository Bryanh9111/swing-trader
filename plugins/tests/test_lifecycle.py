from __future__ import annotations

from typing import Any, Mapping

import msgspec
import pytest

from common.exceptions import CriticalError, ValidationError
from common.interface import Result, ResultStatus
from plugins.interface import PluginCategory, PluginMetadata
from plugins.lifecycle import PluginLifecycleManager, PluginState


class LifecycleConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Configuration struct used by lifecycle plugin tests."""

    value: int


class LifecyclePlugin:
    """Plugin with controllable behaviours for lifecycle tests."""

    metadata = PluginMetadata(
        name="lifecycle_plugin",
        version="1.0.0",
        category=PluginCategory.STRATEGY,
    )

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        fail_validation: bool = False,
        fail_validation_missing_error: bool = False,
        validation_permission: bool = False,
        validation_exception: bool = False,
        raise_validation_error: bool = False,
        fail_init: bool = False,
        degrade_init: bool = False,
        raise_init_permission: bool = False,
        raise_init_exception: bool = False,
        fail_execute: bool = False,
        degrade_execute: bool = False,
        raise_permission: bool = False,
        raise_execute_exception: bool = False,
        fail_cleanup: bool = False,
        degrade_cleanup: bool = False,
        raise_cleanup_permission: bool = False,
        raise_cleanup_exception: bool = False,
    ) -> None:
        self.config = config or {}
        self.fail_validation = fail_validation
        self.fail_validation_missing_error = fail_validation_missing_error
        self.validation_permission = validation_permission
        self.validation_exception = validation_exception
        self.raise_validation_error = raise_validation_error
        self.fail_init = fail_init
        self.degrade_init = degrade_init
        self.raise_init_permission = raise_init_permission
        self.raise_init_exception = raise_init_exception
        self.fail_execute = fail_execute
        self.degrade_execute = degrade_execute
        self.raise_permission = raise_permission
        self.raise_execute_exception = raise_execute_exception
        self.fail_cleanup = fail_cleanup
        self.degrade_cleanup = degrade_cleanup
        self.raise_cleanup_permission = raise_cleanup_permission
        self.raise_cleanup_exception = raise_cleanup_exception
        self.cleaned = False

    def validate_config(self, config: Mapping[str, Any]) -> Result[LifecycleConfig]:
        if self.validation_permission:
            raise PermissionError("validation denied")
        if self.raise_validation_error:
            raise ValidationError(
                "Validation error requested for lifecycle plugin.",
                module="tests.lifecycle",
                reason_code="TEST_VALIDATION_EXCEPTION",
            )
        if self.validation_exception:
            raise RuntimeError("validation crash")
        if self.fail_validation:
            error = ValidationError(
                "Invalid configuration for lifecycle plugin.",
                module="tests.lifecycle",
                reason_code="TEST_INVALID_CONFIG",
            )
            return Result.failed(error, error.reason_code)
        if self.fail_validation_missing_error:
            return Result(status=ResultStatus.FAILED)
        if isinstance(config, Mapping) and "value" in config:
            return Result.success(LifecycleConfig(value=int(config["value"])))
        return Result.success(LifecycleConfig(value=1))

    def init(self, context: Mapping[str, Any] | None = None) -> Result[None]:
        if self.raise_init_permission:
            raise PermissionError("init denied")
        if self.raise_init_exception:
            raise RuntimeError("init failed unexpectedly")
        if self.fail_init:
            error = CriticalError(
                "Initialisation failure requested for test.",
                module="tests.lifecycle",
                reason_code="TEST_INIT_FAILED",
            )
            return Result.failed(error, error.reason_code)
        if self.degrade_init:
            error = CriticalError(
                "Initialisation degraded for test.",
                module="tests.lifecycle",
                reason_code="TEST_INIT_DEGRADED",
            )
            return Result.degraded(data=None, error=error, reason_code=error.reason_code)
        return Result.success(data=None)

    def execute(self, payload: Mapping[str, Any]) -> Result[dict[str, Any]]:
        if self.raise_permission:
            raise PermissionError("permission denied")
        if self.raise_execute_exception:
            raise RuntimeError("execute crashed")
        if self.fail_execute:
            error = CriticalError(
                "Execution failure requested for test.",
                module="tests.lifecycle",
                reason_code="TEST_EXECUTE_FAILED",
            )
            return Result.failed(error, error.reason_code)
        if self.degrade_execute:
            error = CriticalError(
                "Execution degraded for test.",
                module="tests.lifecycle",
                reason_code="TEST_EXECUTE_DEGRADED",
            )
            return Result.degraded(
                data={"payload": dict(payload), "status": "degraded"},
                error=error,
                reason_code=error.reason_code,
            )
        return Result.success({"payload": dict(payload), "status": "ok"})

    def cleanup(self) -> Result[None]:
        if self.raise_cleanup_permission:
            raise PermissionError("cleanup denied")
        if self.raise_cleanup_exception:
            raise RuntimeError("cleanup crashed")
        if self.fail_cleanup:
            error = CriticalError(
                "Cleanup failure requested for test.",
                module="tests.lifecycle",
                reason_code="TEST_CLEANUP_FAILED",
            )
            return Result.failed(error, error.reason_code)
        if self.degrade_cleanup:
            self.cleaned = True
            error = CriticalError(
                "Cleanup degraded for test.",
                module="tests.lifecycle",
                reason_code="TEST_CLEANUP_DEGRADED",
            )
            return Result.degraded(data=None, error=error, reason_code=error.reason_code)
        self.cleaned = True
        return Result.success(data=None)


@pytest.fixture
def manager() -> PluginLifecycleManager:
    return PluginLifecycleManager()


def test_lifecycle_happy_path(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(config={"value": 5})

    init_result = manager.init_plugin(plugin, context={"config": {"value": 5}})
    assert init_result.status is ResultStatus.SUCCESS
    assert manager.get_state(plugin) is PluginState.READY

    exec_result = manager.execute_plugin(plugin, {"foo": "bar"})
    assert exec_result.status is ResultStatus.SUCCESS
    assert exec_result.data["status"] == "ok"  # type: ignore[index]
    assert manager.get_state(plugin) is PluginState.READY

    cleanup_result = manager.cleanup_plugin(plugin)
    assert cleanup_result.status is ResultStatus.SUCCESS
    assert manager.get_state(plugin) is PluginState.CLEANED_UP
    assert plugin.cleaned is True


def test_execute_without_initialisation(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin()

    exec_result = manager.execute_plugin(plugin, {"foo": "bar"})

    assert exec_result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.UNINITIALIZED


def test_validation_failure_sets_failed_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(fail_validation=True)

    init_result = manager.init_plugin(plugin, context={"config": {"value": 2}})

    assert init_result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_permission_error_during_execute(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(raise_permission=True)
    manager.init_plugin(plugin, context={"config": {"value": 1}})

    exec_result = manager.execute_plugin(plugin, {"foo": "bar"})

    assert exec_result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_init_invalid_state_transition(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin()
    first = manager.init_plugin(plugin)
    assert first.status is ResultStatus.SUCCESS
    assert manager.get_state(plugin) is PluginState.READY

    second = manager.init_plugin(plugin)
    assert second.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.READY


def test_init_permission_error_sets_failed_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(raise_init_permission=True)

    result = manager.init_plugin(plugin)

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_init_exception_sets_failed_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(raise_init_exception=True)

    result = manager.init_plugin(plugin)

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_init_degraded_state_keeps_ready(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(degrade_init=True)

    result = manager.init_plugin(plugin)

    assert result.status is ResultStatus.DEGRADED
    assert manager.get_state(plugin) is PluginState.READY


def test_init_failed_result_sets_failed_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(fail_init=True)

    result = manager.init_plugin(plugin)

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_validation_permission_error_wrapped(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(validation_permission=True)

    result = manager.init_plugin(plugin, context={"config": {"value": 3}})

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_validation_error_wrapped(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(raise_validation_error=True)

    result = manager.init_plugin(plugin, context={"config": {"value": 1}})

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_validation_generic_exception_wrapped(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(validation_exception=True)

    result = manager.init_plugin(plugin, context={"config": {"value": 1}})

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_validation_failure_without_error_uses_default(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(fail_validation_missing_error=True)

    result = manager.init_plugin(plugin, context={"config": {"value": 1}})

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == manager._VALIDATION_FAILED_REASON  # type: ignore[attr-defined]
    assert isinstance(result.error, ValidationError)
    assert manager.get_state(plugin) is PluginState.FAILED


def test_execute_degraded_result_resets_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(degrade_execute=True)
    manager.init_plugin(plugin, context={"config": {"value": 1}})

    result = manager.execute_plugin(plugin, {"foo": "bar"})

    assert result.status is ResultStatus.DEGRADED
    assert manager.get_state(plugin) is PluginState.READY


def test_execute_failure_result_sets_failed_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(fail_execute=True)
    manager.init_plugin(plugin, context={"config": {"value": 1}})

    result = manager.execute_plugin(plugin, {"foo": "bar"})

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_execute_exception_sets_failed_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(raise_execute_exception=True)
    manager.init_plugin(plugin, context={"config": {"value": 1}})

    result = manager.execute_plugin(plugin, {"foo": "bar"})

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_execute_after_failure_rejected(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(fail_execute=True)
    manager.init_plugin(plugin, context={"config": {"value": 1}})
    manager.execute_plugin(plugin, {"foo": "bar"})

    repeat = manager.execute_plugin(plugin, {"foo": "bar"})

    assert repeat.status is ResultStatus.FAILED
    assert repeat.reason_code == manager._INVALID_STATE_REASON  # type: ignore[attr-defined]


def test_cleanup_without_initialisation(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin()

    result = manager.cleanup_plugin(plugin)

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.UNINITIALIZED


def test_cleanup_permission_error_sets_failed_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(raise_cleanup_permission=True)
    manager.init_plugin(plugin, context={"config": {"value": 4}})

    result = manager.cleanup_plugin(plugin)

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_cleanup_exception_sets_failed_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(raise_cleanup_exception=True)
    manager.init_plugin(plugin, context={"config": {"value": 4}})

    result = manager.cleanup_plugin(plugin)

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_cleanup_degraded_marks_cleaned(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(degrade_cleanup=True)
    manager.init_plugin(plugin, context={"config": {"value": 2}})

    result = manager.cleanup_plugin(plugin)

    assert result.status is ResultStatus.DEGRADED
    assert manager.get_state(plugin) is PluginState.CLEANED_UP
    assert plugin.cleaned is True


def test_cleanup_failed_result_sets_failed_state(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin(fail_cleanup=True)
    manager.init_plugin(plugin, context={"config": {"value": 2}})

    result = manager.cleanup_plugin(plugin)

    assert result.status is ResultStatus.FAILED
    assert manager.get_state(plugin) is PluginState.FAILED


def test_reinitialise_after_cleanup(manager: PluginLifecycleManager) -> None:
    plugin = LifecyclePlugin()

    first = manager.init_plugin(plugin)
    assert first.status is ResultStatus.SUCCESS

    cleanup = manager.cleanup_plugin(plugin)
    assert cleanup.status is ResultStatus.SUCCESS
    assert manager.get_state(plugin) is PluginState.CLEANED_UP

    second = manager.init_plugin(plugin)
    assert second.status is ResultStatus.SUCCESS
    assert manager.get_state(plugin) is PluginState.READY
