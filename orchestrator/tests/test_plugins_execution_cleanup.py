"""Tests for ExecutionPlugin cleanup functionality."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from common.interface import Result, ResultStatus
from orchestrator.plugins import ExecutionPlugin, ExecutionPluginConfig


class TestExecutionPluginCleanup:
    """Test suite for ExecutionPlugin cleanup with cleanup_strategy."""

    @staticmethod
    def _make_plugin(*, monkeypatch: pytest.MonkeyPatch, config: dict[str, object]) -> ExecutionPlugin:
        plugin = ExecutionPlugin(config=config)
        monkeypatch.setattr(ExecutionPlugin, "_build_manager", lambda *_args, **_kwargs: Result.success(Mock()))
        plugin.validate_config(config)
        return plugin

    @staticmethod
    def _capture_events(plugin: ExecutionPlugin) -> list[tuple[str, str, dict[str, object] | None]]:
        emit_calls: list[tuple[str, str, dict[str, object] | None]] = []

        def capture_emit(event_type: str, module_name: str, data: dict[str, object] | None) -> None:
            emit_calls.append((event_type, module_name, data))

        plugin._emit_event = capture_emit  # noqa: SLF001 - capture emitted events
        return emit_calls

    def test_cleanup_strategy_valid_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_config() accepts valid cleanup_strategy values."""

        for strategy in ("keep_all", "cancel_open_only", "cancel_all"):
            plugin = ExecutionPlugin(config={"enabled": True, "cleanup_strategy": strategy})
            monkeypatch.setattr(ExecutionPlugin, "_build_manager", lambda *_args, **_kwargs: Result.success(Mock()))

            result = plugin.validate_config({"enabled": True, "cleanup_strategy": strategy})

            assert result.status is ResultStatus.SUCCESS
            assert result.data is not None
            assert result.data.cleanup_strategy == strategy

    def test_cleanup_strategy_invalid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_config() rejects invalid cleanup_strategy values."""

        plugin = ExecutionPlugin(config={"enabled": True, "cleanup_strategy": "invalid_strategy"})
        monkeypatch.setattr(ExecutionPlugin, "_build_manager", lambda *_args, **_kwargs: Result.success(Mock()))

        result = plugin.validate_config({"enabled": True, "cleanup_strategy": "invalid_strategy"})

        assert result.status is ResultStatus.FAILED
        assert result.reason_code == "EXECUTION_PLUGIN_CONFIG_INVALID"

    def test_cleanup_strategy_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_config() defaults cleanup_strategy to keep_all."""

        plugin = ExecutionPlugin(config={"enabled": True})
        monkeypatch.setattr(ExecutionPlugin, "_build_manager", lambda *_args, **_kwargs: Result.success(Mock()))

        result = plugin.validate_config({"enabled": True})

        assert result.status is ResultStatus.SUCCESS
        assert result.data is not None
        assert result.data.cleanup_strategy == "keep_all"

    def test_cleanup_keep_all_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup() with keep_all does not cancel orders and still disconnects."""

        plugin = self._make_plugin(monkeypatch=monkeypatch, config={"enabled": True, "cleanup_strategy": "keep_all"})
        plugin._logger = Mock()  # noqa: SLF001 - silence structured logging calls

        mock_manager = Mock()
        mock_adapter = Mock()
        plugin._manager = mock_manager  # noqa: SLF001 - inject stub manager
        plugin._adapter = mock_adapter  # noqa: SLF001 - inject stub adapter
        emit_calls = self._capture_events(plugin)

        result = plugin.cleanup()

        assert result.status is ResultStatus.SUCCESS
        mock_manager.cancel_all_open_orders.assert_not_called()
        mock_adapter.disconnect.assert_called_once()
        assert any(event_type == "module.cleaned_up" for event_type, _module, _data in emit_calls)

    def test_cleanup_cancel_open_only_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup() with cancel_open_only cancels only OPEN_LONG/OPEN_SHORT orders."""

        plugin = self._make_plugin(
            monkeypatch=monkeypatch, config={"enabled": True, "cleanup_strategy": "cancel_open_only"}
        )
        plugin._logger = Mock()  # noqa: SLF001

        mock_manager = Mock()
        mock_manager.cancel_all_open_orders.return_value = Result.success(
            {"total_cancelled": 3, "failed_cancellations": []}
        )
        mock_adapter = Mock()
        plugin._manager = mock_manager  # noqa: SLF001
        plugin._adapter = mock_adapter  # noqa: SLF001
        emit_calls = self._capture_events(plugin)

        result = plugin.cleanup()

        assert result.status is ResultStatus.SUCCESS
        mock_manager.cancel_all_open_orders.assert_called_once_with(intent_types=["OPEN_LONG", "OPEN_SHORT"])

        cleanup_events = [call for call in emit_calls if call[0] == "execution.cleanup"]
        assert len(cleanup_events) == 1
        _event_type, _module_name, payload = cleanup_events[0]
        assert payload is not None
        assert payload["strategy"] == "cancel_open_only"
        assert payload["status"] == "success"

    def test_cleanup_cancel_all_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup() with cancel_all cancels all open orders."""

        plugin = self._make_plugin(monkeypatch=monkeypatch, config={"enabled": True, "cleanup_strategy": "cancel_all"})
        plugin._logger = Mock()  # noqa: SLF001

        mock_manager = Mock()
        mock_manager.cancel_all_open_orders.return_value = Result.success(
            {"total_cancelled": 5, "failed_cancellations": []}
        )
        mock_adapter = Mock()
        plugin._manager = mock_manager  # noqa: SLF001
        plugin._adapter = mock_adapter  # noqa: SLF001
        emit_calls = self._capture_events(plugin)

        result = plugin.cleanup()

        assert result.status is ResultStatus.SUCCESS
        mock_manager.cancel_all_open_orders.assert_called_once_with()

        cleanup_events = [call for call in emit_calls if call[0] == "execution.cleanup"]
        assert len(cleanup_events) == 1
        _event_type, _module_name, payload = cleanup_events[0]
        assert payload is not None
        assert payload["strategy"] == "cancel_all"
        assert payload["status"] == "success"

    def test_cleanup_cancel_failed_continues_disconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup() returns DEGRADED on cancellation failure but still disconnects."""

        plugin = self._make_plugin(monkeypatch=monkeypatch, config={"enabled": True, "cleanup_strategy": "cancel_all"})
        plugin._logger = Mock()  # noqa: SLF001

        mock_manager = Mock()
        mock_manager.cancel_all_open_orders.return_value = Result.failed(RuntimeError("boom"), "CANCEL_FAILED")
        mock_adapter = Mock()
        plugin._manager = mock_manager  # noqa: SLF001
        plugin._adapter = mock_adapter  # noqa: SLF001
        emit_calls = self._capture_events(plugin)

        result = plugin.cleanup()

        assert result.status is ResultStatus.DEGRADED
        mock_adapter.disconnect.assert_called_once()

        cleanup_events = [call for call in emit_calls if call[0] == "execution.cleanup"]
        assert len(cleanup_events) == 1
        _event_type, _module_name, payload = cleanup_events[0]
        assert payload is not None
        assert payload["strategy"] == "cancel_all"
        assert payload["status"] == "failed"

    def test_cleanup_cancel_degraded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup() returns DEGRADED on partial cancellation failures and still disconnects."""

        plugin = self._make_plugin(
            monkeypatch=monkeypatch, config={"enabled": True, "cleanup_strategy": "cancel_open_only"}
        )
        plugin._logger = Mock()  # noqa: SLF001

        mock_manager = Mock()
        mock_manager.cancel_all_open_orders.return_value = Result.degraded(
            {"total_cancelled": 2, "failed_cancellations": ["O-3"]},
            RuntimeError("partial"),
            "PARTIAL_CANCEL",
        )
        mock_adapter = Mock()
        plugin._manager = mock_manager  # noqa: SLF001
        plugin._adapter = mock_adapter  # noqa: SLF001
        emit_calls = self._capture_events(plugin)

        result = plugin.cleanup()

        assert result.status is ResultStatus.DEGRADED
        mock_adapter.disconnect.assert_called_once()

        cleanup_events = [call for call in emit_calls if call[0] == "execution.cleanup"]
        assert len(cleanup_events) == 1
        _event_type, _module_name, payload = cleanup_events[0]
        assert payload is not None
        assert payload["strategy"] == "cancel_open_only"
        assert payload["status"] == "degraded"

    def test_cleanup_cancel_exception_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup() catches cancellation exceptions and returns DEGRADED without raising."""

        plugin = self._make_plugin(monkeypatch=monkeypatch, config={"enabled": True, "cleanup_strategy": "cancel_all"})
        plugin._logger = Mock()  # noqa: SLF001

        mock_manager = Mock()
        mock_manager.cancel_all_open_orders.side_effect = RuntimeError("explode")
        mock_adapter = Mock()
        plugin._manager = mock_manager  # noqa: SLF001
        plugin._adapter = mock_adapter  # noqa: SLF001
        emit_calls = self._capture_events(plugin)

        result = plugin.cleanup()

        assert result.status is ResultStatus.DEGRADED
        mock_adapter.disconnect.assert_called_once()

        cleanup_events = [call for call in emit_calls if call[0] == "execution.cleanup"]
        assert len(cleanup_events) == 1
        _event_type, _module_name, payload = cleanup_events[0]
        assert payload is not None
        assert payload["strategy"] == "cancel_all"
        assert payload["status"] == "exception"

    def test_cleanup_disconnect_exception_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup() catches disconnect exceptions and still emits module.cleaned_up."""

        plugin = ExecutionPlugin(config={"enabled": True})
        plugin._validated = ExecutionPluginConfig(cleanup_strategy="keep_all")  # noqa: SLF001 - avoid manager build
        plugin._logger = Mock()  # noqa: SLF001

        mock_adapter = Mock()
        mock_adapter.disconnect.side_effect = RuntimeError("disconnect boom")
        plugin._adapter = mock_adapter  # noqa: SLF001 - inject stub adapter
        emit_calls = self._capture_events(plugin)

        result = plugin.cleanup()

        assert result.status is ResultStatus.DEGRADED
        assert any(event_type == "module.cleaned_up" for event_type, _module, _data in emit_calls)

    def test_cleanup_emits_cleanup_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cleanup() emits execution.cleanup with strategy + status + cancellation counts."""

        plugin = self._make_plugin(monkeypatch=monkeypatch, config={"enabled": True, "cleanup_strategy": "cancel_all"})
        plugin._logger = Mock()  # noqa: SLF001

        mock_manager = Mock()
        mock_manager.cancel_all_open_orders.return_value = Result.success(
            {"total_cancelled": 3, "failed_cancellations": []}
        )
        mock_adapter = Mock()
        plugin._manager = mock_manager  # noqa: SLF001
        plugin._adapter = mock_adapter  # noqa: SLF001
        emit_calls = self._capture_events(plugin)

        result = plugin.cleanup()

        assert result.status is ResultStatus.SUCCESS

        cleanup_events = [call for call in emit_calls if call[0] == "execution.cleanup"]
        assert len(cleanup_events) == 1
        _event_type, _module_name, payload = cleanup_events[0]
        assert payload is not None
        assert payload["strategy"] == "cancel_all"
        assert payload["status"] == "success"
        assert payload["total_cancelled"] == 3
        assert payload["failed_cancellations"] == 0
