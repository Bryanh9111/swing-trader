"""Tests for DataPlugin statistics emission."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from common.interface import Result, ResultStatus
from data.interface import DataLayerStats
from orchestrator.plugins import DataPlugin


class TestDataPluginStatistics:
    """Test suite for DataPlugin statistics emission."""

    def test_execute_emits_data_stats_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """execute() emits data.stats event with statistics payload."""

        plugin = DataPlugin(config={"enabled": True, "primary_source": "yahoo"})

        monkeypatch.setattr(DataPlugin, "_build_orchestrator", lambda *_args, **_kwargs: Result.success(Mock()))
        plugin.validate_config({"enabled": True, "primary_source": "yahoo"})

        mock_orchestrator = Mock()
        mock_stats = DataLayerStats(
            total_symbols=10,
            freshness_pass_count=7,
            freshness_fail_count=3,
            backfill_attempt_count=2,
            backfill_success_count=2,
            cache_hit_count=5,
            cache_miss_count=5,
        )
        mock_orchestrator.get_run_stats.return_value = mock_stats
        plugin._orchestrator = mock_orchestrator  # noqa: SLF001 - inject stub orchestrator

        emit_calls: list[tuple[str, str, dict[str, object] | None]] = []

        def capture_emit(event_type: str, module_name: str, data: dict[str, object] | None) -> None:
            emit_calls.append((event_type, module_name, data))

        plugin._emit_event = capture_emit  # noqa: SLF001 - capture emitted events

        # Execute with empty universe
        universe_payload = {
            "schema_version": "1.0.0",
            "system_version": "test",
            "asof_timestamp": 1_700_000_000_000_000_000,
            "source": "test",
            "equities": [],
            "filter_criteria": {},
            "total_candidates": 0,
            "total_filtered": 0,
        }
        result = plugin.execute({"universe": universe_payload})
        assert result.status is ResultStatus.SUCCESS

        stats_events = [call for call in emit_calls if call[0] == "data.stats"]
        assert len(stats_events) == 1
        _event_type, _module_name, payload = stats_events[0]
        assert payload is not None

        assert payload["total_symbols"] == 10
        assert payload["freshness_pass_count"] == 7
        assert payload["freshness_rate"] == 0.7
        assert payload["backfill_success_rate"] == 1.0
        assert payload["cache_hit_rate"] == 0.5
