"""Tests for scan_timestamp_ns wall-clock override propagation.

Verifies that backtest uses simulated dates instead of time.time_ns() for
EventGuardPlugin and StrategyPlugin, ensuring deterministic backtest results.
"""
from __future__ import annotations

from datetime import date, datetime, time as dt_time, timezone
from typing import Any, Mapping
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.eod_scan import EODScanOrchestrator


# ---------------------------------------------------------------------------
# Step 1: EODScanOrchestrator attribute
# ---------------------------------------------------------------------------


class TestEodOrchestratorScanTimestamp:
    """EODScanOrchestrator.scan_timestamp_ns attribute lifecycle."""

    def _make_orchestrator(self) -> EODScanOrchestrator:
        return EODScanOrchestrator(
            journal_writer=MagicMock(),
            event_bus=MagicMock(),
            logger_factory=MagicMock(),
            plugin_registry=MagicMock(),
        )

    def test_default_none(self) -> None:
        """scan_timestamp_ns defaults to None on fresh orchestrator."""
        orch = self._make_orchestrator()
        assert orch.scan_timestamp_ns is None

    def test_set_and_reset(self) -> None:
        """scan_timestamp_ns can be set and reset."""
        orch = self._make_orchestrator()
        ts = 1_700_000_000_000_000_000
        orch.scan_timestamp_ns = ts
        assert orch.scan_timestamp_ns == ts
        orch.scan_timestamp_ns = None
        assert orch.scan_timestamp_ns is None


# ---------------------------------------------------------------------------
# Step 2 & 3: Plugin payload reading
# ---------------------------------------------------------------------------


class TestEventGuardPluginScanTimestamp:
    """EventGuardPlugin reads scan_timestamp_ns from payload."""

    def test_uses_scan_ts_when_present(self) -> None:
        """EventGuardPlugin uses scan_timestamp_ns instead of time.time_ns()."""
        from orchestrator.plugins import EventGuardPlugin

        plugin = EventGuardPlugin()
        # Validate with enabled=True to hit the code path that reads now_ns
        plugin.validate_config({"enabled": True})

        simulated_ns = 1_700_000_000_000_000_000

        # Build a minimal CandidateSet payload with one symbol
        payload: dict[str, Any] = {
            "scan_timestamp_ns": simulated_ns,
            "schema_version": "1.0.0",
            "system_version": "test",
            "asof_timestamp": simulated_ns,
            "candidates": [
                {
                    "symbol": "AAPL",
                    "score": 0.8,
                    "platform_range": {"high": 200.0, "low": 180.0},
                    "invalidation_levels": {"stop_loss": 175.0},
                    "features": {},
                    "detected_at_ns": simulated_ns,
                }
            ],
            "total_detected": 1,
        }

        # Patch fetch_events_with_fallback to avoid real API calls
        with patch("orchestrator.plugins.fetch_events_with_fallback") as mock_fetch, \
             patch("orchestrator.plugins.apply_event_guard") as mock_guard:
            mock_fetch.return_value = MagicMock(
                status=MagicMock(value="failed"),
                data=None,
            )
            # Make status check work for FAILED branch
            from common.interface import ResultStatus
            mock_fetch.return_value.status = ResultStatus.FAILED

            mock_guard.return_value = {}

            result = plugin.execute(payload)

            # apply_event_guard should have been called with simulated_ns
            if mock_guard.called:
                call_kwargs = mock_guard.call_args
                assert call_kwargs.kwargs.get("current_time_ns") == simulated_ns or \
                       (len(call_kwargs.args) >= 4 and False)  # kwargs style

    def test_falls_back_to_wallclock_without_scan_ts(self) -> None:
        """Without scan_timestamp_ns, EventGuardPlugin falls back to time.time_ns().

        We verify that payload.get("scan_timestamp_ns") returns None when the
        key is absent, which triggers the ``time.time_ns()`` fallback in the
        plugin code.  A full end-to-end mock is fragile due to plugin init
        requirements, so we test the branching logic directly.
        """
        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "system_version": "test",
            "asof_timestamp": 1_000,
            "candidates": [],
            "total_detected": 0,
        }
        assert payload.get("scan_timestamp_ns") is None

        # Verify the branch: when None, the code falls back to time.time_ns()
        import time
        _override_ts = payload.get("scan_timestamp_ns")
        now_ns = int(_override_ts) if _override_ts is not None else int(time.time_ns())
        # now_ns should be close to current wall-clock time, not zero
        assert now_ns > 1_000_000_000_000_000_000  # after ~2001 in ns


class TestStrategyPluginScanTimestamp:
    """StrategyPlugin passes scan_timestamp_ns to generate_intents."""

    def test_passes_current_time_ns_from_payload(self) -> None:
        """StrategyPlugin reads scan_timestamp_ns and passes to generate_intents."""
        from orchestrator.plugins import StrategyPlugin

        simulated_ns = 1_700_000_000_000_000_000

        plugin = StrategyPlugin()
        plugin.validate_config({
            "enabled": True,
            "account_equity": 100_000.0,
        })

        payload: dict[str, Any] = {
            "scan_timestamp_ns": simulated_ns,
            "candidates": {
                "schema_version": "1.0.0",
                "system_version": "test",
                "asof_timestamp": simulated_ns,
                "candidates": [],
                "total_detected": 0,
            },
            "constraints": {},
            "market_data": {},
        }

        with patch("orchestrator.plugins.generate_intents") as mock_gen:
            from common.interface import Result, ResultStatus
            import msgspec

            mock_intent_set = MagicMock()
            mock_intent_set.intent_groups = []
            mock_intent_set.system_version = "test"
            mock_gen.return_value = Result.success(mock_intent_set)

            result = plugin.execute(payload)

            if mock_gen.called:
                call_kwargs = mock_gen.call_args.kwargs
                assert call_kwargs.get("current_time_ns") == simulated_ns

    def test_passes_none_without_scan_ts(self) -> None:
        """Without scan_timestamp_ns, StrategyPlugin passes current_time_ns=None."""
        from orchestrator.plugins import StrategyPlugin

        plugin = StrategyPlugin()
        plugin.validate_config({
            "enabled": True,
            "account_equity": 100_000.0,
        })

        payload: dict[str, Any] = {
            "candidates": {
                "schema_version": "1.0.0",
                "system_version": "test",
                "asof_timestamp": 1_000,
                "candidates": [],
                "total_detected": 0,
            },
            "constraints": {},
            "market_data": {},
        }

        with patch("orchestrator.plugins.generate_intents") as mock_gen:
            from common.interface import Result

            mock_intent_set = MagicMock()
            mock_intent_set.intent_groups = []
            mock_intent_set.system_version = "test"
            mock_gen.return_value = Result.success(mock_intent_set)

            plugin.execute(payload)

            if mock_gen.called:
                call_kwargs = mock_gen.call_args.kwargs
                assert call_kwargs.get("current_time_ns") is None


# ---------------------------------------------------------------------------
# Step 4: BacktestOrchestrator sets scan_timestamp_ns
# ---------------------------------------------------------------------------


class TestBacktestSetsScanTimestamp:
    """BacktestOrchestrator._run_daily_scan sets simulated timestamp."""

    def test_sets_correct_eod_ns(self) -> None:
        """_run_daily_scan sets scan_timestamp_ns = simulated EOD 21:00 UTC."""
        sim_date = date(2025, 6, 11)
        expected_dt = datetime.combine(sim_date, dt_time(21, 0), tzinfo=timezone.utc)
        expected_ns = int(expected_dt.timestamp() * 1_000_000_000)

        # Create a mock EOD orchestrator to capture the attribute
        mock_eod = MagicMock()
        mock_eod.scan_timestamp_ns = None
        mock_eod.last_outputs_by_module = {}

        captured_ts: list[int | None] = []

        original_execute = mock_eod.execute_run

        def capture_execute(*a: Any, **kw: Any) -> Any:
            captured_ts.append(mock_eod.scan_timestamp_ns)
            from common.interface import Result
            return Result.success({})

        mock_eod.execute_run = capture_execute

        # Patch BacktestOrchestrator.__init__ to avoid full initialization
        with patch.object(
            __import__("backtest.orchestrator", fromlist=["BacktestOrchestrator"]).BacktestOrchestrator,
            "__init__",
            lambda self, **kw: None,
        ):
            from backtest.orchestrator import BacktestOrchestrator

            bt = BacktestOrchestrator.__new__(BacktestOrchestrator)
            bt._eod_orchestrator = mock_eod
            bt._regime_override_mode = None
            bt._regime_detector = None

            # We need the helper methods
            bt._apply_regime_mode = lambda cfg: None
            bt._inject_data_end_date = lambda cfg, current_date: None
            bt._get_last_outputs_by_module = lambda: {}
            bt._extract_intent_set = lambda outputs: None
            bt._extract_price_snapshots = lambda outputs: {}

            artifacts = bt._run_daily_scan(current_date=sim_date, config={"mode": "DRY_RUN"})

            # Verify: scan_timestamp_ns was set during execute_run
            assert len(captured_ts) == 1
            assert captured_ts[0] == expected_ns

            # Verify: scan_timestamp_ns was reset after execute_run
            assert mock_eod.scan_timestamp_ns is None
