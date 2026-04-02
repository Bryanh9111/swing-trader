"""Unit tests for ast_daemon.py — scheduling, health monitor, adapter injection."""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.interface import Result, ResultStatus
from journal.interface import RunType


# ---------------------------------------------------------------------------
# Schedule tests
# ---------------------------------------------------------------------------


def test_schedule_ordering() -> None:
    """RUN_SCHEDULE entries are in chronological order."""

    from ast_daemon import RUN_SCHEDULE

    times = [t for t, _ in RUN_SCHEDULE]
    assert times == sorted(times), f"Schedule not in order: {times}"


def test_schedule_contains_expected_run_types() -> None:
    """All four expected run types are present in the schedule."""

    from ast_daemon import RUN_SCHEDULE

    run_types = {rt for _, rt in RUN_SCHEDULE}
    expected = {
        RunType.PRE_MARKET_FULL_SCAN,
        RunType.INTRADAY_CHECK_1030,
        RunType.INTRADAY_CHECK_1230,
        RunType.INTRADAY_CHECK_1430,
        RunType.AFTER_MARKET_RECON,
    }
    assert run_types == expected


# ---------------------------------------------------------------------------
# Interruptible sleep / shutdown tests
# ---------------------------------------------------------------------------


def test_interruptible_sleep_shutdown() -> None:
    """_run_loop exits promptly when _shutdown is set during a wait."""

    from ast_daemon import _run_loop, _shutdown

    # Reset in case a prior test left it set.
    _shutdown.clear()

    executed: list[str] = []

    # Schedule far in the future — should never execute.
    future_schedule = [("23:59", RunType.AFTER_MARKET_RECON)]

    def fake_execute(run_type):
        executed.append(run_type.value)

    # Set shutdown after a short delay.
    def delayed_shutdown():
        time.sleep(0.3)
        _shutdown.set()

    t = threading.Thread(target=delayed_shutdown, daemon=True)
    t.start()

    start = time.time()
    _run_loop(future_schedule, fake_execute)
    elapsed = time.time() - start

    # Should exit quickly (< 2s), not wait until 23:59.
    assert elapsed < 2.0
    assert executed == []

    # Cleanup.
    _shutdown.clear()


def test_run_loop_skips_past_due() -> None:
    """_run_loop skips runs that are more than 5 minutes past due."""

    from ast_daemon import _run_loop, _shutdown

    _shutdown.clear()

    executed: list[str] = []
    # Use 00:00 — always past due (by >5 min during normal hours).
    schedule = [("00:00", RunType.PRE_MARKET_FULL_SCAN)]

    def fake_execute(run_type):
        executed.append(run_type.value)

    _run_loop(schedule, fake_execute)

    # Should skip (past due > 5 min), not execute.
    assert executed == []
    _shutdown.clear()


# ---------------------------------------------------------------------------
# Health monitor tests
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """Minimal adapter stub for health monitor tests."""

    def __init__(self, *, healthy: bool = True):
        self._healthy = healthy
        self.connect_calls = 0
        self.reset_calls = 0

    def is_healthy(self) -> bool:
        return self._healthy

    def connect(self, *, force: bool = False) -> Result[None]:
        self.connect_calls += 1
        if self._healthy:
            return Result.success(data=None)
        return Result.failed(ConnectionError("still down"), "CONNECTION_FAILED")

    def reset_reconnect_counter(self) -> None:
        self.reset_calls += 1

    def disconnect(self) -> Result[None]:
        return Result.success(data=None)


def test_health_monitor_reconnect() -> None:
    """Health monitor reconnects when adapter reports unhealthy."""

    from ast_daemon import _health_monitor, _shutdown

    _shutdown.clear()

    adapter = _FakeAdapter(healthy=False)
    notifier = MagicMock()
    notifier.enabled = False

    # Run one iteration then shut down.
    def delayed_shutdown():
        time.sleep(0.5)
        _shutdown.set()

    t = threading.Thread(target=delayed_shutdown, daemon=True)
    t.start()

    _health_monitor(adapter, {}, notifier, interval=0.1)

    # Should have attempted at least one connect.
    assert adapter.connect_calls >= 1
    _shutdown.clear()


def test_health_monitor_healthy_noop() -> None:
    """Health monitor does nothing when adapter is healthy."""

    from ast_daemon import _health_monitor, _shutdown

    _shutdown.clear()

    adapter = _FakeAdapter(healthy=True)
    notifier = MagicMock()
    notifier.enabled = False

    def delayed_shutdown():
        time.sleep(0.5)
        _shutdown.set()

    t = threading.Thread(target=delayed_shutdown, daemon=True)
    t.start()

    _health_monitor(adapter, {}, notifier, interval=0.1)

    # Should NOT have called connect (adapter was healthy).
    assert adapter.connect_calls == 0
    _shutdown.clear()


# ---------------------------------------------------------------------------
# execute_single_run adapter injection test
# ---------------------------------------------------------------------------


def test_execute_single_run_injects_adapter() -> None:
    """_execute_single_run places the shared adapter into the execution config."""

    from ast_daemon import _execute_single_run

    fake_adapter = MagicMock()
    fake_adapter.is_connected.return_value = True

    captured_configs: list[dict] = []

    def fake_execute_run(run_config):
        captured_configs.append(dict(run_config))
        return Result.success(data=None)

    config = {
        "strategy": {"account_equity": 50_000},
        "risk_gate": {},
        "scanner": {},
        "universe_builder": {},
        "data": {},
        "event_guard": {},
        "execution": {"adapter": "ibkr", "broker": {"port": 4002}},
    }

    # Patch out heavy infra.
    with (
        patch("ast_daemon.EODScanOrchestrator") as mock_orch_cls,
        patch("ast_daemon.register_real_plugins"),
        patch("ast_daemon.build_run_config") as mock_build,
    ):
        mock_build.return_value = {
            "plugins": {
                "signals": [
                    {"name": "execution_real", "config": {"adapter": "ibkr"}},
                ],
            },
        }
        mock_orch = MagicMock()
        mock_orch.execute_run.side_effect = fake_execute_run
        mock_orch_cls.return_value = mock_orch

        from journal.interface import OperatingMode

        _execute_single_run(
            config,
            OperatingMode.PAPER,
            RunType.PRE_MARKET_FULL_SCAN,
            fake_adapter,
            MagicMock(),
        )

    assert len(captured_configs) == 1
    cfg = captured_configs[0]

    # Verify shared adapter was injected into the plugin config.
    exec_plugin_cfg = None
    for entry in cfg.get("plugins", {}).get("signals", []):
        if entry.get("name") == "execution_real":
            exec_plugin_cfg = entry.get("config", {})
    assert exec_plugin_cfg is not None
    assert exec_plugin_cfg.get("_shared_adapter") is fake_adapter
