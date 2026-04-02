"""IB Gateway health check, auto-restart, and connectivity watchdog.

Used by run_paper_*.py scripts to ensure IB Gateway is alive before
executing the orchestrator pipeline.  If the gateway is unreachable the
module will:

1. Kill any stale java/IBC process.
2. Re-launch IB Gateway via the IBC start script.
3. Wait for the API port to become available.
4. Verify a successful ib_insync connection.

All operations are best-effort and designed to never crash the caller.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

__all__ = ["ensure_ibkr_ready"]

_DEFAULT_PORT = 4002
_DEFAULT_HOST = "127.0.0.1"
_GATEWAY_START_SCRIPT = Path.home() / "opt" / "ibc" / "gatewaystartmacos.sh"
_CONNECT_TIMEOUT = 8
_PORT_WAIT_SECONDS = 90
_PORT_POLL_INTERVAL = 3
_KILL_PATTERN = "ibcalpha.ibc.IbcGateway"


def ensure_ibkr_ready(
    *,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    client_id: int = 98,
    gateway_script: str | Path | None = None,
    notifier: Any | None = None,
) -> bool:
    """Ensure IB Gateway is reachable.  Returns True if healthy.

    If the initial probe fails, attempts one kill-restart cycle.
    Sends a Telegram alert via *notifier* (if provided) on restart.

    Args:
        host: IB Gateway host.
        port: IB Gateway API port.
        client_id: Temporary client ID used for the health probe.
        gateway_script: Path to the IBC gateway start script.
        notifier: Optional ``TelegramNotifier`` instance for alerts.

    Returns:
        ``True`` if the gateway is healthy after the check (possibly
        after a restart), ``False`` if recovery failed.
    """
    script = Path(gateway_script) if gateway_script else _GATEWAY_START_SCRIPT

    # --- first probe ---
    if _probe_connection(host, port, client_id):
        return True

    print(f"[watchdog] IB Gateway unreachable at {host}:{port}, attempting restart...")
    if notifier is not None:
        _safe_send(notifier, f"[AST][WATCHDOG] IB Gateway unreachable at {host}:{port}, restarting...")

    # --- kill stale processes ---
    _kill_stale_gateway()

    # --- restart ---
    if not script.exists():
        print(f"[watchdog] Gateway start script not found: {script}")
        if notifier is not None:
            _safe_send(notifier, f"[AST][WATCHDOG] Gateway restart FAILED — script not found: {script}")
        return False

    try:
        log_dir = Path.home() / "opt" / "ibc" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        gw_log = log_dir / "watchdog-gateway.log"
        gw_fh = gw_log.open("a")
        subprocess.Popen(
            [str(script), "-inline"],
            stdout=gw_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[watchdog] Failed to launch gateway: {exc}")
        if notifier is not None:
            _safe_send(notifier, f"[AST][WATCHDOG] Gateway restart FAILED — {exc}")
        return False

    # --- wait for port ---
    if not _wait_for_port(host, port, timeout=_PORT_WAIT_SECONDS):
        print(f"[watchdog] Port {port} did not open within {_PORT_WAIT_SECONDS}s")
        if notifier is not None:
            _safe_send(notifier, f"[AST][WATCHDOG] Gateway restart FAILED — port {port} timeout after {_PORT_WAIT_SECONDS}s")
        return False

    # --- second probe ---
    # Give a few extra seconds for the API to stabilize after port opens
    time.sleep(5)
    if _probe_connection(host, port, client_id):
        print("[watchdog] IB Gateway restarted and verified OK")
        if notifier is not None:
            _safe_send(notifier, "[AST][WATCHDOG] IB Gateway restarted successfully ✅")
        return True

    print("[watchdog] Gateway restarted but API probe still fails")
    if notifier is not None:
        _safe_send(notifier, "[AST][WATCHDOG] Gateway restarted but API probe FAILED")
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _probe_connection(host: str, port: int, client_id: int) -> bool:
    """Try a quick ib_insync connect/disconnect."""
    try:
        from ib_insync import IB  # noqa: PLC0415

        ib = IB()
        ib.connect(host, port, clientId=client_id, timeout=_CONNECT_TIMEOUT)
        connected = ib.isConnected()
        ib.disconnect()
        return connected
    except Exception:  # noqa: BLE001
        return False


def _kill_stale_gateway() -> None:
    """Find and kill any java process running IBC/IB Gateway.

    Uses the same pgrep pattern as start_ib_gateway.sh for consistency.
    Sends SIGTERM first (graceful, releases port), then SIGKILL if needed.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", _KILL_PATTERN],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]
        if not pids:
            return

        # SIGTERM first — lets Java release the port cleanly
        for pid in pids:
            print(f"[watchdog] Sending SIGTERM to stale gateway PID {pid}")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(3)

        # Check survivors, SIGKILL if still alive
        result2 = subprocess.run(
            ["pgrep", "-f", _KILL_PATTERN],
            capture_output=True,
            text=True,
            timeout=5,
        )
        survivors = [int(p.strip()) for p in result2.stdout.strip().split("\n") if p.strip()]
        for pid in survivors:
            print(f"[watchdog] Force-killing gateway PID {pid}")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if survivors:
            time.sleep(2)
    except Exception as exc:  # noqa: BLE001
        print(f"[watchdog] Error killing stale processes: {exc}")


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    """Poll until *port* accepts a TCP connection or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(_PORT_POLL_INTERVAL)
    return False


def _safe_send(notifier: Any, text: str) -> None:
    """Send a Telegram message, swallowing any error."""
    try:
        notifier.send(text)
    except Exception:  # noqa: BLE001
        pass
