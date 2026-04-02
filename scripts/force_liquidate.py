#!/usr/bin/env python3
"""Force liquidate all positions in an IBKR account.

CRITICAL SAFETY FEATURE:
This script will close ALL open positions via MARKET orders. Use only in emergency
situations or when you must exit all positions immediately.

Usage:
    # Paper trading
    python scripts/force_liquidate.py --paper --reason "Market crash"

    # Live trading (requires explicit --live flag)
    python scripts/force_liquidate.py --live --reason "Emergency exit"

    # Dry run (show positions only, no orders)
    python scripts/force_liquidate.py --paper --dry-run

Notes:
    - Requires IBKR TWS/IB Gateway running with API enabled.
    - Default ports: 7497 (paper), 7496 (live).
    - This script is intentionally interactive and includes multi-step confirmation
      to reduce risk of accidental liquidation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to Python path (match scripts/test_ibkr_connection.py style)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.interface import ResultStatus
from execution.ibkr_adapter import IBKRAdapter
from execution.interface import BrokerConnectionConfig
from order_state_machine.id_generator import IDGenerator


def _fmt_money(value: float | None) -> str:
    """Format a float as a USD-like string (tolerate None)."""

    if value is None:
        return " " * 14
    return f"${value:>13.2f}"


def _fmt_number(value: float | None, width: int = 12) -> str:
    """Format a float with fixed width (tolerate None)."""

    if value is None:
        return " " * width
    return f"{value:>{width}.2f}"


def confirm_liquidation(*, mode: str, positions_count: int, reason: str) -> bool:
    """Multi-step confirmation to prevent accidental liquidation.

    Required inputs:
        1) Type exactly "LIQUIDATE"
        2) Type exactly "YES"
    """

    print("\n" + "=" * 80)
    print("⚠️  WARNING: FORCE LIQUIDATION REQUESTED")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Positions to close: {positions_count}")
    print(f"Reason: {reason}")
    print("\nThis will submit MARKET ORDERS to close ALL positions immediately.")
    print("=" * 80)

    response1 = input("\nType 'LIQUIDATE' to proceed (or anything else to cancel): ").strip()
    if response1 != "LIQUIDATE":
        print("❌ Cancelled by user")
        return False

    print(f"\n⚠️  Final confirmation: Close all {positions_count} positions now?")
    response2 = input("Type 'YES' to confirm: ").strip().upper()
    if response2 != "YES":
        print("❌ Cancelled by user")
        return False

    return True


def display_positions(positions: list[dict[str, Any]]) -> None:
    """Display current positions in a readable table."""

    if not positions:
        print("📊 No open positions found")
        return

    print(f"\n📊 Current Positions ({len(positions)} total):")
    print("-" * 100)
    print(f"{'Symbol':<10} {'Position':<12} {'Market Value':<15} {'Avg Cost':<12} {'Unrealized P&L':<15}")
    print("-" * 100)

    total_value = 0.0
    total_pnl = 0.0

    for pos in positions:
        symbol = str(pos.get("symbol") or "").strip().upper()
        position = pos.get("position")
        market_value = pos.get("market_value")
        avg_cost = pos.get("avg_cost")
        unrealized_pnl = pos.get("unrealized_pnl")

        if isinstance(market_value, (int, float)):
            total_value += abs(float(market_value))
        if isinstance(unrealized_pnl, (int, float)):
            total_pnl += float(unrealized_pnl)

        print(
            f"{symbol:<10} "
            f"{_fmt_number(position, width=12)} "
            f"{_fmt_money(market_value):<15} "
            f"{_fmt_money(avg_cost):<12} "
            f"{_fmt_money(unrealized_pnl):<15}"
        )

    print("-" * 100)
    print(f"{'TOTAL':<10} {'':<12} {_fmt_money(total_value):<15} {'':<12} {_fmt_money(total_pnl):<15}")
    print("-" * 100)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Force liquidate all IBKR positions")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--paper", action="store_true", help="Use paper trading account")
    mode_group.add_argument("--live", action="store_true", help="Use live trading account (DANGEROUS)")

    parser.add_argument(
        "--reason",
        type=str,
        default="MANUAL_LIQUIDATION",
        help="Reason string used for orderRef/audit (default: MANUAL_LIQUIDATION)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show positions only, do not liquidate")

    # Keep parity with scripts/test_ibkr_connection.py defaults but allow overriding.
    parser.add_argument("--host", type=str, default="127.0.0.1", help="TWS/IB Gateway host")
    parser.add_argument("--port", type=int, default=0, help="TWS/IB Gateway port (default: 7497/7496)")
    parser.add_argument(
        "--account",
        type=str,
        default="",
        help="Explicit account id; if omitted uses env/auto-detect",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    mode = "PAPER" if args.paper else "LIVE"
    default_port = 7497 if args.paper else 7496
    port = int(args.port) if int(args.port) > 0 else default_port

    # Prefer explicit CLI account > env var; else None (auto-detect).
    env_key = "IBKR_PAPER_ACCOUNT" if args.paper else "IBKR_LIVE_ACCOUNT"
    env_account = os.environ.get(env_key, "").strip()
    account = (str(args.account).strip() or env_account or None)

    reason = str(args.reason).strip() or "MANUAL_LIQUIDATION"

    print("=" * 80)
    print(f"IBKR Force Liquidation Script - {mode} MODE")
    print("=" * 80)

    # Connection config (same knobs as scripts/test_ibkr_connection.py).
    config = BrokerConnectionConfig(
        host=str(args.host),
        port=port,
        client_id=1,
        account=account,
        readonly=False,
        timeout=20,
        max_reconnect_attempts=5,
        enable_dynamic_client_id=True,
        client_id_range=(1, 32),
    )

    print("\n1. Creating ID Generator...")
    try:
        id_generator = IDGenerator()
        print("   ✅ ID Generator created")
    except Exception as exc:
        print(f"   ❌ Failed to create ID generator: {exc}")
        return 1

    print(f"\n2. Creating IBKR Adapter ({mode})...")
    try:
        adapter = IBKRAdapter(config, id_generator)
        print("   ✅ Adapter created")
    except Exception as exc:
        print(f"   ❌ Failed to create adapter: {exc}")
        return 1

    print(f"\n3. Connecting to TWS ({mode} port {port})...")
    connect_result = adapter.connect()
    if connect_result.status is not ResultStatus.SUCCESS:
        print("   ❌ Connection failed")
        print(f"   Reason: {connect_result.reason_code}")
        if connect_result.error:
            print(f"   Error: {connect_result.error}")
        return 1

    print("   ✅ Connected successfully")
    print(f"   Account: {config.account or 'Auto-detected'}")
    print(f"   Active Client ID: {adapter.active_client_id}")

    try:
        print("\n4. Querying current positions...")
        positions_result = adapter.get_portfolio_positions()
        if positions_result.status is not ResultStatus.SUCCESS:
            print("   ❌ Failed to query positions")
            print(f"   Reason: {positions_result.reason_code}")
            if positions_result.error:
                print(f"   Error: {positions_result.error}")
            return 1

        positions = positions_result.data or []
        print(f"   ✅ Found {len(positions)} positions")
        display_positions(positions)

        if len(positions) == 0:
            print("\n✅ No positions to liquidate")
            return 0

        if args.dry_run:
            print("\n🔍 Dry run mode - no orders will be submitted")
            return 0

        if not confirm_liquidation(mode=mode, positions_count=len(positions), reason=reason):
            return 0

        print("\n5. Submitting liquidation orders...")
        liquidation_result = adapter.liquidate_all_positions(reason=reason)
        if liquidation_result.status is not ResultStatus.SUCCESS:
            print("   ❌ Liquidation failed")
            print(f"   Reason: {liquidation_result.reason_code}")
            if liquidation_result.error:
                print(f"   Error: {liquidation_result.error}")
            return 1

        summary = liquidation_result.data or {}
        print("\n✅ Liquidation completed")
        print(f"   Positions found: {summary.get('positions_count', 0)}")
        print(f"   Orders submitted: {summary.get('orders_submitted', 0)}")
        print(f"   Orders failed: {summary.get('orders_failed', 0)}")
        print(f"   Reason: {summary.get('reason', reason)}")
        orders = summary.get("orders") or []
        if orders:
            print(f"   OrderRef batch: {orders[0].get('order_ref')}")

        if summary.get("errors"):
            print("\n⚠️  Errors encountered:")
            for err in summary.get("errors", []):
                print(f"   - {err}")

        if orders:
            print("\n📋 Submitted orders:")
            for order in orders:
                symbol = order.get("symbol")
                action = order.get("action")
                quantity = order.get("quantity")
                broker_order_id = order.get("broker_order_id")
                order_ref = order.get("order_ref")
                print(f"   {symbol}: {action} {quantity} @ MARKET")
                print(f"      Broker Order ID: {broker_order_id}")
                print(f"      Order Ref: {order_ref}")

        print("\n" + "=" * 80)
        print("✅ LIQUIDATION COMPLETE - Check TWS/IB Gateway for order status")
        print("=" * 80)
        return 0
    finally:
        print("\n6. Disconnecting...")
        disconnect_result = adapter.disconnect()
        if disconnect_result.status is ResultStatus.SUCCESS:
            print("   ✅ Disconnected")
        else:
            print(f"   ⚠️  Disconnect warning: {disconnect_result.reason_code}")


if __name__ == "__main__":
    raise SystemExit(main())
