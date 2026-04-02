"""IBKR broker adapter (Phase 3.5).

This module provides an Interactive Brokers (IBKR) adapter implementation for
AST's Execution layer boundary (``execution.interface.BrokerAdapterProtocol``).

Design notes:
    - Uses ``ib_insync`` for event-driven connectivity to TWS/IB Gateway.
    - All boundary methods return ``common.interface.Result[T]`` (no exceptions
      as control flow).
    - Adapter tracks local order handles (``ib_insync.Trade``) and maintains a
      broker-agnostic view of status/fills through ``IntentOrderMapping`` and
      ``FillDetail``.
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import UTC, datetime
from typing import Any

import msgspec

from common.interface import Result, ResultStatus
from execution.interface import BrokerAdapterProtocol, BrokerConnectionConfig, FillDetail
from order_state_machine import IDGenerator, OrderState
from order_state_machine.interface import IntentOrderMapping
from strategy.interface import IntentType, TradeIntent

try:  # pragma: no cover - dependency availability is environment-specific.
    from ib_insync import IB, LimitOrder, MarketOrder, StopOrder, Stock, Trade
except Exception as exc:  # noqa: BLE001 - boundary returns Result
    IB = Any  # type: ignore[assignment]
    Stock = Any  # type: ignore[assignment]
    MarketOrder = Any  # type: ignore[assignment]
    LimitOrder = Any  # type: ignore[assignment]
    StopOrder = Any  # type: ignore[assignment]
    Trade = Any  # type: ignore[assignment]
    _IB_INSYNC_IMPORT_ERROR: BaseException | None = exc
else:
    _IB_INSYNC_IMPORT_ERROR = None

__all__ = ["IBKRAdapter"]


class IBKRAdapter(BrokerAdapterProtocol):
    """Interactive Brokers adapter implementation using ``ib_insync``.

    Args:
        config: Broker connection configuration for TWS/IB Gateway.
        id_generator: Order/intent id generator used for client order ids.
    """

    _DEFAULT_OCA_TYPE: int = 1  # 1 = Cancel remaining on fill

    _CONNECT_FAILED_REASON = "CONNECTION_FAILED"
    _SUBMIT_FAILED_REASON = "SUBMIT_FAILED"
    _CANCEL_FAILED_REASON = "CANCEL_FAILED"
    _STATUS_FAILED_REASON = "STATUS_FAILED"
    _READONLY_REASON = "READONLY_MODE"
    _ORDER_NOT_FOUND_REASON = "ORDER_NOT_FOUND"
    _DEPENDENCY_MISSING_REASON = "DEPENDENCY_MISSING"
    _ACCOUNT_QUERY_FAILED_REASON = "ACCOUNT_QUERY_FAILED"
    _PORTFOLIO_QUERY_FAILED_REASON = "PORTFOLIO_QUERY_FAILED"
    _LIQUIDATION_FAILED_REASON = "LIQUIDATION_FAILED"

    def __init__(
        self,
        config: BrokerConnectionConfig,
        id_generator: IDGenerator,
        *,
        fills_persistence: Any | None = None,
    ) -> None:
        self._config = config
        self._id_gen = id_generator
        self._ib = IB()

        self._lock = threading.RLock()
        self._trades: dict[str, Trade] = {}
        self._mappings: dict[str, IntentOrderMapping] = {}
        self._fills: dict[str, list[FillDetail]] = {}

        self._fills_persistence = fills_persistence
        self._reconnect_attempts = 0
        self._last_connection_error: tuple[int, str] | None = None
        self._active_client_id: int | None = None
        self._force_disconnecting = False
        self._register_event_handlers()

    @property
    def active_client_id(self) -> int | None:
        """Return the client_id currently in use for the active IBKR session."""

        with self._lock:
            return self._active_client_id

    def get_fills(self, client_order_id: str) -> list[FillDetail]:
        """Return fills for a given client_order_id (thread-safe copy)."""

        with self._lock:
            return list(self._fills.get(client_order_id, []))

    def restore_fills(self, fills_by_order: dict[str, list[FillDetail]]) -> None:
        """Restore persisted fills into memory (called during startup)."""

        with self._lock:
            for client_order_id, fill_list in fills_by_order.items():
                existing = self._fills.setdefault(client_order_id, [])
                seen = {f.execution_id for f in existing}
                for fill in fill_list:
                    if fill.execution_id not in seen:
                        existing.append(fill)
                        seen.add(fill.execution_id)

    def _register_event_handlers(self) -> None:
        """Register ``ib_insync`` event callbacks.

        Events used:
            - connectedEvent / disconnectedEvent: connectivity lifecycle.
            - orderStatusEvent: order status updates.
            - execDetailsEvent: execution fill updates.
            - errorEvent: connection/order errors and info messages.
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            return

        self._ib.connectedEvent += self._on_connected
        self._ib.disconnectedEvent += self._on_disconnected
        self._ib.orderStatusEvent += self._on_order_status
        self._ib.execDetailsEvent += self._on_execution
        self._ib.errorEvent += self._on_error

    def connect(self, *, force: bool = False) -> Result[None]:
        """Connect to IBKR TWS/Gateway with exponential backoff retry.

        When ``config.enable_dynamic_client_id`` is enabled, the adapter will
        detect client_id conflicts (Error 326) and retry connection using the
        next available client_id within ``config.client_id_range``.

        Args:
            force: If True, disconnect any existing session before connecting.
                Useful for clearing zombie sessions (Error 326).

        Retries:
            Up to 5 attempts per client_id (or ``config.max_reconnect_attempts``
            if smaller), with backoff delays: 2s, 4s, 8s, 16s, 32s.

        Returns:
            Result[None]: Success if connected; failed otherwise.
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            err = RuntimeError("ib_insync is not available")
            err.__cause__ = _IB_INSYNC_IMPORT_ERROR
            return Result.failed(err, self._DEPENDENCY_MISSING_REASON)

        with self._lock:
            if self._ib.isConnected():
                if not force:
                    if self._active_client_id is None:
                        current_id = getattr(self._ib, "clientId", None)
                        if isinstance(current_id, int):
                            self._active_client_id = current_id
                    return Result.success(data=None)
                # Force disconnect to clear potential zombie session.
                # Set flag to suppress _on_disconnected auto-reconnect callback.
                try:
                    self._force_disconnecting = True
                    logger.info("Force-disconnecting existing IBKR session before reconnect")
                    self._ib.disconnect()
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    self._force_disconnecting = False

        preferred_client_id = int(self._config.client_id)
        client_ids = [preferred_client_id]
        if bool(self._config.enable_dynamic_client_id):
            start, end = self._config.client_id_range
            lower = int(min(start, end))
            upper = int(max(start, end))
            for candidate in range(lower, upper + 1):
                if candidate != preferred_client_id:
                    client_ids.append(candidate)
        max_attempts = max(1, min(5, int(self._config.max_reconnect_attempts)))
        last_exc: BaseException | None = None

        # Suppress _on_disconnected auto-reconnect during the explicit connect
        # loop.  Without this, a failed attempt (e.g. Error 326) triggers
        # _on_disconnected which re-enters connect() recursively, corrupting
        # state and exhausting retries before the dynamic client_id loop can
        # try the next id.
        self._force_disconnecting = True
        try:
            for client_id in client_ids:
                for attempt in range(1, max_attempts + 1):
                    try:
                        # Clear any stale error so we can accurately classify this attempt.
                        with self._lock:
                            self._last_connection_error = None

                        self._ib.connect(
                            host=self._config.host,
                            port=int(self._config.port),
                            clientId=int(client_id),
                            timeout=int(self._config.timeout),
                            readonly=bool(self._config.readonly),
                        )
                        self._ib.sleep(0)
                        with self._lock:
                            self._reconnect_attempts = 0
                        if self._ib.isConnected():
                            with self._lock:
                                self._active_client_id = int(client_id)
                            return Result.success(data=None)
                        raise ConnectionError("IBKR connect() returned without a live connection.")
                    except BaseException as exc:  # noqa: BLE001 - boundary returns Result
                        last_exc = exc

                        # Clean up ib_insync internal state after a failed connect
                        # attempt. Without this, half-open sockets / event-loop
                        # artefacts can cause subsequent connect() calls to time out
                        # even with a different client_id.
                        try:
                            self._ib.disconnect()
                        except Exception:  # noqa: BLE001
                            pass

                        with self._lock:
                            last_error = self._last_connection_error
                        if last_error is not None and self._is_client_id_conflict(last_error[0], last_error[1]):
                            if not bool(self._config.enable_dynamic_client_id):
                                return Result.failed(last_exc, self._CONNECT_FAILED_REASON)
                            break

                        if attempt >= max_attempts:
                            return Result.failed(last_exc, self._CONNECT_FAILED_REASON)

                        backoff_s = 2**attempt
                        time.sleep(backoff_s)
        finally:
            self._force_disconnecting = False

        return Result.failed(last_exc or RuntimeError("IBKR connection failed."), self._CONNECT_FAILED_REASON)

    def disconnect(self) -> Result[None]:
        """Disconnect from IBKR and release adapter resources.

        Returns:
            Result[None]: Success when disconnected; failed otherwise.
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            err = RuntimeError("ib_insync is not available")
            err.__cause__ = _IB_INSYNC_IMPORT_ERROR
            return Result.failed(err, self._DEPENDENCY_MISSING_REASON)

        try:
            self._force_disconnecting = True
            # Remove event handlers before disconnect to avoid weakref-dead callbacks
            try:
                self._ib.connectedEvent -= self._on_connected
                self._ib.disconnectedEvent -= self._on_disconnected
                self._ib.orderStatusEvent -= self._on_order_status
                self._ib.execDetailsEvent -= self._on_execution
                self._ib.errorEvent -= self._on_error
            except Exception:  # noqa: BLE001
                pass
            self._ib.disconnect()
            self._ib.sleep(0)
            with self._lock:
                self._active_client_id = None
            return Result.success(data=None)
        except BaseException as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, self._CONNECT_FAILED_REASON)
        finally:
            self._force_disconnecting = False

    def is_connected(self) -> bool:
        """Return whether the adapter is currently connected to IBKR."""

        if _IB_INSYNC_IMPORT_ERROR is not None:
            return False
        return bool(self._ib.isConnected())

    def is_healthy(self) -> bool:
        """Quick health probe — True if connected and responsive."""

        try:
            return self._ib.isConnected()
        except Exception:  # noqa: BLE001
            return False

    def reset_reconnect_counter(self) -> None:
        """Reset the reconnect attempt counter.

        Allows external callers (e.g. daemon health monitor) to reset the
        counter after a gateway restart so that subsequent disconnects can
        trigger the built-in reconnect logic again.
        """

        with self._lock:
            self._reconnect_attempts = 0

    def submit_order(self, intent: TradeIntent) -> Result[IntentOrderMapping]:
        """Submit an order for the supplied trade intent.

        Flow:
            1) Generate ``client_order_id`` via ``IDGenerator``.
            2) Build and qualify a ``Stock`` contract (to obtain conId).
            3) Create a Market/Limit order from the intent.
            4) Place order via ``ib.placeOrder`` and begin tracking the Trade.
            5) Return an ``IntentOrderMapping`` in SUBMITTED state.

        Args:
            intent: Trade intent to translate into an IBKR order.

        Returns:
            Result[IntentOrderMapping]: Mapping containing ids, state, and
                broker-specific metadata.
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            err = RuntimeError("ib_insync is not available")
            err.__cause__ = _IB_INSYNC_IMPORT_ERROR
            return Result.failed(err, self._DEPENDENCY_MISSING_REASON)

        if self._config.readonly:
            return Result.failed(
                PermissionError("Adapter is in readonly mode; order submission is disabled."),
                self._READONLY_REASON,
            )

        if not self._ib.isConnected():
            return Result.failed(ConnectionError("Not connected to IBKR."), self._CONNECT_FAILED_REASON)

        try:
            run_id = self._get_run_id(intent)
            client_order_id_result = self._id_gen.generate_order_id(intent.intent_id, run_id)
            if client_order_id_result.status is not ResultStatus.SUCCESS or not client_order_id_result.data:
                error = client_order_id_result.error or RuntimeError("order id generation failed")
                reason = client_order_id_result.reason_code or "ORDER_ID_GENERATION_FAILED"
                return Result.failed(error, reason)
            client_order_id = client_order_id_result.data

            contract = Stock(self._normalize_symbol(intent.symbol), "SMART", "USD")
            qualified = self._ib.qualifyContracts(contract)
            self._ib.sleep(0)
            if not qualified:
                return Result.failed(RuntimeError("Contract qualification failed."), self._SUBMIT_FAILED_REASON)
            contract = qualified[0]

            action = self._infer_action(intent)

            # --- Position Guard: prevent naked shorts and over-selling ---
            guard_result = self._position_guard(intent, action, contract)
            if guard_result is not None:
                return guard_result

            order = self._build_order(intent=intent, action=action, client_order_id=client_order_id)

            trade: Trade = self._ib.placeOrder(contract, order)
            self._ib.sleep(0)

            broker_order_id = None
            order_id_value = getattr(getattr(trade, "order", None), "orderId", None)
            if order_id_value is not None:
                broker_order_id = str(order_id_value)

            created_ns = time.time_ns()
            mapping = IntentOrderMapping(
                intent_id=intent.intent_id,
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
                state=OrderState.SUBMITTED,
                created_at_ns=created_ns,
                updated_at_ns=created_ns,
                intent_snapshot=intent,
                metadata={
                    "symbol": self._normalize_symbol(intent.symbol),
                    "ibkr": {
                        "conId": getattr(contract, "conId", None),
                        "orderId": broker_order_id,
                        "orderType": getattr(getattr(trade, "order", None), "orderType", None),
                    },
                },
            )

            with self._lock:
                self._trades[client_order_id] = trade
                self._mappings[client_order_id] = mapping
                self._fills.setdefault(client_order_id, [])

            return Result.success(mapping)
        except BaseException as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, self._SUBMIT_FAILED_REASON)

    def cancel_order(self, client_order_id: str) -> Result[None]:
        """Cancel an existing order by client order id.

        Args:
            client_order_id: Client order id used to reference the order.

        Returns:
            Result[None]: Success when a cancel request is accepted/queued.
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            err = RuntimeError("ib_insync is not available")
            err.__cause__ = _IB_INSYNC_IMPORT_ERROR
            return Result.failed(err, self._DEPENDENCY_MISSING_REASON)

        if self._config.readonly:
            return Result.failed(
                PermissionError("Adapter is in readonly mode; cancellations are disabled."),
                self._READONLY_REASON,
            )

        if not self._ib.isConnected():
            return Result.failed(ConnectionError("Not connected to IBKR."), self._CONNECT_FAILED_REASON)

        try:
            with self._lock:
                trade = self._trades.get(str(client_order_id))
            if trade is None:
                return Result.failed(KeyError(f"Order not found: {client_order_id}"), self._ORDER_NOT_FOUND_REASON)

            self._ib.cancelOrder(trade.order)
            self._ib.sleep(0)
            return Result.success(data=None)
        except BaseException as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, self._CANCEL_FAILED_REASON)

    def get_order_status(self, client_order_id: str) -> Result[IntentOrderMapping]:
        """Return the latest known mapping view for a given client order id.

        Args:
            client_order_id: Client order id used to reference the order.

        Returns:
            Result[IntentOrderMapping]: Latest mapping view.
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            err = RuntimeError("ib_insync is not available")
            err.__cause__ = _IB_INSYNC_IMPORT_ERROR
            return Result.failed(err, self._DEPENDENCY_MISSING_REASON)

        try:
            self._ib.sleep(0)
            with self._lock:
                mapping = self._mappings.get(str(client_order_id))
                trade = self._trades.get(str(client_order_id))
            if mapping is None:
                return Result.failed(KeyError(f"Order not found: {client_order_id}"), self._ORDER_NOT_FOUND_REASON)

            if trade is None:
                return Result.success(mapping)

            status = getattr(getattr(trade, "orderStatus", None), "status", None)
            if isinstance(status, str) and status:
                mapped_state = self._map_ibkr_status_to_order_state(status)
                mapping = self._replace_mapping_state(mapping, mapped_state, status=status)
                with self._lock:
                    self._mappings[str(client_order_id)] = mapping

            return Result.success(mapping)
        except BaseException as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, self._STATUS_FAILED_REASON)

    def get_all_orders(self) -> Result[list[IntentOrderMapping]]:
        """Return all locally tracked orders as mappings.

        This is intentionally limited to orders submitted via this adapter
        instance (tracked in-memory through ``_mappings``). Brokers may expose
        additional open orders not initiated by AST; those should be surfaced
        by higher-level reconciliation paths if needed.

        Returns:
            Result[list[IntentOrderMapping]]: Mappings for all known orders.
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            err = RuntimeError("ib_insync is not available")
            err.__cause__ = _IB_INSYNC_IMPORT_ERROR
            return Result.failed(err, self._DEPENDENCY_MISSING_REASON)

        try:
            self._ib.sleep(0)
            with self._lock:
                return Result.success(list(self._mappings.values()))
        except BaseException as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, self._STATUS_FAILED_REASON)

    def fetch_broker_order(self, mapping: Any) -> "Result[dict[str, Any] | None]":
        """Fetch broker order as a dict for reconciliation.

        Converts an ib_insync Trade object into the dict format expected by
        the Reconciler (keys: status, broker_order_id, client_order_id, fills).
        """
        try:
            self._ib.sleep(0)
            client_oid = getattr(mapping, "client_order_id", None) or ""
            with self._lock:
                trade = self._trades.get(str(client_oid))
            if trade is None:
                return Result.success(None)

            order_status = getattr(trade, "orderStatus", None)
            status_str = getattr(order_status, "status", "") or ""
            filled_qty = float(getattr(order_status, "filled", 0) or 0)
            remaining = float(getattr(order_status, "remaining", 0) or 0)
            avg_price = float(getattr(order_status, "avgFillPrice", 0) or 0)

            fills_list: list[dict[str, Any]] = []
            for fill in getattr(trade, "fills", []):
                exec_obj = getattr(fill, "execution", None)
                if exec_obj is None:
                    continue
                fills_list.append({
                    "trade_id": str(getattr(exec_obj, "execId", "")),
                    "qty": float(getattr(exec_obj, "shares", 0) or 0),
                    "price": float(getattr(exec_obj, "price", 0) or 0),
                    "time": str(getattr(exec_obj, "time", "")),
                })

            order_obj = getattr(trade, "order", None)
            return Result.success({
                "status": status_str,
                "broker_order_id": str(getattr(order_obj, "orderId", "") if order_obj else ""),
                "client_order_id": str(client_oid),
                "filled_quantity": filled_qty,
                "remaining_quantity": remaining,
                "avg_fill_price": avg_price,
                "fills": fills_list,
            })
        except BaseException as exc:  # noqa: BLE001
            return Result.failed(exc, self._STATUS_FAILED_REASON)

    def get_account_summary(self) -> Result[dict[str, Any]]:
        """Query IBKR account summary including cash, buying power, and equity.

        Uses ib_insync's accountSummary() to fetch key account metrics. The returned
        dictionary contains normalized field names mapped from IBKR's tag-value pairs.

        Returns:
            Result[dict[str, Any]]: Success with account summary dict containing:
                - "total_cash_value": float - Total cash balance (TotalCashValue tag)
                - "buying_power": float - Stock buying power (BuyingPower tag)
                - "net_liquidation": float - Net liquidation value (NetLiquidation tag)
                - "unrealized_pnl": float - Unrealized P&L (UnrealizedPnL tag)
                - "realized_pnl": float - Realized P&L (RealizedPnL tag)
                - "gross_position_value": float - Gross position value (GrossPositionValue tag)

            Failed/degraded results include reason codes:
                - DEPENDENCY_MISSING: ib_insync not available
                - CONNECTION_FAILED: Not connected to IBKR
                - ACCOUNT_QUERY_FAILED: Query failed or timed out
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            err = RuntimeError("ib_insync is not available")
            err.__cause__ = _IB_INSYNC_IMPORT_ERROR
            return Result.failed(err, self._DEPENDENCY_MISSING_REASON)

        if not self._ib.isConnected():
            return Result.failed(ConnectionError("Not connected to IBKR."), self._CONNECT_FAILED_REASON)

        def _to_float(value: Any) -> float | None:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    return float(stripped)
                except ValueError:
                    return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        tag_to_field = {
            # IBKR tag -> normalized field name
            "TotalCashValue": "total_cash_value",
            "BuyingPower": "buying_power",
            "NetLiquidation": "net_liquidation",
            "UnrealizedPnL": "unrealized_pnl",
            "RealizedPnL": "realized_pnl",
            "GrossPositionValue": "gross_position_value",
        }
        summary: dict[str, Any] = dict.fromkeys(tag_to_field.values(), None)

        try:
            account_values = self._ib.accountSummary()
            self._ib.sleep(0)
            for account_value in account_values:
                tag = getattr(account_value, "tag", None)
                if not isinstance(tag, str) or tag not in tag_to_field:
                    continue
                field = tag_to_field[tag]
                summary[field] = _to_float(getattr(account_value, "value", None))

            # Multi-currency cash balances from accountValues().
            currency_balances: dict[str, dict[str, float]] = {}
            try:
                for av in self._ib.accountValues():
                    tag = getattr(av, "tag", "")
                    cur = getattr(av, "currency", "")
                    if tag == "CashBalance" and cur and cur != "BASE":
                        val = _to_float(getattr(av, "value", None))
                        if val is not None and val != 0.0:
                            currency_balances[cur] = {"cash": val}
                    elif tag == "ExchangeRate" and cur and cur != "BASE":
                        val = _to_float(getattr(av, "value", None))
                        if val is not None and cur in currency_balances:
                            currency_balances[cur]["exchange_rate"] = val
            except Exception:  # noqa: BLE001
                pass
            if currency_balances:
                summary["currency_balances"] = currency_balances

            return Result.success(data=summary)
        except BaseException as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, self._ACCOUNT_QUERY_FAILED_REASON)

    def get_portfolio_positions(self) -> Result[list[dict[str, Any]]]:
        """Query current portfolio positions from IBKR.

        Uses ib_insync's portfolio() to fetch all open positions. Each position is
        normalized into a dictionary with consistent field names.

        Returns:
            Result[list[dict[str, Any]]]: Success with list of position dicts, each containing:
                - "symbol": str - Stock symbol
                - "position": float - Position size (positive for long, negative for short)
                - "market_value": float - Current market value
                - "avg_cost": float - Average cost basis
                - "unrealized_pnl": float - Unrealized P&L for this position
                - "realized_pnl": float - Realized P&L for this position (if available)
                - "contract_id": int - IBKR contract ID (conId)

            Failed/degraded results include reason codes:
                - DEPENDENCY_MISSING: ib_insync not available
                - CONNECTION_FAILED: Not connected to IBKR
                - PORTFOLIO_QUERY_FAILED: Query failed or timed out
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            err = RuntimeError("ib_insync is not available")
            err.__cause__ = _IB_INSYNC_IMPORT_ERROR
            return Result.failed(err, self._DEPENDENCY_MISSING_REASON)

        if not self._ib.isConnected():
            return Result.failed(ConnectionError("Not connected to IBKR."), self._CONNECT_FAILED_REASON)

        def _to_float(value: Any) -> float | None:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    return float(stripped)
                except ValueError:
                    return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        try:
            portfolio_items = self._ib.portfolio()
            self._ib.sleep(0)

            positions: list[dict[str, Any]] = []
            for item in portfolio_items:
                contract = getattr(item, "contract", None)
                symbol = getattr(contract, "symbol", None)
                symbol_value = str(symbol) if symbol is not None else ""

                con_id = getattr(contract, "conId", None)
                contract_id: int | None
                if isinstance(con_id, int):
                    contract_id = con_id
                else:
                    try:
                        contract_id = int(con_id) if con_id is not None else None
                    except (TypeError, ValueError):
                        contract_id = None

                positions.append(
                    {
                        "symbol": symbol_value,
                        "position": _to_float(getattr(item, "position", None)),
                        "market_value": _to_float(getattr(item, "marketValue", None)),
                        "avg_cost": _to_float(getattr(item, "averageCost", None)),
                        "unrealized_pnl": _to_float(getattr(item, "unrealizedPNL", None)),
                        "realized_pnl": _to_float(getattr(item, "realizedPNL", None)),
                        "contract_id": contract_id,
                    }
                )

            return Result.success(data=positions)
        except BaseException as exc:  # noqa: BLE001 - boundary returns Result
            return Result.failed(exc, self._PORTFOLIO_QUERY_FAILED_REASON)

    _OPEN_ORDERS_QUERY_FAILED_REASON = "OPEN_ORDERS_QUERY_FAILED"

    def get_open_bracket_orders(self) -> Result[dict[str, dict[str, Any]]]:
        """Query IBKR for all open orders and return SL/TP grouped by symbol.

        Uses ``reqAllOpenOrders`` to fetch orders from **all** client IDs so
        that manually placed or repair-script orders are also visible.

        Returns:
            Result[dict[str, dict]]: ``{symbol: {"sl": {...}, "tp": {...}}}``
            where each sub-dict contains ``price``, ``qty``, and ``status``.
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            return Result.failed(
                RuntimeError("ib_insync is not available"),
                self._DEPENDENCY_MISSING_REASON,
            )

        if not self._ib.isConnected():
            return Result.failed(
                ConnectionError("Not connected to IBKR."),
                self._CONNECT_FAILED_REASON,
            )

        try:
            self._ib.reqAllOpenOrders()
            self._ib.sleep(1)
            trades = self._ib.openTrades()

            result: dict[str, dict[str, Any]] = {}
            for trade in trades:
                order = trade.order
                contract = trade.contract
                status = trade.orderStatus

                if not hasattr(contract, "symbol") or not contract.symbol:
                    continue
                if order.action != "SELL":
                    continue

                symbol = contract.symbol
                entry = result.setdefault(symbol, {})
                qty = float(order.totalQuantity or 0)
                status_str = str(status.status) if status else "UNKNOWN"

                if order.orderType == "STP":
                    price = float(order.auxPrice or 0)
                    entry["sl"] = {"price": price, "qty": qty, "status": status_str}
                elif order.orderType == "LMT":
                    price = float(order.lmtPrice or 0)
                    entry["tp"] = {"price": price, "qty": qty, "status": status_str}

            return Result.success(data=result)
        except BaseException as exc:  # noqa: BLE001
            return Result.failed(exc, self._OPEN_ORDERS_QUERY_FAILED_REASON)

    _BRACKET_REPAIR_FAILED_REASON = "BRACKET_REPAIR_FAILED"

    def repair_bracket_quantities(self) -> Result[dict[str, Any]]:
        """Compare IBKR positions vs open SL/TP orders; cancel+resubmit mismatches.

        This method operates entirely from broker-side data (positions and open
        orders), bypassing local mapping state which may be stale across runs.

        For each symbol where an open SL or TP order quantity differs from the
        actual position size, the mismatched bracket legs are cancelled and
        resubmitted with the correct quantity, preserving price and OCA group.

        Returns:
            Result[dict]: ``aligned``, ``repaired``, ``failures`` counts and details.
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            return Result.failed(
                RuntimeError("ib_insync is not available"),
                self._DEPENDENCY_MISSING_REASON,
            )

        if self._config.readonly:
            return Result.failed(
                PermissionError("Adapter is in readonly mode."),
                self._READONLY_REASON,
            )

        if not self._ib.isConnected():
            return Result.failed(
                ConnectionError("Not connected to IBKR."),
                self._CONNECT_FAILED_REASON,
            )

        try:
            # 1. Snapshot positions: {symbol: abs_qty}
            positions: dict[str, float] = {}
            for p in self._ib.positions():
                sym = getattr(getattr(p, "contract", None), "symbol", None)
                qty = abs(float(getattr(p, "position", 0) or 0))
                if sym and qty > 0:
                    positions[sym] = qty

            # 2. Snapshot open bracket orders grouped by symbol
            self._ib.reqAllOpenOrders()
            self._ib.sleep(1)
            open_trades = self._ib.openTrades()

            # Group SELL orders by symbol → list of (trade, order_type, price, qty, oca_group)
            bracket_orders: dict[str, list[dict[str, Any]]] = {}
            for trade in open_trades:
                order = trade.order
                contract = trade.contract
                sym = getattr(contract, "symbol", None)
                if not sym or getattr(order, "action", "") != "SELL":
                    continue

                qty = float(getattr(order, "totalQuantity", 0) or 0)
                order_type = getattr(order, "orderType", "")
                oca_group = getattr(order, "ocaGroup", "") or ""

                if order_type == "STP":
                    price = float(getattr(order, "auxPrice", 0) or 0)
                    leg_type = "SL"
                elif order_type == "LMT":
                    price = float(getattr(order, "lmtPrice", 0) or 0)
                    leg_type = "TP"
                else:
                    continue

                bracket_orders.setdefault(sym, []).append({
                    "trade": trade,
                    "leg_type": leg_type,
                    "price": price,
                    "qty": qty,
                    "oca_group": oca_group,
                })

            # 3. Compare and repair
            aligned = 0
            repaired: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []

            for sym, legs in bracket_orders.items():
                pos_qty = positions.get(sym, 0)
                if pos_qty <= 0:
                    # Orphan bracket (no position) — skip, not our concern here
                    continue

                for leg in legs:
                    if abs(leg["qty"] - pos_qty) < 0.001:
                        aligned += 1
                        continue

                    # Mismatch: cancel old, resubmit with correct qty
                    old_trade = leg["trade"]
                    old_order = old_trade.order
                    contract = old_trade.contract

                    try:
                        self._ib.cancelOrder(old_order)
                        self._ib.sleep(0.5)
                    except BaseException as cancel_exc:  # noqa: BLE001
                        failures.append({
                            "symbol": sym,
                            "leg_type": leg["leg_type"],
                            "old_qty": leg["qty"],
                            "target_qty": pos_qty,
                            "stage": "cancel",
                            "error": str(cancel_exc),
                        })
                        continue

                    # Build replacement order preserving type, price, OCA
                    try:
                        if leg["leg_type"] == "SL":
                            new_order = StopOrder("SELL", pos_qty, self._round_price(leg["price"]))
                        else:
                            new_order = LimitOrder("SELL", pos_qty, self._round_price(leg["price"]))

                        new_order.tif = "GTC"
                        new_order.outsideRth = False
                        if self._config.account is not None:
                            new_order.account = str(self._config.account)
                        if leg["oca_group"]:
                            new_order.ocaGroup = leg["oca_group"]
                            new_order.ocaType = self._DEFAULT_OCA_TYPE

                        new_trade = self._ib.placeOrder(contract, new_order)
                        self._ib.sleep(0)

                        repaired.append({
                            "symbol": sym,
                            "leg_type": leg["leg_type"],
                            "old_qty": leg["qty"],
                            "new_qty": pos_qty,
                            "price": leg["price"],
                            "new_order_id": getattr(getattr(new_trade, "order", None), "orderId", None),
                        })
                    except BaseException as submit_exc:  # noqa: BLE001
                        failures.append({
                            "symbol": sym,
                            "leg_type": leg["leg_type"],
                            "old_qty": leg["qty"],
                            "target_qty": pos_qty,
                            "stage": "resubmit",
                            "error": str(submit_exc),
                        })

            return Result.success(data={
                "aligned": aligned,
                "repaired": repaired,
                "failures": failures,
            })
        except BaseException as exc:  # noqa: BLE001
            return Result.failed(exc, self._BRACKET_REPAIR_FAILED_REASON)

    def liquidate_all_positions(self, *, reason: str = "MANUAL_LIQUIDATION") -> Result[dict[str, Any]]:
        """Liquidate all open positions at market price.

        This is a critical risk control feature that allows manual intervention
        to close all positions immediately. Use with caution.

        Notes:
            - This method submits one market order per open position.
            - Individual submission failures are captured and do not stop the
              overall liquidation loop. Callers should review the returned
              summary and take follow-up actions if needed.
            - Orders are submitted directly via ``ib.placeOrder`` and are not
              tracked in the adapter's intent/mapping tables (this is an
              out-of-band safety operation).

        Args:
            reason: Reason for liquidation (for audit logging). Default: "MANUAL_LIQUIDATION"

        Returns:
            Result[dict[str, Any]]: Success with liquidation summary dict containing:
                - "positions_count": int - Number of positions found
                - "orders_submitted": int - Number of liquidation orders submitted
                - "orders_failed": int - Number of failed submissions
                - "positions": list[dict] - List of position details
                - "orders": list[dict] - List of submitted order details
                - "errors": list[str] - List of error messages (if any)
                - "reason": str - Liquidation reason

            Failed results include reason codes:
                - DEPENDENCY_MISSING: ib_insync not available
                - CONNECTION_FAILED: Not connected to IBKR
                - READONLY_MODE: Adapter in readonly mode (cannot submit orders)
                - LIQUIDATION_FAILED: Liquidation process failed
        """

        if _IB_INSYNC_IMPORT_ERROR is not None:
            err = RuntimeError("ib_insync is not available")
            err.__cause__ = _IB_INSYNC_IMPORT_ERROR
            return Result.failed(err, self._DEPENDENCY_MISSING_REASON)

        if self._config.readonly:
            return Result.failed(
                PermissionError("Adapter is in readonly mode; liquidation is disabled."),
                self._READONLY_REASON,
            )

        if not self._ib.isConnected():
            return Result.failed(ConnectionError("Not connected to IBKR."), self._CONNECT_FAILED_REASON)

        # Use one consistent orderRef across this liquidation batch for audit
        # and quick identification in TWS/IB Gateway.
        safe_reason = str(reason).strip().upper() if str(reason).strip() else "MANUAL_LIQUIDATION"
        batch_timestamp = int(time.time())
        order_ref = f"LIQUIDATION_{safe_reason}_{batch_timestamp}"

        positions_result = self.get_portfolio_positions()
        if positions_result.status is not ResultStatus.SUCCESS:
            error = positions_result.error or RuntimeError("Failed to query portfolio positions.")
            return Result.failed(error, self._LIQUIDATION_FAILED_REASON)

        positions = positions_result.data or []
        submitted_orders: list[dict[str, Any]] = []
        errors: list[str] = []
        orders_submitted = 0
        orders_failed = 0

        for position_info in positions:
            # Defensive parsing: get_portfolio_positions normalizes fields, but
            # liquidation is safety-critical and should tolerate partial data.
            try:
                symbol = str(position_info.get("symbol") or "").strip().upper()
                raw_position = position_info.get("position")
                position_size = float(raw_position) if raw_position is not None else 0.0
            except BaseException as exc:  # noqa: BLE001 - continue liquidation loop
                orders_failed += 1
                errors.append(f"Failed to parse position record {position_info!r}: {exc}")
                continue

            if not symbol:
                orders_failed += 1
                errors.append(f"Skipping position with missing symbol: {position_info!r}")
                continue

            if position_size == 0:
                # No exposure; nothing to liquidate for this record.
                continue

            action = "SELL" if position_size > 0 else "BUY"
            intent_type = IntentType.CLOSE_LONG if position_size > 0 else IntentType.CLOSE_SHORT
            quantity = abs(position_size)

            # Build a temporary TradeIntent so we can reuse adapter order-building
            # logic (account routing, market-vs-limit selection, etc.).
            intent = TradeIntent(
                intent_id=f"LIQUIDATE_{symbol}_{time.time_ns()}",
                symbol=symbol,
                intent_type=intent_type,
                quantity=float(quantity),
                created_at_ns=time.time_ns(),
                entry_price=None,  # Market order to maximize immediacy.
                stop_loss_price=None,
                take_profit_price=None,
                metadata={"liquidation_reason": safe_reason},
            )

            try:
                contract = Stock(self._normalize_symbol(symbol), "SMART", "USD")
                qualified = self._ib.qualifyContracts(contract)
                self._ib.sleep(0)
                if not qualified:
                    raise RuntimeError("Contract qualification failed.")
                contract = qualified[0]

                order = self._build_order(intent=intent, action=action, client_order_id=intent.intent_id)
                # Override orderRef with the liquidation batch reference as requested.
                order.orderRef = order_ref

                trade = self._ib.placeOrder(contract, order)
                self._ib.sleep(0)

                broker_order_id = getattr(getattr(trade, "order", None), "orderId", None)
                submitted_orders.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "quantity": float(quantity),
                        "order_ref": order_ref,
                        "intent_id": intent.intent_id,
                        "broker_order_id": str(broker_order_id) if broker_order_id is not None else None,
                    }
                )
                orders_submitted += 1
            except BaseException as exc:  # noqa: BLE001 - continue liquidation loop
                orders_failed += 1
                errors.append(f"{symbol}: order submission failed: {exc}")
                continue

        summary: dict[str, Any] = {
            "positions_count": len(positions),
            "orders_submitted": orders_submitted,
            "orders_failed": orders_failed,
            "positions": positions,
            "orders": submitted_orders,
            "errors": errors,
            "reason": safe_reason,
        }
        return Result.success(data=summary)

    _TERMINAL_STATES = frozenset({
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    })

    _ENTRY_INTENT_TYPES = frozenset({IntentType.OPEN_LONG, IntentType.OPEN_SHORT})

    def _on_order_status(self, trade: Trade, *_: Any) -> None:
        """Order status update callback.

        Updates internal mapping state for the associated ``client_order_id``.
        When an entry order transitions to PARTIALLY_FILLED, triggers immediate
        realignment of sibling SL/TP quantities to match the filled quantity.

        Args:
            trade: ``ib_insync.Trade`` with updated order status fields.
        """

        try:
            client_order_id = self._extract_client_order_id(trade)
            if client_order_id is None:
                return

            with self._lock:
                current = self._mappings.get(client_order_id)

            if current is None:
                return

            status = getattr(getattr(trade, "orderStatus", None), "status", None)
            if not isinstance(status, str) or not status:
                return

            new_state = self._map_ibkr_status_to_order_state(status)

            broker_order_id = None
            order_id_value = getattr(getattr(trade, "order", None), "orderId", None)
            if order_id_value is not None:
                broker_order_id = str(order_id_value)

            updated = self._replace_mapping_state(
                current,
                new_state,
                status=status,
                broker_order_id=broker_order_id,
                trade=trade,
            )

            with self._lock:
                self._trades[client_order_id] = trade
                self._mappings[client_order_id] = updated

            # Real-time sibling realignment on partial OR final fill of entry
            # orders.  Without FILLED here, the last partial → full transition
            # would leave SL/TP at the previous partial qty.
            snap = current.intent_snapshot
            snap_intent_type = (
                snap.get("intent_type") if isinstance(snap, dict)
                else getattr(snap, "intent_type", None)
            ) if snap is not None else None
            if (
                new_state in (OrderState.PARTIALLY_FILLED, OrderState.FILLED)
                and snap_intent_type in self._ENTRY_INTENT_TYPES
            ):
                filled_qty = float(getattr(getattr(trade, "orderStatus", None), "filled", 0) or 0)
                if filled_qty > 0:
                    self._realign_sibling_quantities(current.intent_id, filled_qty)
        except BaseException:  # noqa: BLE001 - callbacks should not raise
            return

    def _realign_sibling_quantities(self, entry_intent_id: str, filled_qty: float) -> None:
        """Cancel and resubmit SL/TP siblings whose quantity doesn't match *filled_qty*.

        Called from ``_on_order_status`` when an entry order is partially filled.
        Failures are swallowed — polling serves as safety net.
        """

        if filled_qty <= 0:
            return

        with self._lock:
            siblings = [
                (coid, m)
                for coid, m in self._mappings.items()
                if (
                    m.intent_snapshot is not None
                    and getattr(m.intent_snapshot, "parent_intent_id", None) == entry_intent_id
                    and m.intent_snapshot.intent_type in {IntentType.STOP_LOSS, IntentType.TAKE_PROFIT}
                    and m.state not in self._TERMINAL_STATES
                )
            ]

        for client_order_id, mapping in siblings:
            try:
                current_qty = float(mapping.intent_snapshot.quantity)
                if current_qty == filled_qty:
                    continue

                cancel_result = self.cancel_order(client_order_id)
                if cancel_result.status is not ResultStatus.SUCCESS:
                    continue  # Cancel failed → do NOT resubmit (avoid double-active)

                new_intent = msgspec.structs.replace(
                    mapping.intent_snapshot,
                    quantity=filled_qty,
                )
                self.submit_order(new_intent)
            except BaseException:  # noqa: BLE001 - callback safety
                continue

    def _on_execution(self, trade: Trade, fill: Any, *_: Any) -> None:
        """Execution fill callback.

        Records fill details in ``_fills`` keyed by ``client_order_id``.

        Args:
            trade: ``ib_insync.Trade`` associated with this execution.
            fill: ``ib_insync.objects.Fill`` (or compatible object).
        """

        try:
            client_order_id = self._extract_client_order_id(trade)
            if client_order_id is None:
                return

            detail = self._to_fill_detail(fill)

            with self._lock:
                fills = self._fills.setdefault(client_order_id, [])
                if any(existing.execution_id == detail.execution_id for existing in fills):
                    return
                fills.append(detail)

            if self._fills_persistence is not None:
                try:
                    self._fills_persistence.save_fill(client_order_id, detail)
                except BaseException:  # noqa: BLE001 - persistence failure must not crash callback
                    pass
        except BaseException:  # noqa: BLE001 - callbacks should not raise
            return

    def _on_error(self, reqId: int, errorCode: int, errorString: str, *_: Any) -> None:
        """Error callback from IBKR.

        Categorizes messages into connection-related errors vs informational
        status updates. The adapter does not raise from callbacks; it may update
        internal connection bookkeeping for reconnect workflows.

        Args:
            reqId: Request id.
            errorCode: IB error code.
            errorString: Human-readable error string.
        """

        try:
            _ = reqId
            is_connection_error = self._is_connection_error(int(errorCode), str(errorString))
            if not is_connection_error:
                return
            with self._lock:
                self._last_connection_error = (int(errorCode), str(errorString))
        except BaseException:  # noqa: BLE001 - callbacks should not raise
            return

    def _on_connected(self) -> None:
        """Connection success callback."""

        try:
            with self._lock:
                self._reconnect_attempts = 0
        except BaseException:  # noqa: BLE001 - callbacks should not raise
            return

    def _on_disconnected(self) -> None:
        """Disconnected callback.

        Attempts best-effort reconnection using config max attempts. This is
        intentionally conservative to avoid infinite reconnect loops.
        Skipped during force-disconnect to avoid interfering with the reconnect.
        """

        try:
            if self._force_disconnecting:
                return
            with self._lock:
                self._reconnect_attempts += 1
                attempts = self._reconnect_attempts
            if attempts > max(0, int(self._config.max_reconnect_attempts)):
                return
            self.connect()
        except BaseException:  # noqa: BLE001 - callbacks should not raise
            return

    def _map_ibkr_status_to_order_state(self, status: str) -> OrderState:
        """Map IBKR order status string to AST ``OrderState``.

        Mapping rules (per Execution adapter contract):
            - 'PreSubmitted'/'ApiPending'/'PendingSubmit' -> SUBMITTED
            - 'Submitted' -> SUBMITTED
            - 'Filled' -> FILLED
            - 'Cancelled'/'ApiCancelled'/'Inactive' -> CANCELLED
            - 'PendingCancel' -> SUBMITTED (cancelling)
        """

        normalized = str(status).strip()
        if normalized in {"PreSubmitted", "ApiPending", "PendingSubmit", "Submitted", "PendingCancel"}:
            return OrderState.SUBMITTED
        if normalized == "PartiallyFilled":
            return OrderState.PARTIALLY_FILLED
        if normalized == "Filled":
            return OrderState.FILLED
        if normalized in {"Cancelled", "ApiCancelled", "Inactive"}:
            return OrderState.CANCELLED
        return OrderState.SUBMITTED

    def _extract_client_order_id(self, trade: Trade) -> str | None:
        """Extract AST client_order_id from a trade.

        Prefers ``orderRef`` which this adapter sets to ``client_order_id``.
        """

        order = getattr(trade, "order", None)
        ref = getattr(order, "orderRef", None) if order is not None else None
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
        return None

    def _replace_mapping_state(
        self,
        mapping: IntentOrderMapping,
        state: OrderState,
        *,
        status: str | None = None,
        broker_order_id: str | None = None,
        trade: Trade | None = None,
    ) -> IntentOrderMapping:
        """Return an updated mapping with refreshed state/metadata."""

        metadata = dict(mapping.metadata or {})
        ibkr_meta = dict(metadata.get("ibkr") or {})
        if status is not None:
            ibkr_meta["status"] = status
        if broker_order_id is not None:
            ibkr_meta["orderId"] = broker_order_id
        if trade is not None:
            filled = getattr(getattr(trade, "orderStatus", None), "filled", None)
            remaining = getattr(getattr(trade, "orderStatus", None), "remaining", None)
            avg_fill = getattr(getattr(trade, "orderStatus", None), "avgFillPrice", None)
            if filled is not None:
                ibkr_meta["filled"] = float(filled)
            if remaining is not None:
                ibkr_meta["remaining"] = float(remaining)
            if avg_fill is not None:
                ibkr_meta["avgFillPrice"] = float(avg_fill)
        metadata["ibkr"] = ibkr_meta

        return msgspec.structs.replace(
            mapping,
            state=state,
            broker_order_id=mapping.broker_order_id if broker_order_id is None else broker_order_id,
            updated_at_ns=time.time_ns(),
            metadata=metadata,
        )

    @staticmethod
    def _round_price(price: float) -> float:
        """Round price to IBKR-compatible tick size ($0.01 for stocks >= $1)."""
        return round(price, 2)

    @staticmethod
    def _derive_oca_group(parent_intent_id: str) -> str:
        """Derive a deterministic OCA group name from the parent intent id.

        Same parent_intent_id always yields the same group — idempotent across
        cancel+resubmit cycles.
        """
        digest = hashlib.sha256(parent_intent_id.encode()).hexdigest()[:16]
        return f"AST-OCA-{digest}"

    def _build_order(self, *, intent: TradeIntent, action: str, client_order_id: str) -> Any:
        """Build an IB order for the given intent.

        Args:
            intent: Trade intent to translate.
            action: 'BUY' or 'SELL'.
            client_order_id: AST client order id (stored in IB ``orderRef``).

        Returns:
            Any: ``ib_insync`` order object (MarketOrder/LimitOrder/StopOrder).
        """

        qty = float(intent.quantity)
        if qty <= 0:
            raise ValueError("intent.quantity must be > 0")

        # Determine order type based on intent type:
        # - STOP_LOSS → StopOrder at stop_loss_price
        # - TAKE_PROFIT → LimitOrder at take_profit_price
        # - OPEN_LONG/OPEN_SHORT → LimitOrder at entry_price, or MarketOrder if no price
        if intent.intent_type == IntentType.STOP_LOSS and intent.stop_loss_price is not None:
            order = StopOrder(action, qty, self._round_price(float(intent.stop_loss_price)))
        elif intent.intent_type == IntentType.TAKE_PROFIT and intent.take_profit_price is not None:
            order = LimitOrder(action, qty, self._round_price(float(intent.take_profit_price)))
        elif intent.entry_price is not None:
            order = LimitOrder(action, qty, self._round_price(float(intent.entry_price)))
        else:
            order = MarketOrder(action, qty)

        # Apply common order attributes (best-effort for test doubles).
        try:
            order.tif = "GTC"  # Good Till Cancelled for bracket legs
        except AttributeError:
            pass
        try:
            order.outsideRth = False
        except AttributeError:
            pass

        order.orderRef = str(client_order_id)
        if self._config.account is not None:
            order.account = str(self._config.account)

        # OCA group: link SL/TP siblings so IBKR auto-cancels the other on fill.
        parent_id = getattr(intent, "parent_intent_id", None)
        if parent_id and intent.intent_type in {IntentType.STOP_LOSS, IntentType.TAKE_PROFIT}:
            order.ocaGroup = self._derive_oca_group(str(parent_id))
            order.ocaType = self._DEFAULT_OCA_TYPE

        return order

    _POSITION_GUARD_REASON = "POSITION_GUARD_BLOCKED"

    def _position_guard(self, intent: TradeIntent, action: str, contract: Any) -> Result[IntentOrderMapping] | None:
        """Block SELL orders that would create or increase a short position.

        Safety rules:
            1. SELL is only allowed if we hold a long position in the symbol.
            2. SELL quantity must not exceed current position size.
            3. BUY orders for OPEN_LONG are always allowed (no guard needed).

        Returns:
            ``None`` if the order is safe to proceed.
            ``Result.failed(...)`` if the order must be blocked.
        """

        # Only guard SELL orders — BUY orders for opening longs are always safe.
        if action != "SELL":
            return None

        # Skip guard for bracket legs (SL/TP) linked to an OPEN_LONG parent.
        # These are conditional orders that only trigger after the parent fills,
        # so there WILL be a position when they execute.
        if getattr(intent, "parent_intent_id", None) is not None:
            return None

        # Query current position from broker.
        symbol = str(intent.symbol).upper()
        current_position: float = 0.0
        try:
            positions = self._ib.positions()
            for pos in positions:
                pos_symbol = getattr(getattr(pos, "contract", None), "symbol", "")
                if str(pos_symbol).upper() == symbol:
                    current_position = float(pos.position)
                    break
        except Exception:  # noqa: BLE001
            # If we can't query positions, block the sell as fail-closed.
            return Result.failed(
                RuntimeError(f"Cannot verify position for {symbol}; blocking SELL as safety precaution."),
                self._POSITION_GUARD_REASON,
            )

        order_qty = float(intent.quantity)

        # Rule 1: No selling without an existing long position.
        if current_position <= 0:
            return Result.failed(
                RuntimeError(
                    f"POSITION_GUARD: SELL {order_qty} {symbol} blocked — "
                    f"no long position (current: {current_position}). "
                    f"Would create naked short."
                ),
                self._POSITION_GUARD_REASON,
            )

        # Rule 2: Sell quantity must not exceed current holding.
        if order_qty > current_position:
            return Result.failed(
                RuntimeError(
                    f"POSITION_GUARD: SELL {order_qty} {symbol} blocked — "
                    f"exceeds current position ({current_position}). "
                    f"Would flip to short."
                ),
                self._POSITION_GUARD_REASON,
            )

        return None  # Safe to proceed.

    def _infer_action(self, intent: TradeIntent) -> str:
        """Infer IB order action ('BUY'/'SELL') from ``TradeIntent``."""

        if isinstance(intent.metadata, dict):
            action_hint = intent.metadata.get("action")
            if isinstance(action_hint, str) and action_hint.strip().upper() in {"BUY", "SELL"}:
                return action_hint.strip().upper()

            side_hint = intent.metadata.get("position_side")
            if isinstance(side_hint, str) and intent.intent_type in {
                IntentType.STOP_LOSS,
                IntentType.TAKE_PROFIT,
                IntentType.REDUCE_POSITION,
            }:
                side = side_hint.strip().upper()
                if side == "SHORT":
                    return "BUY"
                if side == "LONG":
                    return "SELL"

        if intent.intent_type in {IntentType.OPEN_LONG, IntentType.CLOSE_SHORT}:
            return "BUY"
        if intent.intent_type in {IntentType.OPEN_SHORT, IntentType.CLOSE_LONG}:
            return "SELL"
        if intent.intent_type in {IntentType.STOP_LOSS, IntentType.TAKE_PROFIT, IntentType.REDUCE_POSITION}:
            return "SELL"

        raise ValueError(f"Unsupported intent_type for IBKR adapter: {intent.intent_type}")

    def _to_fill_detail(self, fill: Any) -> FillDetail:
        """Convert an ib_insync Fill into ``FillDetail``."""

        execution = getattr(fill, "execution", None)
        commission_report = getattr(fill, "commissionReport", None)

        exec_id = getattr(execution, "execId", None)
        if exec_id is None:
            exec_id = getattr(execution, "permId", None)
        if exec_id is None:
            exec_id = f"exec-{time.time_ns()}"

        price = getattr(execution, "price", None)
        shares = getattr(execution, "shares", None)
        dt = getattr(execution, "time", None)

        fill_time_ns = time.time_ns()
        if isinstance(dt, datetime):
            dt_utc = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
            fill_time_ns = int(dt_utc.timestamp() * 1e9)

        commission = getattr(commission_report, "commission", None)
        commission_value: float | None = None
        if commission is not None:
            try:
                commission_value = float(commission)
            except Exception:
                commission_value = None

        return FillDetail(
            execution_id=str(exec_id),
            fill_price=float(price) if price is not None else 0.0,
            fill_quantity=float(shares) if shares is not None else 0.0,
            fill_time_ns=int(fill_time_ns),
            commission=commission_value,
            metadata={},
        )

    def _is_connection_error(self, error_code: int, error_string: str) -> bool:
        """Heuristic classifier for IBKR error messages."""

        connection_codes = {
            502,  # Couldn't connect to TWS
            504,  # Not connected
            326,  # client_id is already in use
            1100,  # Connectivity between IB and TWS has been lost
            1101,  # Connectivity restored - data lost
            1102,  # Connectivity restored - data maintained
            1300,  # Socket closed
        }
        if error_code in connection_codes:
            return True
        text = error_string.lower()
        return "not connected" in text or "connect" in text or "connection" in text

    def _is_client_id_conflict(self, error_code: int, error_string: str) -> bool:
        """Check if error indicates client_id conflict (Error 326)."""

        if error_code == 326:
            return True
        text = error_string.lower()
        return "client id" in text and ("already in use" in text or "in use" in text)

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize input symbol to IBKR-friendly format."""

        return str(symbol).strip().upper()

    def _get_run_id(self, intent: TradeIntent) -> str:
        """Best-effort extraction of run_id for client order id generation."""

        if isinstance(intent.metadata, dict):
            run_id = intent.metadata.get("run_id")
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
        return "unknown"
