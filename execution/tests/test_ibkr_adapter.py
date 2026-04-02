"""Unit tests for the IBKRAdapter implementation (Phase 3.5)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from common.interface import Result, ResultStatus
from execution.ibkr_adapter import IBKRAdapter
from execution.interface import BrokerConnectionConfig, FillDetail
from order_state_machine import IDGenerator, OrderState
from order_state_machine.interface import IntentOrderMapping
from strategy.interface import IntentType, TradeIntent


class _FakeEvent(list):
    """Minimal ib_insync-like event supporting ``+= callback`` registration."""

    def __iadd__(self, callback):  # type: ignore[override]
        self.append(callback)
        return self


@dataclass(slots=True)
class _FakeContract:
    conId: int = 1
    symbol: str = ""


@dataclass(slots=True)
class _FakeOrder:
    action: str
    totalQuantity: float
    lmtPrice: float | None = None
    auxPrice: float | None = None
    orderRef: str | None = None
    orderId: int | None = None
    orderType: str | None = None
    account: str | None = None
    tif: str | None = None
    outsideRth: bool | None = None
    ocaGroup: str | None = None
    ocaType: int | None = None


@dataclass(slots=True)
class _FakeOrderStatus:
    status: str = "Submitted"
    filled: float = 0.0
    remaining: float = 0.0
    avgFillPrice: float = 0.0


@dataclass(slots=True)
class _FakeTrade:
    order: _FakeOrder
    orderStatus: _FakeOrderStatus
    contract: _FakeContract | None = None


class _FakeIB:
    """Minimal IB stub for adapter unit tests."""

    def __init__(self) -> None:
        self.connectedEvent = _FakeEvent()
        self.disconnectedEvent = _FakeEvent()
        self.orderStatusEvent = _FakeEvent()
        self.execDetailsEvent = _FakeEvent()
        self.errorEvent = _FakeEvent()

        self._connected = False
        self.connect_calls: list[dict[str, object]] = []
        self.disconnect_called = False
        self.cancel_calls: list[_FakeOrder] = []

        self.connect_side_effects: list[BaseException | None] = []
        self.qualify_return: list[_FakeContract] = [_FakeContract(conId=101)]
        self.place_order_trade: _FakeTrade | None = None
        self._positions: list[Any] = []
        self._open_trades: list[_FakeTrade] = []
        self.placed_orders: list[tuple[Any, Any]] = []

    def isConnected(self) -> bool:  # noqa: N802 - matches ib_insync
        return self._connected

    def connect(self, **kwargs):  # noqa: ANN001 - test stub
        self.connect_calls.append(dict(kwargs))
        if self.connect_side_effects:
            effect = self.connect_side_effects.pop(0)
            if effect is not None:
                raise effect
        self._connected = True

    def disconnect(self) -> None:
        self.disconnect_called = True
        self._connected = False

    def sleep(self, *_args, **_kwargs) -> None:  # noqa: ANN001 - test stub
        return None

    def qualifyContracts(self, *_args, **_kwargs):  # noqa: ANN001 - test stub
        return list(self.qualify_return)

    def placeOrder(self, contract, order):  # noqa: ANN001 - test stub
        self.placed_orders.append((contract, order))
        if self.place_order_trade is not None:
            return self.place_order_trade
        order.orderType = "MKT" if order.lmtPrice is None else "LMT"
        order.orderId = 9001
        status = _FakeOrderStatus(status="Submitted", filled=0.0, remaining=float(order.totalQuantity))
        return _FakeTrade(order=order, orderStatus=status, contract=contract)

    def cancelOrder(self, order):  # noqa: ANN001 - test stub
        self.cancel_calls.append(order)

    def positions(self):  # noqa: ANN001 - test stub
        return list(self._positions)

    def reqAllOpenOrders(self):  # noqa: ANN001 - test stub
        pass

    def openTrades(self):  # noqa: ANN001 - test stub
        return list(self._open_trades)


def _patch_ib_insync(monkeypatch: pytest.MonkeyPatch) -> _FakeIB:
    """Patch ``execution.ibkr_adapter`` symbols to avoid real ib_insync dependency."""

    import execution.ibkr_adapter as ibkr_mod

    ib = _FakeIB()
    monkeypatch.setattr(ibkr_mod, "_IB_INSYNC_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(ibkr_mod, "IB", lambda: ib, raising=False)
    monkeypatch.setattr(ibkr_mod, "Stock", lambda *_a, **_k: _FakeContract(conId=101), raising=False)

    def _market_order(action: str, qty: float) -> _FakeOrder:
        return _FakeOrder(action=action, totalQuantity=float(qty), lmtPrice=None, orderType="MKT")

    def _limit_order(action: str, qty: float, price: float) -> _FakeOrder:
        return _FakeOrder(action=action, totalQuantity=float(qty), lmtPrice=float(price), orderType="LMT")

    def _stop_order(action: str, qty: float, stop_price: float) -> _FakeOrder:
        return _FakeOrder(action=action, totalQuantity=float(qty), lmtPrice=float(stop_price), orderType="STP")

    monkeypatch.setattr(ibkr_mod, "MarketOrder", _market_order, raising=False)
    monkeypatch.setattr(ibkr_mod, "LimitOrder", _limit_order, raising=False)
    monkeypatch.setattr(ibkr_mod, "StopOrder", _stop_order, raising=False)

    return ib


def _make_intent(
    *,
    intent_id: str = "I-1",
    intent_type: IntentType = IntentType.OPEN_LONG,
    quantity: float = 10.0,
    entry_price: float | None = None,
    metadata: dict[str, object] | None = None,
) -> TradeIntent:
    return TradeIntent(
        intent_id=intent_id,
        symbol="AAPL",
        intent_type=intent_type,
        quantity=float(quantity),
        created_at_ns=1_700_000_000_000_000_000,
        entry_price=entry_price,
        stop_loss_price=None,
        take_profit_price=None,
        parent_intent_id=None,
        linked_intent_ids=[],
        reduce_only=False,
        contingency_type=None,
        ladder_level=None,
        reason_codes=["TEST"],
        metadata=dict(metadata or {"run_id": "run-1", "position_side": "LONG"}),
    )


def test_connect_success(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """connect() returns SUCCESS when IB connects and reports connected."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    result = adapter.connect()

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert adapter.is_connected() is True


def test_register_event_handlers(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """__init__ registers event callbacks when ib_insync is available."""

    # Arrange
    ib = _patch_ib_insync(monkeypatch)

    # Act
    _ = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Assert
    assert len(ib.connectedEvent) == 1
    assert len(ib.disconnectedEvent) == 1
    assert len(ib.orderStatusEvent) == 1
    assert len(ib.execDetailsEvent) == 1
    assert len(ib.errorEvent) == 1


def test_connect_returns_success_when_already_connected(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """connect() is idempotent when already connected."""

    # Arrange
    ib = _patch_ib_insync(monkeypatch)
    ib._connected = True
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    result = adapter.connect()

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert len(ib.connect_calls) == 0


def test_connect_no_live_connection_returns_failed(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """connect() fails when IB.connect returns but isConnected() stays False."""

    # Arrange
    ib = _patch_ib_insync(monkeypatch)
    monkeypatch.setattr(ib, "isConnected", lambda: False)
    monkeypatch.setattr("execution.ibkr_adapter.time.sleep", lambda *_a, **_k: None)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    result = adapter.connect()

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CONNECTION_FAILED"


def test_connect_retry_on_failure(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """connect() retries with backoff when IB.connect raises."""

    # Arrange
    ib = _patch_ib_insync(monkeypatch)
    ib.connect_side_effects = [ConnectionError("boom"), None]
    monkeypatch.setattr("execution.ibkr_adapter.time.sleep", lambda *_a, **_k: None)

    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    result = adapter.connect()

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert len(ib.connect_calls) == 2


def test_connect_exhausts_retries(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """connect() returns FAILED when all retries are exhausted."""

    # Arrange
    ib = _patch_ib_insync(monkeypatch)
    ib.connect_side_effects = [ConnectionError("boom"), ConnectionError("boom2"), ConnectionError("boom3")]
    monkeypatch.setattr("execution.ibkr_adapter.time.sleep", lambda *_a, **_k: None)

    config = BrokerConnectionConfig(
        host=ibkr_adapter_config.host,
        port=ibkr_adapter_config.port,
        client_id=ibkr_adapter_config.client_id,
        readonly=ibkr_adapter_config.readonly,
        account=ibkr_adapter_config.account,
        timeout=ibkr_adapter_config.timeout,
        max_reconnect_attempts=3,
    )
    adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

    # Act
    result = adapter.connect()

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CONNECTION_FAILED"
    assert len(ib.connect_calls) == 3


def test_connect_missing_ib_insync(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """connect() fails with DEPENDENCY_MISSING when ib_insync is unavailable."""

    # Arrange
    import execution.ibkr_adapter as ibkr_mod

    monkeypatch.setattr(ibkr_mod, "IB", lambda: _FakeIB(), raising=False)
    monkeypatch.setattr(ibkr_mod, "_IB_INSYNC_IMPORT_ERROR", ImportError("missing"), raising=False)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    result = adapter.connect()

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "DEPENDENCY_MISSING"


def test_disconnect_success(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """disconnect() returns SUCCESS when IB disconnect succeeds."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter.connect()

    # Act
    result = adapter.disconnect()

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert adapter.is_connected() is False


def test_disconnect_failure_returns_failed(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """disconnect() returns FAILED when the underlying IB.disconnect raises."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    monkeypatch.setattr(adapter._ib, "disconnect", Mock(side_effect=RuntimeError("boom")))  # noqa: SLF001

    # Act
    result = adapter.disconnect()

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CONNECTION_FAILED"


def test_disconnect_missing_ib_insync(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """disconnect() fails with DEPENDENCY_MISSING when ib_insync is unavailable."""

    # Arrange
    import execution.ibkr_adapter as ibkr_mod

    monkeypatch.setattr(ibkr_mod, "IB", lambda: _FakeIB(), raising=False)
    monkeypatch.setattr(ibkr_mod, "_IB_INSYNC_IMPORT_ERROR", ImportError("missing"), raising=False)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    result = adapter.disconnect()

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "DEPENDENCY_MISSING"


class TestDynamicClientID:
    """Tests for dynamic client_id allocation (Error 326 conflict resolution)."""

    def test_init_active_client_id_is_none(self, monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
        """__init__ initializes active_client_id to None."""

        _patch_ib_insync(monkeypatch)
        adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
        assert adapter._active_client_id is None  # noqa: SLF001 - internal state coverage

    @pytest.mark.parametrize(
        ("error_code", "error_string", "expected"),
        [
            (326, "", True),
            (326, "client id already in use", True),
            (999, "client id already in use", True),
            (123, "some unrelated error", False),
        ],
    )
    def test_is_client_id_conflict(self, monkeypatch: pytest.MonkeyPatch, error_code: int, error_string: str, expected: bool) -> None:
        """_is_client_id_conflict recognizes Error 326 and conflict messages."""

        _patch_ib_insync(monkeypatch)
        adapter = IBKRAdapter(config=BrokerConnectionConfig(port=7497), id_generator=IDGenerator())
        assert adapter._is_client_id_conflict(error_code, error_string) is expected  # noqa: SLF001 - internal coverage

    @pytest.mark.parametrize("error_code", [502, 504, 1100, 1101, 1102, 1300])
    def test_is_connection_error_existing_codes(
        self, monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig, error_code: int
    ) -> None:
        """_is_connection_error keeps recognizing existing connection error codes."""

        _patch_ib_insync(monkeypatch)
        adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
        assert adapter._is_connection_error(error_code, "anything") is True  # noqa: SLF001 - internal coverage

    def test_is_connection_error_includes_326(
        self, monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
    ) -> None:
        """_is_connection_error recognizes Error 326 as a connection error."""

        _patch_ib_insync(monkeypatch)
        adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
        assert adapter._is_connection_error(326, "client id already in use") is True  # noqa: SLF001 - internal coverage
        assert adapter._is_connection_error(123, "some unrelated error") is False  # noqa: SLF001 - internal coverage

    def test_connect_dynamic_disabled_success_records_active_client_id(
        self, monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
    ) -> None:
        """connect() records active_client_id when dynamic allocation is disabled and connection succeeds."""

        ib = _patch_ib_insync(monkeypatch)
        adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

        result = adapter.connect()

        assert result.status is ResultStatus.SUCCESS
        assert adapter._active_client_id == int(ibkr_adapter_config.client_id)  # noqa: SLF001 - internal coverage
        assert [call["clientId"] for call in ib.connect_calls] == [int(ibkr_adapter_config.client_id)]

    def test_connect_dynamic_disabled_conflict_fails_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Error 326 fails immediately when dynamic allocation is disabled (no fallback IDs)."""

        ib = _patch_ib_insync(monkeypatch)
        sleep_mock = Mock()
        monkeypatch.setattr("execution.ibkr_adapter.time.sleep", sleep_mock)

        config = BrokerConnectionConfig(port=7497, client_id=1, enable_dynamic_client_id=False, max_reconnect_attempts=5)
        adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

        def connect_side_effect(**kwargs):  # noqa: ANN001 - test shim
            ib.connect_calls.append(dict(kwargs))
            with adapter._lock:  # noqa: SLF001 - test shim
                adapter._last_connection_error = (326, "client id already in use")  # noqa: SLF001 - test shim
            raise ConnectionError("Error 326")

        monkeypatch.setattr(ib, "connect", connect_side_effect)

        result = adapter.connect()

        assert result.status is ResultStatus.FAILED
        assert result.reason_code == "CONNECTION_FAILED"
        assert len(ib.connect_calls) == 1
        sleep_mock.assert_not_called()

    def test_connect_dynamic_disabled_non_326_retries_with_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-326 errors follow existing retry/backoff behavior when dynamic allocation is disabled."""

        ib = _patch_ib_insync(monkeypatch)
        sleep_mock = Mock()
        monkeypatch.setattr("execution.ibkr_adapter.time.sleep", sleep_mock)

        config = BrokerConnectionConfig(port=7497, client_id=1, enable_dynamic_client_id=False, max_reconnect_attempts=2)
        adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

        def connect_side_effect(**kwargs):  # noqa: ANN001 - test shim
            ib.connect_calls.append(dict(kwargs))
            if len(ib.connect_calls) == 1:
                with adapter._lock:  # noqa: SLF001 - test shim
                    adapter._last_connection_error = (504, "Not connected")  # noqa: SLF001 - test shim
                raise ConnectionError("boom")
            ib._connected = True
            return None

        monkeypatch.setattr(ib, "connect", connect_side_effect)

        result = adapter.connect()

        assert result.status is ResultStatus.SUCCESS
        assert [call["clientId"] for call in ib.connect_calls] == [1, 1]
        sleep_mock.assert_called_once_with(2)

    def test_connect_dynamic_enabled_success_records_active_client_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """connect() records active_client_id on success when dynamic allocation is enabled."""

        ib = _patch_ib_insync(monkeypatch)
        config = BrokerConnectionConfig(port=7497, client_id=1, enable_dynamic_client_id=True, client_id_range=(1, 32))
        adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

        result = adapter.connect()

        assert result.status is ResultStatus.SUCCESS
        assert adapter._active_client_id == 1  # noqa: SLF001 - internal coverage
        assert [call["clientId"] for call in ib.connect_calls] == [1]

    def test_connect_dynamic_enabled_fallback_to_next_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dynamic allocation tries next client_id on Error 326 conflict."""

        ib = _patch_ib_insync(monkeypatch)
        sleep_mock = Mock()
        monkeypatch.setattr("execution.ibkr_adapter.time.sleep", sleep_mock)

        config = BrokerConnectionConfig(port=7497, client_id=1, enable_dynamic_client_id=True, client_id_range=(1, 3))
        adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

        def connect_side_effect(**kwargs):  # noqa: ANN001 - test shim
            ib.connect_calls.append(dict(kwargs))
            client_id = int(kwargs["clientId"])
            if client_id == 1:
                with adapter._lock:  # noqa: SLF001 - test shim
                    adapter._last_connection_error = (326, "client id already in use")  # noqa: SLF001 - test shim
                raise ConnectionError("Error 326")
            ib._connected = True
            return None

        monkeypatch.setattr(ib, "connect", connect_side_effect)

        result = adapter.connect()

        assert result.status is ResultStatus.SUCCESS
        assert adapter._active_client_id == 2  # noqa: SLF001 - internal coverage
        assert [call["clientId"] for call in ib.connect_calls] == [1, 2]
        sleep_mock.assert_not_called()

    def test_connect_dynamic_enabled_conflicts_1_to_5_then_6_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When client_id 1-5 conflict, dynamic allocation connects successfully with client_id=6."""

        ib = _patch_ib_insync(monkeypatch)
        sleep_mock = Mock()
        monkeypatch.setattr("execution.ibkr_adapter.time.sleep", sleep_mock)

        config = BrokerConnectionConfig(port=7497, client_id=1, enable_dynamic_client_id=True, client_id_range=(1, 6))
        adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

        def connect_side_effect(**kwargs):  # noqa: ANN001 - test shim
            ib.connect_calls.append(dict(kwargs))
            client_id = int(kwargs["clientId"])
            if client_id < 6:
                with adapter._lock:  # noqa: SLF001 - test shim
                    adapter._last_connection_error = (326, "client id already in use")  # noqa: SLF001 - test shim
                raise ConnectionError("Error 326")
            ib._connected = True
            return None

        monkeypatch.setattr(ib, "connect", connect_side_effect)

        result = adapter.connect()

        assert result.status is ResultStatus.SUCCESS
        assert adapter._active_client_id == 6  # noqa: SLF001 - internal coverage
        assert [call["clientId"] for call in ib.connect_calls] == [1, 2, 3, 4, 5, 6]
        sleep_mock.assert_not_called()

    def test_connect_dynamic_enabled_all_conflicts_returns_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All client_ids conflict -> connect() returns FAILED."""

        ib = _patch_ib_insync(monkeypatch)
        sleep_mock = Mock()
        monkeypatch.setattr("execution.ibkr_adapter.time.sleep", sleep_mock)

        config = BrokerConnectionConfig(port=7497, client_id=1, enable_dynamic_client_id=True, client_id_range=(1, 3))
        adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

        def connect_side_effect(**kwargs):  # noqa: ANN001 - test shim
            ib.connect_calls.append(dict(kwargs))
            with adapter._lock:  # noqa: SLF001 - test shim
                adapter._last_connection_error = (326, "client id already in use")  # noqa: SLF001 - test shim
            raise ConnectionError("Error 326")

        monkeypatch.setattr(ib, "connect", connect_side_effect)

        result = adapter.connect()

        assert result.status is ResultStatus.FAILED
        assert result.reason_code == "CONNECTION_FAILED"
        assert adapter._active_client_id is None  # noqa: SLF001 - internal coverage
        assert [call["clientId"] for call in ib.connect_calls] == [1, 2, 3]
        sleep_mock.assert_not_called()

    def test_connect_dynamic_enabled_non_326_retries_same_client_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-326 errors retry with backoff and do not switch client_id even when dynamic allocation is enabled."""

        ib = _patch_ib_insync(monkeypatch)
        sleep_mock = Mock()
        monkeypatch.setattr("execution.ibkr_adapter.time.sleep", sleep_mock)

        config = BrokerConnectionConfig(port=7497, client_id=1, enable_dynamic_client_id=True, client_id_range=(1, 6), max_reconnect_attempts=2)
        adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

        def connect_side_effect(**kwargs):  # noqa: ANN001 - test shim
            ib.connect_calls.append(dict(kwargs))
            if len(ib.connect_calls) == 1:
                with adapter._lock:  # noqa: SLF001 - test shim
                    adapter._last_connection_error = (504, "Not connected")  # noqa: SLF001 - test shim
                raise ConnectionError("boom")
            ib._connected = True
            return None

        monkeypatch.setattr(ib, "connect", connect_side_effect)

        result = adapter.connect()

        assert result.status is ResultStatus.SUCCESS
        assert adapter._active_client_id == 1  # noqa: SLF001 - internal coverage
        assert [call["clientId"] for call in ib.connect_calls] == [1, 1]
        sleep_mock.assert_called_once_with(2)

    def test_connect_client_id_range_only_preferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """client_id_range that yields no fallbacks only tries the preferred id."""

        ib = _patch_ib_insync(monkeypatch)
        sleep_mock = Mock()
        monkeypatch.setattr("execution.ibkr_adapter.time.sleep", sleep_mock)

        config = BrokerConnectionConfig(port=7497, client_id=1, enable_dynamic_client_id=True, client_id_range=(1, 1))
        adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

        def connect_side_effect(**kwargs):  # noqa: ANN001 - test shim
            ib.connect_calls.append(dict(kwargs))
            with adapter._lock:  # noqa: SLF001 - test shim
                adapter._last_connection_error = (326, "client id already in use")  # noqa: SLF001 - test shim
            raise ConnectionError("Error 326")

        monkeypatch.setattr(ib, "connect", connect_side_effect)

        result = adapter.connect()

        assert result.status is ResultStatus.FAILED
        assert [call["clientId"] for call in ib.connect_calls] == [1]
        sleep_mock.assert_not_called()

    def test_connect_client_id_range_reversed_is_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reversed ranges (e.g. (32, 1)) are handled correctly."""

        ib = _patch_ib_insync(monkeypatch)
        sleep_mock = Mock()
        monkeypatch.setattr("execution.ibkr_adapter.time.sleep", sleep_mock)

        config = BrokerConnectionConfig(port=7497, client_id=3, enable_dynamic_client_id=True, client_id_range=(5, 1))
        adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

        def connect_side_effect(**kwargs):  # noqa: ANN001 - test shim
            ib.connect_calls.append(dict(kwargs))
            client_id = int(kwargs["clientId"])
            if client_id in {3, 1}:
                with adapter._lock:  # noqa: SLF001 - test shim
                    adapter._last_connection_error = (326, "client id already in use")  # noqa: SLF001 - test shim
                raise ConnectionError("Error 326")
            ib._connected = True
            return None

        monkeypatch.setattr(ib, "connect", connect_side_effect)

        result = adapter.connect()

        assert result.status is ResultStatus.SUCCESS
        assert adapter._active_client_id == 2  # noqa: SLF001 - internal coverage
        assert [call["clientId"] for call in ib.connect_calls] == [3, 1, 2]
        sleep_mock.assert_not_called()

    def test_connect_when_already_connected_is_idempotent_and_sets_active_client_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Already-connected connect() returns SUCCESS without calling IB.connect (and captures clientId if available)."""

        ib = _patch_ib_insync(monkeypatch)
        ib._connected = True
        ib.clientId = 7

        adapter = IBKRAdapter(config=BrokerConnectionConfig(port=7497), id_generator=IDGenerator())
        assert adapter._active_client_id is None  # noqa: SLF001 - internal coverage

        result = adapter.connect()

        assert result.status is ResultStatus.SUCCESS
        assert len(ib.connect_calls) == 0
        assert adapter._active_client_id == 7  # noqa: SLF001 - internal coverage

    def test_disconnect_resets_active_client_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """disconnect() resets active_client_id to None."""

        _patch_ib_insync(monkeypatch)
        adapter = IBKRAdapter(config=BrokerConnectionConfig(port=7497, client_id=1), id_generator=IDGenerator())
        adapter.connect()
        assert adapter._active_client_id == 1  # noqa: SLF001 - internal coverage

        result = adapter.disconnect()

        assert result.status is ResultStatus.SUCCESS
        assert adapter._active_client_id is None  # noqa: SLF001 - internal coverage


def test_is_connected(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """is_connected() reflects the underlying IB connection state."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act / Assert
    assert adapter.is_connected() is False
    adapter.connect()
    assert adapter.is_connected() is True


def test_is_connected_missing_ib_insync_returns_false(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """is_connected() returns False when ib_insync is unavailable."""

    import execution.ibkr_adapter as ibkr_mod

    monkeypatch.setattr(ibkr_mod, "IB", lambda: _FakeIB(), raising=False)
    monkeypatch.setattr(ibkr_mod, "_IB_INSYNC_IMPORT_ERROR", ImportError("missing"), raising=False)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    assert adapter.is_connected() is False


def test_submit_order_success(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """submit_order() returns a SUBMITTED IntentOrderMapping and tracks it locally."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    id_gen = Mock(spec=IDGenerator)
    id_gen.generate_order_id.return_value = Result.success("O-1")
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=id_gen)
    adapter.connect()

    intent = _make_intent(intent_id="I-1", entry_price=None)

    # Act
    result = adapter.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.client_order_id == "O-1"
    assert result.data.state is OrderState.SUBMITTED
    assert result.data.metadata["symbol"] == "AAPL"
    assert "O-1" in adapter._mappings  # noqa: SLF001 - internal coverage


def test_submit_order_readonly_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """submit_order() is blocked when config.readonly=True."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    config = BrokerConnectionConfig(port=7497, readonly=True)
    adapter = IBKRAdapter(config=config, id_generator=IDGenerator())
    adapter.connect()
    intent = _make_intent()

    # Act
    result = adapter.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "READONLY_MODE"


def test_submit_order_not_connected(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """submit_order() fails when the adapter is not connected."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    intent = _make_intent()

    # Act
    result = adapter.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CONNECTION_FAILED"


def test_submit_order_id_generation_failure(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """submit_order() fails when ID generation returns no usable order id."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    id_gen = Mock(spec=IDGenerator)
    id_gen.generate_order_id.return_value = Result.degraded(data=None, error=RuntimeError("no id"), reason_code="ID_FAIL")
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=id_gen)
    adapter.connect()

    # Act
    result = adapter.submit_order(_make_intent(intent_id="I-1"))

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ID_FAIL"


def test_submit_order_qualify_fails(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """submit_order() fails when contract qualification returns an empty list."""

    # Arrange
    ib = _patch_ib_insync(monkeypatch)
    ib.qualify_return = []
    id_gen = Mock(spec=IDGenerator)
    id_gen.generate_order_id.return_value = Result.success("O-1")
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=id_gen)
    adapter.connect()
    intent = _make_intent()

    # Act
    result = adapter.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "SUBMIT_FAILED"


def test_submit_order_place_order_exception(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """submit_order() returns SUBMIT_FAILED when IB.placeOrder raises."""

    # Arrange
    ib = _patch_ib_insync(monkeypatch)
    id_gen = Mock(spec=IDGenerator)
    id_gen.generate_order_id.return_value = Result.success("O-1")
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=id_gen)
    adapter.connect()
    monkeypatch.setattr(ib, "placeOrder", Mock(side_effect=RuntimeError("boom")))

    # Act
    result = adapter.submit_order(_make_intent(intent_id="I-1"))

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "SUBMIT_FAILED"


def test_submit_order_missing_ib_insync(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """submit_order() fails with DEPENDENCY_MISSING when ib_insync is unavailable."""

    # Arrange
    import execution.ibkr_adapter as ibkr_mod

    monkeypatch.setattr(ibkr_mod, "IB", lambda: _FakeIB(), raising=False)
    monkeypatch.setattr(ibkr_mod, "_IB_INSYNC_IMPORT_ERROR", ImportError("missing"), raising=False)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    intent = _make_intent()

    # Act
    result = adapter.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "DEPENDENCY_MISSING"


def test_submit_order_generates_client_order_id(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """submit_order() calls IDGenerator.generate_order_id(intent_id, run_id)."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    id_gen = Mock(spec=IDGenerator)
    id_gen.generate_order_id.return_value = Result.success("O-XYZ")

    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=id_gen)
    adapter.connect()
    intent = _make_intent(intent_id="I-XYZ", metadata={"run_id": "run-9"})

    # Act
    result = adapter.submit_order(intent)

    # Assert
    assert result.status is ResultStatus.SUCCESS
    id_gen.generate_order_id.assert_called_once_with("I-XYZ", "run-9")


def test_cancel_order_success(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """cancel_order() calls IB.cancelOrder for a tracked order."""

    # Arrange
    ib = _patch_ib_insync(monkeypatch)
    id_gen = Mock(spec=IDGenerator)
    id_gen.generate_order_id.return_value = Result.success("O-1")
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=id_gen)
    adapter.connect()
    mapping = adapter.submit_order(_make_intent()).data
    assert mapping is not None

    # Act
    result = adapter.cancel_order(mapping.client_order_id)

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert len(ib.cancel_calls) == 1


def test_cancel_order_readonly_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """cancel_order() is blocked when config.readonly=True."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    config = BrokerConnectionConfig(port=7497, readonly=True)
    adapter = IBKRAdapter(config=config, id_generator=IDGenerator())
    adapter.connect()

    # Act
    result = adapter.cancel_order("O-1")

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "READONLY_MODE"


def test_cancel_order_not_found(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """cancel_order() fails when client_order_id is unknown."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter.connect()

    # Act
    result = adapter.cancel_order("O-404")

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ORDER_NOT_FOUND"


def test_cancel_order_not_connected(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """cancel_order() fails when the adapter is not connected."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    result = adapter.cancel_order("O-1")

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CONNECTION_FAILED"


def test_cancel_order_exception_returns_failed(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """cancel_order() returns CANCEL_FAILED when IB.cancelOrder raises."""

    # Arrange
    ib = _patch_ib_insync(monkeypatch)
    id_gen = Mock(spec=IDGenerator)
    id_gen.generate_order_id.return_value = Result.success("O-1")
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=id_gen)
    adapter.connect()
    mapping = adapter.submit_order(_make_intent()).data
    assert mapping is not None
    monkeypatch.setattr(ib, "cancelOrder", Mock(side_effect=RuntimeError("boom")))

    # Act
    result = adapter.cancel_order(mapping.client_order_id)

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CANCEL_FAILED"


def test_get_order_status_success(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """get_order_status() refreshes mapping state from the underlying Trade when present."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    intent = _make_intent(intent_id="I-1")
    mapping = IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id="O-1",
        broker_order_id="B-1",
        state=OrderState.SUBMITTED,
        created_at_ns=1,
        updated_at_ns=1,
        intent_snapshot=intent,
        metadata={},
    )
    trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=10.0, orderRef="O-1", orderId=42),
        orderStatus=_FakeOrderStatus(status="Filled", filled=10.0, remaining=0.0, avgFillPrice=101.0),
    )
    adapter._mappings["O-1"] = mapping  # noqa: SLF001 - internal coverage
    adapter._trades["O-1"] = trade  # noqa: SLF001 - internal coverage

    # Act
    result = adapter.get_order_status("O-1")

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.state is OrderState.FILLED
    assert result.data.metadata["ibkr"]["status"] == "Filled"


def test_get_order_status_trade_missing_returns_cached(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """get_order_status() returns the cached mapping when no Trade is tracked."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    mapping = IntentOrderMapping(
        intent_id="I-1",
        client_order_id="O-1",
        broker_order_id="B-1",
        state=OrderState.SUBMITTED,
        created_at_ns=1,
        updated_at_ns=1,
        intent_snapshot=_make_intent(intent_id="I-1"),
        metadata={},
    )
    adapter._mappings["O-1"] = mapping  # noqa: SLF001 - internal coverage

    # Act
    result = adapter.get_order_status("O-1")

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert result.data == mapping


def test_get_order_status_exception_returns_failed(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """get_order_status() returns STATUS_FAILED when ib.sleep raises unexpectedly."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    monkeypatch.setattr(adapter._ib, "sleep", Mock(side_effect=RuntimeError("boom")))  # noqa: SLF001

    # Act
    result = adapter.get_order_status("O-1")

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "STATUS_FAILED"


def test_get_order_status_not_found(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """get_order_status() fails with ORDER_NOT_FOUND when mapping is missing."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    result = adapter.get_order_status("O-404")

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ORDER_NOT_FOUND"


def test_get_all_orders_success(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """get_all_orders() returns all locally tracked mappings."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter._mappings["O-1"] = IntentOrderMapping(  # noqa: SLF001 - internal coverage
        intent_id="I-1",
        client_order_id="O-1",
        broker_order_id="B-1",
        state=OrderState.SUBMITTED,
        created_at_ns=1,
        updated_at_ns=1,
        intent_snapshot=_make_intent(),
        metadata={},
    )

    # Act
    result = adapter.get_all_orders()

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert len(result.data) == 1


def test_get_all_orders_exception_returns_failed(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """get_all_orders() returns STATUS_FAILED when an unexpected exception occurs."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    monkeypatch.setattr(adapter._ib, "sleep", Mock(side_effect=RuntimeError("boom")))  # noqa: SLF001

    # Act
    result = adapter.get_all_orders()

    # Assert
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "STATUS_FAILED"


def test_get_all_orders_empty(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """get_all_orders() returns an empty list when no mappings exist."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    result = adapter.get_all_orders()

    # Assert
    assert result.status is ResultStatus.SUCCESS
    assert result.data == []


def test_on_order_status_updates_mapping(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_order_status updates mapping state and metadata for a tracked trade."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    intent = _make_intent(intent_id="I-1")
    mapping = IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id="O-1",
        broker_order_id=None,
        state=OrderState.SUBMITTED,
        created_at_ns=1,
        updated_at_ns=1,
        intent_snapshot=intent,
        metadata={},
    )
    adapter._mappings["O-1"] = mapping  # noqa: SLF001 - internal coverage
    trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=10.0, orderRef="O-1", orderId=777),
        orderStatus=_FakeOrderStatus(status="Filled", filled=10.0, remaining=0.0, avgFillPrice=101.0),
    )

    # Act
    adapter._on_order_status(trade)

    # Assert
    updated = adapter._mappings["O-1"]  # noqa: SLF001 - internal coverage
    assert updated.state is OrderState.FILLED
    assert updated.broker_order_id == "777"
    assert updated.metadata["ibkr"]["avgFillPrice"] == 101.0


def test_on_order_status_partially_filled(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_order_status transitions to PARTIALLY_FILLED and records filled qty."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    intent = _make_intent(intent_id="I-PF")
    mapping = IntentOrderMapping(
        intent_id=intent.intent_id,
        client_order_id="O-PF",
        broker_order_id=None,
        state=OrderState.SUBMITTED,
        created_at_ns=1,
        updated_at_ns=1,
        intent_snapshot=intent,
        metadata={},
    )
    adapter._mappings["O-PF"] = mapping  # noqa: SLF001
    trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=51.0, orderRef="O-PF", orderId=888),
        orderStatus=_FakeOrderStatus(status="PartiallyFilled", filled=30.0, remaining=21.0, avgFillPrice=50.5),
    )

    adapter._on_order_status(trade)

    updated = adapter._mappings["O-PF"]  # noqa: SLF001
    assert updated.state is OrderState.PARTIALLY_FILLED
    assert updated.metadata["ibkr"]["filled"] == 30.0
    assert updated.metadata["ibkr"]["remaining"] == 21.0


def test_on_execution_records_fill(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_execution appends FillDetail to adapter._fills."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    trade = _FakeTrade(order=_FakeOrder(action="BUY", totalQuantity=1.0, orderRef="O-1"), orderStatus=_FakeOrderStatus())
    monkeypatch.setattr(
        adapter,
        "_to_fill_detail",
        lambda *_a, **_k: FillDetail(execution_id="E-1", fill_price=1.0, fill_quantity=1.0, fill_time_ns=1),
    )

    # Act
    adapter._on_execution(trade, fill=SimpleNamespace())

    # Assert
    assert adapter._fills["O-1"][0].execution_id == "E-1"  # noqa: SLF001 - internal coverage


def test_on_execution_deduplicates_fills(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_execution ignores duplicate fills with the same execution_id."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    trade = _FakeTrade(order=_FakeOrder(action="BUY", totalQuantity=1.0, orderRef="O-1"), orderStatus=_FakeOrderStatus())
    monkeypatch.setattr(
        adapter,
        "_to_fill_detail",
        lambda *_a, **_k: FillDetail(execution_id="E-1", fill_price=1.0, fill_quantity=1.0, fill_time_ns=1),
    )

    # Act
    adapter._on_execution(trade, fill=SimpleNamespace())
    adapter._on_execution(trade, fill=SimpleNamespace())

    # Assert
    assert len(adapter._fills["O-1"]) == 1  # noqa: SLF001 - internal coverage


def test_on_error_connection_error(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_error stores the last connection error when the message indicates connectivity failure."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    adapter._on_error(reqId=0, errorCode=504, errorString="Not connected")

    # Assert
    assert adapter._last_connection_error == (504, "Not connected")  # noqa: SLF001 - internal coverage


def test_on_error_non_connection_ignored(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_error ignores non-connection informational messages."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act
    adapter._on_error(reqId=0, errorCode=2104, errorString="Market data farm is OK")

    # Assert
    assert adapter._last_connection_error is None  # noqa: SLF001 - internal coverage


def test_on_connected_resets_attempts(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_connected resets reconnect attempt counter to 0."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter._reconnect_attempts = 3  # noqa: SLF001 - internal coverage

    # Act
    adapter._on_connected()

    # Assert
    assert adapter._reconnect_attempts == 0  # noqa: SLF001 - internal coverage


def test_on_disconnected_triggers_reconnect(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_disconnected triggers reconnect while attempts <= max_reconnect_attempts."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    config = BrokerConnectionConfig(port=ibkr_adapter_config.port, max_reconnect_attempts=1)
    adapter = IBKRAdapter(config=config, id_generator=IDGenerator())
    adapter.connect = Mock(return_value=Result.success(None))  # type: ignore[method-assign]

    # Act
    adapter._on_disconnected()
    adapter._on_disconnected()  # exceeds max attempts; should not call connect again

    # Assert
    assert adapter.connect.call_count == 1


def test_on_disconnected_exceeds_max_attempts_no_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """_on_disconnected stops reconnecting once max attempts is exceeded."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    config = BrokerConnectionConfig(port=7497, max_reconnect_attempts=0)
    adapter = IBKRAdapter(config=config, id_generator=IDGenerator())
    adapter.connect = Mock(return_value=Result.success(None))  # type: ignore[method-assign]

    # Act
    adapter._on_disconnected()

    # Assert
    adapter.connect.assert_not_called()


def test_infer_action_metadata_hints(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_infer_action honors metadata hints when present."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act / Assert
    assert adapter._infer_action(_make_intent(metadata={"action": "SELL", "run_id": "run-1"})) == "SELL"
    assert (
        adapter._infer_action(
            _make_intent(intent_type=IntentType.STOP_LOSS, metadata={"position_side": "SHORT", "run_id": "run-1"})
        )
        == "BUY"
    )


def test_build_order_limit_and_account(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_order builds a limit order and sets orderRef/account."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    config = BrokerConnectionConfig(port=7497, account="DU123")
    adapter = IBKRAdapter(config=config, id_generator=IDGenerator())

    # Act
    order = adapter._build_order(intent=_make_intent(entry_price=123.0), action="BUY", client_order_id="O-1")

    # Assert
    assert getattr(order, "orderRef") == "O-1"
    assert getattr(order, "account") == "DU123"
    assert getattr(order, "lmtPrice") == 123.0


def test_build_order_invalid_quantity_raises(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_build_order raises ValueError when intent.quantity <= 0."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    with pytest.raises(ValueError, match="intent\\.quantity must be > 0"):
        adapter._build_order(intent=_make_intent(quantity=0.0), action="BUY", client_order_id="O-1")


def test_to_fill_detail_datetime_and_commission_handling(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """_to_fill_detail normalizes time and commission values."""

    from datetime import datetime

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    fill = SimpleNamespace(
        execution=SimpleNamespace(execId="E-1", price=101.0, shares=2.0, time=datetime(2025, 1, 1, 0, 0, 0)),
        commissionReport=SimpleNamespace(commission="not-a-number"),
    )
    detail = adapter._to_fill_detail(fill)
    assert detail.execution_id == "E-1"
    assert detail.fill_price == 101.0
    assert detail.fill_quantity == 2.0
    assert detail.commission is None


def test_import_error_branch_is_executable_for_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module-level ib_insync import fallback branch can be executed (coverage)."""

    import builtins
    import importlib.util
    import sys

    import execution.ibkr_adapter as ibkr_mod

    original_import = builtins.__import__

    def failing_import(name: str, *args, **kwargs):  # noqa: ANN001 - test shim
        if name == "ib_insync":
            raise ImportError("forced missing ib_insync")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = failing_import
    try:
        spec = importlib.util.spec_from_file_location(
            "execution._ibkr_adapter_import_fail_test",
            ibkr_mod.__file__,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assert getattr(module, "_IB_INSYNC_IMPORT_ERROR") is not None
    finally:
        builtins.__import__ = original_import


def test_cancel_and_status_dependency_missing_paths(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """cancel_order/get_order_status/get_all_orders fail with DEPENDENCY_MISSING when ib_insync is unavailable."""

    import execution.ibkr_adapter as ibkr_mod

    monkeypatch.setattr(ibkr_mod, "IB", lambda: _FakeIB(), raising=False)
    monkeypatch.setattr(ibkr_mod, "_IB_INSYNC_IMPORT_ERROR", ImportError("missing"), raising=False)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    assert adapter.cancel_order("O-1").reason_code == "DEPENDENCY_MISSING"
    assert adapter.get_order_status("O-1").reason_code == "DEPENDENCY_MISSING"
    assert adapter.get_all_orders().reason_code == "DEPENDENCY_MISSING"


def test_on_order_status_early_return_paths(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_order_status returns early for missing client id, unknown mapping, or missing status."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    trade_no_ref = _FakeTrade(order=_FakeOrder(action="BUY", totalQuantity=1.0, orderRef=None), orderStatus=_FakeOrderStatus())
    adapter._on_order_status(trade_no_ref)

    trade_unknown = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=1.0, orderRef="O-404"),
        orderStatus=_FakeOrderStatus(status="Filled"),
    )
    adapter._on_order_status(trade_unknown)

    intent = _make_intent(intent_id="I-1")
    adapter._mappings["O-1"] = IntentOrderMapping(  # noqa: SLF001 - internal coverage
        intent_id=intent.intent_id,
        client_order_id="O-1",
        broker_order_id=None,
        state=OrderState.SUBMITTED,
        created_at_ns=1,
        updated_at_ns=1,
        intent_snapshot=intent,
        metadata={},
    )
    trade_no_status = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=1.0, orderRef="O-1"),
        orderStatus=_FakeOrderStatus(status=""),
    )
    adapter._on_order_status(trade_no_status)


def test_on_execution_missing_client_id_and_exception_paths(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """_on_execution returns early for missing client id and swallows exceptions."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    trade_no_ref = _FakeTrade(order=_FakeOrder(action="BUY", totalQuantity=1.0, orderRef=None), orderStatus=_FakeOrderStatus())
    adapter._on_execution(trade_no_ref, fill=SimpleNamespace())

    trade_with_ref = _FakeTrade(order=_FakeOrder(action="BUY", totalQuantity=1.0, orderRef="O-1"), orderStatus=_FakeOrderStatus())
    monkeypatch.setattr(adapter, "_to_fill_detail", Mock(side_effect=RuntimeError("boom")))
    adapter._on_execution(trade_with_ref, fill=SimpleNamespace())


def test_on_error_exception_path(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_on_error swallows unexpected exceptions."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    monkeypatch.setattr(adapter, "_is_connection_error", Mock(side_effect=RuntimeError("boom")))
    adapter._on_error(reqId=0, errorCode=504, errorString="Not connected")


def test_on_connected_and_disconnected_exception_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """_on_connected/_on_disconnected swallow unexpected exceptions."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=BrokerConnectionConfig(port=7497, max_reconnect_attempts=1), id_generator=IDGenerator())

    class _ExplodingLock:
        def __enter__(self):  # noqa: ANN001 - test stub
            raise RuntimeError("boom")

        def __exit__(self, *_a, **_k):  # noqa: ANN001 - test stub
            return False

    adapter._lock = _ExplodingLock()  # type: ignore[assignment]  # noqa: SLF001
    adapter._on_connected()

    adapter._lock = threading.RLock()  # type: ignore[assignment]  # noqa: SLF001
    adapter.connect = Mock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    adapter._on_disconnected()


def test_infer_action_long_hint_and_unsupported_intent(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_infer_action covers LONG hint and raises for unsupported intent types."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    assert (
        adapter._infer_action(
            _make_intent(intent_type=IntentType.STOP_LOSS, metadata={"position_side": "LONG", "run_id": "run-1"})
        )
        == "SELL"
    )
    with pytest.raises(ValueError, match="Unsupported intent_type"):
        adapter._infer_action(_make_intent(intent_type=IntentType.CANCEL_PENDING))


def test_to_fill_detail_exec_id_fallbacks(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_to_fill_detail falls back to permId and generated ids when execId is missing."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    fill_perm = SimpleNamespace(execution=SimpleNamespace(execId=None, permId="P-1", price=None, shares=None, time=None), commissionReport=None)
    detail_perm = adapter._to_fill_detail(fill_perm)
    assert detail_perm.execution_id == "P-1"

    fill_gen = SimpleNamespace(execution=SimpleNamespace(execId=None, permId=None, price=None, shares=None, time=None), commissionReport=None)
    detail_gen = adapter._to_fill_detail(fill_gen)
    assert detail_gen.execution_id.startswith("exec-")


def test_get_run_id_fallback(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_get_run_id falls back to 'unknown' when intent metadata is missing/blank."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    intent = _make_intent(metadata={"run_id": "  "})
    assert adapter._get_run_id(intent) == "unknown"

def test_map_ibkr_status_submitted(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_map_ibkr_status_to_order_state maps submitted-like statuses to SUBMITTED."""

    # Arrange
    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Act / Assert
    assert adapter._map_ibkr_status_to_order_state("PreSubmitted") is OrderState.SUBMITTED
    assert adapter._map_ibkr_status_to_order_state("ApiPending") is OrderState.SUBMITTED
    assert adapter._map_ibkr_status_to_order_state("PendingSubmit") is OrderState.SUBMITTED
    assert adapter._map_ibkr_status_to_order_state("PendingCancel") is OrderState.SUBMITTED
    assert adapter._map_ibkr_status_to_order_state("Submitted") is OrderState.SUBMITTED


def test_map_ibkr_status_partially_filled(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_map_ibkr_status_to_order_state maps PartiallyFilled to PARTIALLY_FILLED."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    assert adapter._map_ibkr_status_to_order_state("PartiallyFilled") is OrderState.PARTIALLY_FILLED


def test_map_ibkr_status_filled(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_map_ibkr_status_to_order_state maps Filled to FILLED."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    assert adapter._map_ibkr_status_to_order_state("Filled") is OrderState.FILLED


def test_map_ibkr_status_cancelled(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_map_ibkr_status_to_order_state maps cancelled-like statuses to CANCELLED."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    assert adapter._map_ibkr_status_to_order_state("Cancelled") is OrderState.CANCELLED
    assert adapter._map_ibkr_status_to_order_state("ApiCancelled") is OrderState.CANCELLED
    assert adapter._map_ibkr_status_to_order_state("Inactive") is OrderState.CANCELLED


def test_map_ibkr_status_pending_cancel(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_map_ibkr_status_to_order_state maps PendingCancel to SUBMITTED (cancelling)."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    assert adapter._map_ibkr_status_to_order_state("PendingCancel") is OrderState.SUBMITTED


def test_map_ibkr_status_unknown_defaults(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """_map_ibkr_status_to_order_state defaults unknown statuses to SUBMITTED."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    assert adapter._map_ibkr_status_to_order_state("WeirdStatus") is OrderState.SUBMITTED


# --- OCA Group tests ---


def test_derive_oca_group_deterministic() -> None:
    """Same parent_intent_id always yields same OCA group (idempotent)."""
    g1 = IBKRAdapter._derive_oca_group("ENTRY-123")
    g2 = IBKRAdapter._derive_oca_group("ENTRY-123")
    assert g1 == g2
    assert g1.startswith("AST-OCA-")
    assert len(g1) == 24  # "AST-OCA-" (8) + 16 hex chars


def test_derive_oca_group_differs_for_different_parents() -> None:
    """Different parent ids produce different OCA groups."""
    g1 = IBKRAdapter._derive_oca_group("ENTRY-123")
    g2 = IBKRAdapter._derive_oca_group("ENTRY-456")
    assert g1 != g2


def test_build_order_sl_with_parent_gets_oca(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """SL intent with parent_intent_id gets ocaGroup and ocaType on the order."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    sl_intent = TradeIntent(
        intent_id="SL-1",
        symbol="AAPL",
        intent_type=IntentType.STOP_LOSS,
        quantity=50.0,
        created_at_ns=1,
        stop_loss_price=95.0,
        parent_intent_id="ENTRY-1",
        metadata={"run_id": "run-1", "position_side": "LONG"},
    )

    order = adapter._build_order(intent=sl_intent, action="SELL", client_order_id="O-SL-1")
    assert order.ocaGroup == IBKRAdapter._derive_oca_group("ENTRY-1")
    assert order.ocaType == 1


def test_build_order_tp_with_parent_gets_oca(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """TP intent with parent_intent_id gets same ocaGroup as its SL sibling."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    tp_intent = TradeIntent(
        intent_id="TP-1",
        symbol="AAPL",
        intent_type=IntentType.TAKE_PROFIT,
        quantity=50.0,
        created_at_ns=1,
        take_profit_price=110.0,
        parent_intent_id="ENTRY-1",
        metadata={"run_id": "run-1", "position_side": "LONG"},
    )

    order = adapter._build_order(intent=tp_intent, action="SELL", client_order_id="O-TP-1")
    expected_group = IBKRAdapter._derive_oca_group("ENTRY-1")
    assert order.ocaGroup == expected_group
    assert order.ocaType == 1


def test_build_order_entry_no_oca(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """Entry (OPEN_LONG) intent does NOT get ocaGroup."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    intent = _make_intent(intent_id="E-1", intent_type=IntentType.OPEN_LONG, entry_price=100.0)

    order = adapter._build_order(intent=intent, action="BUY", client_order_id="O-E-1")
    assert not hasattr(order, "ocaGroup") or getattr(order, "ocaGroup", None) is None


def test_build_order_sl_no_parent_no_oca(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """Repair-generated SL (no parent_intent_id) does NOT get ocaGroup."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    sl_intent = TradeIntent(
        intent_id="REPAIR-SL-1",
        symbol="AAPL",
        intent_type=IntentType.STOP_LOSS,
        quantity=50.0,
        created_at_ns=1,
        stop_loss_price=95.0,
        parent_intent_id=None,
        metadata={"run_id": "run-1", "position_side": "LONG"},
    )

    order = adapter._build_order(intent=sl_intent, action="SELL", client_order_id="O-RSL-1")
    assert not hasattr(order, "ocaGroup") or getattr(order, "ocaGroup", None) is None


# --- Real-time sibling realignment tests ---


def _setup_entry_with_siblings(
    monkeypatch: pytest.MonkeyPatch,
    ibkr_adapter_config: BrokerConnectionConfig,
    *,
    entry_qty: float = 100.0,
    sl_qty: float = 100.0,
    tp_qty: float | None = None,
) -> tuple:
    """Helper: create an adapter with an OPEN_LONG entry and SL (optionally TP) siblings."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter._ib._connected = True  # noqa: SLF001

    entry_intent = TradeIntent(
        intent_id="ENTRY-1",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=entry_qty,
        created_at_ns=1,
        entry_price=100.0,
        metadata={"run_id": "run-1"},
    )
    entry_mapping = IntentOrderMapping(
        intent_id="ENTRY-1",
        client_order_id="O-ENTRY-1",
        broker_order_id="B-1",
        state=OrderState.SUBMITTED,
        created_at_ns=1,
        updated_at_ns=1,
        intent_snapshot=entry_intent,
        metadata={},
    )
    adapter._mappings["O-ENTRY-1"] = entry_mapping  # noqa: SLF001

    sl_intent = TradeIntent(
        intent_id="SL-1",
        symbol="AAPL",
        intent_type=IntentType.STOP_LOSS,
        quantity=sl_qty,
        created_at_ns=1,
        stop_loss_price=95.0,
        parent_intent_id="ENTRY-1",
        metadata={"run_id": "run-1", "position_side": "LONG"},
    )
    sl_mapping = IntentOrderMapping(
        intent_id="SL-1",
        client_order_id="O-SL-1",
        broker_order_id="B-2",
        state=OrderState.SUBMITTED,
        created_at_ns=1,
        updated_at_ns=1,
        intent_snapshot=sl_intent,
        metadata={},
    )
    sl_trade = _FakeTrade(
        order=_FakeOrder(action="SELL", totalQuantity=sl_qty, orderRef="O-SL-1", orderId=9002),
        orderStatus=_FakeOrderStatus(status="Submitted"),
    )
    adapter._mappings["O-SL-1"] = sl_mapping  # noqa: SLF001
    adapter._trades["O-SL-1"] = sl_trade  # noqa: SLF001

    tp_mapping = None
    if tp_qty is not None:
        tp_intent = TradeIntent(
            intent_id="TP-1",
            symbol="AAPL",
            intent_type=IntentType.TAKE_PROFIT,
            quantity=tp_qty,
            created_at_ns=1,
            take_profit_price=110.0,
            parent_intent_id="ENTRY-1",
            metadata={"run_id": "run-1", "position_side": "LONG"},
        )
        tp_mapping = IntentOrderMapping(
            intent_id="TP-1",
            client_order_id="O-TP-1",
            broker_order_id="B-3",
            state=OrderState.SUBMITTED,
            created_at_ns=1,
            updated_at_ns=1,
            intent_snapshot=tp_intent,
            metadata={},
        )
        tp_trade = _FakeTrade(
            order=_FakeOrder(action="SELL", totalQuantity=tp_qty, orderRef="O-TP-1", orderId=9003),
            orderStatus=_FakeOrderStatus(status="Submitted"),
        )
        adapter._mappings["O-TP-1"] = tp_mapping  # noqa: SLF001
        adapter._trades["O-TP-1"] = tp_trade  # noqa: SLF001

    return adapter, entry_mapping, sl_mapping, tp_mapping


def test_on_order_status_partial_fill_triggers_realignment(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """Entry PartiallyFilled → auto cancel+resubmit SL with filled qty."""

    adapter, _, sl_mapping, _ = _setup_entry_with_siblings(monkeypatch, ibkr_adapter_config)

    cancel_calls: list[str] = []
    submit_calls: list[TradeIntent] = []
    original_cancel = adapter.cancel_order
    original_submit = adapter.submit_order

    def mock_cancel(coid: str) -> Result:
        cancel_calls.append(coid)
        return Result.success(data=None)

    def mock_submit(intent: TradeIntent) -> Result:
        submit_calls.append(intent)
        return Result.success(data=None)

    monkeypatch.setattr(adapter, "cancel_order", mock_cancel)
    monkeypatch.setattr(adapter, "submit_order", mock_submit)

    entry_trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=100.0, orderRef="O-ENTRY-1", orderId=9001),
        orderStatus=_FakeOrderStatus(status="PartiallyFilled", filled=60.0, remaining=40.0),
    )
    adapter._on_order_status(entry_trade)

    assert cancel_calls == ["O-SL-1"]
    assert len(submit_calls) == 1
    assert submit_calls[0].quantity == 60.0
    assert submit_calls[0].intent_type is IntentType.STOP_LOSS
    assert submit_calls[0].parent_intent_id == "ENTRY-1"


def test_on_order_status_partial_fill_both_sl_tp_realigned(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """SL and TP are both realigned when entry is partially filled."""

    adapter, _, _, _ = _setup_entry_with_siblings(
        monkeypatch, ibkr_adapter_config, sl_qty=100.0, tp_qty=100.0
    )

    cancel_calls: list[str] = []
    submit_calls: list[TradeIntent] = []

    def mock_cancel(coid: str) -> Result:
        cancel_calls.append(coid)
        return Result.success(data=None)

    def mock_submit(intent: TradeIntent) -> Result:
        submit_calls.append(intent)
        return Result.success(data=None)

    monkeypatch.setattr(adapter, "cancel_order", mock_cancel)
    monkeypatch.setattr(adapter, "submit_order", mock_submit)

    entry_trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=100.0, orderRef="O-ENTRY-1", orderId=9001),
        orderStatus=_FakeOrderStatus(status="PartiallyFilled", filled=40.0, remaining=60.0),
    )
    adapter._on_order_status(entry_trade)

    assert sorted(cancel_calls) == ["O-SL-1", "O-TP-1"]
    assert len(submit_calls) == 2
    for intent in submit_calls:
        assert intent.quantity == 40.0


def test_on_order_status_partial_fill_cancel_fails_no_resubmit(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """Cancel failure → do NOT resubmit (avoid double-active orders)."""

    adapter, _, _, _ = _setup_entry_with_siblings(monkeypatch, ibkr_adapter_config)

    submit_calls: list[TradeIntent] = []

    def mock_cancel(coid: str) -> Result:
        return Result.failed(RuntimeError("cancel failed"), "CANCEL_FAILED")

    def mock_submit(intent: TradeIntent) -> Result:
        submit_calls.append(intent)
        return Result.success(data=None)

    monkeypatch.setattr(adapter, "cancel_order", mock_cancel)
    monkeypatch.setattr(adapter, "submit_order", mock_submit)

    entry_trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=100.0, orderRef="O-ENTRY-1", orderId=9001),
        orderStatus=_FakeOrderStatus(status="PartiallyFilled", filled=60.0, remaining=40.0),
    )
    adapter._on_order_status(entry_trade)

    assert len(submit_calls) == 0  # No resubmit after cancel failure


def test_on_order_status_filled_entry_no_realignment(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """FILLED entry does NOT trigger realignment (qty should already match)."""

    adapter, _, _, _ = _setup_entry_with_siblings(monkeypatch, ibkr_adapter_config)

    cancel_calls: list[str] = []
    monkeypatch.setattr(adapter, "cancel_order", lambda coid: cancel_calls.append(coid) or Result.success(data=None))

    entry_trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=100.0, orderRef="O-ENTRY-1", orderId=9001),
        orderStatus=_FakeOrderStatus(status="Filled", filled=100.0, remaining=0.0),
    )
    adapter._on_order_status(entry_trade)

    assert cancel_calls == []


def test_on_order_status_partial_fill_qty_already_correct_no_action(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """SL qty already matches filled qty → no cancel/resubmit needed."""

    adapter, _, _, _ = _setup_entry_with_siblings(
        monkeypatch, ibkr_adapter_config, sl_qty=60.0
    )

    cancel_calls: list[str] = []
    monkeypatch.setattr(adapter, "cancel_order", lambda coid: cancel_calls.append(coid) or Result.success(data=None))

    entry_trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=100.0, orderRef="O-ENTRY-1", orderId=9001),
        orderStatus=_FakeOrderStatus(status="PartiallyFilled", filled=60.0, remaining=40.0),
    )
    adapter._on_order_status(entry_trade)

    assert cancel_calls == []


# ---------------------------------------------------------------------------
# repair_bracket_quantities — broker-direct bracket qty repair
# ---------------------------------------------------------------------------


def _make_position(symbol: str, qty: float) -> SimpleNamespace:
    """Build a fake IBKR position row."""
    return SimpleNamespace(contract=_FakeContract(conId=1, symbol=symbol), position=qty)


def _make_bracket_trade(
    symbol: str,
    order_type: str,
    qty: float,
    price: float,
    oca_group: str = "OCA-1",
) -> _FakeTrade:
    """Build a fake open bracket Trade (SL or TP)."""
    order = _FakeOrder(
        action="SELL",
        totalQuantity=qty,
        orderType=order_type,
        lmtPrice=price if order_type == "LMT" else None,
        auxPrice=price if order_type == "STP" else None,
        ocaGroup=oca_group,
        ocaType=1,
        orderId=5000,
    )
    contract = _FakeContract(conId=1, symbol=symbol)
    status = _FakeOrderStatus(status="PreSubmitted")
    return _FakeTrade(order=order, orderStatus=status, contract=contract)


def test_repair_bracket_quantities_fixes_mismatch(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """SL/TP qty=491 but position=400 → cancel+resubmit with qty=400."""
    ib = _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter.connect()

    # Position: BTOG 400 shares
    ib._positions = [_make_position("BTOG", 400.0)]

    # Open orders: SL and TP both at 491
    ib._open_trades = [
        _make_bracket_trade("BTOG", "STP", 491.0, 2.58, "OCA-BTOG"),
        _make_bracket_trade("BTOG", "LMT", 491.0, 3.04, "OCA-BTOG"),
    ]

    result = adapter.repair_bracket_quantities()

    assert result.status is ResultStatus.SUCCESS
    data = result.data
    assert data["aligned"] == 0
    assert len(data["repaired"]) == 2
    assert data["failures"] == []

    # Verify both legs were cancelled
    assert len(ib.cancel_calls) == 2

    # Verify both resubmitted with qty=400
    assert len(ib.placed_orders) == 2
    for _contract, order in ib.placed_orders:
        assert order.totalQuantity == 400.0
        assert order.ocaGroup == "OCA-BTOG"

    # Check repaired details
    sl_repair = [r for r in data["repaired"] if r["leg_type"] == "SL"][0]
    assert sl_repair["old_qty"] == 491.0
    assert sl_repair["new_qty"] == 400.0
    assert sl_repair["symbol"] == "BTOG"

    tp_repair = [r for r in data["repaired"] if r["leg_type"] == "TP"][0]
    assert tp_repair["old_qty"] == 491.0
    assert tp_repair["new_qty"] == 400.0


def test_repair_bracket_quantities_already_aligned(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """SL/TP qty matches position → no action taken."""
    ib = _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter.connect()

    ib._positions = [_make_position("CHRW", 7.0)]
    ib._open_trades = [
        _make_bracket_trade("CHRW", "STP", 7.0, 174.08),
        _make_bracket_trade("CHRW", "LMT", 7.0, 205.19),
    ]

    result = adapter.repair_bracket_quantities()

    assert result.status is ResultStatus.SUCCESS
    assert result.data["aligned"] == 2
    assert result.data["repaired"] == []
    assert ib.cancel_calls == []
    assert ib.placed_orders == []


def test_repair_bracket_quantities_no_position_skips(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """Bracket orders for symbol with no position → skipped (not our concern)."""
    ib = _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter.connect()

    ib._positions = []  # No positions
    ib._open_trades = [
        _make_bracket_trade("GHOST", "STP", 100.0, 50.0),
    ]

    result = adapter.repair_bracket_quantities()

    assert result.status is ResultStatus.SUCCESS
    assert result.data["aligned"] == 0
    assert result.data["repaired"] == []
    assert ib.cancel_calls == []


def test_repair_bracket_quantities_cancel_failure(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """Cancel fails → recorded as failure, no resubmit."""
    ib = _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter.connect()

    ib._positions = [_make_position("BTOG", 400.0)]
    ib._open_trades = [
        _make_bracket_trade("BTOG", "STP", 491.0, 2.58),
    ]

    # Make cancelOrder raise
    def _cancel_raises(order):
        raise RuntimeError("cancel rejected")

    ib.cancelOrder = _cancel_raises

    result = adapter.repair_bracket_quantities()

    assert result.status is ResultStatus.SUCCESS
    assert len(result.data["failures"]) == 1
    assert result.data["failures"][0]["stage"] == "cancel"
    assert result.data["repaired"] == []
    assert ib.placed_orders == []


def test_repair_bracket_quantities_readonly_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readonly adapter → repair blocked."""
    ib = _patch_ib_insync(monkeypatch)
    config = BrokerConnectionConfig(host="127.0.0.1", port=4002, client_id=99, readonly=True)
    adapter = IBKRAdapter(config=config, id_generator=IDGenerator())
    adapter.connect()

    result = adapter.repair_bracket_quantities()

    assert result.status is ResultStatus.FAILED
    assert "readonly" in str(result.error).lower()


def test_repair_bracket_quantities_mixed_symbols(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """Multiple symbols: one mismatched, one aligned → only mismatch repaired."""
    ib = _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter.connect()

    ib._positions = [
        _make_position("BTOG", 400.0),
        _make_position("CHRW", 7.0),
    ]
    ib._open_trades = [
        _make_bracket_trade("BTOG", "STP", 491.0, 2.58),
        _make_bracket_trade("BTOG", "LMT", 491.0, 3.04),
        _make_bracket_trade("CHRW", "STP", 7.0, 174.08),
        _make_bracket_trade("CHRW", "LMT", 7.0, 205.19),
    ]

    result = adapter.repair_bracket_quantities()

    assert result.status is ResultStatus.SUCCESS
    assert result.data["aligned"] == 2  # CHRW SL + TP
    assert len(result.data["repaired"]) == 2  # BTOG SL + TP
    assert all(r["symbol"] == "BTOG" for r in result.data["repaired"])


def test_realign_zero_filled_skip(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """filled_qty=0 → _realign_sibling_quantities does nothing."""

    adapter, _, _, _ = _setup_entry_with_siblings(monkeypatch, ibkr_adapter_config)

    cancel_calls: list[str] = []
    monkeypatch.setattr(adapter, "cancel_order", lambda coid: cancel_calls.append(coid) or Result.success(data=None))

    adapter._realign_sibling_quantities("ENTRY-1", 0.0)

    assert cancel_calls == []


# ---------------------------------------------------------------------------
# is_healthy / reset_reconnect_counter tests
# ---------------------------------------------------------------------------


def test_is_healthy_connected(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """is_healthy() returns True when the adapter is connected."""

    ib = _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    adapter.connect()  # sets ib._connected = True
    assert adapter.is_healthy() is True


def test_is_healthy_disconnected(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """is_healthy() returns False when the adapter is disconnected."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    # Never connect
    assert adapter.is_healthy() is False


def test_is_healthy_exception(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """is_healthy() returns False when isConnected() raises."""

    ib = _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())
    monkeypatch.setattr(ib, "isConnected", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert adapter.is_healthy() is False


def test_reset_reconnect_counter(monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig) -> None:
    """reset_reconnect_counter() zeroes the internal counter."""

    _patch_ib_insync(monkeypatch)
    adapter = IBKRAdapter(config=ibkr_adapter_config, id_generator=IDGenerator())

    # Simulate several reconnect bumps.
    adapter._reconnect_attempts = 5
    adapter.reset_reconnect_counter()
    assert adapter._reconnect_attempts == 0


def test_on_order_status_filled_triggers_realignment(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """Entry Filled (after partial) → auto realign SL/TP to total filled qty.

    Scenario: 491 shares ordered → 400 partial fill → SL/TP adjusted to 400
    → remaining 91 fill → status = "Filled", filled=491 → SL/TP must realign to 491.
    """

    # Set up with SL qty = 400 (simulating previous partial realignment).
    adapter, _, sl_mapping, _ = _setup_entry_with_siblings(
        monkeypatch, ibkr_adapter_config, entry_qty=491.0, sl_qty=400.0,
    )

    cancel_calls: list[str] = []
    submit_calls: list[TradeIntent] = []

    def mock_cancel(coid: str) -> Result:
        cancel_calls.append(coid)
        return Result.success(data=None)

    def mock_submit(intent: TradeIntent) -> Result:
        submit_calls.append(intent)
        return Result.success(data=None)

    monkeypatch.setattr(adapter, "cancel_order", mock_cancel)
    monkeypatch.setattr(adapter, "submit_order", mock_submit)

    # Simulate the final fill callback: status="Filled", filled=491.
    entry_trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=491.0, orderRef="O-ENTRY-1", orderId=9001),
        orderStatus=_FakeOrderStatus(status="Filled", filled=491.0, remaining=0.0),
    )
    adapter._on_order_status(entry_trade)

    # SL was 400, should be cancelled and resubmitted at 491.
    assert "O-SL-1" in cancel_calls
    assert len(submit_calls) == 1
    assert submit_calls[0].quantity == 491.0


def test_on_order_status_filled_no_change_skips_realignment(
    monkeypatch: pytest.MonkeyPatch, ibkr_adapter_config: BrokerConnectionConfig
) -> None:
    """Entry Filled with SL already at correct qty → no cancel/resubmit."""

    # SL qty already matches the total fill qty.
    adapter, _, sl_mapping, _ = _setup_entry_with_siblings(
        monkeypatch, ibkr_adapter_config, entry_qty=100.0, sl_qty=100.0,
    )

    cancel_calls: list[str] = []
    submit_calls: list[TradeIntent] = []

    monkeypatch.setattr(adapter, "cancel_order", lambda coid: cancel_calls.append(coid) or Result.success(data=None))
    monkeypatch.setattr(adapter, "submit_order", lambda intent: submit_calls.append(intent) or Result.success(data=None))

    entry_trade = _FakeTrade(
        order=_FakeOrder(action="BUY", totalQuantity=100.0, orderRef="O-ENTRY-1", orderId=9001),
        orderStatus=_FakeOrderStatus(status="Filled", filled=100.0, remaining=0.0),
    )
    adapter._on_order_status(entry_trade)

    # Already aligned → no action.
    assert cancel_calls == []
    assert submit_calls == []
