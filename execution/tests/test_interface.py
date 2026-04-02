"""Schema and protocol tests for the Execution layer interface contracts."""

from __future__ import annotations

from typing import Any

import msgspec
import msgspec.json

from common.interface import Result
from execution import (
    BrokerAdapterProtocol,
    BrokerConnectionConfig,
    ExecutionReport,
    FillDetail,
    IBKRAdapter,
    OrderManager,
)
from order_state_machine.interface import OrderState, StateTransition
from strategy.interface import TradeIntent

if not hasattr(msgspec.json, "DecodeError"):
    setattr(msgspec.json, "DecodeError", msgspec.DecodeError)
if not hasattr(msgspec.json, "EncodeError"):
    setattr(msgspec.json, "EncodeError", msgspec.EncodeError)


def test_fill_detail_schema_valid() -> None:
    """FillDetail supports msgspec JSON roundtrip with required fields."""

    # Arrange
    fill = FillDetail(
        execution_id="E-1",
        fill_price=100.25,
        fill_quantity=10.0,
        fill_time_ns=1_700_000_000_000_000_000,
        commission=0.42,
        metadata={"venue": "TEST"},
    )

    # Act
    encoded = msgspec.json.encode(fill)
    decoded = msgspec.json.decode(encoded, type=FillDetail)

    # Assert
    assert decoded == fill


def test_fill_detail_optional_fields() -> None:
    """FillDetail optional fields default to None/empty mapping."""

    # Arrange / Act
    fill = FillDetail(
        execution_id="E-1",
        fill_price=100.0,
        fill_quantity=1.0,
        fill_time_ns=1,
    )

    # Assert
    assert fill.commission is None
    assert fill.metadata == {}


def test_execution_report_schema_valid() -> None:
    """ExecutionReport supports msgspec JSON roundtrip for a terminal report."""

    # Arrange
    report = ExecutionReport(
        run_id="run-1",
        intent_id="I-1",
        client_order_id="O-1",
        broker_order_id="B-1",
        symbol="AAPL",
        final_state=OrderState.FILLED,
        filled_quantity=10.0,
        remaining_quantity=0.0,
        avg_fill_price=100.0,
        commissions=0.5,
        fills=[
            FillDetail(
                execution_id="E-1",
                fill_price=100.0,
                fill_quantity=10.0,
                fill_time_ns=1,
                commission=0.5,
            )
        ],
        state_transitions=[
            StateTransition(
                intent_id="I-1",
                from_state=OrderState.SUBMITTED,
                to_state=OrderState.FILLED,
                timestamp_ns=1,
                reason="TEST",
            )
        ],
        executed_at_ns=2,
        metadata={"source": "pytest"},
    )

    # Act
    encoded = msgspec.json.encode(report)
    decoded = msgspec.json.decode(encoded, type=ExecutionReport)

    # Assert
    assert decoded == report


def test_execution_report_terminal_states() -> None:
    """ExecutionReport supports all terminal OrderState values."""

    for state in OrderState.terminal_states():
        report = ExecutionReport(
            run_id="run-1",
            intent_id="I-1",
            client_order_id="O-1",
            broker_order_id="B-1",
            symbol="AAPL",
            final_state=state,
            filled_quantity=0.0,
            remaining_quantity=0.0,
            executed_at_ns=1,
        )
        assert report.final_state == state


def test_broker_connection_config_defaults() -> None:
    """BrokerConnectionConfig provides stable defaults for optional fields."""

    config = BrokerConnectionConfig(port=7497)
    assert config.host == "127.0.0.1"
    assert config.client_id == 1
    assert config.readonly is False
    assert config.account is None
    assert config.timeout == 20
    assert config.max_reconnect_attempts == 5


def test_broker_connection_config_custom() -> None:
    """BrokerConnectionConfig accepts custom values for all fields."""

    config = BrokerConnectionConfig(
        host="localhost",
        port=4002,
        client_id=7,
        readonly=True,
        account="DU123",
        timeout=3,
        max_reconnect_attempts=9,
    )
    assert config.host == "localhost"
    assert config.port == 4002
    assert config.client_id == 7
    assert config.readonly is True
    assert config.account == "DU123"
    assert config.timeout == 3
    assert config.max_reconnect_attempts == 9


def test_broker_adapter_protocol_runtime_check() -> None:
    """BrokerAdapterProtocol is runtime-checkable for adapter instances."""

    class Dummy(BrokerAdapterProtocol):  # type: ignore[misc]
        def connect(self) -> Result[None]:  # type: ignore[override]
            return Result.success(None)

        def disconnect(self) -> Result[None]:  # type: ignore[override]
            return Result.success(None)

        def is_connected(self) -> bool:  # type: ignore[override]
            return True

        def submit_order(self, intent: TradeIntent) -> Result[Any]:  # type: ignore[override]
            return Result.failed(RuntimeError("not implemented"), "NOT_IMPLEMENTED")

        def cancel_order(self, client_order_id: str) -> Result[None]:  # type: ignore[override]
            return Result.success(None)

        def get_order_status(self, client_order_id: str) -> Result[Any]:  # type: ignore[override]
            return Result.failed(RuntimeError("not implemented"), "NOT_IMPLEMENTED")

        def get_all_orders(self) -> Result[list[Any]]:  # type: ignore[override]
            return Result.success([])

    dummy = Dummy()
    assert isinstance(dummy, BrokerAdapterProtocol)


def test_execution_package_exports() -> None:
    """The execution package exports the public API surface (import coverage)."""

    import execution as execution_pkg

    assert execution_pkg.BrokerConnectionConfig is BrokerConnectionConfig
    assert execution_pkg.FillDetail is FillDetail
    assert execution_pkg.ExecutionReport is ExecutionReport
    assert execution_pkg.IBKRAdapter is IBKRAdapter
    assert execution_pkg.OrderManager is OrderManager

