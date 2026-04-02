"""Shared pytest fixtures for the Execution layer test suite (Phase 3.5).

These fixtures follow the Phase 3.4 testing style used across the repo:
- ``msgspec.Struct`` value objects as canonical test inputs.
- ``unittest.mock.Mock`` for external boundaries (adapter/persistence/reconciler).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.interface import Result
from execution.interface import BrokerAdapterProtocol, BrokerConnectionConfig
from order_state_machine import OrderState, Persistence, Reconciler
from order_state_machine.interface import IntentOrderMapping, ReconciliationResult
from strategy.interface import IntentType, TradeIntent


@pytest.fixture()
def sample_trade_intent() -> TradeIntent:
    """Return a fully-populated TradeIntent fixture (all optional fields set)."""

    return TradeIntent(
        intent_id="I-EXEC-1",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=1_700_000_000_000_000_000,
        entry_price=101.25,
        stop_loss_price=95.0,
        take_profit_price=110.0,
        parent_intent_id=None,
        linked_intent_ids=["I-EXEC-1-SL", "I-EXEC-1-TP"],
        reduce_only=False,
        contingency_type="OTO",
        ladder_level=0,
        reason_codes=["TEST"],
        metadata={"source": "pytest", "run_id": "run-1", "action": "BUY", "position_side": "LONG"},
    )


@pytest.fixture()
def sample_intent_order_mapping(sample_trade_intent: TradeIntent) -> IntentOrderMapping:
    """Return an IntentOrderMapping in SUBMITTED state for the sample intent."""

    now_ns = int(time.time_ns())
    return IntentOrderMapping(
        intent_id=sample_trade_intent.intent_id,
        client_order_id="O-EXEC-1",
        broker_order_id="B-EXEC-1",
        state=OrderState.SUBMITTED,
        created_at_ns=now_ns,
        updated_at_ns=now_ns,
        intent_snapshot=sample_trade_intent,
        metadata={"symbol": "AAPL"},
    )


@pytest.fixture()
def mock_adapter() -> Mock:
    """Return a BrokerAdapterProtocol mock with sensible defaults."""

    adapter = Mock(spec=BrokerAdapterProtocol)
    adapter.is_connected.return_value = True
    return adapter


@pytest.fixture()
def mock_persistence() -> Mock:
    """Return a Persistence mock with successful defaults."""

    persistence = Mock(spec=Persistence)
    persistence.load_all.return_value = Result.success([])
    persistence.save.return_value = Result.success(None)
    return persistence


@pytest.fixture()
def mock_reconciler() -> Mock:
    """Return a Reconciler mock with successful defaults."""

    reconciler = Mock(spec=Reconciler)
    reconciler.reconcile_startup.return_value = Result.success(ReconciliationResult(reconciled_at_ns=0))
    reconciler.reconcile_periodic.return_value = Result.success(ReconciliationResult(reconciled_at_ns=0))
    return reconciler


@pytest.fixture()
def ibkr_adapter_config() -> BrokerConnectionConfig:
    """Return a paper-trading-friendly BrokerConnectionConfig."""

    return BrokerConnectionConfig(
        port=7497,
        client_id=1,
        readonly=False,
        account=None,
        timeout=5,
        max_reconnect_attempts=3,
    )
