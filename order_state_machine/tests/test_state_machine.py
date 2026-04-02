from __future__ import annotations

from typing import Any

import pytest

from common.exceptions import SystemStateError
from common.interface import ResultStatus
from order_state_machine.interface import IntentOrderMapping, OrderState
from order_state_machine.state_machine import StateMachine
from strategy.interface import TradeIntent


def _make_mapping(*, intent: TradeIntent, state: OrderState) -> IntentOrderMapping:
    return IntentOrderMapping(
        intent_id="I-1",
        client_order_id="O-1",
        broker_order_id=None,
        state=state,
        created_at_ns=10,
        updated_at_ns=10,
        intent_snapshot=intent,
        metadata={"base": True},
    )


def test_transition_valid_transitions_all_paths_recorded(mock_trade_intent: TradeIntent) -> None:
    sm = StateMachine()
    rules = sm._build_transition_rules()
    start_ts = 1_000

    for from_state, allowed in rules.items():
        for to_state in allowed:
            mapping = _make_mapping(intent=mock_trade_intent, state=from_state)
            result = sm.transition(mapping, to_state, reason="test", timestamp_ns=start_ts)
            assert result.status is ResultStatus.SUCCESS
            assert result.data is not None
            assert result.data.state is to_state
            assert result.data.updated_at_ns == start_ts

    # One transition record per successful non-idempotent transition.
    assert len(sm.transition_log) == sum(len(v) for v in rules.values())


def test_transition_idempotent_from_state_equals_to_state_no_log(mock_trade_intent: TradeIntent) -> None:
    sm = StateMachine()
    mapping = _make_mapping(intent=mock_trade_intent, state=OrderState.SUBMITTED)
    result = sm.transition(mapping, OrderState.SUBMITTED, reason="repeat", timestamp_ns=123)
    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "IDEMPOTENT"
    assert result.data == mapping
    assert sm.transition_log == ()


def test_transition_invalid_terminal_to_non_self_returns_system_state_error(mock_trade_intent: TradeIntent) -> None:
    sm = StateMachine()
    for terminal in OrderState.terminal_states():
        mapping = _make_mapping(intent=mock_trade_intent, state=terminal)
        result = sm.transition(mapping, OrderState.ACCEPTED, reason="illegal", timestamp_ns=1)
        assert result.status is ResultStatus.FAILED
        assert isinstance(result.error, SystemStateError)
        assert result.reason_code == "SYSTEM_STATE_ERROR"


def test_transition_invalid_non_terminal_pair_returns_failed(mock_trade_intent: TradeIntent) -> None:
    sm = StateMachine()
    # PENDING → FILLED is illegal (must go through SUBMITTED first).
    mapping = _make_mapping(intent=mock_trade_intent, state=OrderState.PENDING)
    result = sm.transition(mapping, OrderState.FILLED, reason="illegal", timestamp_ns=1)
    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, SystemStateError)


def test_submitted_to_partially_filled_and_filled_valid(mock_trade_intent: TradeIntent) -> None:
    """IBKR can skip ACCEPTED; SUBMITTED → PARTIALLY_FILLED and → FILLED must be valid."""
    sm = StateMachine()
    # SUBMITTED → PARTIALLY_FILLED
    mapping = _make_mapping(intent=mock_trade_intent, state=OrderState.SUBMITTED)
    result = sm.transition(mapping, OrderState.PARTIALLY_FILLED, reason="partial", timestamp_ns=1)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.state is OrderState.PARTIALLY_FILLED

    # SUBMITTED → FILLED
    mapping2 = _make_mapping(intent=mock_trade_intent, state=OrderState.SUBMITTED)
    result2 = sm.transition(mapping2, OrderState.FILLED, reason="fill", timestamp_ns=2)
    assert result2.status is ResultStatus.SUCCESS
    assert result2.data is not None
    assert result2.data.state is OrderState.FILLED


def test_is_valid_transition_matrix_all_combinations() -> None:
    sm = StateMachine()
    rules = sm._build_transition_rules()
    states = list(OrderState)

    for a in states:
        for b in states:
            actual = sm.is_valid_transition(a, b)
            if a == b:
                assert actual is True
                continue
            if a in sm.terminal_states:
                assert actual is False
                continue
            expected = b in rules.get(a, set())
            assert actual is expected


def test_build_transition_rules_and_terminal_states_coverage() -> None:
    sm = StateMachine()
    rules = sm._build_transition_rules()
    assert OrderState.PENDING in rules
    assert OrderState.FILLED not in rules
    assert sm.terminal_states == OrderState.terminal_states()


def test_transition_log_drain_returns_and_clears(mock_trade_intent: TradeIntent) -> None:
    sm = StateMachine()
    mapping = _make_mapping(intent=mock_trade_intent, state=OrderState.PENDING)
    r1 = sm.transition(mapping, OrderState.SUBMITTED, reason="submit", timestamp_ns=1, metadata={"a": 1})
    assert r1.status is ResultStatus.SUCCESS and r1.data is not None
    r2 = sm.transition(r1.data, OrderState.ACCEPTED, reason="ack", timestamp_ns=2)
    assert r2.status is ResultStatus.SUCCESS

    assert len(sm.transition_log) == 2
    drained = sm.drain_transition_log()
    assert len(drained) == 2
    assert sm.transition_log == ()


def test_transition_timestamp_override_used_for_audit_record(mock_trade_intent: TradeIntent) -> None:
    sm = StateMachine()
    mapping = _make_mapping(intent=mock_trade_intent, state=OrderState.PENDING)
    result = sm.transition(mapping, OrderState.SUBMITTED, reason="submit", timestamp_ns=999)
    assert result.status is ResultStatus.SUCCESS

    record = sm.transition_log[-1]
    assert record.timestamp_ns == 999
    assert record.metadata == {}


def test_transition_merges_metadata_without_mutating_input(mock_trade_intent: TradeIntent) -> None:
    sm = StateMachine()
    mapping = _make_mapping(intent=mock_trade_intent, state=OrderState.PENDING)
    meta: dict[str, Any] = {"x": 1}
    result = sm.transition(mapping, OrderState.SUBMITTED, reason="submit", timestamp_ns=20, metadata=meta)
    assert result.status is ResultStatus.SUCCESS and result.data is not None
    assert result.data.metadata["base"] is True
    assert result.data.metadata["x"] == 1
    assert meta == {"x": 1}


def test_transition_exception_is_caught_and_returns_failed(mock_trade_intent: TradeIntent, monkeypatch: pytest.MonkeyPatch) -> None:
    sm = StateMachine()
    monkeypatch.setattr(sm, "_create_updated_mapping", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    mapping = _make_mapping(intent=mock_trade_intent, state=OrderState.PENDING)
    result = sm.transition(mapping, OrderState.SUBMITTED, reason="submit", timestamp_ns=1)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "STATE_MACHINE_ERROR"


def test_is_valid_transition_missing_rules_entry_returns_false() -> None:
    sm = StateMachine()
    sm._rules.pop(OrderState.PENDING, None)
    assert sm.is_valid_transition(OrderState.PENDING, OrderState.SUBMITTED) is False


def test_validate_transition_rules_rejects_invalid_definitions() -> None:
    sm = StateMachine()
    with pytest.raises(ValueError):
        sm._validate_transition_rules({OrderState.FILLED: {OrderState.CANCELLED}})
    with pytest.raises(ValueError):
        sm._validate_transition_rules({OrderState.PENDING: set()})
    with pytest.raises(ValueError):
        sm._validate_transition_rules({OrderState.PENDING: {OrderState.PENDING}})
    with pytest.raises(ValueError):
        sm._validate_transition_rules({OrderState.SUBMITTED: {OrderState.PENDING}})
