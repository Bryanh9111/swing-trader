from __future__ import annotations

import msgspec
import pytest

from common.interface import Result, ResultStatus
from risk_gate import engine as rg_engine
from risk_gate.interface import (
    CheckResult,
    CheckStatus,
    DecisionType,
    RiskCheckContext,
    RiskDecisionSet,
    RiskGateConfig,
    SafeModeState,
)
from strategy.interface import IntentGroup, IntentType, OrderIntentSet, TradeIntent


def _intent(
    *,
    intent_id: str,
    symbol: str = "AAPL",
    intent_type: IntentType = IntentType.OPEN_LONG,
    quantity: float = 1.0,
    reduce_only: bool = False,
) -> TradeIntent:
    return TradeIntent(
        intent_id=intent_id,
        symbol=symbol,
        intent_type=intent_type,
        quantity=float(quantity),
        created_at_ns=1,
        reduce_only=bool(reduce_only),
    )


def _intent_set(*intents: TradeIntent) -> OrderIntentSet:
    group = IntentGroup(group_id="g-1", symbol=intents[0].symbol, intents=list(intents), created_at_ns=1)
    return OrderIntentSet(
        schema_version="3.2.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=[group],
        constraints_applied={},
        source_candidates=[group.symbol],
    )


class _PassCheck:
    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        return Result.success(CheckResult(status=CheckStatus.PASS, reason_codes=[], details={"check": "pass"}))


class _FailCheck:
    def __init__(self, *, safe_mode_trigger: bool = False, reason_codes: list[str] | None = None) -> None:
        self._trigger = bool(safe_mode_trigger)
        self._reason_codes = list(reason_codes or ["FAIL"])

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        details = {"check": "fail"}
        if self._trigger:
            details["safe_mode_trigger"] = True
        return Result.success(
            CheckResult(status=CheckStatus.FAIL, reason_codes=list(self._reason_codes), details=details)
        )


class _FailedResultCheck:
    def __init__(self, *, reason_code: str = "CHECK_BROKE") -> None:
        self._reason_code = str(reason_code)

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        return Result.failed(RuntimeError("boom"), self._reason_code)


class _ExceptionCheck:
    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        raise ValueError("kaboom")


def test_evaluate_intents_basic_flow_allow() -> None:
    intents = _intent_set(_intent(intent_id="i-1"))
    context = RiskCheckContext(account_equity=100.0)
    res = rg_engine.evaluate_intents(intents, context, RiskGateConfig(), checks=[_PassCheck()], current_time_ns=123)

    assert res.status is ResultStatus.SUCCESS
    assert isinstance(res.data, RiskDecisionSet)
    assert res.data.safe_mode_active is False
    assert res.data.safe_mode_reason is None
    assert len(res.data.decisions) == 1
    assert res.data.decisions[0].decision_type is DecisionType.ALLOW
    assert res.data.decisions[0].checked_at_ns == 123
    assert res.data.constraints_snapshot["portfolio"]["max_leverage"] == 1.5


def test_safe_mode_halted_blocks_all_intents() -> None:
    intents = _intent_set(_intent(intent_id="i-1", intent_type=IntentType.CLOSE_LONG))
    context = RiskCheckContext(account_equity=100.0, safe_mode_state=SafeModeState.HALTED)
    res = rg_engine.evaluate_intents(intents, context, RiskGateConfig(), checks=[_PassCheck()], current_time_ns=1)

    assert res.status is ResultStatus.DEGRADED
    assert res.data is not None
    assert res.data.safe_mode_active is True
    assert res.data.safe_mode_reason == "SAFE_MODE_HALTED"
    assert res.data.decisions[0].decision_type is DecisionType.BLOCK
    assert "SAFE_MODE_HALTED" in res.data.decisions[0].reason_codes
    assert res.data.decisions[0].details["safe_mode_state"] == "HALTED"
    assert "checks" not in res.data.decisions[0].details


def test_safe_reducing_blocks_opening_intents() -> None:
    intents = _intent_set(_intent(intent_id="i-1", intent_type=IntentType.OPEN_LONG))
    context = RiskCheckContext(account_equity=100.0, safe_mode_state=SafeModeState.SAFE_REDUCING)
    res = rg_engine.evaluate_intents(intents, context, RiskGateConfig(), checks=[_PassCheck()], current_time_ns=1)

    assert res.status is ResultStatus.DEGRADED
    assert res.data is not None
    decision = res.data.decisions[0]
    assert decision.decision_type is DecisionType.BLOCK
    assert "SAFE_MODE_REDUCING_ONLY" in decision.reason_codes
    assert decision.details["safe_mode_state"] == "SAFE_REDUCING"


def test_decision_rules_block_vs_downgrade() -> None:
    intents = _intent_set(_intent(intent_id="i-1"))
    context = RiskCheckContext(account_equity=100.0, safe_mode_state=SafeModeState.ACTIVE)

    res = rg_engine.evaluate_intents(intents, context, RiskGateConfig(), checks=[_FailCheck()], current_time_ns=1)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.decisions[0].decision_type is DecisionType.BLOCK

    res = rg_engine.evaluate_intents(
        intents, context, RiskGateConfig(), checks=[_FailCheck(safe_mode_trigger=True)], current_time_ns=1
    )
    assert res.status is ResultStatus.DEGRADED
    assert res.data is not None
    assert res.data.decisions[0].decision_type is DecisionType.DOWNGRADE


def test_fail_closed_behavior_for_failed_check_results() -> None:
    intents = _intent_set(_intent(intent_id="i-1"))
    context = RiskCheckContext(account_equity=100.0)

    res = rg_engine.evaluate_intents(intents, context, RiskGateConfig(), checks=[_FailedResultCheck()], current_time_ns=1)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    decision = res.data.decisions[0]
    assert decision.decision_type is DecisionType.BLOCK
    assert "CHECK_BROKE" in decision.reason_codes

    checks_payload = decision.details["checks"]
    assert checks_payload[" _FailedResultCheck".strip()]["status"] == "FAIL"
    assert checks_payload[" _FailedResultCheck".strip()]["reason_codes"] == ["CHECK_BROKE"]
    assert checks_payload[" _FailedResultCheck".strip()]["details"]["result_status"] == "failed"
    assert checks_payload[" _FailedResultCheck".strip()]["details"]["error_type"] == "RuntimeError"


def test_check_exception_handled_as_check_error() -> None:
    intents = _intent_set(_intent(intent_id="i-1"))
    context = RiskCheckContext(account_equity=100.0)
    res = rg_engine.evaluate_intents(intents, context, RiskGateConfig(), checks=[_ExceptionCheck()], current_time_ns=1)

    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    decision = res.data.decisions[0]
    assert decision.decision_type is DecisionType.BLOCK
    assert "CHECK_ERROR" in decision.reason_codes
    payload = decision.details["checks"]["_ExceptionCheck"]
    assert payload["status"] == "FAIL"
    assert payload["details"]["error_type"] == "ValueError"


def test_safe_mode_trigger_detects_reason_codes() -> None:
    intents = _intent_set(_intent(intent_id="i-1"))
    context = RiskCheckContext(account_equity=100.0)
    res = rg_engine.evaluate_intents(
        intents, context, RiskGateConfig(), checks=[_FailCheck(reason_codes=["SAFE_MODE.TRIGGERED"])], current_time_ns=1
    )

    assert res.status is ResultStatus.DEGRADED
    assert res.data is not None
    assert res.data.decisions[0].decision_type is DecisionType.DOWNGRADE


def test_is_safe_reducing_allowed_logic() -> None:
    assert rg_engine._is_safe_reducing_allowed(_intent(intent_id="i-1", intent_type=IntentType.CLOSE_LONG)) is True
    assert rg_engine._is_safe_reducing_allowed(_intent(intent_id="i-1", intent_type=IntentType.STOP_LOSS)) is True
    assert rg_engine._is_safe_reducing_allowed(
        _intent(intent_id="i-1", intent_type=IntentType.OPEN_LONG, reduce_only=True)
    ) is True
    assert rg_engine._is_safe_reducing_allowed(_intent(intent_id="i-1", intent_type=IntentType.OPEN_LONG)) is False


def test_reason_code_dedupe_and_details_aggregation() -> None:
    decision = rg_engine._aggregate_check_results(
        intent_id="i-1",
        decision_type=DecisionType.BLOCK,
        check_outcomes=[
            ("A", CheckResult(status=CheckStatus.FAIL, reason_codes=["X", "X", ""], details={"a": 1})),
            ("B", CheckResult(status=CheckStatus.FAIL, reason_codes=["Y", "X"], details={})),
        ],
        checked_at_ns=1,
        safe_mode_state=SafeModeState.ACTIVE,
        extra_reason_codes=["Y", "Z", ""],
        extra_details={"intent_type": "OPEN_LONG"},
    )

    assert decision.reason_codes == ["X", "Y", "Z"]
    assert decision.details["checks"]["A"]["details"]["a"] == 1
    assert decision.details["intent_type"] == "OPEN_LONG"
    assert decision.details["safe_mode_state"] == "ACTIVE"


def test_evaluate_intents_failure_returns_failed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    intents = _intent_set(_intent(intent_id="i-1"))
    context = RiskCheckContext(account_equity=100.0)
    monkeypatch.setattr(rg_engine, "_evaluate_single_intent", _boom)

    res = rg_engine.evaluate_intents(intents, context, RiskGateConfig(), checks=[_PassCheck()], current_time_ns=1)
    assert res.status is ResultStatus.FAILED
    assert res.reason_code == "RISK_GATE_EVALUATION_ERROR"


def test_constraints_snapshot_fallback_path(monkeypatch: pytest.MonkeyPatch) -> None:
    original = msgspec.to_builtins

    def _patched(obj: object, *args: object, **kwargs: object) -> object:
        if isinstance(obj, RiskGateConfig):
            raise RuntimeError("nope")
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(rg_engine.msgspec, "to_builtins", _patched)

    intents = _intent_set(_intent(intent_id="i-1"))
    context = RiskCheckContext(account_equity=100.0)
    res = rg_engine.evaluate_intents(intents, context, RiskGateConfig(), checks=[_PassCheck()], current_time_ns=123)

    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert set(res.data.constraints_snapshot) == {"portfolio", "symbol", "operational"}
