"""Risk Gate core evaluation engine (Phase 3.3).

This module implements the Risk Gate "gateway" logic that evaluates Strategy
Engine intents against a pluggable set of risk checks and produces a
deterministic :class:`~risk_gate.interface.RiskDecisionSet`.

Design goals (see ``docs/sessions/session-2025-12-19-risk-gate.md``):
    - Unified entry point: :func:`evaluate_intents` processes an entire
      :class:`~strategy.interface.OrderIntentSet`.
    - Result-first propagation: all public functions return
      :class:`common.interface.Result` objects.
    - Default BLOCK on uncertainty: exceptions or failed results from checks
      are treated as check FAIL outcomes.
    - Minimal Safe Mode integration: SAFE_REDUCING allows only
      reduce/cancel/protect intents; HALTED blocks all intents.
    - Degradation-first decisioning: checks may recommend Safe Mode escalation
      using ``details["safe_mode_trigger"]=True`` or ``SAFE_MODE.*`` reason codes.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import msgspec

from common.interface import Result, ResultStatus
from risk_gate.interface import (
    CheckResult,
    CheckStatus,
    DecisionType,
    RiskCheckContext,
    RiskCheckProtocol,
    RiskDecision,
    RiskDecisionSet,
    RiskGateConfig,
    SafeModeState,
)
from strategy.interface import IntentType, OrderIntentSet, TradeIntent

__all__ = [
    "evaluate_intents",
]

_SCHEMA_VERSION = "3.3.0"


def evaluate_intents(
    intents: OrderIntentSet,
    context: RiskCheckContext,
    config: RiskGateConfig,
    checks: Sequence[RiskCheckProtocol],
    *,
    current_time_ns: int | None = None,
) -> Result[RiskDecisionSet]:
    """Evaluate all intents in an :class:`~strategy.interface.OrderIntentSet`.

    Args:
        intents: Strategy Engine output snapshot containing intent groups.
        context: Point-in-time risk context (equity, positions, market data,
            and current Safe Mode state).
        config: Risk Gate configuration thresholds.
        checks: Ordered list of risk checks to apply to each intent.
        current_time_ns: Optional timestamp override for tests; defaults to
            ``time.time_ns()``.

    Returns:
        Result[RiskDecisionSet]:
            - ``SUCCESS``: Risk evaluation completed normally.
            - ``DEGRADED``: Safe Mode is active or a check requested downgrade.
            - ``FAILED``: Unexpected exception prevented evaluation.
    """

    try:
        checked_at_ns = int(time.time_ns() if current_time_ns is None else current_time_ns)
        decisions: list[RiskDecision] = []
        for group in list(intents.intent_groups or []):
            group_context = context
            try:
                group_context = msgspec.structs.replace(group_context, intents=list(group.intents or []))
            except Exception:  # pragma: no cover - tolerate older context shapes
                group_context = context

            for intent in list(group.intents or []):
                decisions.append(
                    _evaluate_single_intent(
                        intent=intent,
                        context=group_context,
                        config=config,
                        checks=checks,
                        checked_at_ns=checked_at_ns,
                    )
                )

        safe_mode_active = context.safe_mode_state != SafeModeState.ACTIVE
        safe_mode_reason = (
            None if not safe_mode_active else f"SAFE_MODE_{context.safe_mode_state.value}"
        )

        constraints_snapshot: dict[str, Any]
        try:
            constraints_snapshot = msgspec.to_builtins(config)
        except Exception:  # noqa: BLE001 - tolerate optional msgspec behavior drift
            constraints_snapshot = {
                "portfolio": msgspec.to_builtins(config.portfolio),
                "symbol": msgspec.to_builtins(config.symbol),
                "operational": msgspec.to_builtins(config.operational),
            }

        decision_set = RiskDecisionSet(
            schema_version=_SCHEMA_VERSION,
            system_version=intents.system_version,
            asof_timestamp=checked_at_ns,
            decisions=decisions,
            safe_mode_active=safe_mode_active,
            safe_mode_reason=safe_mode_reason,
            constraints_snapshot=constraints_snapshot,
        )

        downgraded = any(dec.decision_type == DecisionType.DOWNGRADE for dec in decisions)
        if safe_mode_active or downgraded:
            return Result.degraded(decision_set, RuntimeError("safe mode active"), "SAFE_MODE_ACTIVE")
        return Result.success(decision_set)
    except Exception as exc:  # noqa: BLE001 - engine must not crash orchestrator
        return Result.failed(exc, "RISK_GATE_EVALUATION_ERROR")


def _evaluate_single_intent(
    *,
    intent: TradeIntent,
    context: RiskCheckContext,
    config: RiskGateConfig,
    checks: Sequence[RiskCheckProtocol],
    checked_at_ns: int,
) -> RiskDecision:
    """Evaluate one intent against Safe Mode and all configured checks.

    Decision rules:
        - All checks PASS => ALLOW (subject to Safe Mode gating).
        - Any check FAIL + Safe Mode trigger => DOWNGRADE.
        - Any check FAIL (non Safe Mode) => BLOCK.

    Default BLOCK on uncertainty:
        - A check returning ``Result.failed`` is treated as a FAIL outcome.
        - A check raising an exception is caught and treated as FAIL with
          reason code ``CHECK_ERROR``.
    """

    safe_mode_state = context.safe_mode_state
    if safe_mode_state == SafeModeState.HALTED:
        return _aggregate_check_results(
            intent_id=intent.intent_id,
            decision_type=DecisionType.BLOCK,
            check_outcomes=[],
            checked_at_ns=checked_at_ns,
            safe_mode_state=safe_mode_state,
            extra_reason_codes=["SAFE_MODE_HALTED"],
            extra_details={"safe_mode_state": safe_mode_state.value},
        )

    if safe_mode_state == SafeModeState.SAFE_REDUCING and not _is_safe_reducing_allowed(intent):
        return _aggregate_check_results(
            intent_id=intent.intent_id,
            decision_type=DecisionType.BLOCK,
            check_outcomes=[],
            checked_at_ns=checked_at_ns,
            safe_mode_state=safe_mode_state,
            extra_reason_codes=["SAFE_MODE_REDUCING_ONLY"],
            extra_details={"safe_mode_state": safe_mode_state.value},
        )

    check_outcomes: list[tuple[str, CheckResult]] = []
    for check in checks:
        check_name = getattr(check, "__class__", type(check)).__name__
        try:
            result = check.check(intent, context, config)
        except Exception as exc:  # noqa: BLE001 - default BLOCK on uncertainty
            check_outcomes.append(
                (
                    check_name,
                    CheckResult(
                        status=CheckStatus.FAIL,
                        reason_codes=["CHECK_ERROR"],
                        details={
                            "check": check_name,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    ),
                )
            )
            continue

        check_outcomes.append((check_name, _coerce_check_result(result, check_name)))

    decision_type = _decide(check_outcomes, safe_mode_state=safe_mode_state)
    return _aggregate_check_results(
        intent_id=intent.intent_id,
        decision_type=decision_type,
        check_outcomes=check_outcomes,
        checked_at_ns=checked_at_ns,
        safe_mode_state=safe_mode_state,
        extra_details={"symbol": intent.symbol, "intent_type": intent.intent_type.value},
    )


def _aggregate_check_results(
    *,
    intent_id: str,
    decision_type: DecisionType,
    check_outcomes: Iterable[tuple[str, CheckResult]],
    checked_at_ns: int,
    safe_mode_state: SafeModeState,
    extra_reason_codes: Sequence[str] | None = None,
    extra_details: dict[str, Any] | None = None,
) -> RiskDecision:
    """Aggregate check outcomes into a single :class:`~risk_gate.interface.RiskDecision`.

    Aggregation rules:
        - ``reason_codes`` are de-duplicated while preserving order.
        - ``details`` includes a structured per-check payload under
          ``details["checks"]`` plus any provided ``extra_details``.
    """

    reason_codes: list[str] = []
    details: dict[str, Any] = {}
    checks_payload: dict[str, Any] = {}

    for check_name, outcome in check_outcomes:
        checks_payload[str(check_name)] = {
            "status": outcome.status.value,
            "reason_codes": list(outcome.reason_codes or []),
            "details": dict(outcome.details or {}),
        }
        for code in outcome.reason_codes or []:
            if code:
                reason_codes.append(str(code))

    if extra_reason_codes:
        for code in extra_reason_codes:
            if code:
                reason_codes.append(str(code))

    if checks_payload:
        details["checks"] = checks_payload
    details["safe_mode_state"] = safe_mode_state.value

    if extra_details:
        # Extra details take precedence over generic fields.
        details.update(extra_details)

    deduped_codes = _dedupe_preserve_order(reason_codes)
    if decision_type != DecisionType.ALLOW and not deduped_codes:
        deduped_codes = ["RISK_BLOCKED"]

    return RiskDecision(
        intent_id=str(intent_id),
        decision_type=decision_type,
        reason_codes=deduped_codes,
        details=details,
        checked_at_ns=int(checked_at_ns),
    )


def _coerce_check_result(result: Result[CheckResult], check_name: str) -> CheckResult:
    """Normalize a check invocation output into a concrete :class:`CheckResult`.

    The Risk Gate treats non-success check results as FAIL by default
    (fail-closed). If a degraded result includes data, the underlying
    ``CheckResult`` is preserved, but missing data is treated as FAIL.
    """

    if result.status == ResultStatus.SUCCESS and result.data is not None:
        return result.data

    if result.status == ResultStatus.DEGRADED and result.data is not None:
        return result.data

    reason = result.reason_code or (
        "CHECK_FAILED" if result.status == ResultStatus.FAILED else "CHECK_DEGRADED"
    )
    base_details = {
        "check": check_name,
        "result_status": result.status.value,
    }
    if result.error is not None:
        base_details["error_type"] = type(result.error).__name__
        base_details["error"] = str(result.error)
    return CheckResult(status=CheckStatus.FAIL, reason_codes=[str(reason)], details=base_details)


def _decide(
    check_outcomes: Sequence[tuple[str, CheckResult]],
    *,
    safe_mode_state: SafeModeState,
) -> DecisionType:
    """Apply decision policy to aggregated per-check outcomes."""

    any_fail = any(outcome.status == CheckStatus.FAIL for _, outcome in check_outcomes)
    if not any_fail:
        return DecisionType.ALLOW

    safe_mode_triggered = safe_mode_state == SafeModeState.SAFE_REDUCING or _any_safe_mode_trigger(
        check_outcomes
    )
    return DecisionType.DOWNGRADE if safe_mode_triggered else DecisionType.BLOCK


def _any_safe_mode_trigger(check_outcomes: Sequence[tuple[str, CheckResult]]) -> bool:
    """Detect Safe Mode trigger requests embedded by risk checks."""

    for _, outcome in check_outcomes:
        if outcome.status != CheckStatus.FAIL:
            continue
        details = outcome.details or {}
        if bool(details.get("safe_mode_trigger")):
            return True
        for code in outcome.reason_codes or []:
            code_str = str(code)
            if code_str == "SAFE_MODE_TRIGGER" or code_str.startswith("SAFE_MODE."):
                return True
    return False


def _is_safe_reducing_allowed(intent: TradeIntent) -> bool:
    """Return True when an intent is allowed during SAFE_REDUCING mode."""

    if bool(getattr(intent, "reduce_only", False)):
        return True

    return intent.intent_type in (
        IntentType.CLOSE_LONG,
        IntentType.CLOSE_SHORT,
        IntentType.REDUCE_POSITION,
        IntentType.CANCEL_PENDING,
        IntentType.STOP_LOSS,
        IntentType.TAKE_PROFIT,
    )


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        v = str(value)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out
