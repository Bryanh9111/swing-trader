from __future__ import annotations

import msgspec
import msgspec.json

from risk_gate.interface import (
    CheckResult,
    CheckStatus,
    DecisionType,
    OperationalRiskConfig,
    PortfolioRiskConfig,
    RiskCheckContext,
    RiskDecision,
    RiskDecisionSet,
    RiskGateConfig,
    SafeModeState,
    SymbolRiskConfig,
)


if not hasattr(msgspec.json, "DecodeError"):
    setattr(msgspec.json, "DecodeError", msgspec.DecodeError)
if not hasattr(msgspec.json, "EncodeError"):
    setattr(msgspec.json, "EncodeError", msgspec.EncodeError)


def test_enums_have_expected_values() -> None:
    assert DecisionType.ALLOW.value == "ALLOW"
    assert DecisionType.BLOCK.value == "BLOCK"
    assert DecisionType.DOWNGRADE.value == "DOWNGRADE"

    assert SafeModeState.ACTIVE.value == "ACTIVE"
    assert SafeModeState.SAFE_REDUCING.value == "SAFE_REDUCING"
    assert SafeModeState.HALTED.value == "HALTED"

    assert CheckStatus.PASS.value == "PASS"
    assert CheckStatus.FAIL.value == "FAIL"


def test_checkresult_riskdecision_and_decisionset_roundtrip_json() -> None:
    check = CheckResult(
        status=CheckStatus.FAIL,
        reason_codes=["A", "B"],
        details={"x": 1, "safe_mode_trigger": True},
    )
    decision = RiskDecision(
        intent_id="intent-1",
        decision_type=DecisionType.DOWNGRADE,
        reason_codes=["A"],
        details={"checks": {"Dummy": {"status": "FAIL"}}},
        checked_at_ns=123,
    )
    decision_set = RiskDecisionSet(
        schema_version="3.3.0",
        system_version="test",
        asof_timestamp=123,
        decisions=[decision],
        safe_mode_active=True,
        safe_mode_reason="SAFE_MODE_SAFE_REDUCING",
        constraints_snapshot={"portfolio": {"max_leverage": 1.5}},
    )

    encoded = msgspec.json.encode(check)
    decoded = msgspec.json.decode(encoded, type=CheckResult)
    assert decoded == check

    encoded = msgspec.json.encode(decision)
    decoded = msgspec.json.decode(encoded, type=RiskDecision)
    assert decoded == decision

    encoded = msgspec.json.encode(decision_set)
    decoded = msgspec.json.decode(encoded, type=RiskDecisionSet)
    assert decoded == decision_set


def test_config_defaults() -> None:
    portfolio = PortfolioRiskConfig()
    assert portfolio.max_leverage == 1.5
    assert portfolio.max_drawdown_pct == 0.2
    assert portfolio.max_daily_loss_pct == 0.02
    assert portfolio.max_concentration_pct == 0.25

    symbol = SymbolRiskConfig()
    assert symbol.max_position_size == 10_000.0
    assert symbol.max_price_band_bps == 250.0
    assert symbol.max_volatility_z_score == 3.0
    assert symbol.event_window_hours == 24.0

    operational = OperationalRiskConfig()
    assert operational.max_orders_per_run == 50
    assert operational.max_new_positions_per_day == 15
    assert operational.rate_limit_per_second == 5.0
    assert operational.max_order_count == 200


def test_risk_gate_config_and_context_msgspec_compatibility() -> None:
    config = RiskGateConfig()
    builtins = msgspec.to_builtins(config)
    assert builtins["portfolio"]["max_leverage"] == 1.5
    assert builtins["symbol"]["max_price_band_bps"] == 250.0

    encoded = msgspec.json.encode(config)
    decoded = msgspec.json.decode(encoded, type=RiskGateConfig)
    assert decoded == config

    context = RiskCheckContext(
        account_equity=100.0,
        portfolio_snapshot={"peak_equity": 100.0, "day_start_equity": 100.0},
        positions={"AAPL": {"qty": 1.0}},
        market_data={"AAPL": {"note": "untyped snapshot in early phases"}},
        safe_mode_state=SafeModeState.ACTIVE,
    )
    encoded = msgspec.json.encode(context)
    decoded = msgspec.json.decode(encoded, type=RiskCheckContext)
    assert decoded == context
