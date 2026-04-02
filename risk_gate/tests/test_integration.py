from __future__ import annotations

import msgspec
import msgspec.json
import pytest

from common.interface import ResultStatus
from data.interface import PriceBar, PriceSeriesSnapshot
from risk_gate.checks import (
    ConcentrationCheck,
    DailyLossCheck,
    DrawdownCheck,
    LeverageCheck,
    MaxPositionCheck,
    OrderCountCheck,
    PriceBandCheck,
    RateLimitCheck,
    ReduceOnlyCheck,
    VolatilityHaltCheck,
)
from risk_gate.engine import evaluate_intents
from risk_gate.interface import RiskCheckContext, RiskDecisionSet, RiskGateConfig, SafeModeState
from strategy.interface import IntentGroup, IntentType, OrderIntentSet, TradeIntent


if not hasattr(msgspec.json, "DecodeError"):
    setattr(msgspec.json, "DecodeError", msgspec.DecodeError)
if not hasattr(msgspec.json, "EncodeError"):
    setattr(msgspec.json, "EncodeError", msgspec.EncodeError)


def _series(symbol: str, closes: list[float]) -> PriceSeriesSnapshot:
    bars = [PriceBar(timestamp=i + 1, open=c, high=c, low=c, close=c, volume=1000) for i, c in enumerate(closes)]
    return PriceSeriesSnapshot(
        schema_version="2.2.0",
        system_version="test",
        asof_timestamp=1,
        symbol=symbol,
        timeframe="1d",
        bars=bars,
        source="test",
        quality_flags={},
    )


def _intent_set(groups: list[IntentGroup]) -> OrderIntentSet:
    return OrderIntentSet(
        schema_version="3.2.0",
        system_version="test",
        asof_timestamp=1,
        intent_groups=groups,
        constraints_applied={},
        source_candidates=[g.symbol for g in groups],
    )


@pytest.mark.integration
def test_integration_full_risk_gate_workflow_allow() -> None:
    checks = [
        LeverageCheck(),
        DrawdownCheck(),
        DailyLossCheck(),
        ConcentrationCheck(),
        MaxPositionCheck(),
        PriceBandCheck(),
        VolatilityHaltCheck(lookback_returns=5),
        ReduceOnlyCheck(),
        OrderCountCheck(),
        RateLimitCheck(),
    ]

    entry = TradeIntent(
        intent_id="i-entry",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=1,
        entry_price=100.0,
    )
    stop = TradeIntent(
        intent_id="i-sl",
        symbol="AAPL",
        intent_type=IntentType.STOP_LOSS,
        quantity=10.0,
        created_at_ns=1,
        stop_loss_price=95.0,
        reduce_only=True,
        parent_intent_id="i-entry",
    )
    take = TradeIntent(
        intent_id="i-tp",
        symbol="AAPL",
        intent_type=IntentType.TAKE_PROFIT,
        quantity=10.0,
        created_at_ns=1,
        take_profit_price=110.0,
        reduce_only=True,
        parent_intent_id="i-entry",
    )
    intents = _intent_set(
        [IntentGroup(group_id="g-1", symbol="AAPL", intents=[entry, stop, take], created_at_ns=1)]
    )

    context = RiskCheckContext(
        account_equity=100_000.0,
        portfolio_snapshot={"peak_equity": 100_000.0, "day_start_equity": 100_000.0},
        positions={"AAPL": 0.0},
        market_data={"AAPL": _series("AAPL", [99.5, 100.0, 100.2, 100.1, 100.0, 100.0])},
        safe_mode_state=SafeModeState.ACTIVE,
    )
    config = RiskGateConfig(
        symbol=msgspec.structs.replace(
            RiskGateConfig().symbol,
            max_price_band_bps=2_000.0,
            max_position_size=100_000.0,
            max_volatility_z_score=10.0,
        )
    )

    res = evaluate_intents(intents, context, config, checks, current_time_ns=123)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert len(res.data.decisions) == 3
    assert all(d.decision_type.value == "ALLOW" for d in res.data.decisions)


@pytest.mark.integration
def test_integration_multi_intent_batch_processing() -> None:
    groups = [
        IntentGroup(
            group_id="g-1",
            symbol="AAPL",
            intents=[
                TradeIntent(intent_id="a1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1.0, created_at_ns=1),
                TradeIntent(intent_id="a2", symbol="AAPL", intent_type=IntentType.CANCEL_PENDING, quantity=0.0, created_at_ns=1),
            ],
            created_at_ns=1,
        ),
        IntentGroup(
            group_id="g-2",
            symbol="MSFT",
            intents=[
                TradeIntent(intent_id="m1", symbol="MSFT", intent_type=IntentType.OPEN_LONG, quantity=1.0, created_at_ns=1),
            ],
            created_at_ns=1,
        ),
    ]
    intents = _intent_set(groups)
    context = RiskCheckContext(
        account_equity=100_000.0,
        market_data={
            "AAPL": _series("AAPL", [100.0]),
            "MSFT": _series("MSFT", [100.0]),
        },
    )
    res = evaluate_intents(intents, context, RiskGateConfig(), checks=[LeverageCheck()], current_time_ns=1)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert len(res.data.decisions) == 3


@pytest.mark.integration
def test_integration_safe_mode_trigger_changes_decision_type() -> None:
    intents = _intent_set(
        [
            IntentGroup(
                group_id="g-1",
                symbol="AAPL",
                intents=[
                    TradeIntent(intent_id="i-1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1.0, created_at_ns=1)
                ],
                created_at_ns=1,
            )
        ]
    )
    context = RiskCheckContext(
        account_equity=95.0,
        portfolio_snapshot={"day_start_equity": 100.0},
        market_data={"AAPL": _series("AAPL", [100.0])},
        safe_mode_state=SafeModeState.ACTIVE,
    )
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_daily_loss_pct=0.02))

    res = evaluate_intents(intents, context, config, checks=[DailyLossCheck()], current_time_ns=1)
    assert res.status is ResultStatus.DEGRADED
    assert res.data is not None
    assert res.data.decisions[0].decision_type.value == "DOWNGRADE"


@pytest.mark.integration
def test_integration_aggregates_multiple_failing_checks_with_deduped_reasons() -> None:
    intents = _intent_set(
        [
            IntentGroup(
                group_id="g-1",
                symbol="AAPL",
                intents=[
                    TradeIntent(
                        intent_id="i-1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1.0, created_at_ns=1
                    )
                ],
                created_at_ns=1,
            )
        ]
    )
    context = RiskCheckContext(account_equity=100.0, market_data={})
    res = evaluate_intents(intents, context, RiskGateConfig(), checks=[LeverageCheck(), PriceBandCheck()], current_time_ns=1)

    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    decision = res.data.decisions[0]
    assert decision.decision_type.value == "BLOCK"
    assert decision.reason_codes.count("MISSING_MARKET_DATA") == 1
    assert set(decision.details["checks"]) == {"LeverageCheck", "PriceBandCheck"}


@pytest.mark.integration
def test_integration_risk_decision_set_is_journal_compatible() -> None:
    intents = _intent_set(
        [
            IntentGroup(
                group_id="g-1",
                symbol="AAPL",
                intents=[
                    TradeIntent(intent_id="i-1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1.0, created_at_ns=1)
                ],
                created_at_ns=1,
            )
        ]
    )
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", [100.0])})
    res = evaluate_intents(intents, context, RiskGateConfig(), checks=[LeverageCheck()], current_time_ns=123)
    assert res.status is ResultStatus.SUCCESS
    assert isinstance(res.data, RiskDecisionSet)

    payload = msgspec.json.encode(res.data)
    decoded = msgspec.json.decode(payload, type=RiskDecisionSet)
    assert decoded == res.data
