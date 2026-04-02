from __future__ import annotations

import pytest
from typing import Any

import msgspec

from strategy.engine import generate_intents
from strategy.interface import IntentGroup, IntentType, OrderIntentSet, StrategyEngineConfig, TradeIntent
from strategy.pricing import ATRBracketPolicy, LadderEntryPolicy, TrailingStopPolicy
from strategy.sizing import FixedPercentSizer, FixedRiskSizer, VolatilityScaledSizer

from common.interface import ResultStatus
from data.interface import PriceSeriesSnapshot
from event_guard.interface import TradeConstraints
from scanner.interface import CandidateSet, PlatformCandidate

from .conftest import (  # noqa: F401
    default_config,
    sample_candidate,
    sample_candidate_set,
    sample_constraints,
    sample_market_data,
)


def _clone_candidate(candidate: PlatformCandidate, *, symbol: str) -> PlatformCandidate:
    return msgspec.structs.replace(candidate, symbol=symbol)


def _clone_candidate_set(base: CandidateSet, *, candidates: list[PlatformCandidate]) -> CandidateSet:
    return msgspec.structs.replace(
        base,
        candidates=candidates,
        total_scanned=len(candidates),
        total_detected=len(candidates),
    )


def _clone_market_data(snapshot: PriceSeriesSnapshot, *, symbol: str) -> PriceSeriesSnapshot:
    return msgspec.structs.replace(snapshot, symbol=symbol)


def _clone_constraints(
    base: TradeConstraints,
    *,
    symbol: str,
    can_open_new: bool | None = None,
    max_position_size: float | None = None,
) -> TradeConstraints:
    return TradeConstraints(
        symbol=symbol,
        can_open_new=base.can_open_new if can_open_new is None else can_open_new,
        can_increase=base.can_increase,
        can_decrease=base.can_decrease,
        max_position_size=base.max_position_size if max_position_size is None else max_position_size,
        no_trade_windows=base.no_trade_windows,
        reason_codes=base.reason_codes,
    )


def _groups_for_symbol(intent_set: OrderIntentSet, *, symbol: str) -> list[IntentGroup]:
    return [g for g in intent_set.intent_groups if g.symbol == symbol]


def _intent_by_type(group: IntentGroup, intent_type: IntentType) -> TradeIntent:
    matches = [intent for intent in group.intents if intent.intent_type is intent_type]
    assert matches, f"Expected intent_type={intent_type} to exist in group_id={group.group_id}"
    assert len(matches) == 1, f"Expected 1 intent_type={intent_type} in group_id={group.group_id}, got {len(matches)}"
    return matches[0]


def _extract_entry_sl_tp(group: IntentGroup) -> tuple[TradeIntent, TradeIntent, TradeIntent]:
    entry = _intent_by_type(group, IntentType.OPEN_LONG)
    stop = _intent_by_type(group, IntentType.STOP_LOSS)
    take = _intent_by_type(group, IntentType.TAKE_PROFIT)
    return entry, stop, take


def _normalize_intent_set_for_compare(intent_set: OrderIntentSet) -> dict[str, Any]:
    payload: dict[str, Any] = msgspec.to_builtins(intent_set)
    payload["asof_timestamp"] = 0
    for group in payload.get("intent_groups", []):
        group["created_at_ns"] = 0
        for intent in group.get("intents", []):
            intent["created_at_ns"] = 0
    return payload


def _entry_price_with_costs(entry_price: float, config: StrategyEngineConfig) -> float:
    tx_costs = config.transaction_costs
    return float(entry_price) * (1.0 + float(tx_costs.spread_pct) + float(tx_costs.slippage_pct))


def _exit_price_with_costs(exit_price: float, config: StrategyEngineConfig) -> float:
    tx_costs = config.transaction_costs
    return float(exit_price) * (1.0 - float(tx_costs.spread_pct) - float(tx_costs.slippage_pct))


@pytest.mark.integration
def test_integration_single_candidate_happy_path(
    sample_candidate_set: CandidateSet,
    sample_constraints: TradeConstraints,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
) -> None:
    config = msgspec.structs.replace(
        default_config,
        position_sizer="fixed_risk",
        price_policy="atr_bracket",
        atr_multiplier=2.0,
        risk_per_trade_pct=0.01,
        max_position_pct=1.0,
    )
    constraints = _clone_constraints(sample_constraints, symbol="AAPL", max_position_size=10_000.0)
    policy = ATRBracketPolicy()
    sizer = FixedRiskSizer()
    account_equity = 100_000.0

    result = generate_intents(
        sample_candidate_set,
        {"AAPL": constraints},
        {"AAPL": sample_market_data},
        account_equity,
        config,
        sizer,
        policy,
        current_time_ns=1_700_000_000_000_000_000,
    )

    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 1, f"Expected 1 IntentGroup, got {len(result.data.intent_groups)}"

    group = result.data.intent_groups[0]
    entry_intent, sl_intent, tp_intent = _extract_entry_sl_tp(group)

    base_entry = float(sample_market_data.bars[-1].close)
    entry = _entry_price_with_costs(base_entry, config)
    assert entry_intent.entry_price == pytest.approx(entry), "Entry price should include transaction costs"

    atr_pct = float(getattr(sample_market_data, "atr_pct", 0.0) or 0.0)
    assert atr_pct > 0, "Test fixture must provide a positive atr_pct"
    atr_sl_pct = atr_pct * float(config.atr_multiplier)
    final_sl_pct = min(float(atr_sl_pct), float(config.stop_loss_pct))
    expected_sl = _exit_price_with_costs(entry * (1.0 - final_sl_pct), config)
    expected_tp = _exit_price_with_costs(entry * (1.0 + final_sl_pct * float(config.tp_sl_ratio)), config)
    assert sl_intent.stop_loss_price == pytest.approx(expected_sl), "SL should apply transaction costs on exit"
    assert tp_intent.take_profit_price == pytest.approx(expected_tp), "TP should apply transaction costs on exit"

    risk_money = account_equity * float(config.risk_per_trade_pct)
    risk_points = abs(float(entry_intent.entry_price or 0.0) - float(sl_intent.stop_loss_price or 0.0))
    expected_qty = float(int(risk_money / risk_points))
    assert entry_intent.quantity == pytest.approx(expected_qty), "Quantity should follow FixedRiskSizer sizing math"


@pytest.mark.integration
def test_integration_multiple_candidates(
    sample_candidate: PlatformCandidate,
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    sample_constraints: TradeConstraints,
    default_config: StrategyEngineConfig,
) -> None:
    candidates = [
        _clone_candidate(sample_candidate, symbol="AAPL"),
        _clone_candidate(sample_candidate, symbol="MSFT"),
        _clone_candidate(sample_candidate, symbol="GOOGL"),
    ]
    candidate_set = _clone_candidate_set(sample_candidate_set, candidates=candidates)

    market_data = {
        "AAPL": _clone_market_data(sample_market_data, symbol="AAPL"),
        "MSFT": _clone_market_data(sample_market_data, symbol="MSFT"),
        "GOOGL": _clone_market_data(sample_market_data, symbol="GOOGL"),
    }
    constraints = {
        "AAPL": _clone_constraints(sample_constraints, symbol="AAPL"),
        "MSFT": _clone_constraints(sample_constraints, symbol="MSFT"),
        "GOOGL": _clone_constraints(sample_constraints, symbol="GOOGL"),
    }

    result = generate_intents(
        candidate_set,
        constraints,
        market_data,
        100_000.0,
        default_config,
        FixedPercentSizer(),
        ATRBracketPolicy(),
        current_time_ns=1_700_000_000_000_000_000,
    )

    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 3, f"Expected 3 IntentGroups, got {len(result.data.intent_groups)}"

    symbols = {g.symbol for g in result.data.intent_groups}
    assert symbols == {"AAPL", "MSFT", "GOOGL"}, f"Expected 3 symbols, got {symbols}"
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        groups = _groups_for_symbol(result.data, symbol=symbol)
        assert len(groups) == 1, f"Expected exactly 1 IntentGroup for symbol={symbol}, got {len(groups)}"
        entry_intent, sl_intent, tp_intent = _extract_entry_sl_tp(groups[0])
        assert entry_intent.symbol == symbol, f"Entry intent symbol mismatch for {symbol}"
        assert sl_intent.symbol == symbol, f"SL intent symbol mismatch for {symbol}"
        assert tp_intent.symbol == symbol, f"TP intent symbol mismatch for {symbol}"


@pytest.mark.integration
def test_integration_constraints_filtering(
    sample_candidate: PlatformCandidate,
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    sample_constraints: TradeConstraints,
    default_config: StrategyEngineConfig,
) -> None:
    candidates = [
        _clone_candidate(sample_candidate, symbol="AAPL"),
        _clone_candidate(sample_candidate, symbol="MSFT"),
    ]
    candidate_set = _clone_candidate_set(sample_candidate_set, candidates=candidates)
    market_data = {
        "AAPL": _clone_market_data(sample_market_data, symbol="AAPL"),
        "MSFT": _clone_market_data(sample_market_data, symbol="MSFT"),
    }
    constraints = {
        "AAPL": _clone_constraints(sample_constraints, symbol="AAPL", can_open_new=True),
        "MSFT": _clone_constraints(sample_constraints, symbol="MSFT", can_open_new=False),
    }

    result = generate_intents(
        candidate_set,
        constraints,
        market_data,
        100_000.0,
        default_config,
        FixedPercentSizer(),
        ATRBracketPolicy(),
        current_time_ns=1_700_000_000_000_000_000,
    )

    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 1, f"Expected 1 IntentGroup, got {len(result.data.intent_groups)}"
    assert result.data.intent_groups[0].symbol == "AAPL", "Expected MSFT to be filtered by can_open_new=False"


@pytest.mark.integration
def test_integration_ladder_entry(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    sample_constraints: TradeConstraints,
    default_config: StrategyEngineConfig,
) -> None:
    config = msgspec.structs.replace(
        default_config,
        entry_strategy="ladder",
        ladder_levels=3,
        ladder_spacing_pct=0.01,
    )

    result = generate_intents(
        sample_candidate_set,
        {"AAPL": sample_constraints},
        {"AAPL": sample_market_data},
        100_000.0,
        config,
        FixedPercentSizer(),
        LadderEntryPolicy(),
        current_time_ns=1_700_000_000_000_000_000,
    )

    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"

    groups = _groups_for_symbol(result.data, symbol="AAPL")
    assert len(groups) == 3, f"Expected 3 IntentGroups for ladder entry, got {len(groups)}"

    base = float(sample_market_data.bars[-1].close)
    spacing = float(config.ladder_spacing_pct)

    entries: list[TradeIntent] = []
    for group in groups:
        entry_intent, _, _ = _extract_entry_sl_tp(group)
        entries.append(entry_intent)

    entries_by_level = sorted(entries, key=lambda i: int(i.ladder_level or 0))
    assert [e.ladder_level for e in entries_by_level] == [0, 1, 2], "Expected ladder_level to be [0,1,2]"

    expected_prices = [base * (1.0 - spacing * float(level)) for level in [0, 1, 2]]
    actual_prices = [float(e.entry_price or 0.0) for e in entries_by_level]
    assert actual_prices == pytest.approx(expected_prices), "Expected ladder entry prices to decrease by spacing_pct"

    quantities = [float(e.quantity) for e in entries_by_level]
    assert quantities[0] == pytest.approx(quantities[1]) and quantities[1] == pytest.approx(
        quantities[2]
    ), "Expected equal quantity per ladder level (total split into 3 legs)"

    total_entry_qty = sum(quantities)
    expected_total_qty = float(int((100_000.0 * float(config.position_size_pct)) / float(base)))
    assert total_entry_qty == pytest.approx(
        expected_total_qty
    ), "Expected total ladder quantity to equal the FixedPercentSizer total budget"


@pytest.mark.integration
def test_integration_missing_market_data_degraded(
    sample_candidate_set: CandidateSet,
    default_config: StrategyEngineConfig,
) -> None:
    result = generate_intents(
        sample_candidate_set,
        {},
        {},
        100_000.0,
        default_config,
        FixedPercentSizer(),
        ATRBracketPolicy(),
        current_time_ns=1_700_000_000_000_000_000,
    )

    assert result.status is ResultStatus.DEGRADED, f"Expected DEGRADED, got {result.status}"
    assert result.reason_code == "MISSING_MARKET_DATA", f"Expected MISSING_MARKET_DATA, got {result.reason_code}"
    assert result.data is not None, "Expected OrderIntentSet payload on DEGRADED"
    assert len(result.data.intent_groups) == 1, f"Expected 1 protective IntentGroup, got {len(result.data.intent_groups)}"

    group = result.data.intent_groups[0]
    intent_types = {intent.intent_type for intent in group.intents}
    assert IntentType.REDUCE_POSITION in intent_types, "Expected protective REDUCE_POSITION intent in degraded mode"
    assert all(intent.reduce_only for intent in group.intents), "Expected protective intents to be reduce_only"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("sizer_name", "sizer", "config_overrides"),
    [
        ("fixed_percent", FixedPercentSizer(), {"position_size_pct": 0.02, "max_position_pct": 1.0}),
        ("fixed_risk", FixedRiskSizer(), {"risk_per_trade_pct": 0.01, "max_position_pct": 1.0}),
        ("volatility_scaled", VolatilityScaledSizer(), {"position_size_pct": 0.03, "max_position_pct": 1.0}),
    ],
)
def test_integration_different_position_sizers(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    sample_constraints: TradeConstraints,
    default_config: StrategyEngineConfig,
    sizer_name: str,
    sizer: Any,
    config_overrides: dict[str, Any],
) -> None:
    config = msgspec.structs.replace(
        default_config,
        position_sizer=sizer_name,
        price_policy="atr_bracket",
        atr_multiplier=2.0,
        **config_overrides,
    )

    result = generate_intents(
        sample_candidate_set,
        {"AAPL": _clone_constraints(sample_constraints, symbol="AAPL", max_position_size=10_000.0)},
        {"AAPL": sample_market_data},
        100_000.0,
        config,
        sizer,
        ATRBracketPolicy(),
        current_time_ns=1_700_000_000_000_000_000,
    )

    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 1, f"Expected 1 IntentGroup, got {len(result.data.intent_groups)}"

    entry_intent, sl_intent, _ = _extract_entry_sl_tp(result.data.intent_groups[0])
    assert entry_intent.quantity > 0, f"Expected positive quantity for sizer={sizer_name}"
    entry = float(entry_intent.entry_price or 0.0)
    equity = 100_000.0
    if sizer_name == "fixed_percent":
        expected = float(int((equity * float(config.position_size_pct)) / entry))
    elif sizer_name == "volatility_scaled":
        atr_pct = float(getattr(sample_market_data, "atr_pct", 0.0) or 0.0)
        if atr_pct > 0:
            risk_distance = entry * atr_pct * float(config.atr_multiplier)
            risk_money = equity * float(config.risk_per_trade_pct)
            expected = float(int(risk_money / risk_distance))
        else:
            expected = float(int((equity * float(config.position_size_pct)) / entry))
    else:
        risk_points = abs(entry - float(sl_intent.stop_loss_price or 0.0))
        risk_money = equity * float(config.risk_per_trade_pct)
        expected = float(int(risk_money / risk_points))
    assert float(entry_intent.quantity) == pytest.approx(expected), f"Unexpected quantity for sizer={sizer_name}"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("policy_name", "policy", "config_overrides"),
    [
        ("atr_bracket", ATRBracketPolicy(), {"atr_multiplier": 2.0}),
        ("trailing_stop", TrailingStopPolicy(), {"trailing_offset_pct": 0.05}),
        ("ladder_entry", LadderEntryPolicy(), {"atr_multiplier": 3.0}),
    ],
)
def test_integration_different_price_policies(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    sample_constraints: TradeConstraints,
    default_config: StrategyEngineConfig,
    policy_name: str,
    policy: Any,
    config_overrides: dict[str, Any],
) -> None:
    config = msgspec.structs.replace(
        default_config,
        price_policy=policy_name,
        **config_overrides,
    )

    result = generate_intents(
        sample_candidate_set,
        {"AAPL": _clone_constraints(sample_constraints, symbol="AAPL", max_position_size=10_000.0)},
        {"AAPL": sample_market_data},
        100_000.0,
        config,
        FixedPercentSizer(),
        policy,
        current_time_ns=1_700_000_000_000_000_000,
    )

    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 1, f"Expected 1 IntentGroup, got {len(result.data.intent_groups)}"

    entry_intent, sl_intent, tp_intent = _extract_entry_sl_tp(result.data.intent_groups[0])
    sl = float(sl_intent.stop_loss_price or 0.0)
    tp = float(tp_intent.take_profit_price or 0.0)
    assert sl > 0 and tp > 0, f"Expected positive SL/TP for policy={policy_name}"
    entry = float(entry_intent.entry_price or 0.0)
    if policy_name == "trailing_stop":
        expected_sl = entry * (1.0 - float(config.trailing_offset_pct))
        expected_tp = entry * (1.0 + 2.0 * float(config.trailing_offset_pct))
    else:
        atr_pct = float(getattr(sample_market_data, "atr_pct", 0.0) or 0.0)
        atr_sl_pct = atr_pct * float(config.atr_multiplier)
        final_sl_pct = min(float(atr_sl_pct), float(config.stop_loss_pct))
        expected_sl = _exit_price_with_costs(entry * (1.0 - final_sl_pct), config)
        expected_tp = _exit_price_with_costs(entry * (1.0 + final_sl_pct * float(config.tp_sl_ratio)), config)
    assert sl == pytest.approx(expected_sl), f"Unexpected stop loss for policy={policy_name}"
    assert tp == pytest.approx(expected_tp), f"Unexpected take profit for policy={policy_name}"


@pytest.mark.integration
def test_integration_replay_determinism(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    sample_constraints: TradeConstraints,
    default_config: StrategyEngineConfig,
) -> None:
    config = msgspec.structs.replace(
        default_config,
        position_sizer="fixed_percent",
        price_policy="atr_bracket",
        atr_multiplier=2.0,
    )
    constraints = {"AAPL": _clone_constraints(sample_constraints, symbol="AAPL", max_position_size=10_000.0)}
    market_data = {"AAPL": sample_market_data}

    first = generate_intents(
        sample_candidate_set,
        constraints,
        market_data,
        100_000.0,
        config,
        FixedPercentSizer(),
        ATRBracketPolicy(),
        current_time_ns=1_700_000_000_000_000_000,
    )
    second = generate_intents(
        sample_candidate_set,
        constraints,
        market_data,
        100_000.0,
        config,
        FixedPercentSizer(),
        ATRBracketPolicy(),
        current_time_ns=1_700_000_000_000_000_123,
    )

    assert first.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {first.status}"
    assert second.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {second.status}"
    assert first.data is not None and second.data is not None, "Expected OrderIntentSet payloads on SUCCESS"

    first_intents = sorted(
        [intent for g in first.data.intent_groups for intent in g.intents],
        key=lambda i: i.intent_id,
    )
    second_intents = sorted(
        [intent for g in second.data.intent_groups for intent in g.intents],
        key=lambda i: i.intent_id,
    )
    assert [i.intent_id for i in first_intents] == [
        i.intent_id for i in second_intents
    ], "Expected deterministic intent_ids across replays"

    def _prices_qty(intents: list[TradeIntent]) -> list[tuple[float, float, float, float]]:
        rows: list[tuple[float, float, float, float]] = []
        for intent in intents:
            rows.append(
                (
                    float(intent.quantity),
                    float(intent.entry_price or 0.0),
                    float(intent.stop_loss_price or 0.0),
                    float(intent.take_profit_price or 0.0),
                )
            )
        return rows

    assert _prices_qty(first_intents) == _prices_qty(second_intents), "Expected identical prices/quantities across replays"

    first_norm = _normalize_intent_set_for_compare(first.data)
    second_norm = _normalize_intent_set_for_compare(second.data)
    assert first_norm == second_norm, "Expected identical OrderIntentSet output after normalizing timestamps"
