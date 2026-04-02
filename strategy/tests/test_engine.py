from __future__ import annotations

import re
from typing import Any

import msgspec
import pytest

from indicators.interface import MACDResult
from strategy import engine as engine_module
from strategy.engine import generate_intents
from strategy.interface import IndicatorsConfig, IntentType, OrderIntentSet, StrategyEngineConfig, TradeIntent

from common.interface import Result, ResultStatus
from data.interface import PriceSeriesSnapshot
from event_guard.interface import TradeConstraints
from scanner.interface import CandidateSet, PlatformCandidate


class MockPositionSizer:
    def __init__(
        self,
        *,
        result: Result[float] | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._result = result or Result.success(10.0)
        self._raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        config: StrategyEngineConfig,
        constraints: TradeConstraints | None,
        market_data: PriceSeriesSnapshot | None = None,
    ) -> Result[float]:
        self.calls.append(
            {
                "symbol": symbol,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "account_equity": account_equity,
                "config": config,
                "constraints": constraints,
                "market_data": market_data,
            }
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._result


class MockPricePolicy:
    def __init__(
        self,
        *,
        entry_result: Result[float] | None = None,
        stop_result: Result[float] | None = None,
        take_result: Result[float] | None = None,
        raise_on: str | None = None,
    ) -> None:
        self._entry_result = entry_result or Result.success(100.0)
        self._stop_result = stop_result or Result.success(95.0)
        self._take_result = take_result or Result.success(105.0)
        self._raise_on = raise_on
        self.calls: list[dict[str, Any]] = []

    def calculate_entry_price(
        self,
        symbol: str,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        self.calls.append(
            {
                "method": "entry",
                "symbol": symbol,
                "side": side,
                "config": config,
                "market_data": market_data,
            }
        )
        if self._raise_on == "entry":
            raise RuntimeError("boom-entry")
        return self._entry_result

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        self.calls.append(
            {
                "method": "stop",
                "symbol": symbol,
                "entry_price": entry_price,
                "side": side,
                "config": config,
                "market_data": market_data,
            }
        )
        if self._raise_on == "stop":
            raise RuntimeError("boom-stop")
        return self._stop_result

    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        self.calls.append(
            {
                "method": "take",
                "symbol": symbol,
                "entry_price": entry_price,
                "side": side,
                "config": config,
                "market_data": market_data,
            }
        )
        if self._raise_on == "take":
            raise RuntimeError("boom-take")
        return self._take_result


@pytest.fixture()
def mock_position_sizer() -> MockPositionSizer:
    return MockPositionSizer(result=Result.success(10.0))


@pytest.fixture()
def mock_price_policy() -> MockPricePolicy:
    return MockPricePolicy(
        entry_result=Result.success(100.0),
        stop_result=Result.success(95.0),
        take_result=Result.success(105.0),
    )


def _clone_market_data(snapshot: PriceSeriesSnapshot, *, symbol: str, bars: list[Any] | None = None) -> PriceSeriesSnapshot:
    payload = msgspec.to_builtins(snapshot)
    payload["symbol"] = symbol
    if bars is not None:
        payload["bars"] = bars
    return snapshot.__class__(**payload)  # type: ignore[call-arg]


def _clone_candidate(candidate: PlatformCandidate, *, symbol: str, score: float | None = None) -> PlatformCandidate:
    updated = msgspec.structs.replace(candidate, symbol=symbol)
    if score is not None:
        updated = msgspec.structs.replace(updated, score=score)
    return updated


def _make_candidate_set(base: CandidateSet, *, candidates: list[PlatformCandidate]) -> CandidateSet:
    return msgspec.structs.replace(
        base,
        candidates=candidates,
        total_scanned=len(candidates),
        total_detected=len(candidates),
    )


def _intents_by_type(intent_set: OrderIntentSet, *, symbol: str) -> dict[IntentType, TradeIntent]:
    groups = [g for g in intent_set.intent_groups if g.symbol == symbol]
    assert groups, f"Expected at least 1 IntentGroup for symbol={symbol}, got 0"
    assert len(groups) == 1, f"Expected exactly 1 IntentGroup for symbol={symbol}, got {len(groups)}"
    intents = groups[0].intents
    mapping: dict[IntentType, TradeIntent] = {intent.intent_type: intent for intent in intents}
    assert len(mapping) == len(intents), "Expected intent types to be unique within a group"
    return mapping


def _extract_entry_sl_tp(intent_set: OrderIntentSet, *, symbol: str) -> tuple[TradeIntent, TradeIntent, TradeIntent]:
    mapping = _intents_by_type(intent_set, symbol=symbol)
    assert IntentType.OPEN_LONG in mapping, "Expected OPEN_LONG intent to exist"
    assert IntentType.STOP_LOSS in mapping, "Expected STOP_LOSS intent to exist"
    assert IntentType.TAKE_PROFIT in mapping, "Expected TAKE_PROFIT intent to exist"
    return mapping[IntentType.OPEN_LONG], mapping[IntentType.STOP_LOSS], mapping[IntentType.TAKE_PROFIT]


def test_generate_intents_single_candidate_success(
    sample_candidate_set: CandidateSet,
    sample_constraints: TradeConstraints,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    result = generate_intents(
        sample_candidate_set,
        {"AAPL": sample_constraints},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 1, f"Expected 1 IntentGroup, got {len(result.data.intent_groups)}"
    assert result.data.intent_groups[0].symbol == "AAPL", "Expected IntentGroup.symbol to match candidate symbol"

    intent_types = [intent.intent_type for intent in result.data.intent_groups[0].intents]
    assert intent_types == [
        IntentType.OPEN_LONG,
        IntentType.STOP_LOSS,
        IntentType.TAKE_PROFIT,
    ], f"Expected [OPEN_LONG, STOP_LOSS, TAKE_PROFIT], got {intent_types}"


def test_generate_intents_multiple_candidates_success(
    sample_candidate: PlatformCandidate,
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    candidates = [
        _clone_candidate(sample_candidate, symbol="AAPL"),
        _clone_candidate(sample_candidate, symbol="MSFT"),
        _clone_candidate(sample_candidate, symbol="TSLA"),
    ]
    candidate_set = _make_candidate_set(sample_candidate_set, candidates=candidates)
    market_data = {
        "AAPL": _clone_market_data(sample_market_data, symbol="AAPL"),
        "MSFT": _clone_market_data(sample_market_data, symbol="MSFT"),
        "TSLA": _clone_market_data(sample_market_data, symbol="TSLA"),
    }

    result = generate_intents(
        candidate_set,
        {},
        market_data,
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    symbols = {g.symbol for g in result.data.intent_groups}
    assert symbols == {"AAPL", "MSFT", "TSLA"}, f"Expected 3 symbols, got {symbols}"
    assert len(result.data.intent_groups) == 3, f"Expected 3 IntentGroups, got {len(result.data.intent_groups)}"


def test_generate_intents_intent_id_deterministic(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    inputs = dict(
        candidates=sample_candidate_set,
        constraints={},
        market_data={"AAPL": sample_market_data},
        account_equity=100_000.0,
        config=default_config,
        position_sizer=mock_position_sizer,
        price_policy=mock_price_policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    first = generate_intents(**inputs)
    second = generate_intents(**inputs)

    assert first.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {first.status}"
    assert second.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {second.status}"
    assert first.data is not None and second.data is not None, "Expected intent-set payloads for both runs"

    first_ids = [intent.intent_id for intent in first.data.intent_groups[0].intents]
    second_ids = [intent.intent_id for intent in second.data.intent_groups[0].intents]
    assert first_ids == second_ids, f"Expected deterministic intent_ids, got {first_ids} vs {second_ids}"


def test_generate_intents_intent_group_linkage(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"

    entry, stop_loss, take_profit = _extract_entry_sl_tp(result.data, symbol="AAPL")
    assert entry.linked_intent_ids == [
        stop_loss.intent_id,
        take_profit.intent_id,
    ], "Expected Entry.linked_intent_ids to reference [sl_id, tp_id]"
    assert stop_loss.parent_intent_id == entry.intent_id, "Expected SL.parent_intent_id == entry_id"
    assert take_profit.parent_intent_id == entry.intent_id, "Expected TP.parent_intent_id == entry_id"
    assert stop_loss.reduce_only is True, "Expected SL.reduce_only == True"
    assert take_profit.reduce_only is True, "Expected TP.reduce_only == True"


def test_generate_intents_quantity_from_position_sizer(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_price_policy: MockPricePolicy,
) -> None:
    sizer = MockPositionSizer(result=Result.success(42.0))
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        sizer,
        mock_price_policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    assert len(sizer.calls) == 1, f"Expected 1 sizing call, got {len(sizer.calls)}"
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    entry, stop_loss, take_profit = _extract_entry_sl_tp(result.data, symbol="AAPL")
    assert entry.quantity == pytest.approx(42.0), f"Expected entry.quantity==42.0, got {entry.quantity}"
    assert stop_loss.quantity == pytest.approx(42.0), f"Expected sl.quantity==42.0, got {stop_loss.quantity}"
    assert take_profit.quantity == pytest.approx(42.0), f"Expected tp.quantity==42.0, got {take_profit.quantity}"


def test_generate_intents_prices_from_price_policy(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
) -> None:
    policy = MockPricePolicy(
        entry_result=Result.success(123.0),
        stop_result=Result.success(120.0),
        take_result=Result.success(130.0),
    )
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    entry, stop_loss, take_profit = _extract_entry_sl_tp(result.data, symbol="AAPL")
    assert entry.entry_price == pytest.approx(123.0), f"Expected entry_price==123.0, got {entry.entry_price}"
    assert stop_loss.stop_loss_price == pytest.approx(120.0), f"Expected stop_loss==120.0, got {stop_loss.stop_loss_price}"
    assert (
        take_profit.take_profit_price == pytest.approx(130.0)
    ), f"Expected take_profit==130.0, got {take_profit.take_profit_price}"

    methods = [call["method"] for call in policy.calls]
    assert methods.count("entry") == 1, f"Expected 1 entry price call, got {methods.count('entry')}"
    assert methods.count("stop") == 2, f"Expected 2 stop-loss calls (base+bracket), got {methods.count('stop')}"
    assert methods.count("take") == 1, f"Expected 1 take-profit call, got {methods.count('take')}"


def test_generate_intents_constraints_recorded(
    sample_candidate_set: CandidateSet,
    sample_constraints: TradeConstraints,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    result = generate_intents(
        sample_candidate_set,
        {"AAPL": sample_constraints},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert "AAPL" in result.data.constraints_applied, "Expected constraints_applied to include key=symbol"
    assert result.data.constraints_applied["AAPL"] == msgspec.to_builtins(
        sample_constraints
    ), "Expected constraints_applied[symbol] == msgspec.to_builtins(constraints)"


def test_generate_intents_empty_candidates_success(
    sample_candidate_set: CandidateSet,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    empty_set = _make_candidate_set(sample_candidate_set, candidates=[])
    result = generate_intents(
        empty_set,
        {},
        {},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected empty intent_groups for empty candidates"
    assert result.data.source_candidates == [], "Expected empty source_candidates for empty candidates"


def test_generate_intents_missing_market_data_degraded(
    sample_candidate: PlatformCandidate,
    sample_candidate_set: CandidateSet,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    candidate_set = _make_candidate_set(sample_candidate_set, candidates=[_clone_candidate(sample_candidate, symbol="MSFT")])
    result = generate_intents(
        candidate_set,
        {},
        {},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    assert result.status is ResultStatus.DEGRADED, f"Expected DEGRADED, got {result.status}"
    assert result.reason_code == "MISSING_MARKET_DATA", f"Expected MISSING_MARKET_DATA, got {result.reason_code}"
    assert isinstance(result.error, RuntimeError), "Expected a RuntimeError explaining missing market data"
    assert result.data is not None, "Expected protective OrderIntentSet payload on DEGRADED"
    assert len(result.data.intent_groups) == 1, f"Expected 1 protective group, got {len(result.data.intent_groups)}"


def test_generate_intents_degraded_on_empty_bars(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    empty_bars_snapshot = _clone_market_data(sample_market_data, symbol="AAPL", bars=[])
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": empty_bars_snapshot},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    assert result.status is ResultStatus.DEGRADED, f"Expected DEGRADED, got {result.status}"
    assert result.reason_code == "MISSING_MARKET_DATA", f"Expected MISSING_MARKET_DATA, got {result.reason_code}"
    assert result.data is not None, "Expected protective OrderIntentSet payload on DEGRADED"
    assert result.data.source_candidates == ["AAPL"], f"Expected protective source_candidates ['AAPL'], got {result.data.source_candidates}"


def test_generate_intents_protective_intents_structure(
    sample_candidate: PlatformCandidate,
    sample_candidate_set: CandidateSet,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    candidate_set = _make_candidate_set(sample_candidate_set, candidates=[_clone_candidate(sample_candidate, symbol="MSFT")])
    result = generate_intents(
        candidate_set,
        {},
        {},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=1_700_000_000_000_000_000,
    )
    assert result.status is ResultStatus.DEGRADED, f"Expected DEGRADED, got {result.status}"
    assert result.data is not None, "Expected protective OrderIntentSet payload on DEGRADED"
    group = result.data.intent_groups[0]
    intent_types = {intent.intent_type for intent in group.intents}
    assert IntentType.REDUCE_POSITION in intent_types, f"Expected REDUCE_POSITION in protective intents, got {intent_types}"
    reduce_intent = next(intent for intent in group.intents if intent.intent_type is IntentType.REDUCE_POSITION)
    assert reduce_intent.reduce_only is True, "Expected protective REDUCE_POSITION intent to set reduce_only=True"
    assert "DEGRADED_PROTECTIVE_MODE" in (
        reduce_intent.reason_codes or []
    ), "Expected protective intent to include reason code DEGRADED_PROTECTIVE_MODE"


def test_generate_intents_constraints_can_open_new_false(
    sample_candidate_set: CandidateSet,
    sample_constraints: TradeConstraints,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    blocked = msgspec.structs.replace(sample_constraints, can_open_new=False)
    result = generate_intents(
        sample_candidate_set,
        {"AAPL": blocked},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected candidate to be skipped when can_open_new=False"
    assert "AAPL" in result.data.constraints_applied, "Expected constraints_applied to be recorded even when skipping"


def test_generate_intents_position_sizer_failed_skips_candidate(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_price_policy: MockPricePolicy,
) -> None:
    sizer = MockPositionSizer(result=Result.failed(RuntimeError("no size"), "SIZER_FAILED"))
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS with empty intents, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected no intents when position_sizer returns FAILED"


def test_generate_intents_position_sizer_exception_returns_failed(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_price_policy: MockPricePolicy,
) -> None:
    sizer = MockPositionSizer(raise_exc=RuntimeError("boom-sizer"))
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.FAILED, f"Expected FAILED, got {result.status}"
    assert result.reason_code == "POSITION_SIZER_EXCEPTION", f"Expected POSITION_SIZER_EXCEPTION, got {result.reason_code}"
    assert isinstance(result.error, RuntimeError), "Expected RuntimeError captured in Result.error"


def test_generate_intents_price_policy_entry_failed_skips_candidate(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
) -> None:
    policy = MockPricePolicy(entry_result=Result.failed(RuntimeError("no entry"), "NO_ENTRY"))
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS with empty intents, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected candidate skipped when entry pricing returns FAILED"


def test_generate_intents_price_policy_stop_loss_failed_skips_candidate(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
) -> None:
    policy = MockPricePolicy(stop_result=Result.failed(RuntimeError("no stop"), "NO_STOP"))
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS with empty intents, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected candidate skipped when stop-loss pricing returns FAILED"


def test_generate_intents_price_policy_take_profit_failed_skips_candidate(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
) -> None:
    policy = MockPricePolicy(take_result=Result.failed(RuntimeError("no take"), "NO_TAKE"))
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS with empty intents, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected candidate skipped when take-profit pricing returns FAILED"


def test_generate_intents_quantity_zero_after_constraints_skips_candidate(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    sample_constraints: TradeConstraints,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    capped = msgspec.structs.replace(sample_constraints, max_position_size=0.0)
    result = generate_intents(
        sample_candidate_set,
        {"AAPL": capped},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS with empty intents, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected candidate skipped when constraints cap quantity to 0"


def test_generate_intents_ladder_entry_single_level(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    config = StrategyEngineConfig(entry_strategy="single")
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 1, f"Expected 1 IntentGroup for single entry, got {len(result.data.intent_groups)}"


def test_generate_intents_ladder_entry_multiple_levels(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
) -> None:
    config = StrategyEngineConfig(entry_strategy="ladder", ladder_levels=3, ladder_spacing_pct=0.02)
    sizer = MockPositionSizer(result=Result.success(30.0))
    policy = MockPricePolicy(
        entry_result=Result.success(100.0),
        stop_result=Result.success(95.0),
        take_result=Result.success(105.0),
    )
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        config,
        sizer,
        policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 3, f"Expected 3 IntentGroups for ladder_levels=3, got {len(result.data.intent_groups)}"

    entry_prices = [next(i for i in g.intents if i.intent_type is IntentType.OPEN_LONG).entry_price for g in result.data.intent_groups]
    assert entry_prices == sorted(entry_prices, reverse=True), f"Expected ladder prices descending, got {entry_prices}"
    assert len(set(entry_prices)) == 3, f"Expected unique entry prices per ladder level, got {entry_prices}"

    entry_ids = [next(i for i in g.intents if i.intent_type is IntentType.OPEN_LONG).intent_id for g in result.data.intent_groups]
    assert len(set(entry_ids)) == 3, f"Expected unique entry intent_id per ladder level, got {entry_ids}"

    quantities = [next(i for i in g.intents if i.intent_type is IntentType.OPEN_LONG).quantity for g in result.data.intent_groups]
    assert quantities == pytest.approx([10.0, 10.0, 10.0]), f"Expected per-leg qty 10.0, got {quantities}"


def test_generate_intents_ladder_entry_prices_spacing(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
) -> None:
    config = StrategyEngineConfig(entry_strategy="ladder", ladder_levels=3, ladder_spacing_pct=0.02)
    sizer = MockPositionSizer(result=Result.success(30.0))
    policy = MockPricePolicy(entry_result=Result.success(100.0))
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        config,
        sizer,
        policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    entry_prices = [next(i for i in g.intents if i.intent_type is IntentType.OPEN_LONG).entry_price for g in result.data.intent_groups]
    assert entry_prices == pytest.approx([100.0, 98.0, 96.0]), f"Expected ladder prices [100,98,96], got {entry_prices}"


def test_generate_intents_indicator_checks_disabled_does_not_filter(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Indicator compute function should not be called when indicators.enabled=False")

    monkeypatch.setattr(engine_module, "compute_rsi_last", _boom)
    monkeypatch.setattr(engine_module, "compute_macd_last", _boom)

    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 1, f"Expected candidate to pass, got {len(result.data.intent_groups)} intent groups"
    assert result.data.degradation_events == [], "Expected no degradation_events when indicators are disabled"


def test_generate_intents_indicator_rsi_overbought_blocks_candidate(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StrategyEngineConfig(
        indicators=IndicatorsConfig(
            enabled=True,
            rsi_enabled=True,
            macd_enabled=False,
            rsi_overbought=70.0,
        )
    )
    monkeypatch.setattr(engine_module, "compute_rsi_last", lambda *_args, **_kwargs: 80.0)

    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected candidate blocked by RSI overbought"
    assert len(result.data.degradation_events) == 1, f"Expected 1 degradation event, got {result.data.degradation_events}"
    assert result.data.degradation_events[0]["event_type"] == "INDICATOR_FILTER_BLOCK"
    assert result.data.degradation_events[0]["symbol"] == "AAPL"
    assert result.data.degradation_events[0]["reason_codes"] == ["RSI_OVERBOUGHT"]


def test_generate_intents_indicator_macd_not_bullish_blocks_candidate(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StrategyEngineConfig(
        indicators=IndicatorsConfig(
            enabled=True,
            rsi_enabled=False,
            macd_enabled=True,
            macd_require_bullish=True,
        )
    )
    monkeypatch.setattr(
        engine_module,
        "compute_macd_last",
        lambda *_args, **_kwargs: MACDResult(macd=0.0, signal=0.0, histogram=-0.01),
    )

    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected candidate blocked by MACD not bullish"
    assert len(result.data.degradation_events) == 1, f"Expected 1 degradation event, got {result.data.degradation_events}"
    assert result.data.degradation_events[0]["event_type"] == "INDICATOR_FILTER_BLOCK"
    assert result.data.degradation_events[0]["symbol"] == "AAPL"
    assert result.data.degradation_events[0]["reason_codes"] == ["MACD_NOT_BULLISH"]


def test_generate_intents_indicator_data_insufficient_graceful_degradation(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = StrategyEngineConfig(
        indicators=IndicatorsConfig(
            enabled=True,
            rsi_enabled=True,
            macd_enabled=True,
        )
    )
    monkeypatch.setattr(engine_module, "compute_rsi_last", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine_module, "compute_macd_last", lambda *_args, **_kwargs: None)

    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 1, f"Expected candidate to pass, got {len(result.data.intent_groups)} intent groups"
    assert result.data.degradation_events == [], "Expected no degradation events when indicator data is insufficient"


def test__generate_intent_id_format() -> None:
    bar_ts_ns = 1_700_000_000_000_000_000
    intent_id = engine_module._generate_intent_id("STRAT", "AAPL", bar_ts_ns, IntentType.OPEN_LONG)
    assert intent_id.startswith("O-"), f"Expected id to start with 'O-', got {intent_id}"
    assert "-STRAT-AAPL-OPEN_LONG-" in intent_id, f"Expected id to include strategy/symbol/type, got {intent_id}"
    assert re.match(r"^O-\d{8}T\d{6}-STRAT-AAPL-OPEN_LONG-[0-9a-f]{8}$", intent_id), (
        f"Expected id format O-<datetime>-<strategy>-<symbol>-<type>-<hash>, got {intent_id}"
    )


def test__generate_intent_id_deterministic() -> None:
    bar_ts_ns = 1_700_000_000_000_000_000
    first = engine_module._generate_intent_id("STRAT", "AAPL", bar_ts_ns, IntentType.OPEN_LONG, ladder_level=2)
    second = engine_module._generate_intent_id("STRAT", "AAPL", bar_ts_ns, IntentType.OPEN_LONG, ladder_level=2)
    assert first == second, f"Expected deterministic ids, got {first} vs {second}"


def test__build_intent_group_linkage() -> None:
    entry = TradeIntent(
        intent_id="E-1",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=1.0,
        created_at_ns=1,
        entry_price=100.0,
    )
    stop_loss = TradeIntent(
        intent_id="SL-1",
        symbol="AAPL",
        intent_type=IntentType.STOP_LOSS,
        quantity=1.0,
        created_at_ns=1,
        stop_loss_price=95.0,
    )
    take_profit = TradeIntent(
        intent_id="TP-1",
        symbol="AAPL",
        intent_type=IntentType.TAKE_PROFIT,
        quantity=1.0,
        created_at_ns=1,
        take_profit_price=105.0,
    )
    group = engine_module._build_intent_group("AAPL", entry, stop_loss, take_profit, created_at_ns=123)
    assert group.group_id == "G-E-1", f"Expected group_id to prefix entry id, got {group.group_id}"
    assert group.contingency_type == "OUO", f"Expected group contingency_type OUO, got {group.contingency_type}"

    entry_out = group.intents[0]
    stop_out = group.intents[1]
    take_out = group.intents[2]
    assert entry_out.linked_intent_ids == ["SL-1", "TP-1"], "Expected entry intent to link SL/TP ids"
    assert entry_out.contingency_type == "OTO", f"Expected entry contingency_type OTO, got {entry_out.contingency_type}"
    assert stop_out.parent_intent_id == "E-1", "Expected stop-loss to reference entry via parent_intent_id"
    assert take_out.parent_intent_id == "E-1", "Expected take-profit to reference entry via parent_intent_id"
    assert stop_out.reduce_only is True, "Expected stop-loss reduce_only=True"
    assert take_out.reduce_only is True, "Expected take-profit reduce_only=True"


def test__apply_constraints_max_position_size(sample_constraints: TradeConstraints) -> None:
    constraints = msgspec.structs.replace(sample_constraints, max_position_size=10.0)
    qty, reasons = engine_module._apply_constraints(25.0, constraints, IntentType.OPEN_LONG)
    assert qty == pytest.approx(10.0), f"Expected quantity capped to 10.0, got {qty}"
    assert "CONSTRAINT_MAX_POSITION_SIZE" in reasons, f"Expected reason code for cap, got {reasons}"


def test__apply_constraints_can_open_new_false(sample_constraints: TradeConstraints) -> None:
    constraints = msgspec.structs.replace(sample_constraints, can_open_new=False)
    qty, reasons = engine_module._apply_constraints(25.0, constraints, IntentType.OPEN_LONG)
    assert qty == pytest.approx(0.0), f"Expected quantity=0.0 when can_open_new=False, got {qty}"
    assert "CONSTRAINT_NO_NEW_POSITIONS" in reasons, f"Expected CONSTRAINT_NO_NEW_POSITIONS, got {reasons}"


def test__apply_constraints_reason_codes_for_decrease(sample_constraints: TradeConstraints) -> None:
    constraints = msgspec.structs.replace(sample_constraints, can_decrease=False)
    qty, reasons = engine_module._apply_constraints(25.0, constraints, IntentType.REDUCE_POSITION)
    assert qty == pytest.approx(0.0), f"Expected quantity=0.0 when can_decrease=False, got {qty}"
    assert "CONSTRAINT_NO_DECREASE" in reasons, f"Expected CONSTRAINT_NO_DECREASE, got {reasons}"


def test__generate_protective_intents_structure() -> None:
    protective = engine_module._generate_protective_intents(
        [" aapl ", "AAPL", "msft"],
        "MISSING_MARKET_DATA",
        1_700_000_000_000_000_000,
        system_version="deadbeef",
    )
    assert protective.system_version == "deadbeef", "Expected protective system_version propagated"
    assert protective.source_candidates == ["AAPL", "MSFT"], f"Expected normalized, deduped symbols, got {protective.source_candidates}"
    assert len(protective.intent_groups) == 2, f"Expected 2 protective groups, got {len(protective.intent_groups)}"

    group = next(g for g in protective.intent_groups if g.symbol == "AAPL")
    types = {intent.intent_type for intent in group.intents}
    assert types == {
        IntentType.CANCEL_PENDING,
        IntentType.REDUCE_POSITION,
        IntentType.STOP_LOSS,
    }, f"Expected 3 protective intent types, got {types}"
    assert all(intent.reduce_only is True for intent in group.intents), "Expected all protective intents reduce_only=True"
    assert all(
        "DEGRADED_PROTECTIVE_MODE" in (intent.reason_codes or []) for intent in group.intents
    ), "Expected all protective intents to include DEGRADED_PROTECTIVE_MODE"


def test_generate_intents_current_time_ns_override(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    override_ns = 123456789
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=override_ns,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.asof_timestamp == override_ns, f"Expected asof_timestamp=={override_ns}, got {result.data.asof_timestamp}"
    entry, _, _ = _extract_entry_sl_tp(result.data, symbol="AAPL")
    assert entry.created_at_ns == override_ns, f"Expected created_at_ns=={override_ns}, got {entry.created_at_ns}"


def test_generate_intents_source_candidates_recorded(
    sample_candidate: PlatformCandidate,
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    candidates = [
        _clone_candidate(sample_candidate, symbol="AAPL", score=0.5),
        _clone_candidate(sample_candidate, symbol="MSFT", score=0.6),
        _clone_candidate(sample_candidate, symbol="AAPL", score=0.9),
    ]
    candidate_set = _make_candidate_set(sample_candidate_set, candidates=candidates)
    market_data = {
        "AAPL": _clone_market_data(sample_market_data, symbol="AAPL"),
        "MSFT": _clone_market_data(sample_market_data, symbol="MSFT"),
    }
    result = generate_intents(
        candidate_set,
        {},
        market_data,
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert set(result.data.source_candidates) == {"AAPL", "MSFT"}, (
        f"Expected source_candidates to record all unique symbols, got {result.data.source_candidates}"
    )


def test_generate_intents_system_version_propagated(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert (
        result.data.system_version == sample_candidate_set.system_version
    ), f"Expected system_version propagated, got {result.data.system_version}"


def test_generate_intents_exception_handling_returns_failed(
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
) -> None:
    policy = MockPricePolicy(raise_on="entry")
    result = generate_intents(
        sample_candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.FAILED, f"Expected FAILED, got {result.status}"
    assert (
        result.reason_code == "PRICE_POLICY_ENTRY_EXCEPTION"
    ), f"Expected PRICE_POLICY_ENTRY_EXCEPTION, got {result.reason_code}"
    assert isinstance(result.error, RuntimeError), "Expected RuntimeError captured in Result.error"


def test_generate_intents_best_candidate_selected_per_symbol(
    sample_candidate: PlatformCandidate,
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    lower = _clone_candidate(sample_candidate, symbol="AAPL", score=0.1)
    higher = _clone_candidate(sample_candidate, symbol="AAPL", score=0.9)
    candidate_set = _make_candidate_set(sample_candidate_set, candidates=[lower, higher])
    result = generate_intents(
        candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert len(result.data.intent_groups) == 1, f"Expected 1 IntentGroup after de-dup, got {len(result.data.intent_groups)}"

    entry, _, _ = _extract_entry_sl_tp(result.data, symbol="AAPL")
    assert any(
        code.startswith("SCAN_") for code in (entry.reason_codes or [])
    ), f"Expected entry.reason_codes to include SCAN_* codes, got {entry.reason_codes}"


def test_generate_intents_skips_invalid_symbol_candidate(
    sample_candidate: PlatformCandidate,
    sample_candidate_set: CandidateSet,
    sample_market_data: PriceSeriesSnapshot,
    default_config: StrategyEngineConfig,
    mock_position_sizer: MockPositionSizer,
    mock_price_policy: MockPricePolicy,
) -> None:
    bad = _clone_candidate(sample_candidate, symbol="  ")
    candidate_set = _make_candidate_set(sample_candidate_set, candidates=[bad])
    result = generate_intents(
        candidate_set,
        {},
        {"AAPL": sample_market_data},
        100_000.0,
        default_config,
        mock_position_sizer,
        mock_price_policy,
        current_time_ns=123,
    )
    assert result.status is ResultStatus.SUCCESS, f"Expected SUCCESS, got {result.status}"
    assert result.data is not None, "Expected OrderIntentSet payload on SUCCESS"
    assert result.data.intent_groups == [], "Expected invalid symbol candidate to be skipped"
