from __future__ import annotations

from typing import Any

import msgspec
import msgspec.json
import pytest

from common.interface import Result
from data.interface import PriceSeriesSnapshot
from event_guard.interface import TradeConstraints
from journal.interface import SnapshotBase
from strategy.interface import (
    IntentGroup,
    IntentSnapshot,
    IntentType,
    OrderIntentSet,
    PositionSizerProtocol,
    PricePolicyProtocol,
    StrategyEngineConfig,
    TradeIntent,
)


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (IntentType.OPEN_LONG, "OPEN_LONG"),
        (IntentType.OPEN_SHORT, "OPEN_SHORT"),
        (IntentType.CLOSE_LONG, "CLOSE_LONG"),
        (IntentType.CLOSE_SHORT, "CLOSE_SHORT"),
        (IntentType.STOP_LOSS, "STOP_LOSS"),
        (IntentType.TAKE_PROFIT, "TAKE_PROFIT"),
        (IntentType.REDUCE_POSITION, "REDUCE_POSITION"),
        (IntentType.CANCEL_PENDING, "CANCEL_PENDING"),
    ],
)
def test_intent_type_enum_values(member: IntentType, value: str) -> None:
    assert member.value == value, f"IntentType.{member.name} should equal {value}, got {member.value}"
    assert IntentType(value) is member, f"IntentType should roundtrip from {value} to the same enum member"


def test_trade_intent_creation() -> None:
    intent = TradeIntent(
        intent_id="I-1",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=100.0,
        created_at_ns=1_700_000_000_000_000_000,
        entry_price=50.0,
        stop_loss_price=48.0,
        take_profit_price=55.0,
        parent_intent_id=None,
        linked_intent_ids=["I-2", "I-3"],
        reduce_only=False,
        contingency_type="OTO",
        ladder_level=None,
        reason_codes=["SCAN_SIGNAL"],
        metadata={"source": "test"},
    )
    assert intent.symbol == "AAPL", "TradeIntent.symbol should preserve the provided symbol"
    assert intent.intent_type is IntentType.OPEN_LONG, "TradeIntent.intent_type should preserve enum values"
    assert intent.quantity == pytest.approx(100.0), "TradeIntent.quantity should preserve the provided float"
    assert intent.entry_price == pytest.approx(50.0), "TradeIntent.entry_price should preserve the provided float"
    assert intent.linked_intent_ids == ["I-2", "I-3"], "TradeIntent.linked_intent_ids should preserve linkage"

    payload: dict[str, Any] = {
        "intent_id": "I-1",
        "symbol": "AAPL",
        "intent_type": "OPEN_LONG",
        "quantity": 100.0,
        "created_at_ns": 1_700_000_000_000_000_000,
    }
    converted = msgspec.convert(payload, type=TradeIntent)
    assert converted.intent_id == "I-1", "msgspec.convert should create TradeIntent from builtin payloads"

    for missing in ("intent_id", "symbol", "intent_type", "quantity", "created_at_ns"):
        bad = dict(payload)
        bad.pop(missing)
        with pytest.raises(msgspec.ValidationError, match=missing):
            msgspec.convert(bad, type=TradeIntent)


def test_trade_intent_serialization() -> None:
    intent = TradeIntent(
        intent_id="I-1",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=100.0,
        created_at_ns=1_700_000_000_000_000_000,
        entry_price=50.0,
    )
    as_builtins = msgspec.to_builtins(intent)
    assert isinstance(as_builtins, dict), "msgspec.to_builtins should return dict payloads for Structs"
    assert as_builtins["intent_id"] == "I-1", "Serialized payload should include the intent_id"

    decoded = msgspec.convert(as_builtins, type=TradeIntent)
    assert decoded == intent, "TradeIntent should roundtrip via msgspec.to_builtins + msgspec.convert"


def test_intent_group_creation() -> None:
    intents = [
        TradeIntent(
            intent_id="I-1",
            symbol="AAPL",
            intent_type=IntentType.OPEN_LONG,
            quantity=100.0,
            created_at_ns=1,
        ),
        TradeIntent(
            intent_id="I-2",
            symbol="AAPL",
            intent_type=IntentType.STOP_LOSS,
            quantity=100.0,
            created_at_ns=1,
            parent_intent_id="I-1",
            reduce_only=True,
        ),
        TradeIntent(
            intent_id="I-3",
            symbol="AAPL",
            intent_type=IntentType.TAKE_PROFIT,
            quantity=100.0,
            created_at_ns=1,
            parent_intent_id="I-1",
            reduce_only=True,
        ),
    ]
    group = IntentGroup(
        group_id="G-1",
        symbol="AAPL",
        intents=intents,
        created_at_ns=1,
        contingency_type="OUO",
    )
    assert group.group_id == "G-1", "IntentGroup.group_id should preserve deterministic ids"
    assert len(group.intents) == 3, "IntentGroup should contain the provided intents"
    assert [i.intent_id for i in group.intents] == ["I-1", "I-2", "I-3"], "IntentGroup should preserve intent ordering"


def test_intent_group_linkage() -> None:
    entry = TradeIntent(
        intent_id="I-1",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=100.0,
        created_at_ns=1,
        linked_intent_ids=["I-2", "I-3"],
        contingency_type="OTO",
    )
    stop = TradeIntent(
        intent_id="I-2",
        symbol="AAPL",
        intent_type=IntentType.STOP_LOSS,
        quantity=100.0,
        created_at_ns=1,
        parent_intent_id="I-1",
        reduce_only=True,
    )
    take = TradeIntent(
        intent_id="I-3",
        symbol="AAPL",
        intent_type=IntentType.TAKE_PROFIT,
        quantity=100.0,
        created_at_ns=1,
        parent_intent_id="I-1",
        reduce_only=True,
    )
    group = IntentGroup(group_id="G-1", symbol="AAPL", intents=[entry, stop, take], created_at_ns=1)

    assert group.intents[0].linked_intent_ids == ["I-2", "I-3"], "Entry intent should reference SL/TP intent ids"
    assert group.intents[1].parent_intent_id == "I-1", "Stop-loss intent should reference entry via parent_intent_id"
    assert group.intents[2].parent_intent_id == "I-1", "Take-profit intent should reference entry via parent_intent_id"


def test_order_intent_set_creation(sample_constraints: TradeConstraints) -> None:
    group = IntentGroup(
        group_id="G-1",
        symbol="AAPL",
        intents=[
            TradeIntent(
                intent_id="I-1",
                symbol="AAPL",
                intent_type=IntentType.OPEN_LONG,
                quantity=100.0,
                created_at_ns=1,
            )
        ],
        created_at_ns=1,
    )
    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        intent_groups=[group],
        constraints_applied={"AAPL": msgspec.to_builtins(sample_constraints)},
        source_candidates=["AAPL"],
    )
    assert intent_set.intent_groups[0].group_id == "G-1", "OrderIntentSet should embed IntentGroup outputs"
    assert "AAPL" in intent_set.constraints_applied, "OrderIntentSet should record applied constraints by symbol"
    assert intent_set.source_candidates == ["AAPL"], "OrderIntentSet should record upstream candidate symbols"


def test_order_intent_set_snapshot_base() -> None:
    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
    )
    assert isinstance(intent_set, SnapshotBase), "OrderIntentSet should inherit SnapshotBase for journaling metadata"
    assert intent_set.schema_version == "1.0.0", "OrderIntentSet should expose SnapshotBase.schema_version"
    assert intent_set.intent_groups == [], "OrderIntentSet.intent_groups should default to empty list"
    assert intent_set.constraints_applied == {}, "OrderIntentSet.constraints_applied should default to empty mapping"
    assert intent_set.source_candidates == [], "OrderIntentSet.source_candidates should default to empty list"


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("position_sizer", "fixed_percent"),
        ("position_size_pct", 0.02),
        ("risk_per_trade_pct", 0.01),
        ("max_position_pct", 0.1),
        ("price_policy", "atr_bracket"),
        ("atr_multiplier", 2.0),
        ("trailing_offset_pct", 0.05),
        ("entry_strategy", "single"),
        ("ladder_levels", 3),
        ("ladder_spacing_pct", 0.02),
        ("use_conservative_defaults", True),
    ],
)
def test_strategy_engine_config_defaults(default_config: StrategyEngineConfig, field: str, expected: object) -> None:
    actual = getattr(default_config, field)
    assert actual == expected, f"StrategyEngineConfig.{field} default should be {expected!r}, got {actual!r}"


def test_strategy_engine_config_custom() -> None:
    config = StrategyEngineConfig(
        position_sizer="fixed_risk",
        position_size_pct=0.03,
        risk_per_trade_pct=0.015,
        max_position_pct=0.2,
        price_policy="trailing_stop",
        atr_multiplier=3.0,
        trailing_offset_pct=0.075,
        entry_strategy="ladder",
        ladder_levels=5,
        ladder_spacing_pct=0.01,
        use_conservative_defaults=False,
    )
    assert config.position_sizer == "fixed_risk", "Custom StrategyEngineConfig.position_sizer should be preserved"
    assert config.price_policy == "trailing_stop", "Custom StrategyEngineConfig.price_policy should be preserved"
    assert config.ladder_levels == 5, "Custom StrategyEngineConfig.ladder_levels should be preserved"

    encoded = msgspec.json.encode(config)
    decoded = msgspec.json.decode(encoded, type=StrategyEngineConfig)
    assert decoded == config, "StrategyEngineConfig should roundtrip via msgspec JSON encode/decode"


def test_intent_snapshot_creation(sample_constraints: TradeConstraints) -> None:
    group = IntentGroup(
        group_id="G-1",
        symbol="AAPL",
        intents=[
            TradeIntent(
                intent_id="I-1",
                symbol="AAPL",
                intent_type=IntentType.OPEN_LONG,
                quantity=100.0,
                created_at_ns=1,
            )
        ],
        created_at_ns=1,
    )
    intent_set = OrderIntentSet(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        intent_groups=[group],
        constraints_applied={"AAPL": msgspec.to_builtins(sample_constraints)},
        source_candidates=["AAPL"],
    )
    snapshot = IntentSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        intent_sets=[intent_set],
    )
    assert snapshot.intent_sets[0] == intent_set, "IntentSnapshot should embed one or more OrderIntentSet payloads"
    assert snapshot.degradation_events == [], "IntentSnapshot.degradation_events should default to empty list"


def test_intent_snapshot_degradation_events() -> None:
    snapshot = IntentSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        degradation_events=[
            {"symbol": "AAPL", "reason_code": "MISSING_MARKET_DATA"},
            {"symbol": "MSFT", "reason_code": "CONSTRAINT_BLOCK"},
        ],
    )
    assert len(snapshot.degradation_events) == 2, "IntentSnapshot should persist degradation events verbatim"
    assert snapshot.degradation_events[0]["symbol"] == "AAPL", "Degradation events should preserve per-event payloads"


def test_position_sizer_protocol_runtime_checkable() -> None:
    class DummySizer:
        def calculate_size(  # type: ignore[override]
            self,
            symbol: str,
            entry_price: float,
            stop_loss_price: float,
            account_equity: float,
            config: StrategyEngineConfig,
            constraints: TradeConstraints | None = None,
            market_data: PriceSeriesSnapshot | None = None,
        ) -> Result[float]:
            return Result.success(1.0)

    assert isinstance(DummySizer(), PositionSizerProtocol), "PositionSizerProtocol should be runtime-checkable via isinstance"


def test_price_policy_protocol_runtime_checkable(sample_market_data: PriceSeriesSnapshot, default_config: StrategyEngineConfig) -> None:
    class DummyPolicy:
        def calculate_entry_price(  # type: ignore[override]
            self,
            symbol: str,
            side: str,
            config: StrategyEngineConfig,
            market_data: PriceSeriesSnapshot,
        ) -> Result[float]:
            return Result.success(float(market_data.bars[-1].close))

        def calculate_stop_loss(  # type: ignore[override]
            self,
            symbol: str,
            entry_price: float,
            side: str,
            config: StrategyEngineConfig,
            market_data: PriceSeriesSnapshot,
        ) -> Result[float]:
            return Result.success(entry_price * 0.95)

        def calculate_take_profit(  # type: ignore[override]
            self,
            symbol: str,
            entry_price: float,
            side: str,
            config: StrategyEngineConfig,
            market_data: PriceSeriesSnapshot,
        ) -> Result[float]:
            return Result.success(entry_price * 1.05)

    policy = DummyPolicy()
    assert isinstance(policy, PricePolicyProtocol), "PricePolicyProtocol should be runtime-checkable via isinstance"
    assert policy.calculate_entry_price("AAPL", "LONG", default_config, sample_market_data).status.name == "SUCCESS"


def test_trade_intent_frozen() -> None:
    intent = TradeIntent(
        intent_id="I-1",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=100.0,
        created_at_ns=1,
    )
    with pytest.raises((AttributeError, TypeError)):
        intent.symbol = "MSFT"  # type: ignore[misc]


def test_intent_group_contingency_default() -> None:
    group = IntentGroup(
        group_id="G-1",
        symbol="AAPL",
        intents=[
            TradeIntent(
                intent_id="I-1",
                symbol="AAPL",
                intent_type=IntentType.OPEN_LONG,
                quantity=100.0,
                created_at_ns=1,
            )
        ],
        created_at_ns=1,
    )
    assert group.contingency_type == "OUO", 'IntentGroup.contingency_type should default to "OUO"'
