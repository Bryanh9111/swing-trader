from __future__ import annotations

import msgspec
import pytest

from common.interface import ResultStatus
from data.interface import PriceBar, PriceSeriesSnapshot
from indicators.interface import compute_atr_last
from strategy.interface import StrategyEngineConfig, TransactionCostConfig
from strategy.pricing import (
    ATRBracketPolicy,
    LadderEntryPolicy,
    TrailingStopPolicy,
    create_price_policy,
)


def _with_market_data_atr(market_data: PriceSeriesSnapshot, atr_pct: float | None) -> PriceSeriesSnapshot:
    payload = msgspec.to_builtins(market_data)
    payload["atr_pct"] = atr_pct
    return market_data.__class__(**payload)  # type: ignore[call-arg]


def _make_market_data(
    *,
    bars: list[PriceBar],
    symbol: str = "AAPL",
    atr_pct: float | None = None,
) -> PriceSeriesSnapshot:
    class PriceSeriesSnapshotWithATR(PriceSeriesSnapshot, frozen=True, kw_only=True):
        atr_pct: float | None = msgspec.field(default=None)

    return PriceSeriesSnapshotWithATR(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1_700_000_000_000_000_000,
        symbol=symbol,
        timeframe="1d",
        bars=bars,
        source="test",
        quality_flags={},
        atr_pct=atr_pct,
    )


def _make_flat_bars(*, count: int = 15, close: float = 100.0, high: float = 102.0, low: float = 98.0) -> list[PriceBar]:
    bars: list[PriceBar] = []
    for idx in range(count):
        bars.append(
            PriceBar(
                timestamp=1_700_000_000_000_000_000 + idx * 60_000_000_000,
                open=close,
                high=high,
                low=low,
                close=close,
                volume=1_000_000,
            )
        )
    return bars


def test_atr_bracket_entry_price_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0)
    )
    result = ATRBracketPolicy().calculate_entry_price("AAPL", "LONG", config, sample_market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(float(sample_market_data.bars[-1].close))


def test_atr_bracket_entry_price_empty_bars_fails() -> None:
    market_data = _make_market_data(bars=[], atr_pct=0.02)
    result = ATRBracketPolicy().calculate_entry_price("AAPL", "LONG", StrategyEngineConfig(), market_data)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "EMPTY_MARKET_DATA"


def test_atr_bracket_entry_price_invalid_price_fails(sample_market_data: PriceSeriesSnapshot) -> None:
    bad_last = PriceBar(
        timestamp=sample_market_data.bars[-1].timestamp + 60_000_000_000,
        open=0.0,
        high=0.0,
        low=0.0,
        close=0.0,
        volume=1,
    )
    market_data = _make_market_data(bars=list(sample_market_data.bars[:-1]) + [bad_last], atr_pct=0.02)
    result = ATRBracketPolicy().calculate_entry_price("AAPL", "LONG", StrategyEngineConfig(), market_data)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_ENTRY_PRICE"


def test_atr_bracket_stop_loss_long_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(
        atr_multiplier=2.0,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )
    entry = float(sample_market_data.bars[-1].close)
    atr_pct = 0.02
    market_data = _with_market_data_atr(sample_market_data, atr_pct)

    result = ATRBracketPolicy().calculate_stop_loss("AAPL", entry, "LONG", config, market_data)
    assert result.status is ResultStatus.SUCCESS

    dist = entry * atr_pct * config.atr_multiplier
    expected = entry - dist
    assert result.data == pytest.approx(expected)


def test_atr_bracket_stop_loss_short_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(atr_multiplier=2.0)
    entry = float(sample_market_data.bars[-1].close)
    atr_pct = 0.02
    market_data = _with_market_data_atr(sample_market_data, atr_pct)

    result = ATRBracketPolicy().calculate_stop_loss("AAPL", entry, "SHORT", config, market_data)
    assert result.status is ResultStatus.SUCCESS

    dist = entry * atr_pct * config.atr_multiplier
    expected = entry + dist
    assert result.data == pytest.approx(expected)


def test_atr_bracket_stop_loss_uses_market_data_atr_pct(sample_market_data: PriceSeriesSnapshot) -> None:
    entry = float(sample_market_data.bars[-1].close)
    config = StrategyEngineConfig(
        atr_multiplier=2.0,
        stop_loss_pct=1.0,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )

    bars = _make_flat_bars(count=15, close=entry, high=entry * 1.001, low=entry * 0.999)
    market_data = _make_market_data(bars=bars, atr_pct=0.10)

    result = ATRBracketPolicy().calculate_stop_loss("AAPL", entry, "LONG", config, market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(entry - (entry * 0.10 * config.atr_multiplier))


def test_atr_bracket_stop_loss_uses_fallback_atr() -> None:
    config = StrategyEngineConfig(
        atr_multiplier=2.0,
        stop_loss_pct=1.0,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )
    bars = _make_flat_bars(count=15, close=100.0, high=102.0, low=98.0)
    market_data = _make_market_data(bars=bars, atr_pct=None)

    result = ATRBracketPolicy().calculate_stop_loss("AAPL", 100.0, "LONG", config, market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(92.0)


@pytest.mark.parametrize(
    ("atr_pct", "bars_count"),
    [
        (0.0, 0),
        (None, 0),
        (None, 10),
    ],
)
def test_atr_bracket_stop_loss_fallback_when_atr_unavailable(atr_pct: float | None, bars_count: int) -> None:
    bars = _make_flat_bars(count=bars_count) if bars_count else []
    market_data = _make_market_data(bars=bars, atr_pct=atr_pct)
    config = StrategyEngineConfig(
        stop_loss_pct=0.06,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )
    result = ATRBracketPolicy().calculate_stop_loss("AAPL", 100.0, "LONG", config, market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(94.0)


def test_atr_bracket_stop_loss_invalid_sl_fails() -> None:
    config = StrategyEngineConfig(
        atr_multiplier=2.0,
        stop_loss_pct=2.0,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )
    market_data = _make_market_data(bars=_make_flat_bars(close=50.0), atr_pct=1.0)
    result = ATRBracketPolicy().calculate_stop_loss("AAPL", 50.0, "LONG", config, market_data)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_STOP_LOSS"


def test_atr_bracket_take_profit_long_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(
        atr_multiplier=2.0,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )
    entry = float(sample_market_data.bars[-1].close)
    atr_pct = 0.02
    market_data = _with_market_data_atr(sample_market_data, atr_pct)

    result = ATRBracketPolicy().calculate_take_profit("AAPL", entry, "LONG", config, market_data)
    assert result.status is ResultStatus.SUCCESS

    # Calculate expected TP using the new TP/SL ratio (default 1.5:1).
    # SL distance: entry * atr_pct * atr_multiplier
    # TP distance: SL distance * tp_sl_ratio
    sl_dist = entry * atr_pct * config.atr_multiplier  # Stop loss distance
    tp_dist = sl_dist * config.tp_sl_ratio  # Take profit distance (1.5x SL by default)
    expected = entry + tp_dist
    assert result.data == pytest.approx(expected)


def test_atr_bracket_take_profit_short_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(atr_multiplier=2.0)
    entry = float(sample_market_data.bars[-1].close)
    atr_pct = 0.02
    market_data = _with_market_data_atr(sample_market_data, atr_pct)

    result = ATRBracketPolicy().calculate_take_profit("AAPL", entry, "SHORT", config, market_data)
    assert result.status is ResultStatus.SUCCESS

    # Calculate expected TP using the new TP/SL ratio (default 1.5:1).
    # SL distance: entry * atr_pct * atr_multiplier
    # TP distance: SL distance * tp_sl_ratio
    sl_dist = entry * atr_pct * config.atr_multiplier  # Stop loss distance
    tp_dist = sl_dist * config.tp_sl_ratio  # Take profit distance (1.5x SL by default)
    expected = entry - tp_dist
    assert result.data == pytest.approx(expected)


@pytest.mark.parametrize(
    ("atr_pct", "bars_count"),
    [
        (0.0, 0),
        (None, 0),
        (None, 10),
    ],
)
def test_atr_bracket_take_profit_fallback_when_atr_unavailable(atr_pct: float | None, bars_count: int) -> None:
    bars = _make_flat_bars(count=bars_count) if bars_count else []
    market_data = _make_market_data(bars=bars, atr_pct=atr_pct)
    config = StrategyEngineConfig(
        stop_loss_pct=0.06,
        tp_sl_ratio=1.5,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )
    result = ATRBracketPolicy().calculate_take_profit("AAPL", 100.0, "LONG", config, market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(109.0)


def test_atr_bracket_take_profit_invalid_tp_fails() -> None:
    config = StrategyEngineConfig(
        use_fixed_tp=True,
        take_profit_pct=2.0,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )
    market_data = _make_market_data(bars=_make_flat_bars(close=50.0), atr_pct=0.02)
    result = ATRBracketPolicy().calculate_take_profit("AAPL", 50.0, "SHORT", config, market_data)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_TAKE_PROFIT"


def test_atr_bracket_policy_respects_regime_sl_constraint() -> None:
    """Test that ATR-based SL is capped by regime stop_loss_pct."""

    market_data = _make_market_data(bars=_make_flat_bars(close=100.0), atr_pct=0.10)
    config = StrategyEngineConfig(
        atr_multiplier=1.5,
        stop_loss_pct=0.06,
        tp_sl_ratio=2.0,
        dynamic_tp_sl_enabled=False,  # Disable dynamic ratio to test fixed tp_sl_ratio
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )

    sl = ATRBracketPolicy().calculate_stop_loss("TEST", 100.0, "LONG", config, market_data)
    assert sl.status is ResultStatus.SUCCESS
    assert sl.data == pytest.approx(94.0)

    tp = ATRBracketPolicy().calculate_take_profit("TEST", 100.0, "LONG", config, market_data)
    assert tp.status is ResultStatus.SUCCESS
    assert tp.data == pytest.approx(112.0)


def test_atr_bracket_policy_uses_atr_when_lower() -> None:
    """Test that ATR-based SL is used when lower than regime constraint."""

    market_data = _make_market_data(bars=_make_flat_bars(close=100.0), atr_pct=0.02)
    config = StrategyEngineConfig(
        atr_multiplier=1.5,
        stop_loss_pct=0.06,
        tp_sl_ratio=2.0,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )

    sl = ATRBracketPolicy().calculate_stop_loss("TEST", 100.0, "LONG", config, market_data)
    assert sl.status is ResultStatus.SUCCESS
    assert sl.data == pytest.approx(97.0)

    tp = ATRBracketPolicy().calculate_take_profit("TEST", 100.0, "LONG", config, market_data)
    assert tp.status is ResultStatus.SUCCESS
    assert tp.data == pytest.approx(106.0)


def test_atr_bracket_policy_fallback_when_atr_unavailable() -> None:
    """Test fallback to regime stop_loss_pct when ATR is unavailable."""

    market_data = _make_market_data(bars=[], atr_pct=None)
    config = StrategyEngineConfig(
        stop_loss_pct=0.06,
        tp_sl_ratio=1.5,
        transaction_costs=TransactionCostConfig(spread_pct=0.0, slippage_pct=0.0, commission_per_trade=0.0),
    )

    sl = ATRBracketPolicy().calculate_stop_loss("TEST", 100.0, "LONG", config, market_data)
    assert sl.status is ResultStatus.SUCCESS
    assert sl.data == pytest.approx(94.0)

    tp = ATRBracketPolicy().calculate_take_profit("TEST", 100.0, "LONG", config, market_data)
    assert tp.status is ResultStatus.SUCCESS
    assert tp.data == pytest.approx(109.0)


def test_trailing_stop_entry_price_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    result = TrailingStopPolicy().calculate_entry_price("AAPL", "LONG", StrategyEngineConfig(), sample_market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(float(sample_market_data.bars[-1].close))


def test_entry_price_with_transaction_costs() -> None:
    """Test that entry price is increased by spread + slippage for LONG positions."""

    market_data = _make_market_data(
        bars=[
            PriceBar(
                timestamp=1_700_000_000_000_000_000,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1_000_000,
            )
        ],
        symbol="TEST",
    )
    config = StrategyEngineConfig(
        transaction_costs=TransactionCostConfig(spread_pct=0.001, slippage_pct=0.0005, commission_per_trade=1.0)
    )

    result = ATRBracketPolicy().calculate_entry_price("TEST", "LONG", config, market_data)
    assert result.status is ResultStatus.SUCCESS

    expected_price = 100.0 * (1.0 + 0.001 + 0.0005)
    assert result.data == pytest.approx(expected_price)


def test_stop_loss_price_with_transaction_costs() -> None:
    """Test that stop-loss price is decreased by spread + slippage for LONG exits."""

    market_data = _make_market_data(
        bars=_make_flat_bars(count=20, close=100.0, high=102.0, low=98.0),
        symbol="TEST",
        atr_pct=0.02,
    )
    entry_price = 100.0
    config = StrategyEngineConfig(
        atr_multiplier=2.0,
        transaction_costs=TransactionCostConfig(spread_pct=0.001, slippage_pct=0.0005, commission_per_trade=1.0),
    )

    result = ATRBracketPolicy().calculate_stop_loss("TEST", entry_price, "LONG", config, market_data)
    assert result.status is ResultStatus.SUCCESS

    base_sl = entry_price - (entry_price * 0.02 * 2.0)
    expected_sl = base_sl * (1.0 - 0.001 - 0.0005)
    assert result.data == pytest.approx(expected_sl)


def test_take_profit_price_with_transaction_costs() -> None:
    """Test that take-profit price is decreased by spread + slippage for LONG exits."""

    market_data = _make_market_data(
        bars=_make_flat_bars(count=20, close=100.0, high=102.0, low=98.0),
        symbol="TEST",
        atr_pct=0.02,
    )
    entry_price = 100.0
    config = StrategyEngineConfig(
        atr_multiplier=2.0,
        transaction_costs=TransactionCostConfig(spread_pct=0.001, slippage_pct=0.0005, commission_per_trade=1.0),
    )

    result = ATRBracketPolicy().calculate_take_profit("TEST", entry_price, "LONG", config, market_data)
    assert result.status is ResultStatus.SUCCESS

    # Calculate base TP using the new TP/SL ratio (default 1.5:1), then apply transaction costs.
    sl_dist = entry_price * 0.02 * 2.0  # Stop loss distance
    tp_dist = sl_dist * config.tp_sl_ratio  # Take profit distance (1.5x SL by default)
    base_tp = entry_price + tp_dist
    expected_tp = base_tp * (1.0 - 0.001 - 0.0005)
    assert result.data == pytest.approx(expected_tp)


def test_trailing_stop_stop_loss_long_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(trailing_offset_pct=0.05)
    entry = float(sample_market_data.bars[-1].close)
    result = TrailingStopPolicy().calculate_stop_loss("AAPL", entry, "LONG", config, sample_market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(entry * (1.0 - 0.05))


def test_trailing_stop_stop_loss_short_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(trailing_offset_pct=0.05)
    entry = float(sample_market_data.bars[-1].close)
    result = TrailingStopPolicy().calculate_stop_loss("AAPL", entry, "SHORT", config, sample_market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(entry * (1.0 + 0.05))


def test_trailing_stop_take_profit_long_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(trailing_offset_pct=0.05)
    entry = float(sample_market_data.bars[-1].close)
    result = TrailingStopPolicy().calculate_take_profit("AAPL", entry, "LONG", config, sample_market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(entry * (1.0 + 2.0 * 0.05))


def test_trailing_stop_take_profit_short_happy_path(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(trailing_offset_pct=0.05)
    entry = float(sample_market_data.bars[-1].close)
    result = TrailingStopPolicy().calculate_take_profit("AAPL", entry, "SHORT", config, sample_market_data)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(entry * (1.0 - 2.0 * 0.05))


def test_ladder_entry_price_level_0(sample_market_data: PriceSeriesSnapshot, default_config: StrategyEngineConfig) -> None:
    base = float(sample_market_data.bars[-1].close)
    result = LadderEntryPolicy().calculate_entry_price("AAPL", "LONG", default_config, sample_market_data, ladder_level=0)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(base)


def test_ladder_entry_price_level_1_long(sample_market_data: PriceSeriesSnapshot, default_config: StrategyEngineConfig) -> None:
    base = float(sample_market_data.bars[-1].close)
    spacing = float(default_config.ladder_spacing_pct)
    result = LadderEntryPolicy().calculate_entry_price("AAPL", "LONG", default_config, sample_market_data, ladder_level=1)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(base * (1.0 - spacing))


def test_ladder_entry_price_level_2_short(sample_market_data: PriceSeriesSnapshot, default_config: StrategyEngineConfig) -> None:
    base = float(sample_market_data.bars[-1].close)
    spacing = float(default_config.ladder_spacing_pct)
    result = LadderEntryPolicy().calculate_entry_price("AAPL", "SHORT", default_config, sample_market_data, ladder_level=2)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(base * (1.0 + 2.0 * spacing))


def test_ladder_stop_loss_delegates_to_atr(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(atr_multiplier=2.0)
    entry = float(sample_market_data.bars[-1].close)
    ladder_result = LadderEntryPolicy().calculate_stop_loss("AAPL", entry, "LONG", config, sample_market_data)
    atr_result = ATRBracketPolicy().calculate_stop_loss("AAPL", entry, "LONG", config, sample_market_data)
    assert ladder_result.status is atr_result.status
    assert ladder_result.reason_code == atr_result.reason_code
    assert ladder_result.data == pytest.approx(float(atr_result.data))


def test_ladder_take_profit_delegates_to_atr(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(atr_multiplier=2.0)
    entry = float(sample_market_data.bars[-1].close)
    ladder_result = LadderEntryPolicy().calculate_take_profit("AAPL", entry, "LONG", config, sample_market_data)
    atr_result = ATRBracketPolicy().calculate_take_profit("AAPL", entry, "LONG", config, sample_market_data)
    assert ladder_result.status is atr_result.status
    assert ladder_result.reason_code == atr_result.reason_code
    assert ladder_result.data == pytest.approx(float(atr_result.data))


def test_create_atr_bracket_policy() -> None:
    policy = create_price_policy(StrategyEngineConfig(price_policy="atr_bracket"))
    assert isinstance(policy, ATRBracketPolicy)


def test_create_trailing_stop_policy() -> None:
    policy = create_price_policy(StrategyEngineConfig(price_policy="trailing_stop"))
    assert isinstance(policy, TrailingStopPolicy)


def test_create_ladder_entry_policy() -> None:
    policy = create_price_policy(StrategyEngineConfig(price_policy="ladder_entry"))
    assert isinstance(policy, LadderEntryPolicy)


def test_create_price_policy_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown price_policy type"):
        create_price_policy(StrategyEngineConfig(price_policy="nope"))


def test_compute_atr_last_happy_path() -> None:
    bars = _make_flat_bars(count=15, close=100.0, high=102.0, low=98.0)
    atr_pct = compute_atr_last(bars, period=14, percentage=True, reference_price=100.0)
    assert atr_pct == pytest.approx(0.04)


def test_compute_atr_last_insufficient_bars() -> None:
    bars = _make_flat_bars(count=14)
    assert compute_atr_last(bars, period=14, percentage=True, reference_price=100.0) is None


def test_compute_atr_last_tr_calculation() -> None:
    bars: list[PriceBar] = []
    for idx in range(15):
        if idx == 0:
            bars.append(
                PriceBar(
                    timestamp=1_700_000_000_000_000_000,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1,
                )
            )
            continue
        if idx < 14:
            bars.append(
                PriceBar(
                    timestamp=1_700_000_000_000_000_000 + idx * 60_000_000_000,
                    open=100.0,
                    high=101.0,
                    low=100.0,
                    close=100.0,
                    volume=1,
                )
            )
            continue
        bars.append(
            PriceBar(
                timestamp=1_700_000_000_000_000_000 + idx * 60_000_000_000,
                open=119.0,
                high=120.0,
                low=119.0,
                close=119.0,
                volume=1,
            )
        )

    atr_pct = compute_atr_last(bars, period=14, percentage=True, reference_price=100.0)
    assert atr_pct == pytest.approx(((13.0 * 1.0) + 20.0) / 14.0 / 100.0)
