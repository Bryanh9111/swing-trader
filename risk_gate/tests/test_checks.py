from __future__ import annotations

import msgspec
import pytest

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
    check_daily_new_position_limit,
)
from risk_gate.interface import CheckStatus, RiskCheckContext, RiskGateConfig
from strategy.interface import IntentType, TradeIntent


def _intent(*, intent_id: str, symbol: str, intent_type: IntentType, quantity: float, entry_price: float | None = None) -> TradeIntent:
    return TradeIntent(
        intent_id=intent_id,
        symbol=symbol,
        intent_type=intent_type,
        quantity=float(quantity),
        created_at_ns=1,
        entry_price=entry_price,
    )


def _series(symbol: str, closes: list[float]) -> PriceSeriesSnapshot:
    bars = [
        PriceBar(timestamp=i + 1, open=c, high=c, low=c, close=c, volume=1000)  # type: ignore[arg-type]
        for i, c in enumerate(closes)
    ]
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


def test_leverage_check_happy_path_passes_with_sufficient_headroom() -> None:
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_leverage=10.0))
    context = RiskCheckContext(
        account_equity=1_000.0,
        positions={"AAPL": {"qty": 1.0}},
        market_data={"AAPL": _series("AAPL", [50.0])},
    )
    res = LeverageCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["gross_exposure_after"] > res.data.details["gross_exposure_before"]


def test_leverage_check_fails_closed_on_invalid_equity() -> None:
    res = LeverageCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1),
        RiskCheckContext(account_equity=0.0),
        RiskGateConfig(),
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "MISSING_ACCOUNT" in res.data.reason_codes


def test_leverage_check_fails_closed_on_missing_market_data_for_existing_positions() -> None:
    context = RiskCheckContext(account_equity=100.0, positions={"AAPL": 1.0}, market_data={})
    res = LeverageCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1),
        context,
        RiskGateConfig(),
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "MISSING_MARKET_DATA" in res.data.reason_codes
    assert "AAPL" in (res.data.details.get("missing_symbols") or [])


def test_drawdown_check_happy_path_passes_below_threshold() -> None:
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_drawdown_pct=0.2))
    context = RiskCheckContext(account_equity=90.0, portfolio_snapshot={"peak_equity": 100.0})
    res = DrawdownCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS


def test_drawdown_check_fails_closed_on_missing_peak_equity() -> None:
    context = RiskCheckContext(account_equity=90.0, portfolio_snapshot={})
    res = DrawdownCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1),
        context,
        RiskGateConfig(),
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "MISSING_ACCOUNT" in res.data.reason_codes


def test_daily_loss_check_happy_path_passes_below_threshold() -> None:
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_daily_loss_pct=0.02))
    context = RiskCheckContext(account_equity=99.0, portfolio_snapshot={"day_start_equity": 100.0})
    res = DailyLossCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details.get("safe_mode_trigger") is None


def test_daily_loss_check_fails_closed_on_missing_day_start_equity() -> None:
    context = RiskCheckContext(account_equity=99.0, portfolio_snapshot={})
    res = DailyLossCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1),
        context,
        RiskGateConfig(),
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "MISSING_ACCOUNT" in res.data.reason_codes
    assert res.data.details.get("reason") == "missing_day_start_equity"


def test_concentration_check_happy_path_passes() -> None:
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_concentration_pct=0.25))
    context = RiskCheckContext(
        account_equity=1_000.0,
        positions={"AAPL": 0.0},
        market_data={"AAPL": _series("AAPL", [10.0])},
    )
    res = ConcentrationCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=10),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS


def test_concentration_check_projection_allows_reducing_exposure() -> None:
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_concentration_pct=0.05))
    context = RiskCheckContext(
        account_equity=1_000.0,
        positions={"AAPL": 200.0},
        market_data={"AAPL": _series("AAPL", [10.0])},
    )
    res = ConcentrationCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.CLOSE_LONG, quantity=200),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["projected_qty"] == 0.0


def test_max_position_check_boundary_allows_equal_threshold() -> None:
    config = RiskGateConfig(symbol=msgspec.structs.replace(RiskGateConfig().symbol, max_position_size=10.0))
    context = RiskCheckContext(account_equity=100.0, positions={"AAPL": 5.0})
    res = MaxPositionCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=5),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["projected_qty"] == 10.0


def test_price_band_check_allows_market_orders() -> None:
    config = RiskGateConfig(symbol=msgspec.structs.replace(RiskGateConfig().symbol, max_price_band_bps=1.0))
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", [100.0])})
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=None),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details.get("skipped") == "market_order"


def test_price_band_check_fails_closed_on_missing_stop_loss_price() -> None:
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", [100.0])})
    res = PriceBandCheck().check(
        TradeIntent(intent_id="1", symbol="AAPL", intent_type=IntentType.STOP_LOSS, quantity=1.0, created_at_ns=1),
        context,
        RiskGateConfig(),
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "CHECK_FAILED" in res.data.reason_codes


def test_price_band_check_boundary_allows_equal_bps_threshold() -> None:
    config = RiskGateConfig(symbol=msgspec.structs.replace(RiskGateConfig().symbol, max_price_band_bps=250.0))
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", [100.0])})
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=102.5),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["dev_bps"] == pytest.approx(250.0)


def test_volatility_halt_check_happy_path_passes_for_stable_returns() -> None:
    config = RiskGateConfig(symbol=msgspec.structs.replace(RiskGateConfig().symbol, max_volatility_z_score=3.0))
    closes = [100.0, 100.1, 100.05, 100.0, 100.08, 100.1]
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", closes)})
    res = VolatilityHaltCheck(lookback_returns=5).check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=100.1),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS


def test_volatility_halt_check_fails_closed_on_insufficient_bars() -> None:
    config = RiskGateConfig(symbol=msgspec.structs.replace(RiskGateConfig().symbol, max_volatility_z_score=3.0))
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", [100.0, 101.0])})
    res = VolatilityHaltCheck(lookback_returns=5).check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=101.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "MISSING_MARKET_DATA" in res.data.reason_codes
    assert res.data.details.get("reason") == "insufficient_bars"


def test_reduce_only_check_happy_path_allows_reduction() -> None:
    intent = TradeIntent(
        intent_id="1",
        symbol="AAPL",
        intent_type=IntentType.REDUCE_POSITION,
        quantity=5.0,
        created_at_ns=1,
        reduce_only=True,
    )
    context = RiskCheckContext(account_equity=100.0, positions={"AAPL": 10.0})
    res = ReduceOnlyCheck().check(intent, context, RiskGateConfig())
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["projected_qty"] == 5.0


@pytest.mark.parametrize(
    ("intent_type", "price_field", "price_value"),
    [
        (IntentType.STOP_LOSS, "stop_loss_price", 95.0),
        (IntentType.TAKE_PROFIT, "take_profit_price", 105.0),
    ],
)
def test_reduce_only_check_allows_bracket_leg_when_parent_opens_position(
    intent_type: IntentType, price_field: str, price_value: float
) -> None:
    entry = TradeIntent(
        intent_id="i-entry",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=1,
    )
    leg_kwargs = {
        "intent_id": f"i-{intent_type.value.lower()}",
        "symbol": "AAPL",
        "intent_type": intent_type,
        "quantity": 10.0,
        "created_at_ns": 1,
        "reduce_only": True,
        "parent_intent_id": "i-entry",
        price_field: price_value,
    }
    leg = TradeIntent(**leg_kwargs)  # type: ignore[arg-type]
    context = RiskCheckContext(account_equity=100.0, positions={}, intents=[entry, leg])

    res = ReduceOnlyCheck().check(leg, context, RiskGateConfig())
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details.get("skipped") == "bracket_parent_opens_position"
    assert res.data.details.get("parent_intent_id") == "i-entry"


def test_reduce_only_check_falls_back_when_parent_missing_and_no_position() -> None:
    leg = TradeIntent(
        intent_id="i-sl",
        symbol="AAPL",
        intent_type=IntentType.STOP_LOSS,
        quantity=10.0,
        created_at_ns=1,
        stop_loss_price=95.0,
        reduce_only=True,
        parent_intent_id="i-entry",
    )
    context = RiskCheckContext(account_equity=100.0, positions={}, intents=[])

    res = ReduceOnlyCheck().check(leg, context, RiskGateConfig())
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert res.data.details.get("reason") == "missing_current_position"


def test_reduce_only_check_does_not_skip_when_parent_is_reduce_only() -> None:
    entry = TradeIntent(
        intent_id="i-entry",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=10.0,
        created_at_ns=1,
        reduce_only=True,
    )
    leg = TradeIntent(
        intent_id="i-sl",
        symbol="AAPL",
        intent_type=IntentType.STOP_LOSS,
        quantity=10.0,
        created_at_ns=1,
        stop_loss_price=95.0,
        reduce_only=True,
        parent_intent_id="i-entry",
    )
    context = RiskCheckContext(account_equity=100.0, positions={}, intents=[entry, leg])

    res = ReduceOnlyCheck().check(leg, context, RiskGateConfig())
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert res.data.details.get("reason") == "missing_current_position"


def test_order_count_check_does_not_double_count_same_intent_id() -> None:
    config = RiskGateConfig(operational=msgspec.structs.replace(RiskGateConfig().operational, max_orders_per_run=1))
    check = OrderCountCheck()
    context = RiskCheckContext(account_equity=100.0)
    intent = _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1)
    assert check.check(intent, context, config).data.status == CheckStatus.PASS
    assert check.check(intent, context, config).data.status == CheckStatus.PASS


def test_rate_limit_check_uses_custom_rate_limit_and_initial_count() -> None:
    check = RateLimitCheck(initial_count=1, rate_limit=2)
    context = RiskCheckContext(account_equity=100.0)
    config = RiskGateConfig()
    assert check.check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config).data.status == CheckStatus.PASS
    assert check.check(_intent(intent_id="2", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config).data.status == CheckStatus.FAIL


def test_daily_new_position_limit_passes_when_under_limit() -> None:
    config = RiskGateConfig(operational=msgspec.structs.replace(RiskGateConfig().operational, max_new_positions_per_day=2))
    intents = [
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1),
        _intent(intent_id="2", symbol="MSFT", intent_type=IntentType.OPEN_LONG, quantity=1),
    ]
    res = check_daily_new_position_limit(intents, config, market_data={})
    assert res.status == CheckStatus.PASS
    assert res.details["new_position_count"] == 2
    assert res.details["max_new_positions_per_day"] == 2


def test_daily_new_position_limit_keeps_top_scored_symbols() -> None:
    config = RiskGateConfig(operational=msgspec.structs.replace(RiskGateConfig().operational, max_new_positions_per_day=2))
    aapl = msgspec.structs.replace(_series("AAPL", [100.0]), quality_flags={"score": 0.90})
    msft = msgspec.structs.replace(_series("MSFT", [100.0]), quality_flags={"score": 0.80})
    tsla = msgspec.structs.replace(_series("TSLA", [100.0]), quality_flags={"score": 0.95})
    intents = [
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1),
        _intent(intent_id="2", symbol="MSFT", intent_type=IntentType.OPEN_LONG, quantity=1),
        _intent(intent_id="3", symbol="TSLA", intent_type=IntentType.OPEN_LONG, quantity=1),
    ]
    res = check_daily_new_position_limit(intents, config, market_data={"AAPL": aapl, "MSFT": msft, "TSLA": tsla})
    assert res.status == CheckStatus.FAIL
    assert res.details["kept_symbols"] == ["TSLA", "AAPL"]
    assert res.details["rejected_symbols"] == ["MSFT"]
    assert res.details["rejected_count"] == 1


def test_leverage_check_blocks_when_projected_leverage_exceeds_limit() -> None:
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_leverage=0.5))
    context = RiskCheckContext(
        account_equity=100.0,
        positions={"AAPL": 1.0},
        market_data={"AAPL": _series("AAPL", [50.0])},
    )
    res = LeverageCheck().check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config)
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "PORTFOLIO.MAX_LEVERAGE" in res.data.reason_codes


def test_drawdown_check_blocks_when_drawdown_exceeds_limit() -> None:
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_drawdown_pct=0.2))
    context = RiskCheckContext(
        account_equity=70.0,
        portfolio_snapshot={"peak_equity": 100.0},
    )
    res = DrawdownCheck().check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config)
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "PORTFOLIO.DRAWDOWN_LIMIT" in res.data.reason_codes


def test_daily_loss_check_triggers_safe_mode_on_limit_breach() -> None:
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_daily_loss_pct=0.02))
    context = RiskCheckContext(
        account_equity=95.0,
        portfolio_snapshot={"day_start_equity": 100.0},
    )
    res = DailyLossCheck().check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config)
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "PORTFOLIO.DAILY_LOSS_LIMIT" in res.data.reason_codes
    assert res.data.details.get("safe_mode_trigger") is True


def test_concentration_check_blocks_when_projected_concentration_exceeds_limit() -> None:
    config = RiskGateConfig(portfolio=msgspec.structs.replace(RiskGateConfig().portfolio, max_concentration_pct=0.25))
    context = RiskCheckContext(
        account_equity=100.0,
        positions={"AAPL": 0.0},
        market_data={"AAPL": _series("AAPL", [10.0])},
    )
    res = ConcentrationCheck().check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=30), context, config)
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "PORTFOLIO.CONCENTRATION_LIMIT" in res.data.reason_codes


def test_max_position_check_blocks_when_projected_position_exceeds_limit() -> None:
    config = RiskGateConfig(symbol=msgspec.structs.replace(RiskGateConfig().symbol, max_position_size=10.0))
    context = RiskCheckContext(account_equity=100.0, positions={"AAPL": 5.0})
    res = MaxPositionCheck().check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=6), context, config)
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "SYMBOL.MAX_POSITION" in res.data.reason_codes


def test_price_band_check_fails_closed_on_missing_market_data() -> None:
    res = PriceBandCheck().check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=100.0), RiskCheckContext(account_equity=100.0), RiskGateConfig())
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "MISSING_MARKET_DATA" in res.data.reason_codes


def test_price_band_check_blocks_when_deviation_exceeds_threshold() -> None:
    config = RiskGateConfig(symbol=msgspec.structs.replace(RiskGateConfig().symbol, max_price_band_bps=250.0))
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", [100.0])})
    res = PriceBandCheck().check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=103.0), context, config)
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "SYMBOL.PRICE_BAND" in res.data.reason_codes


def test_volatility_halt_check_triggers_safe_mode() -> None:
    config = RiskGateConfig(symbol=msgspec.structs.replace(RiskGateConfig().symbol, max_volatility_z_score=3.0))
    closes = [100.0] * 25 + [120.0]
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", closes)})
    res = VolatilityHaltCheck(lookback_returns=20).check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=120.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "SYMBOL.VOLATILITY_HALT" in res.data.reason_codes
    assert res.data.details.get("safe_mode_trigger") is True


def test_reduce_only_check_blocks_on_position_increase() -> None:
    intent = TradeIntent(
        intent_id="1",
        symbol="AAPL",
        intent_type=IntentType.OPEN_LONG,
        quantity=1.0,
        created_at_ns=1,
        entry_price=100.0,
        reduce_only=True,
    )
    context = RiskCheckContext(account_equity=100.0, positions={"AAPL": 10.0})
    res = ReduceOnlyCheck().check(intent, context, RiskGateConfig())
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "SYMBOL.REDUCE_ONLY_VIOLATION" in res.data.reason_codes


def test_order_count_check_blocks_when_exceeding_limit() -> None:
    config = RiskGateConfig(operational=msgspec.structs.replace(RiskGateConfig().operational, max_orders_per_run=2))
    check = OrderCountCheck()
    context = RiskCheckContext(account_equity=100.0)
    assert check.check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config).data.status == CheckStatus.PASS
    assert check.check(_intent(intent_id="2", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config).data.status == CheckStatus.PASS
    assert check.check(_intent(intent_id="3", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config).data.status == CheckStatus.FAIL


def test_rate_limit_check_blocks_when_exceeding_limit() -> None:
    config = RiskGateConfig(operational=msgspec.structs.replace(RiskGateConfig().operational, max_order_count=2))
    check = RateLimitCheck()
    context = RiskCheckContext(account_equity=100.0)
    assert check.check(_intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config).data.status == CheckStatus.PASS
    assert check.check(_intent(intent_id="2", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config).data.status == CheckStatus.PASS
    assert check.check(_intent(intent_id="3", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1), context, config).data.status == CheckStatus.FAIL


def _series_with_tr(*, symbol: str, close: float, tr_abs: float, bars_count: int = 20) -> PriceSeriesSnapshot:
    snapshot = _series(symbol, [close] * bars_count)
    half_range = float(tr_abs) / 2.0
    bars = [
        PriceBar(
            timestamp=i + 1,
            open=float(close),
            high=float(close) + half_range,
            low=float(close) - half_range,
            close=float(close),
            volume=1000,
        )
        for i in range(bars_count)
    ]
    return msgspec.structs.replace(snapshot, bars=bars)


def test_price_band_check_dynamic_threshold_happy_path() -> None:
    config = RiskGateConfig(
        symbol=msgspec.structs.replace(
            RiskGateConfig().symbol,
            use_dynamic_price_band=True,
            max_price_band_bps=250.0,
            atr_multiplier=3.0,
            atr_period=14,
        )
    )
    context = RiskCheckContext(
        account_equity=100.0,
        market_data={"AAPL": _series_with_tr(symbol="AAPL", close=100.0, tr_abs=2.0, bars_count=20)},
    )
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=103.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.reason_codes == []
    assert res.data.details["dev_bps"] == pytest.approx(300.0)
    assert res.data.details["max_price_band_bps"] == pytest.approx(600.0)
    assert "dynamic_price_band" in res.data.details
    dpb = res.data.details["dynamic_price_band"]
    assert isinstance(dpb, dict)
    assert dpb["enabled"] is True
    assert dpb["atr_pct"] == pytest.approx(0.02)
    assert dpb["atr_multiplier"] == pytest.approx(3.0)
    assert dpb["dynamic_bps"] == pytest.approx(600.0)
    assert dpb["effective_bps"] == pytest.approx(600.0)
    assert dpb["base_bps"] == pytest.approx(250.0)


def test_price_band_check_dynamic_threshold_blocks_excessive_deviation() -> None:
    config = RiskGateConfig(
        symbol=msgspec.structs.replace(
            RiskGateConfig().symbol,
            use_dynamic_price_band=True,
            max_price_band_bps=150.0,
            atr_multiplier=2.0,
            atr_period=14,
        )
    )
    context = RiskCheckContext(
        account_equity=100.0,
        market_data={"AAPL": _series_with_tr(symbol="AAPL", close=100.0, tr_abs=1.0, bars_count=20)},
    )
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=105.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.FAIL
    assert "SYMBOL.PRICE_BAND" in res.data.reason_codes
    assert res.data.details["dev_bps"] == pytest.approx(500.0)
    assert res.data.details["max_price_band_bps"] == pytest.approx(200.0)
    dpb = res.data.details["dynamic_price_band"]
    assert dpb["enabled"] is True
    assert dpb["atr_pct"] == pytest.approx(0.01)
    assert dpb["atr_multiplier"] == pytest.approx(2.0)
    assert dpb["dynamic_bps"] == pytest.approx(200.0)
    assert dpb["effective_bps"] == pytest.approx(200.0)


def test_price_band_check_dynamic_fallback_on_insufficient_data() -> None:
    config = RiskGateConfig(
        symbol=msgspec.structs.replace(
            RiskGateConfig().symbol,
            use_dynamic_price_band=True,
            max_price_band_bps=250.0,
            atr_multiplier=3.0,
            atr_period=14,
        )
    )
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", [100.0] * 5)})
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=102.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["dev_bps"] == pytest.approx(200.0)
    assert res.data.details["max_price_band_bps"] == pytest.approx(250.0)
    dpb = res.data.details["dynamic_price_band"]
    assert dpb["enabled"] is True
    assert dpb["atr_calculation_failed"] is True
    assert dpb["fallback_to_base"] is True
    assert dpb["effective_bps"] == pytest.approx(250.0)


def test_price_band_check_dynamic_fallback_on_missing_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    import risk_gate.checks as checks

    monkeypatch.setattr(checks, "_get_symbol_reference_price", lambda _context, _symbol: 100.0)

    class MarketDataWithoutBars:
        pass

    config = RiskGateConfig(
        symbol=msgspec.structs.replace(
            RiskGateConfig().symbol,
            use_dynamic_price_band=True,
            max_price_band_bps=250.0,
            atr_multiplier=3.0,
            atr_period=14,
        )
    )
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": MarketDataWithoutBars()})
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=102.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["dev_bps"] == pytest.approx(200.0)
    assert res.data.details["max_price_band_bps"] == pytest.approx(250.0)
    dpb = res.data.details["dynamic_price_band"]
    assert dpb["enabled"] is True
    assert dpb["insufficient_bars"] is True
    assert dpb["fallback_to_base"] is True
    assert dpb["effective_bps"] == pytest.approx(250.0)


def test_price_band_check_backward_compatible_when_disabled() -> None:
    config = RiskGateConfig(
        symbol=msgspec.structs.replace(
            RiskGateConfig().symbol,
            use_dynamic_price_band=False,
            max_price_band_bps=250.0,
        )
    )
    context = RiskCheckContext(account_equity=100.0, market_data={"AAPL": _series("AAPL", [100.0] * 20)})
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=102.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["dev_bps"] == pytest.approx(200.0)
    assert res.data.details["max_price_band_bps"] == pytest.approx(250.0)
    assert "dynamic_price_band" not in res.data.details


def test_price_band_check_dynamic_threshold_widens_for_volatile_stocks() -> None:
    config = RiskGateConfig(
        symbol=msgspec.structs.replace(
            RiskGateConfig().symbol,
            use_dynamic_price_band=True,
            max_price_band_bps=250.0,
            atr_multiplier=3.0,
            atr_period=14,
        )
    )
    context = RiskCheckContext(
        account_equity=100.0,
        market_data={"AAPL": _series_with_tr(symbol="AAPL", close=100.0, tr_abs=4.0, bars_count=20)},
    )
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=108.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["dev_bps"] == pytest.approx(800.0)
    assert res.data.details["max_price_band_bps"] == pytest.approx(1200.0)
    dpb = res.data.details["dynamic_price_band"]
    assert dpb["enabled"] is True
    assert dpb["atr_pct"] == pytest.approx(0.04)
    assert dpb["atr_multiplier"] == pytest.approx(3.0)
    assert dpb["dynamic_bps"] == pytest.approx(1200.0)
    assert dpb["effective_bps"] == pytest.approx(1200.0)


def test_price_band_check_dynamic_threshold_floor_is_base() -> None:
    config = RiskGateConfig(
        symbol=msgspec.structs.replace(
            RiskGateConfig().symbol,
            use_dynamic_price_band=True,
            max_price_band_bps=250.0,
            atr_multiplier=2.0,
            atr_period=14,
        )
    )
    context = RiskCheckContext(
        account_equity=100.0,
        market_data={"AAPL": _series_with_tr(symbol="AAPL", close=100.0, tr_abs=0.5, bars_count=20)},
    )
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=102.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["dev_bps"] == pytest.approx(200.0)
    assert res.data.details["max_price_band_bps"] == pytest.approx(250.0)
    dpb = res.data.details["dynamic_price_band"]
    assert dpb["enabled"] is True
    assert dpb["atr_pct"] == pytest.approx(0.005)
    assert dpb["atr_multiplier"] == pytest.approx(2.0)
    assert dpb["dynamic_bps"] == pytest.approx(100.0)
    assert dpb["effective_bps"] == pytest.approx(250.0)
    assert dpb["base_bps"] == pytest.approx(250.0)


def test_price_band_check_dynamic_fallback_on_atr_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import indicators.interface as indicators

    monkeypatch.setattr(indicators, "compute_atr_last", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    config = RiskGateConfig(
        symbol=msgspec.structs.replace(
            RiskGateConfig().symbol,
            use_dynamic_price_band=True,
            max_price_band_bps=250.0,
            atr_multiplier=3.0,
            atr_period=14,
        )
    )
    context = RiskCheckContext(
        account_equity=100.0,
        market_data={"AAPL": _series_with_tr(symbol="AAPL", close=100.0, tr_abs=2.0, bars_count=20)},
    )
    res = PriceBandCheck().check(
        _intent(intent_id="1", symbol="AAPL", intent_type=IntentType.OPEN_LONG, quantity=1, entry_price=102.0),
        context,
        config,
    )
    assert res.data is not None
    assert res.data.status == CheckStatus.PASS
    assert res.data.details["max_price_band_bps"] == pytest.approx(250.0)
    dpb = res.data.details["dynamic_price_band"]
    assert dpb["enabled"] is True
    assert dpb["atr_calculation_failed"] is True
    assert "boom" in str(dpb.get("error"))
    assert dpb["fallback_to_base"] is True
    assert dpb["effective_bps"] == pytest.approx(250.0)
