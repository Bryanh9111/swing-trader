from __future__ import annotations

import msgspec
import pytest

from common.interface import ResultStatus
from data.interface import PriceSeriesSnapshot
from event_guard.interface import TradeConstraints
from portfolio.interface import AllocationResult
from strategy.interface import StrategyEngineConfig, TransactionCostConfig
from strategy.sizing import (
    AdaptivePositionSizer,
    FixedPercentSizer,
    FixedRiskSizer,
    QualityScaledSizer,
    VolatilityScaledSizer,
    create_position_sizer,
)


@pytest.mark.parametrize(
    "entry_price",
    [
        0.0,
        -1.0,
    ],
)
def test_fixed_percent_sizer_negative_entry_price_fails(
    default_config: StrategyEngineConfig,
    entry_price: float,
) -> None:
    result = FixedPercentSizer().calculate_size(
        "AAPL",
        entry_price,
        0.0,
        100_000.0,
        default_config,
        None,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_INPUTS"


@pytest.mark.parametrize(
    "account_equity",
    [
        0.0,
        -1.0,
    ],
)
def test_fixed_percent_sizer_negative_equity_fails(
    default_config: StrategyEngineConfig,
    account_equity: float,
) -> None:
    result = FixedPercentSizer().calculate_size(
        "AAPL",
        50.0,
        0.0,
        account_equity,
        default_config,
        None,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_INPUTS"


def test_fixed_percent_sizer_happy_path(default_config: StrategyEngineConfig) -> None:
    config = StrategyEngineConfig(
        position_size_pct=default_config.position_size_pct,
        max_position_pct=default_config.max_position_pct,
    )
    result = FixedPercentSizer().calculate_size("AAPL", 50.0, 0.0, 100_000.0, config, None)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(40.0)


def test_fixed_percent_sizer_applies_max_position_pct_cap() -> None:
    config = StrategyEngineConfig(position_size_pct=0.5, max_position_pct=0.1)
    result = FixedPercentSizer().calculate_size("AAPL", 50.0, 0.0, 100_000.0, config, None)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(200.0)


def test_fixed_percent_sizer_applies_constraints_max_position_size(sample_constraints: TradeConstraints) -> None:
    config = StrategyEngineConfig(position_size_pct=0.5, max_position_pct=1.0)
    constraints = TradeConstraints(
        symbol=sample_constraints.symbol,
        can_open_new=sample_constraints.can_open_new,
        can_increase=sample_constraints.can_increase,
        can_decrease=sample_constraints.can_decrease,
        max_position_size=50.0,
        no_trade_windows=sample_constraints.no_trade_windows,
        reason_codes=sample_constraints.reason_codes,
    )
    result = FixedPercentSizer().calculate_size("AAPL", 50.0, 0.0, 100_000.0, config, constraints)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(50.0)


def test_fixed_percent_sizer_zero_shares_fails(default_config: StrategyEngineConfig) -> None:
    config = StrategyEngineConfig(position_size_pct=0.02, max_position_pct=1.0)
    result = FixedPercentSizer().calculate_size("AAPL", 100.0, 0.0, 10.0, config, None)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_SHARES"


@pytest.mark.parametrize("entry_price", [0.0, -1.0])
def test_fixed_risk_sizer_negative_entry_fails(
    default_config: StrategyEngineConfig,
    entry_price: float,
) -> None:
    result = FixedRiskSizer().calculate_size(
        "AAPL",
        entry_price,
        45.0,
        100_000.0,
        default_config,
        None,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_INPUTS"


@pytest.mark.parametrize("stop_loss_price", [0.0, -1.0])
def test_fixed_risk_sizer_negative_stop_fails(
    default_config: StrategyEngineConfig,
    stop_loss_price: float,
) -> None:
    result = FixedRiskSizer().calculate_size(
        "AAPL",
        50.0,
        stop_loss_price,
        100_000.0,
        default_config,
        None,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_INPUTS"


@pytest.mark.parametrize("account_equity", [0.0, -1.0])
def test_fixed_risk_sizer_negative_equity_fails(
    default_config: StrategyEngineConfig,
    account_equity: float,
) -> None:
    result = FixedRiskSizer().calculate_size(
        "AAPL",
        50.0,
        45.0,
        account_equity,
        default_config,
        None,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_INPUTS"


def test_fixed_risk_sizer_happy_path() -> None:
    config = StrategyEngineConfig(risk_per_trade_pct=0.01, max_position_pct=0.01)
    result = FixedRiskSizer().calculate_size("AAPL", 50.0, 45.0, 100_000.0, config, None)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(20.0)


def test_fixed_risk_sizer_zero_risk_distance_fails(default_config: StrategyEngineConfig) -> None:
    result = FixedRiskSizer().calculate_size("AAPL", 50.0, 50.0, 100_000.0, default_config, None)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_RISK_DISTANCE"


def test_fixed_risk_sizer_applies_max_position_pct_cap() -> None:
    config = StrategyEngineConfig(risk_per_trade_pct=0.01, max_position_pct=0.005)
    result = FixedRiskSizer().calculate_size("AAPL", 50.0, 45.0, 100_000.0, config, None)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(10.0)


def test_fixed_risk_sizer_applies_constraints_max_position_size(sample_constraints: TradeConstraints) -> None:
    config = StrategyEngineConfig(risk_per_trade_pct=0.01, max_position_pct=1.0)
    constraints = TradeConstraints(
        symbol=sample_constraints.symbol,
        can_open_new=sample_constraints.can_open_new,
        can_increase=sample_constraints.can_increase,
        can_decrease=sample_constraints.can_decrease,
        max_position_size=10.0,
        no_trade_windows=sample_constraints.no_trade_windows,
        reason_codes=sample_constraints.reason_codes,
    )
    result = FixedRiskSizer().calculate_size("AAPL", 50.0, 45.0, 100_000.0, config, constraints)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(10.0)


def test_fixed_risk_sizer_zero_shares_fails() -> None:
    config = StrategyEngineConfig(risk_per_trade_pct=0.0001, max_position_pct=1.0)
    result = FixedRiskSizer().calculate_size("AAPL", 50.0, 1.0, 10.0, config, None)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_SHARES"


def test_fixed_risk_sizer_risk_money_calculation() -> None:
    config = StrategyEngineConfig(risk_per_trade_pct=0.01, max_position_pct=1.0)
    result = FixedRiskSizer().calculate_size("AAPL", 50.0, 49.0, 12_345.0, config, None)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(123.0)


@pytest.mark.parametrize(
    "entry_price",
    [
        0.0,
        -1.0,
    ],
)
def test_volatility_scaled_sizer_negative_entry_fails(
    default_config: StrategyEngineConfig,
    entry_price: float,
) -> None:
    result = VolatilityScaledSizer().calculate_size(
        "AAPL",
        entry_price,
        0.0,
        100_000.0,
        default_config,
        None,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_INPUTS"


@pytest.mark.parametrize("account_equity", [0.0, -1.0])
def test_volatility_scaled_sizer_negative_equity_fails(
    default_config: StrategyEngineConfig,
    account_equity: float,
) -> None:
    result = VolatilityScaledSizer().calculate_size(
        "AAPL",
        50.0,
        0.0,
        account_equity,
        default_config,
        None,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_INPUTS"

def test_volatility_scaled_sizer_happy_path_fallback() -> None:
    config = StrategyEngineConfig(position_size_pct=0.02, max_position_pct=0.1)
    result = VolatilityScaledSizer().calculate_size("AAPL", 50.0, 0.0, 100_000.0, config, None)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(40.0)


def test_volatility_scaled_sizer_applies_max_position_pct_cap() -> None:
    config = StrategyEngineConfig(position_size_pct=0.5, max_position_pct=0.1)
    result = VolatilityScaledSizer().calculate_size("AAPL", 50.0, 0.0, 100_000.0, config, None)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(200.0)


def test_volatility_scaled_sizer_applies_constraints_max_position_size(sample_constraints: TradeConstraints) -> None:
    config = StrategyEngineConfig(position_size_pct=0.5, max_position_pct=1.0)
    constraints = TradeConstraints(
        symbol=sample_constraints.symbol,
        can_open_new=sample_constraints.can_open_new,
        can_increase=sample_constraints.can_increase,
        can_decrease=sample_constraints.can_decrease,
        max_position_size=50.0,
        no_trade_windows=sample_constraints.no_trade_windows,
        reason_codes=sample_constraints.reason_codes,
    )
    result = VolatilityScaledSizer().calculate_size("AAPL", 50.0, 0.0, 100_000.0, config, constraints)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(50.0)


def test_volatility_scaled_sizer_zero_shares_fails() -> None:
    config = StrategyEngineConfig(position_size_pct=0.02, max_position_pct=1.0)
    result = VolatilityScaledSizer().calculate_size("AAPL", 100.0, 0.0, 10.0, config, None)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_SHARES"


def test_volatility_scaled_sizer_uses_atr_pct_when_available(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(
        position_size_pct=0.02,
        risk_per_trade_pct=0.01,
        atr_multiplier=2.0,
        max_position_pct=1.0,
    )
    result = VolatilityScaledSizer().calculate_size(
        "AAPL",
        50.0,
        0.0,
        100_000.0,
        config,
        None,
        market_data=sample_market_data,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(500.0)


def test_volatility_scaled_sizer_invalid_atr_pct_falls_back(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(
        position_size_pct=0.02,
        risk_per_trade_pct=0.01,
        atr_multiplier=2.0,
        max_position_pct=1.0,
    )
    payload = msgspec.to_builtins(sample_market_data)
    payload["atr_pct"] = 0.0
    market_data = type(sample_market_data)(**payload)

    result = VolatilityScaledSizer().calculate_size(
        "AAPL",
        50.0,
        0.0,
        100_000.0,
        config,
        None,
        market_data=market_data,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(40.0)


def test_volatility_scaled_sizer_atr_path_applies_constraints_cap(
    sample_market_data: PriceSeriesSnapshot, sample_constraints: TradeConstraints
) -> None:
    config = StrategyEngineConfig(
        position_size_pct=0.02,
        risk_per_trade_pct=0.01,
        atr_multiplier=2.0,
        max_position_pct=1.0,
    )
    constraints = TradeConstraints(
        symbol=sample_constraints.symbol,
        can_open_new=sample_constraints.can_open_new,
        can_increase=sample_constraints.can_increase,
        can_decrease=sample_constraints.can_decrease,
        max_position_size=25.0,
        no_trade_windows=sample_constraints.no_trade_windows,
        reason_codes=sample_constraints.reason_codes,
    )
    result = VolatilityScaledSizer().calculate_size(
        "AAPL",
        50.0,
        0.0,
        100_000.0,
        config,
        constraints,
        market_data=sample_market_data,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(25.0)


def test_create_fixed_percent_sizer(default_config: StrategyEngineConfig) -> None:
    config = StrategyEngineConfig(position_sizer="fixed_percent")
    sizer = create_position_sizer(config)
    assert isinstance(sizer, FixedPercentSizer)


def test_create_fixed_risk_sizer() -> None:
    config = StrategyEngineConfig(position_sizer="fixed_risk")
    sizer = create_position_sizer(config)
    assert isinstance(sizer, FixedRiskSizer)


def test_create_volatility_scaled_sizer() -> None:
    config = StrategyEngineConfig(position_sizer="volatility_scaled")
    sizer = create_position_sizer(config)
    assert isinstance(sizer, VolatilityScaledSizer)


def _clone_market_data_with_quality_flags(
    market_data: PriceSeriesSnapshot, *, quality_flags: dict[str, object]
) -> PriceSeriesSnapshot:
    payload = msgspec.to_builtins(market_data)
    payload["quality_flags"] = quality_flags
    return type(market_data)(**payload)


def _make_quality_scaled_config(**overrides: object) -> StrategyEngineConfig:
    defaults: dict[str, object] = {
        "position_sizer": "quality_scaled",
        "base_position_pct": 0.05,
        "min_position_pct": 0.025,
        "max_position_pct": 0.20,
        "min_score_threshold": 0.85,
        "quality_threshold_excellent": 0.95,
        "quality_threshold_good": 0.90,
        "quality_threshold_acceptable": 0.85,
        "quality_multiplier_excellent": 2.0,
        "quality_multiplier_good": 1.5,
        "quality_multiplier_acceptable": 1.0,
        "max_commission_ratio": 0.02,
        "transaction_costs": TransactionCostConfig(
            spread_pct=0.001,
            slippage_pct=0.0005,
            commission_per_trade=1.0,
        ),
    }
    return StrategyEngineConfig(**(defaults | overrides))


def test_quality_scaled_sizer_excellent_tier(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test QualityScaledSizer with excellent quality score (≥0.95)"""
    config = _make_quality_scaled_config(base_position_pct=0.05, min_position_pct=0.025)
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.96})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(20.0)


def test_quality_scaled_sizer_good_tier(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test QualityScaledSizer with good quality score (0.90-0.95)"""
    config = _make_quality_scaled_config(base_position_pct=0.05, min_position_pct=0.025)
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.92})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(15.0)


def test_quality_scaled_sizer_acceptable_tier(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test QualityScaledSizer with acceptable quality score (0.85-0.90)"""
    config = _make_quality_scaled_config(base_position_pct=0.05, min_position_pct=0.025)
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.87})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(10.0)


def test_quality_scaled_sizer_min_position_floor(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test that min_position_pct floor is applied"""
    config = _make_quality_scaled_config(base_position_pct=0.001, min_position_pct=0.025)
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.96})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(5.0)


def test_quality_scaled_sizer_max_position_cap(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test that max_position_pct cap is enforced"""
    config = _make_quality_scaled_config(
        base_position_pct=0.20,
        min_position_pct=0.0,
        max_position_pct=0.20,
    )
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.96})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(40.0)


@pytest.mark.parametrize(
    ("score", "expected_shares"),
    [
        (0.95, 20.0),  # exactly excellent threshold -> excellent tier
        (0.90, 15.0),  # exactly good threshold -> good tier
        (0.85, 10.0),  # exactly acceptable threshold -> acceptable tier
    ],
)
def test_quality_scaled_sizer_score_at_threshold(
    sample_market_data: PriceSeriesSnapshot, score: float, expected_shares: float
) -> None:
    """Test behavior when score exactly at threshold boundaries"""
    config = _make_quality_scaled_config(base_position_pct=0.05, min_position_pct=0.025)
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": score})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(expected_shares)


def test_quality_scaled_sizer_commission_ratio_rejection(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test rejection when commission_ratio > max_commission_ratio"""
    config = _make_quality_scaled_config(
        base_position_pct=0.0005,
        min_position_pct=0.0,
        max_position_pct=1.0,
        max_commission_ratio=0.02,
        transaction_costs=TransactionCostConfig(commission_per_trade=1.0),
    )
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.96})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 10.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "COMMISSION_TOO_HIGH"


def test_quality_scaled_sizer_commission_ratio_acceptable(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test acceptance when commission_ratio <= max_commission_ratio"""
    config = _make_quality_scaled_config(
        base_position_pct=0.01,
        min_position_pct=0.0,
        max_position_pct=1.0,
        max_commission_ratio=0.02,
        transaction_costs=TransactionCostConfig(commission_per_trade=1.0),
    )
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.87})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(2.0)


def test_quality_scaled_sizer_missing_score(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test fallback to default score (0.90) when quality_flags missing score"""
    config = _make_quality_scaled_config(base_position_pct=0.05, min_position_pct=0.025)
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(15.0)


def test_quality_scaled_sizer_invalid_score_falls_back(sample_market_data: PriceSeriesSnapshot) -> None:
    config = _make_quality_scaled_config(base_position_pct=0.05, min_position_pct=0.025)
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": "nope"})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(15.0)


def test_quality_scaled_sizer_no_market_data() -> None:
    """Test fallback when market_data is None"""
    config = _make_quality_scaled_config(base_position_pct=0.05, min_position_pct=0.025)
    result = QualityScaledSizer().calculate_size("AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=None)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(15.0)


def test_quality_scaled_sizer_negative_price() -> None:
    """Test rejection of negative entry_price"""
    config = _make_quality_scaled_config()
    result = QualityScaledSizer().calculate_size("AAPL", -10.0, 0.0, 10_000.0, config, None, market_data=None)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_INPUTS"


def test_quality_scaled_sizer_zero_equity() -> None:
    """Test rejection of zero account_equity"""
    config = _make_quality_scaled_config()
    result = QualityScaledSizer().calculate_size("AAPL", 50.0, 0.0, 0.0, config, None, market_data=None)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_INPUTS"


def test_quality_scaled_sizer_below_threshold(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test rejection when score < min_score_threshold"""
    config = _make_quality_scaled_config(min_score_threshold=0.85)
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.80})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_SHARES"


def test_quality_scaled_sizer_below_acceptable_tier(sample_market_data: PriceSeriesSnapshot) -> None:
    config = _make_quality_scaled_config(min_score_threshold=0.80, quality_threshold_acceptable=0.85)
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.84})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_SHARES"


def test_quality_scaled_sizer_zero_shares_after_floor(sample_market_data: PriceSeriesSnapshot) -> None:
    """Test rejection when calculated shares round to zero"""
    config = _make_quality_scaled_config(
        base_position_pct=0.00001,
        min_position_pct=0.0,
        max_position_pct=1.0,
    )
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.87})
    result = QualityScaledSizer().calculate_size(
        "AAPL", 50.0, 0.0, 10_000.0, config, None, market_data=market_data
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_SHARES"


def test_create_quality_scaled_sizer() -> None:
    config = StrategyEngineConfig(position_sizer="quality_scaled")
    sizer = create_position_sizer(config)
    assert isinstance(sizer, QualityScaledSizer)


def test_create_position_sizer_invalid_type_raises() -> None:
    config = StrategyEngineConfig(position_sizer="nope")
    with pytest.raises(ValueError, match="Unknown position_sizer type"):
        create_position_sizer(config)


def test_adaptive_sizer_with_allocation_context() -> None:
    config = StrategyEngineConfig(
        base_position_pct=0.10,
        max_position_pct=1.0,
        transaction_costs=TransactionCostConfig(commission_per_trade=1.0),
    )
    allocation_context = AllocationResult(
        available_capital=10_000.0,
        reserved_capital=0.0,
        max_new_positions=10,
        per_position_budget=1_000.0,
        current_exposure=0.0,
        current_position_count=0,
        total_equity=100_000.0,
    )
    result = AdaptivePositionSizer().calculate_size(
        "AAPL",
        50.0,
        0.0,
        100_000.0,
        config,
        None,
        market_data=None,
        allocation_context=allocation_context,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(20.0)


def test_adaptive_sizer_atr_based_sizing(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(
        base_position_pct=0.50,
        risk_per_trade_pct=0.01,
        atr_multiplier=2.0,
        max_position_pct=1.0,
        quality_scaling_enabled=False,
    )
    result = AdaptivePositionSizer().calculate_size(
        "AAPL",
        50.0,
        0.0,
        100_000.0,
        config,
        None,
        market_data=sample_market_data,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(500.0)


def test_adaptive_sizer_commission_check() -> None:
    config = StrategyEngineConfig(
        base_position_pct=0.01,
        max_position_pct=1.0,
        max_commission_ratio=0.005,
        transaction_costs=TransactionCostConfig(commission_per_trade=1.0),
    )
    result = AdaptivePositionSizer().calculate_size(
        "AAPL",
        10.0,
        0.0,
        10_000.0,
        config,
        None,
        market_data=None,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "COMMISSION_TOO_HIGH"


def test_adaptive_sizer_quality_scaling(sample_market_data: PriceSeriesSnapshot) -> None:
    config = StrategyEngineConfig(
        base_position_pct=0.05,
        risk_per_trade_pct=0.01,
        atr_multiplier=2.0,
        max_position_pct=1.0,
        quality_scaling_enabled=True,
        min_score_threshold=0.85,
        quality_threshold_excellent=0.95,
        quality_threshold_good=0.90,
        quality_threshold_acceptable=0.85,
        quality_multiplier_excellent=2.0,
        quality_multiplier_good=1.5,
        quality_multiplier_acceptable=1.0,
        transaction_costs=TransactionCostConfig(commission_per_trade=1.0),
    )
    market_data = _clone_market_data_with_quality_flags(sample_market_data, quality_flags={"score": 0.96})
    result = AdaptivePositionSizer().calculate_size(
        "AAPL",
        50.0,
        0.0,
        10_000.0,
        config,
        None,
        market_data=market_data,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(20.0)
