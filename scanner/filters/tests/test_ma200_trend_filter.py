from __future__ import annotations

from types import SimpleNamespace

from common.interface import ResultStatus
from data.interface import PriceBar
from scanner.filters import MA200TrendFilter
from scanner.interface import MA200TrendFilterConfig, ScannerConfig


def _bars(*closes: float) -> list[PriceBar]:
    bars: list[PriceBar] = []
    for idx, close in enumerate(closes):
        value = float(close)
        bars.append(
            PriceBar(
                timestamp=idx,
                open=value,
                high=value,
                low=value,
                close=value,
                volume=1,
            )
        )
    return bars


def _config(
    *,
    enabled: bool = True,
    period: int = 5,
    fallback_periods: list[int] | None = None,
    require_above: bool = True,
    tolerance_pct: float = 0.02,
) -> ScannerConfig:
    if fallback_periods is None:
        derived_fallback = period // 2
        fallback_periods = [period] if derived_fallback <= 0 else [period, derived_fallback]
    return ScannerConfig(
        ma200_trend_filter=MA200TrendFilterConfig(
            enabled=enabled,
            period=period,
            fallback_periods=fallback_periods,
            require_above=require_above,
            tolerance_pct=tolerance_pct,
        )
    )


def test_passes_when_close_above_ma() -> None:
    bars = _bars(100, 100, 100, 100, 102)
    result = MA200TrendFilter().apply(bars, _config(period=5, tolerance_pct=0.0))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.reason_code == "OK"
    assert result.data.features["ma"] == (100 + 100 + 100 + 100 + 102) / 5.0
    assert result.data.features["close"] == 102.0
    assert result.data.metadata["period"] == 5


def test_passes_when_close_within_tolerance_below_ma() -> None:
    bars = _bars(100, 100, 100, 100, 99)
    result = MA200TrendFilter().apply(bars, _config(period=5, tolerance_pct=0.02))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.reason_code == "OK"


def test_fails_when_close_below_ma_beyond_tolerance() -> None:
    bars = _bars(100, 100, 100, 100, 97)
    result = MA200TrendFilter().apply(bars, _config(period=5, tolerance_pct=0.02))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "MA_BELOW_TREND"
    assert result.data.reason == "MA_BELOW_TREND"


def test_requires_below_when_require_above_false() -> None:
    pass_bars = _bars(100, 100, 100, 100, 101)
    pass_result = MA200TrendFilter().apply(pass_bars, _config(period=5, require_above=False, tolerance_pct=0.02))
    assert pass_result.status is ResultStatus.SUCCESS
    assert pass_result.data is not None
    assert pass_result.data.passed is True

    fail_bars = _bars(100, 100, 100, 100, 103)
    fail_result = MA200TrendFilter().apply(fail_bars, _config(period=5, require_above=False, tolerance_pct=0.02))
    assert fail_result.status is ResultStatus.SUCCESS
    assert fail_result.data is not None
    assert fail_result.data.passed is False
    assert fail_result.reason_code == "MA_BELOW_TREND"


def test_falls_back_to_shorter_ma_when_primary_period_insufficient() -> None:
    bars = _bars(100, 100, 100, 100)
    result = MA200TrendFilter().apply(bars, _config(period=5, fallback_periods=[5, 3]))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.reason_code == "OK"
    assert result.data.metadata["period"] == 3


def test_insufficient_bars_fails_when_all_fallback_periods_insufficient() -> None:
    bars = _bars(100, 100, 100)
    result = MA200TrendFilter().apply(bars, _config(period=5, fallback_periods=[5, 4]))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "INSUFFICIENT_BARS"
    assert result.data.reason == "INSUFFICIENT_BARS"
    assert result.data.metadata["required_periods"] == [5, 4]


def test_invalid_period_fails() -> None:
    bars = _bars(100, 100, 100, 100, 100)
    result = MA200TrendFilter().apply(bars, _config(period=0))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_MA_PERIOD"


def test_invalid_tolerance_pct_fails() -> None:
    bars = _bars(100, 100, 100, 100, 100)
    result = MA200TrendFilter().apply(bars, _config(tolerance_pct=-0.01))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_TOLERANCE_PCT"


def test_disabled_filter_passes() -> None:
    bars = _bars(100, 100, 100, 100, 97)
    result = MA200TrendFilter(enabled=False).apply(bars, _config(period=5))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.reason_code == "FILTER_DISABLED"


def test_missing_config_is_treated_as_disabled() -> None:
    bars = _bars(100, 100, 100, 100, 97)
    config = SimpleNamespace(ma200_trend_filter=None)
    result = MA200TrendFilter().apply(bars, config)  # type: ignore[arg-type]
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.reason_code == "OK"
