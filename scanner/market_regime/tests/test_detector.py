from __future__ import annotations

import math

import pytest

from common.interface import ResultStatus
from scanner.market_regime import detector as detector_module
from scanner.market_regime.detector import MarketRegimeDetector
from scanner.market_regime.interface import MarketRegime

from .conftest import make_series_snapshot


def test__extract_bars_returns_empty_on_error() -> None:
    class _BadSnapshot:
        @property
        def bars(self) -> list[object]:
            raise RuntimeError("boom")

    assert detector_module._extract_bars(_BadSnapshot()) == []


def test__get_float_attr_returns_nan_on_bad_value() -> None:
    class _Bar:
        high = "nope"

    assert math.isnan(detector_module._get_float_attr(_Bar(), "high"))


def test__get_int_attr_returns_zero_on_bad_value() -> None:
    class _Bar:
        volume = "nope"

    assert detector_module._get_int_attr(_Bar(), "volume") == 0


def test__std_returns_nan_for_empty_values() -> None:
    assert math.isnan(detector_module._std([]))


def test__compute_volatility_last_raises_on_invalid_lookback() -> None:
    with pytest.raises(ValueError, match="lookback must be > 1"):
        detector_module._compute_volatility_last([100.0, 101.0], lookback=1)


def test__compute_ma50_slope_raises_on_invalid_slope_days() -> None:
    with pytest.raises(ValueError, match="slope_days must be > 0"):
        detector_module._compute_ma50_slope([100.0] * 100, slope_days=0)


def test__compute_ma50_slope_requires_enough_data() -> None:
    assert detector_module._compute_ma50_slope(list(range(40))) is None


def test__compute_volatility_last_requires_enough_data() -> None:
    assert detector_module._compute_volatility_last([100.0] * 10, lookback=20) is None


def test__compute_adx_last_returns_finite_value() -> None:
    closes = [100.0 + i for i in range(60)]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    adx = detector_module._compute_adx_last(highs, lows, closes, period=14)
    assert adx is not None
    assert math.isfinite(adx)
    assert 0.0 <= adx <= 100.0


def test__compute_adx_last_returns_none_for_nonfinite_inputs() -> None:
    closes = [100.0 + i for i in range(40)]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    highs[10] = float("nan")
    assert detector_module._compute_adx_last(highs, lows, closes, period=14) is None


def test__compute_adx_last_handles_zero_directional_movement() -> None:
    highs = [100.0 for _ in range(80)]
    lows = [100.0 for _ in range(80)]
    closes = [100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(80)]
    adx = detector_module._compute_adx_last(highs, lows, closes, period=14)
    assert adx is not None
    assert math.isfinite(adx)


def test__compute_adx_last_returns_none_when_tr_is_zero() -> None:
    highs = [0.0 for _ in range(40)]
    lows = [0.0 for _ in range(40)]
    closes = [0.0 for _ in range(40)]
    assert detector_module._compute_adx_last(highs, lows, closes, period=14) is None


def test_detector_returns_failed_when_no_market_data() -> None:
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({})
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NO_MARKET_INDEX_DATA"


def test_detector_returns_failed_for_insufficient_bars() -> None:
    snapshot = make_series_snapshot("SPY", closes=[100.0 + i for i in range(10)])
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"SPY": snapshot})
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BARS"


def test_detector_returns_failed_for_nonfinite_ohlc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    closes = [100.0 + i for i in range(80)]
    closes[10] = float("nan")
    snapshot = make_series_snapshot("SPY", closes=closes)
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"SPY": snapshot})
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NONFINITE_OHLC"


def test_detector_returns_failed_when_adx_computation_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: None)
    snapshot = make_series_snapshot("SPY", closes=[100.0 + i for i in range(80)])
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"SPY": snapshot})
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ADX_FAILED"


def test_detector_classifies_choppy_when_adx_low_and_slope_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHOPPY requires both low ADX AND flat MA slope (new priority: MA slope > ADX)."""
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 10.0)
    monkeypatch.setattr(detector_module, "_compute_ma50_slope", lambda *args, **kwargs: 0.005)  # flat slope
    snapshot = make_series_snapshot("SPY", closes=[100.0 + i for i in range(80)])
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"SPY": snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.CHOPPY


def test_detector_classifies_bull_when_adx_low_but_slope_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """BULL even with low ADX if MA slope is strongly positive (narrow bull market)."""
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 10.0)
    snapshot = make_series_snapshot("SPY", closes=[100.0 + i for i in range(80)])  # positive slope
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"SPY": snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.BULL  # MA slope > 0.01 → BULL


def test_detector_classifies_bull_when_trending_and_slope_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(80)])
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"SPY": snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.BULL


def test_detector_classifies_bear_when_trending_and_slope_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    snapshot = make_series_snapshot("SPY", closes=[140.0 - i * 0.5 for i in range(80)])
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"SPY": snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.BEAR


def test_detector_returns_unknown_when_ma_slope_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    snapshot = make_series_snapshot("SPY", closes=[100.0 + i for i in range(40)])
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"SPY": snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.UNKNOWN


def test_detector_classifies_choppy_when_not_trending_unstable_and_slope_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHOPPY when ADX moderate, high volatility, AND MA slope flat."""
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 22.0)
    monkeypatch.setattr(detector_module, "_compute_volatility_last", lambda *args, **kwargs: 0.10)
    monkeypatch.setattr(detector_module, "_compute_ma50_slope", lambda *args, **kwargs: 0.005)  # flat slope
    snapshot = make_series_snapshot("SPY", closes=[100.0 + i for i in range(80)])
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"SPY": snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.CHOPPY


def test_detector_classifies_choppy_when_slope_within_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    monkeypatch.setattr(detector_module, "_compute_ma50_slope", lambda *args, **kwargs: 0.005)
    monkeypatch.setattr(detector_module, "_compute_volatility_last", lambda *args, **kwargs: 0.01)

    snapshot = make_series_snapshot("SPY", closes=[100.0 + i for i in range(80)])
    detector = MarketRegimeDetector(index_symbols=("SPY",), ma_slope_bull_threshold=0.01, ma_slope_bear_threshold=-0.01)
    result = detector.detect({"SPY": snapshot})

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.CHOPPY


def test_detector_selects_first_available_symbol_when_index_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    snapshot = make_series_snapshot("QQQ", closes=[100.0 + i * 0.5 for i in range(80)])
    detector = MarketRegimeDetector(index_symbols=("SPY",))
    result = detector.detect({"QQQ": snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.market_symbol == "QQQ"


def test_detector_applies_tracker_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    from scanner.market_regime.interface import RegimeConfirmationConfig
    from scanner.market_regime.regime_tracker import RegimeTransitionTracker

    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    monkeypatch.setattr(detector_module, "_compute_volatility_last", lambda *args, **kwargs: 0.01)

    slopes = iter([0.02, -0.02, -0.02])
    monkeypatch.setattr(detector_module, "_compute_ma50_slope", lambda *args, **kwargs: next(slopes))

    snapshot = make_series_snapshot("SPY", closes=[100.0 + i for i in range(80)])
    tracker = RegimeTransitionTracker(config=RegimeConfirmationConfig(confirmation_days=2, reset_on_opposite=True))
    detector = MarketRegimeDetector(index_symbols=("SPY",), tracker=tracker)

    first = detector.detect({"SPY": snapshot})
    assert first.status is ResultStatus.SUCCESS
    assert first.data is not None
    assert first.data.regime is MarketRegime.BULL

    second = detector.detect({"SPY": snapshot})
    assert second.status is ResultStatus.SUCCESS
    assert second.data is not None
    # Still BULL while BEAR is pending.
    assert second.data.regime is MarketRegime.BULL

    third = detector.detect({"SPY": snapshot})
    assert third.status is ResultStatus.SUCCESS
    assert third.data is not None
    assert third.data.regime is MarketRegime.BEAR


# === VIX Proxy Detection Tests ===


def test__compute_daily_change_returns_none_for_insufficient_bars() -> None:
    assert detector_module._compute_daily_change([]) is None
    assert detector_module._compute_daily_change([100.0]) is None


def test__compute_daily_change_returns_none_for_nonfinite_values() -> None:
    class _Bar:
        def __init__(self, close: float):
            self.close = close

    bars = [_Bar(float("nan")), _Bar(100.0)]
    assert detector_module._compute_daily_change(bars) is None

    bars = [_Bar(100.0), _Bar(float("inf"))]
    assert detector_module._compute_daily_change(bars) is None


def test__compute_daily_change_returns_none_for_zero_prev_close() -> None:
    class _Bar:
        def __init__(self, close: float):
            self.close = close

    bars = [_Bar(0.0), _Bar(100.0)]
    assert detector_module._compute_daily_change(bars) is None


def test__compute_daily_change_calculates_correct_percentage() -> None:
    class _Bar:
        def __init__(self, close: float):
            self.close = close

    bars = [_Bar(100.0), _Bar(110.0)]
    change = detector_module._compute_daily_change(bars)
    assert change is not None
    assert abs(change - 0.10) < 0.0001  # 10% increase

    bars = [_Bar(100.0), _Bar(90.0)]
    change = detector_module._compute_daily_change(bars)
    assert change is not None
    assert abs(change - (-0.10)) < 0.0001  # 10% decrease


def test_detector_detects_vix_spike_from_proxy_etf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    spy_snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(80)])
    # VIXY with 15% daily increase (above 10% threshold)
    vixy_closes = [20.0 for _ in range(79)] + [23.0]  # 15% spike on last day
    vixy_snapshot = make_series_snapshot("VIXY", closes=vixy_closes)

    detector = MarketRegimeDetector(
        index_symbols=("SPY",),
        vix_proxy_symbols=("VIXY",),
        vix_spike_threshold=0.10,
        vix_spike_forces_choppy=True,
    )
    result = detector.detect({"SPY": spy_snapshot, "VIXY": vixy_snapshot})

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    # VIX spike should override to CHOPPY
    assert result.data.regime is MarketRegime.CHOPPY
    assert result.data.vix_proxy_symbol == "VIXY"
    assert result.data.vix_proxy_change is not None
    assert result.data.vix_proxy_change >= 0.10
    assert result.data.vix_spike_detected is True


def test_detector_does_not_force_choppy_when_vix_spike_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    spy_snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(80)])
    # VIXY with 15% daily increase
    vixy_closes = [20.0 for _ in range(79)] + [23.0]
    vixy_snapshot = make_series_snapshot("VIXY", closes=vixy_closes)

    detector = MarketRegimeDetector(
        index_symbols=("SPY",),
        vix_proxy_symbols=("VIXY",),
        vix_spike_threshold=0.10,
        vix_spike_forces_choppy=False,  # Disabled
    )
    result = detector.detect({"SPY": spy_snapshot, "VIXY": vixy_snapshot})

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    # Should be BULL because VIX spike forcing is disabled
    assert result.data.regime is MarketRegime.BULL
    assert result.data.vix_spike_detected is True


def test_detector_no_vix_spike_when_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    spy_snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(80)])
    # VIXY with only 5% increase (below 10% threshold)
    vixy_closes = [20.0 for _ in range(79)] + [21.0]  # 5% increase
    vixy_snapshot = make_series_snapshot("VIXY", closes=vixy_closes)

    detector = MarketRegimeDetector(
        index_symbols=("SPY",),
        vix_proxy_symbols=("VIXY",),
        vix_spike_threshold=0.10,
        vix_spike_forces_choppy=True,
    )
    result = detector.detect({"SPY": spy_snapshot, "VIXY": vixy_snapshot})

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    # Should be BULL because VIX didn't spike
    assert result.data.regime is MarketRegime.BULL
    assert result.data.vix_proxy_symbol == "VIXY"
    assert result.data.vix_spike_detected is False


def test_detector_vix_proxy_falls_back_to_next_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    spy_snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(80)])
    # VXX with 12% increase (VIXY not available)
    vxx_closes = [15.0 for _ in range(79)] + [16.8]  # 12% spike
    vxx_snapshot = make_series_snapshot("VXX", closes=vxx_closes)

    detector = MarketRegimeDetector(
        index_symbols=("SPY",),
        vix_proxy_symbols=("VIXY", "VXX"),  # VIXY first but not available
        vix_spike_threshold=0.10,
        vix_spike_forces_choppy=True,
    )
    result = detector.detect({"SPY": spy_snapshot, "VXX": vxx_snapshot})

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.CHOPPY
    assert result.data.vix_proxy_symbol == "VXX"
    assert result.data.vix_spike_detected is True


def test_detector_no_vix_data_returns_none_vix_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    spy_snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(80)])

    detector = MarketRegimeDetector(
        index_symbols=("SPY",),
        vix_proxy_symbols=("VIXY", "VXX"),
    )
    # No VIX proxy data provided
    result = detector.detect({"SPY": spy_snapshot})

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    # Should still classify based on ADX/MA50
    assert result.data.regime is MarketRegime.BULL
    # VIX fields should be None/False
    assert result.data.vix_proxy_symbol is None
    assert result.data.vix_proxy_change is None
    assert result.data.vix_spike_detected is False
