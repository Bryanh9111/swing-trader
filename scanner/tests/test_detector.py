from __future__ import annotations

import numpy as np
import pytest

from common.interface import ResultStatus
from scanner import detector as detector_module
from scanner.interface import ScannerConfig

from .conftest import DummyPriceBar, make_bar, make_bars, make_downtrend_series, make_platform_series


def test__as_float_array_success() -> None:
    result = detector_module._as_float_array([1, 2.5, "3"], name="values")
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.dtype == float
    assert result.data.shape == (3,)


@pytest.mark.parametrize(
    ("values", "name", "reason_code"),
    [
        ([], "open", "EMPTY_OPEN_ARRAY"),
        (["x"], "high", "INVALID_HIGH_ARRAY"),
        ([1.0, float("nan")], "close", "NONFINITE_CLOSE"),
    ],
)
def test__as_float_array_failures(values: list[object], name: str, reason_code: str) -> None:
    result = detector_module._as_float_array(values, name=name)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == reason_code
    assert isinstance(result.error, BaseException)


def test__pct_change_success() -> None:
    values = np.asarray([100.0, 110.0, 99.0], dtype=float)
    result = detector_module._pct_change(values)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.shape == (2,)
    assert result.data[0] == pytest.approx(0.10)
    assert result.data[1] == pytest.approx(-0.10)


@pytest.mark.parametrize(
    ("values", "reason_code"),
    [
        (np.asarray([100.0], dtype=float), "INSUFFICIENT_VALUES"),
        (np.asarray([0.0, 1.0], dtype=float), "ZERO_PREV_VALUE"),
    ],
)
def test__pct_change_failures(values: np.ndarray, reason_code: str) -> None:
    result = detector_module._pct_change(values)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == reason_code


def test__safe_mean_success() -> None:
    values = np.asarray([1.0, 2.0, 3.0], dtype=float)
    result = detector_module._safe_mean(values, name="sample")
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("values", "name", "reason_code"),
    [
        (np.asarray([], dtype=float), "sample", "EMPTY_SAMPLE"),
        (np.asarray([float("inf")], dtype=float), "sample", "NONFINITE_SAMPLE_MEAN"),
    ],
)
def test__safe_mean_failures(values: np.ndarray, name: str, reason_code: str) -> None:
    result = detector_module._safe_mean(values, name=name)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == reason_code


def test__coefficient_of_variation_success() -> None:
    values = np.asarray([10.0, 10.0, 10.0], dtype=float)
    result = detector_module._coefficient_of_variation(values, mean_value=10.0, name="volume")
    assert result.status is ResultStatus.SUCCESS
    assert result.data == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("values", "mean_value", "name", "reason_code"),
    [
        (np.asarray([0.0, 0.0], dtype=float), 0.0, "volume", "ZERO_VOLUME_MEAN"),
        (np.asarray([float("inf"), 1.0], dtype=float), 1.0, "volume", "NONFINITE_VOLUME_STD"),
    ],
)
def test__coefficient_of_variation_failures(
    values: np.ndarray, mean_value: float, name: str, reason_code: str
) -> None:
    result = detector_module._coefficient_of_variation(values, mean_value=mean_value, name=name)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == reason_code


def test__window_slice_success() -> None:
    bars = [make_bar(close=100.0, volume=1.0) for _ in range(10)]
    result = detector_module._window_slice(bars, window=5)
    assert result.status is ResultStatus.SUCCESS
    assert result.data == bars[-5:]


@pytest.mark.parametrize(
    ("window", "expected_reason"),
    [(1, "INVALID_WINDOW"), (20, "INSUFFICIENT_BARS")],
)
def test__window_slice_failures(window: int, expected_reason: str) -> None:
    bars = [make_bar(close=100.0, volume=1.0) for _ in range(10)]
    result = detector_module._window_slice(bars, window=window)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == expected_reason


def test__extract_ohlcv_success() -> None:
    bars = make_bars([100.0, 101.0], volume=[100.0, 200.0])
    result = detector_module._extract_ohlcv(bars)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert set(result.data.keys()) == {"open", "high", "low", "close", "volume"}
    assert result.data["close"].tolist() == pytest.approx([100.0, 101.0])


def test__extract_ohlcv_failure_from_nonfinite_field() -> None:
    bars = [
        DummyPriceBar(open=100.0, high=101.0, low=99.0, close=float("nan"), volume=100.0),
        DummyPriceBar(open=100.0, high=101.0, low=99.0, close=100.0, volume=100.0),
    ]
    result = detector_module._extract_ohlcv(bars)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CLOSE_EXTRACTION_FAILED"


def test_detect_price_platform_happy_path(platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig) -> None:
    result = detector_module.detect_price_platform(platform_bars, window=30, config=scanner_config)
    assert result.status is ResultStatus.SUCCESS
    is_platform, features = result.data
    assert is_platform is True
    assert 0 <= features["box_range"] <= scanner_config.box_threshold
    assert features["box_low"] > 0
    assert features["box_high"] > features["box_low"]
    assert 0 <= features["ma_diff"] <= scanner_config.ma_diff_threshold
    assert 0 <= features["volatility"] <= scanner_config.volatility_threshold


@pytest.mark.parametrize("window", [1, 999])
def test_detect_price_platform_invalid_or_insufficient_window_returns_failed(window: int, scanner_config: ScannerConfig) -> None:
    bars = make_platform_series(total_bars=120)
    result = detector_module.detect_price_platform(bars, window=window, config=scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_PRICE_BARS"


def test_detect_price_platform_empty_bars_returns_failed(scanner_config: ScannerConfig) -> None:
    result = detector_module.detect_price_platform([], window=20, config=scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_PRICE_BARS"


def test_detect_price_platform_threshold_boundary_is_inclusive(scanner_config: ScannerConfig) -> None:
    config = ScannerConfig(box_threshold=0.10, ma_diff_threshold=0.02, volatility_threshold=0.025)
    window_bars = [
        make_bar(close=105.0, open=105.0, high=110.0, low=100.0, volume=1_000_000.0)
        for _ in range(30)
    ]
    result = detector_module.detect_price_platform(window_bars, window=30, config=config)
    assert result.status is ResultStatus.SUCCESS
    is_platform, features = result.data
    assert is_platform is True
    assert features["box_range"] == pytest.approx(0.10)


def test_detect_price_platform_ohlcv_error_path(scanner_config: ScannerConfig) -> None:
    bars = [
        DummyPriceBar(open=100.0, high=101.0, low=99.0, close=float("nan"), volume=100.0)
        for _ in range(30)
    ]
    result = detector_module.detect_price_platform(bars, window=30, config=scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "PRICE_OHLCV_FAILED"


def test_detect_volume_platform_happy_path(platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig) -> None:
    result = detector_module.detect_volume_platform(platform_bars, window=30, config=scanner_config)
    assert result.status is ResultStatus.SUCCESS
    volume_ok, features = result.data
    assert volume_ok is True
    assert 0 <= features["volume_change_ratio"] <= scanner_config.volume_change_threshold
    assert 0 <= features["volume_stability"] <= scanner_config.volume_stability_threshold
    assert features["volume_stability_robust"] >= 0
    assert features["volume_trend"] == pytest.approx(features["volume_change_ratio"] - 1.0)
    assert 0.0 <= features["trend_score"] <= 1.0
    assert 0.0 <= features["volume_quality"] <= 1.0
    assert features["avg_dollar_volume"] > 0


def test_detect_volume_platform_insufficient_bars_returns_failed(scanner_config: ScannerConfig) -> None:
    bars = make_bars([100.0] * 10, volume=1_000_000.0)
    result = detector_module.detect_volume_platform(bars, window=30, config=scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_VOLUME_BARS"


def test_detect_volume_platform_invalid_window_returns_failed(scanner_config: ScannerConfig) -> None:
    bars = make_platform_series(total_bars=120)
    result = detector_module.detect_volume_platform(bars, window=1, config=scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_WINDOW"


def test_detect_volume_platform_zero_previous_volume_mean_returns_failed(scanner_config: ScannerConfig) -> None:
    closes = [100.0] * 60
    prev_volumes = [0.0] * 30
    recent_volumes = [1_000_000.0] * 30
    bars = make_bars(closes, volume=prev_volumes + recent_volumes)
    result = detector_module.detect_volume_platform(bars, window=30, config=scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_PREVIOUS_VOLUME_MEAN"


def test_detect_volume_platform_recent_mean_zero_triggers_stability_failed(scanner_config: ScannerConfig) -> None:
    closes = [100.0] * 60
    prev_volumes = [1_000_000.0] * 30
    recent_volumes = [0.0] * 30
    bars = make_bars(closes, volume=prev_volumes + recent_volumes)
    result = detector_module.detect_volume_platform(bars, window=30, config=scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "VOLUME_STABILITY_FAILED"


def test_detect_volume_platform_negative_ratio_returns_failed(scanner_config: ScannerConfig) -> None:
    closes = [100.0] * 60
    prev_volumes = [1_000_000.0] * 30
    recent_volumes = [-1_000_000.0] * 30
    bars = make_bars(closes, volume=prev_volumes + recent_volumes)
    result = detector_module.detect_volume_platform(bars, window=30, config=scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_VOLUME_RATIO"


def test_detect_volume_platform_dollar_volume_nonfinite_returns_failed(scanner_config: ScannerConfig) -> None:
    closes = [1e308] * 60
    volumes = [1e308] * 60
    bars = make_bars(closes, volume=volumes, high_pct=0.0, low_pct=0.0)
    result = detector_module.detect_volume_platform(bars, window=30, config=scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "DOLLAR_VOLUME_FAILED"


def test_calculate_atr_happy_path(platform_bars: list[DummyPriceBar]) -> None:
    result = detector_module.calculate_atr(platform_bars, period=14)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data > 0


@pytest.mark.parametrize(
    ("period", "bars_len", "reason_code"),
    [
        (1, 20, "INVALID_ATR_PERIOD"),
        (14, 10, "INSUFFICIENT_ATR_BARS"),
    ],
)
def test_calculate_atr_validation_failures(period: int, bars_len: int, reason_code: str) -> None:
    bars = make_bars([100.0] * bars_len, volume=1_000_000.0)
    result = detector_module.calculate_atr(bars, period=period)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == reason_code


def test_calculate_atr_ohlcv_failure_returns_failed() -> None:
    bars = [
        DummyPriceBar(open=100.0, high=101.0, low=99.0, close=float("nan"), volume=100.0)
        for _ in range(20)
    ]
    result = detector_module.calculate_atr(bars, period=14)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ATR_OHLCV_FAILED"


def test_calculate_atr_zero_true_range_returns_failed() -> None:
    bars = [
        DummyPriceBar(open=100.0, high=100.0, low=100.0, close=100.0, volume=100.0)
        for _ in range(20)
    ]
    result = detector_module.calculate_atr(bars, period=14)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_ATR"


def test_detect_support_resistance_happy_path(platform_bars: list[DummyPriceBar]) -> None:
    window_bars = list(platform_bars[-30:])
    result = detector_module.detect_support_resistance(window_bars, tolerance=0.02)
    assert result.status is ResultStatus.SUCCESS
    data = result.data
    assert data["support_level"] is not None
    assert data["resistance_level"] is not None
    assert data["resistance_level"] > data["support_level"]
    assert 0.0 <= float(data["box_quality"]) <= 1.0


@pytest.mark.parametrize("tolerance", [0.0, -0.01, 0.2, 0.5])
def test_detect_support_resistance_invalid_tolerance_returns_failed(tolerance: float) -> None:
    bars = make_bars([100.0] * 10, volume=1_000_000.0)
    result = detector_module.detect_support_resistance(bars, tolerance=tolerance)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_TOLERANCE"


def test_detect_support_resistance_insufficient_bars_returns_failed() -> None:
    bars = make_bars([100.0] * 4, volume=1_000_000.0)
    result = detector_module.detect_support_resistance(bars, tolerance=0.02)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BOX_BARS"


def test_detect_support_resistance_ohlcv_failed_returns_failed() -> None:
    bars = [
        DummyPriceBar(open=100.0, high=101.0, low=99.0, close=float("nan"), volume=100.0)
        for _ in range(10)
    ]
    result = detector_module.detect_support_resistance(bars, tolerance=0.02)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "BOX_OHLCV_FAILED"


def test_detect_support_resistance_invalid_support_resistance_returns_failed() -> None:
    bars = [
        DummyPriceBar(open=1.0, high=1.0, low=0.0, close=1.0, volume=100.0) for _ in range(10)
    ]
    result = detector_module.detect_support_resistance(bars, tolerance=0.02)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_SUPPORT_RESISTANCE"


def test_detect_support_resistance_resistance_not_above_support_returns_failed() -> None:
    bars = [
        DummyPriceBar(open=100.0, high=100.0, low=100.0, close=100.0, volume=100.0)
        for _ in range(10)
    ]
    result = detector_module.detect_support_resistance(bars, tolerance=0.02)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_BOX_LEVELS"


def test_detect_platform_candidate_happy_path(platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig) -> None:
    result = detector_module.detect_platform_candidate(
        symbol="AAPL",
        bars=platform_bars,
        window=30,
        config=scanner_config,
        detected_at=123,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    candidate = result.data
    assert candidate.symbol == "AAPL"
    assert candidate.window == 30
    assert candidate.detected_at == 123
    assert 0.0 <= candidate.score <= 1.0
    assert candidate.invalidation_level > 0
    assert candidate.target_level > candidate.invalidation_level
    assert candidate.features.support_level is not None
    assert candidate.features.resistance_level is not None


@pytest.mark.parametrize(
    ("bars_factory", "expected_reason"),
    [
        (lambda: make_downtrend_series(total_bars=120), "PRICE_RULES_NOT_MET"),
        (
            lambda: make_platform_series(
                total_bars=120,
                prev_volume=1_000_000.0,
                recent_volume_high=2_000_000.0,
                recent_volume_low=2_000_000.0,
            ),
            "VOLUME_RULES_NOT_MET",
        ),
    ],
)
def test_detect_platform_candidate_filters_produce_none_with_reason_code(
    bars_factory, expected_reason: str, scanner_config: ScannerConfig
) -> None:
    bars = bars_factory()
    result = detector_module.detect_platform_candidate(
        symbol="TEST",
        bars=bars,
        window=30,
        config=scanner_config,
        detected_at=0,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == expected_reason


def test_detect_platform_candidate_marks_low_liquidity_but_does_not_filter(scanner_config: ScannerConfig) -> None:
    base = make_platform_series(total_bars=120)
    bars: list[DummyPriceBar] = [
        DummyPriceBar(
            open=bar.open,
            high=bar.high,
            low=bar.close * 0.96,
            close=bar.close,
            volume=bar.volume * 0.001,
        )
        for bar in base
    ]

    result = detector_module.detect_platform_candidate(
        symbol="ILLQ",
        bars=bars,
        window=30,
        config=scanner_config,
        detected_at=0,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert isinstance(result.data.meta.get("liquidity_metrics"), dict)
    assert result.data.meta["liquidity_metrics"]["liquidity_score"] < 0.3


def test_detect_platform_candidate_filters_on_atr(scanner_config: ScannerConfig) -> None:
    config = ScannerConfig(min_atr_pct=0.01, max_atr_pct=0.05)
    bars = make_platform_series(total_bars=120, high_pct=0.0002, low_pct=0.0002)

    result = detector_module.detect_platform_candidate(
        symbol="LOWATR",
        bars=bars,
        window=30,
        config=config,
        detected_at=0,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "ATR_FILTERED"


def test_detect_platform_candidate_filters_on_box_quality(scanner_config: ScannerConfig) -> None:
    config = ScannerConfig(
        min_box_quality=0.95,
        min_atr_pct=0.0,
        max_atr_pct=1.0,
    )
    base = make_platform_series(total_bars=120)
    bars: list[DummyPriceBar] = list(base)
    for idx in range(90, 120):
        bar = bars[idx]
        bars[idx] = DummyPriceBar(
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=150.0,  # force containment loss while keeping tight highs/lows
            volume=bar.volume,
        )
    result = detector_module.detect_platform_candidate(
        symbol="BOX",
        bars=bars,
        window=30,
        config=config,
        detected_at=0,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "BOX_QUALITY_FILTERED"


def test_detect_platform_candidate_event_guard_filters_earnings_window(scanner_config: ScannerConfig) -> None:
    bars = make_platform_series(total_bars=120)
    now_ns = 1_700_000_000_000_000_000
    config = ScannerConfig(use_event_guard_filter=True, earnings_window_days=10, require_breakout_confirmation=False, require_breakout_volume_spike=False)
    constraints = {
        "TEST": {
            "no_trade_windows": [(now_ns + 1 * 86_400 * 1_000_000_000, now_ns + 2 * 86_400 * 1_000_000_000)],
            "reason_codes": ["EARNINGS_BLACKOUT", "EVENT_TYPE_EARNINGS"],
        }
    }

    result = detector_module.detect_platform_candidate(
        symbol="TEST",
        bars=bars,
        window=30,
        config=config,
        detected_at=now_ns,
        event_constraints=constraints,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "SKIPPED_EARNINGS_WINDOW"


def test_detect_platform_candidate_market_cap_filter_excludes_micro_cap(scanner_config: ScannerConfig) -> None:
    bars = make_platform_series(total_bars=120)
    config = ScannerConfig(use_market_cap_filter=True)

    result = detector_module.detect_platform_candidate(
        symbol="MICRO",
        bars=bars,
        window=30,
        config=config,
        detected_at=0,
        meta={"market_cap": 100_000_000.0},
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "MICRO_CAP_EXCLUDED"


def test_detect_platform_candidate_insufficient_bars_returns_success_with_none(scanner_config: ScannerConfig) -> None:
    """Insufficient bars should return SUCCESS with None data to avoid backtest failures."""
    bars = make_bars([100.0] * 10, volume=1_000_000.0)
    result = detector_module.detect_platform_candidate(
        symbol="SHORT",
        bars=bars,
        window=30,
        config=scanner_config,
        detected_at=0,
    )
    # Insufficient bars now returns SUCCESS(None) for backtest tolerance
    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "INSUFFICIENT_BARS"


def test_detect_platform_candidate_levels_error_returns_failed(
    platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = detector_module._CandidateInputs(
        price_features={"box_range": 0.02, "box_low": 99.0, "box_high": 101.0, "ma_diff": 0.0, "volatility": 0.0},
        volume_features={"volume_change_ratio": 0.5, "volume_stability": 0.0, "avg_dollar_volume": 100_000_000.0},
        atr_pct=0.02,
        support=float("nan"),
        resistance=101.0,
        box_quality=1.0,
    )

    monkeypatch.setattr(detector_module, "_gather_candidate_inputs", lambda *_args, **_kwargs: detector_module.Result.success(inputs))

    legacy_config = ScannerConfig(use_filter_chain_detector=False)
    result = detector_module.detect_platform_candidate(
        symbol="BADLVL",
        bars=platform_bars,
        window=30,
        config=legacy_config,
        detected_at=0,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "LEVELS_FAILED"
    assert isinstance(result.error, BaseException)


def test_snapshot_to_bars_handles_exceptions() -> None:
    class BadSnapshot:
        @property
        def bars(self) -> list[DummyPriceBar]:
            raise RuntimeError("boom")

    assert detector_module._snapshot_to_bars(BadSnapshot()) == []
