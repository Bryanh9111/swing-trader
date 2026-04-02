from __future__ import annotations

import pytest
import numpy as np

from common.interface import Result, ResultStatus
from scanner import detector as detector_module
from scanner.filters import (
    ATRFilter,
    BoxQualityFilter,
    BreakthroughFilter,
    LowPositionFilter,
    PricePlatformFilter,
    RapidDeclineFilter,
    VolumePlatformFilter,
)
from scanner.interface import ScannerConfig

from .conftest import DummyPriceBar, make_bar, make_bars, make_platform_series


def test_price_platform_filter_happy_path(
    platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig
) -> None:
    filt = PricePlatformFilter(window=30)
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.score == 1.0
    assert set(result.data.features.keys()) >= {"box_range", "ma_diff", "volatility", "ma_values"}


def test_price_platform_filter_propagates_detector_failure(
    platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args, **_kwargs):
        return Result.failed(ValueError("boom"), "DETECTOR_FAILED")

    monkeypatch.setattr(detector_module, "detect_price_platform", _boom)
    filt = PricePlatformFilter(window=30)
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "DETECTOR_FAILED"
    assert isinstance(result.error, BaseException)


def test_price_platform_filter_wraps_detector_function(
    platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, object] = {}

    def _stub(bars, window, config):
        called["bars"] = bars
        called["window"] = window
        called["config"] = config
        return Result.success(
            (
                True,
                {
                    "box_range": 0.01,
                    "ma_diff": 0.01,
                    "volatility": 0.01,
                    "ma_values": {"5": 100.0},
                },
            ),
            reason_code="OK",
        )

    monkeypatch.setattr(detector_module, "detect_price_platform", _stub)
    filt = PricePlatformFilter(window=30)
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert called["window"] == 30
    assert isinstance(called["bars"], list)
    assert called["config"] is scanner_config


def test_volume_platform_filter_happy_path(
    platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig
) -> None:
    filt = VolumePlatformFilter(window=30)
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.score == 1.0
    assert set(result.data.features.keys()) >= {
        "volume_change_ratio",
        "volume_stability",
        "volume_stability_robust",
        "volume_trend",
        "trend_score",
        "volume_quality",
        "avg_dollar_volume",
    }


def test_volume_platform_filter_returns_passed_false_when_rules_not_met(
    platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _stub(*_args, **_kwargs):
        return Result.success(
            (
                False,
                {"volume_change_ratio": 2.0, "volume_stability": 2.0, "avg_dollar_volume": 1.0},
            ),
            reason_code="VOLUME_RULES_NOT_MET",
        )

    monkeypatch.setattr(detector_module, "detect_volume_platform", _stub)
    filt = VolumePlatformFilter(window=30)
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "VOLUME_RULES_NOT_MET"
    assert result.data is not None
    assert result.data.passed is False
    assert result.data.score == 0.0


def test_volume_platform_filter_propagates_detector_failure(
    platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args, **_kwargs):
        return Result.failed(ValueError("boom"), "DETECTOR_FAILED")

    monkeypatch.setattr(detector_module, "detect_volume_platform", _boom)
    filt = VolumePlatformFilter(window=30)
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "DETECTOR_FAILED"


def test_atr_filter_happy_path(platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig) -> None:
    filt = ATRFilter()
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.score == 1.0
    assert set(result.data.features.keys()) >= {"atr_value", "atr_pct", "close_price"}


def test_atr_filter_returns_passed_false_when_out_of_range(platform_bars: list[DummyPriceBar]) -> None:
    config = ScannerConfig(min_atr_pct=0.5, max_atr_pct=1.0)
    filt = ATRFilter()
    result = filt.apply(platform_bars, config)

    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "ATR_FILTERED"
    assert result.data is not None
    assert result.data.passed is False
    assert result.data.score == 0.0


def test_atr_filter_propagates_calculate_atr_failure(
    platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args, **_kwargs):
        return Result.failed(ValueError("boom"), "ATR_OHLCV_FAILED")

    monkeypatch.setattr(detector_module, "calculate_atr", _boom)
    filt = ATRFilter()
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ATR_OHLCV_FAILED"


def test_box_quality_filter_happy_path(platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig) -> None:
    filt = BoxQualityFilter(window=30)
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert 0.0 <= result.data.score <= 1.0
    assert set(result.data.features.keys()) >= {
        "support",
        "resistance",
        "box_quality",
        "touch_score",
        "support_score",
        "resistance_score",
        "containment",
        "distribution_score",
    }


def test_box_quality_filter_returns_passed_false_when_below_threshold() -> None:
    config = ScannerConfig(min_box_quality=0.95, min_atr_pct=0.0, max_atr_pct=1.0)
    base = make_platform_series(total_bars=120)
    bars: list[DummyPriceBar] = list(base)
    for idx in range(90, 120):
        bar = bars[idx]
        bars[idx] = DummyPriceBar(
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=150.0,
            volume=bar.volume,
        )

    filt = BoxQualityFilter(window=30)
    result = filt.apply(bars, config)

    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "BOX_QUALITY_FILTERED"
    assert result.data is not None
    assert result.data.passed is False
    assert 0.0 <= result.data.score <= 1.0


def test_box_quality_filter_wraps_support_resistance_detector(
    platform_bars: list[DummyPriceBar], scanner_config: ScannerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, object] = {}

    def _stub(bars, tolerance=0.0):
        called["bars_len"] = len(bars)
        called["tolerance"] = tolerance
        return Result.success(
            {
                "support_level": 99.0,
                "resistance_level": 101.0,
                "box_quality": 1.0,
                "touch_score": 1.0,
                "containment": 1.0,
            }
        )

    monkeypatch.setattr(detector_module, "detect_support_resistance", _stub)
    filt = BoxQualityFilter(window=30, tolerance=0.02)
    result = filt.apply(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert called["bars_len"] == 30
    assert called["tolerance"] == pytest.approx(0.02)


def test_low_position_filter_happy_path(scanner_config: ScannerConfig) -> None:
    bars = make_bars(np.linspace(100.0, 70.0, 50), volume=1_000_000.0)
    filt = LowPositionFilter()
    result = filt.apply(bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.score == 1.0
    assert set(result.data.features.keys()) >= {
        "historical_high",
        "current_close",
        "decline_pct",
        "periods_from_high",
    }


def test_low_position_filter_boundary_conditions() -> None:
    bars = make_bars(np.linspace(100.0, 70.0, 50), volume=1_000_000.0)
    config = ScannerConfig(decline_threshold=0.3, min_periods=49)
    filt = LowPositionFilter()
    result = filt.apply(bars, config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.features["decline_pct"] == pytest.approx(0.3)
    assert result.data.features["periods_from_high"] == 49


def test_low_position_filter_insufficient_bars_failed(scanner_config: ScannerConfig) -> None:
    bars = make_bars(np.linspace(100.0, 95.0, 10), volume=1_000_000.0)
    filt = LowPositionFilter()
    result = filt.apply(bars, scanner_config)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BARS"


def test_low_position_filter_invalid_data_failed(scanner_config: ScannerConfig) -> None:
    bars = make_bars(np.linspace(100.0, 90.0, 30), volume=1_000_000.0)
    bars[-1] = make_bar(close=float("nan"), volume=1_000_000.0)
    filt = LowPositionFilter()
    result = filt.apply(bars, scanner_config)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NONFINITE_CLOSE"


def test_rapid_decline_filter_happy_path() -> None:
    closes = [100.0] * 10 + [120.0] + [110.0, 105.0, 100.0, 95.0, 90.0] + [90.0] * 24
    assert len(closes) == 40
    bars = make_bars(closes, volume=1_000_000.0)
    config = ScannerConfig(rapid_decline_days=30, rapid_decline_threshold=0.15)

    filt = RapidDeclineFilter()
    result = filt.apply(bars, config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.score == 1.0
    assert result.data.features["decline_start_idx"] == 10
    assert result.data.features["max_rapid_decline"] >= 0.15


def test_rapid_decline_filter_threshold_boundary() -> None:
    closes = [100.0] * 10 + [120.0] + [110.0, 105.0, 100.0, 95.0, 90.0] + [90.0] * 24
    bars = make_bars(closes, volume=1_000_000.0)
    initial = RapidDeclineFilter().apply(bars, ScannerConfig(rapid_decline_days=30, rapid_decline_threshold=0.0))

    assert initial.status is ResultStatus.SUCCESS
    assert initial.data is not None
    max_decline = float(initial.data.features["max_rapid_decline"])
    boundary = RapidDeclineFilter().apply(
        bars,
        ScannerConfig(rapid_decline_days=30, rapid_decline_threshold=max_decline),
    )

    assert boundary.status is ResultStatus.SUCCESS
    assert boundary.data is not None
    assert boundary.data.passed is True


def test_rapid_decline_filter_insufficient_bars_failed(scanner_config: ScannerConfig) -> None:
    bars = make_bars(np.linspace(100.0, 95.0, 10), volume=1_000_000.0)
    filt = RapidDeclineFilter()
    result = filt.apply(bars, scanner_config)

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BARS"


def test_rapid_decline_filter_invalid_data_failed() -> None:
    bars = make_bars([100.0] * 30, volume=1_000_000.0)
    bars[5] = DummyPriceBar(open=100.0, high=-1.0, low=99.0, close=100.0, volume=1_000_000.0)

    filt = RapidDeclineFilter()
    result = filt.apply(bars, ScannerConfig(rapid_decline_days=30))

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NONPOSITIVE_PRICE"


def test_breakthrough_filter_happy_path() -> None:
    closes = [98.0] * 14 + [100.0] + [99.0] * 5
    volumes = [100.0] * 15 + [130.0] * 5
    bars = make_bars(closes, volume=volumes)

    filt = BreakthroughFilter()
    # Disable P0/P1 breakout confirmation for legacy behavior test
    result = filt.apply(
        bars,
        ScannerConfig(
            resistance_proximity_pct=0.02,
            volume_increase_ratio=1.2,
            require_breakout_confirmation=False,
            require_breakout_volume_spike=False,
        ),
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.features["near_resistance"] is True
    assert result.data.features["volume_increasing"] is True
    assert 0.0 <= result.data.features["breakthrough_score"] <= 1.0


def test_breakthrough_filter_threshold_boundary() -> None:
    closes = [98.0] * 14 + [100.0] + [99.0] * 5
    volumes = [100.0] * 15 + [130.0] * 5
    bars = make_bars(closes, volume=volumes)

    # Disable P0/P1 breakout confirmation for legacy behavior test
    initial = BreakthroughFilter().apply(
        bars,
        ScannerConfig(
            resistance_proximity_pct=0.2,
            volume_increase_ratio=1.0,
            require_breakout_confirmation=False,
            require_breakout_volume_spike=False,
        ),
    )
    assert initial.status is ResultStatus.SUCCESS
    assert initial.data is not None

    proximity = float(initial.data.metadata["price_distance_pct"])
    volume_ratio = float(initial.data.metadata["volume_ratio"])

    boundary = BreakthroughFilter().apply(
        bars,
        ScannerConfig(
            resistance_proximity_pct=proximity,
            volume_increase_ratio=volume_ratio,
            require_breakout_confirmation=False,
            require_breakout_volume_spike=False,
        ),
    )
    assert boundary.status is ResultStatus.SUCCESS
    assert boundary.data is not None
    assert boundary.data.passed is True


def test_breakthrough_filter_insufficient_bars_failed() -> None:
    bars = make_bars([100.0] * 14, volume=100.0)
    filt = BreakthroughFilter()
    result = filt.apply(bars, ScannerConfig())

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BARS"


def test_breakthrough_filter_invalid_data_failed() -> None:
    closes = [100.0] * 20
    volumes = [100.0] * 20
    bars = make_bars(closes, volume=volumes)
    bars[-1] = DummyPriceBar(open=100.0, high=101.0, low=99.0, close=100.0, volume=-1.0)

    filt = BreakthroughFilter()
    result = filt.apply(bars, ScannerConfig())

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_VOLUME"
