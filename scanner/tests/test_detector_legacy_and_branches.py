from __future__ import annotations

import numpy as np
import pytest

from common.interface import Result, ResultStatus
from scanner import detector as detector_module
from scanner.interface import ScannerConfig

from .conftest import DummyPriceBar, make_bar, make_bars, make_downtrend_series, make_platform_series


def test_detect_platform_candidate_legacy_happy_path(platform_bars: list[DummyPriceBar]) -> None:
    config = ScannerConfig(use_filter_chain_detector=False)
    result = detector_module.detect_platform_candidate("AAPL", platform_bars, 30, config, detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None


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
def test_detect_platform_candidate_legacy_filters_match_reason_codes(bars_factory, expected_reason: str) -> None:
    config = ScannerConfig(use_filter_chain_detector=False)
    bars = bars_factory()

    result = detector_module.detect_platform_candidate("TEST", bars, 30, config, detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == expected_reason


def test_detect_platform_candidate_legacy_marks_low_liquidity_but_does_not_filter() -> None:
    config = ScannerConfig(use_filter_chain_detector=False)
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

    result = detector_module.detect_platform_candidate("ILLQ", bars, 30, config, detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert isinstance(result.data.meta.get("liquidity_metrics"), dict)
    assert result.data.meta["liquidity_metrics"]["liquidity_score"] < 0.3


def test_detect_platform_candidate_legacy_filters_on_atr() -> None:
    config = ScannerConfig(use_filter_chain_detector=False, min_atr_pct=0.01, max_atr_pct=0.05)
    bars = make_platform_series(total_bars=120, high_pct=0.0002, low_pct=0.0002)

    result = detector_module.detect_platform_candidate("LOWATR", bars, 30, config, detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "ATR_FILTERED"


def test_detect_platform_candidate_legacy_filters_on_box_quality() -> None:
    config = ScannerConfig(use_filter_chain_detector=False, min_box_quality=0.95, min_atr_pct=0.0, max_atr_pct=1.0)
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

    result = detector_module.detect_platform_candidate("BOX", bars, 30, config, detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "BOX_QUALITY_FILTERED"


def test_detect_platform_candidate_propagates_chain_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detector_module,
        "_detect_platform_candidate_via_chain",
        lambda *_args, **_kwargs: Result.degraded(None, ValueError("boom"), "CHAIN_FILTER_FAILED"),
    )
    result = detector_module.detect_platform_candidate(
        "AAPL",
        make_platform_series(total_bars=120),
        30,
        ScannerConfig(),
        detected_at=1,
    )
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "CHAIN_FILTER_FAILED"


def test_detect_platform_candidate_propagates_chain_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detector_module,
        "_detect_platform_candidate_via_chain",
        lambda *_args, **_kwargs: Result.failed(ValueError("boom"), "CHAIN_INVALID_LOGIC"),
    )
    result = detector_module.detect_platform_candidate(
        "AAPL",
        make_platform_series(total_bars=120),
        30,
        ScannerConfig(),
        detected_at=1,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CHAIN_INVALID_LOGIC"


def test_extract_ohlcv_propagates_nonfinite_open() -> None:
    bars = [
        make_bar(close=100.0, volume=1.0, open=float("nan")),
        make_bar(close=100.0, volume=1.0),
    ]
    result = detector_module._extract_ohlcv(bars)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "OPEN_EXTRACTION_FAILED"


def test_extract_ohlcv_propagates_nonfinite_high() -> None:
    bars = [
        make_bar(close=100.0, volume=1.0, high=float("nan")),
        make_bar(close=100.0, volume=1.0),
    ]
    result = detector_module._extract_ohlcv(bars)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "HIGH_EXTRACTION_FAILED"


def test_extract_ohlcv_propagates_nonfinite_low() -> None:
    bars = [
        make_bar(close=100.0, volume=1.0, low=float("nan")),
        make_bar(close=100.0, volume=1.0),
    ]
    result = detector_module._extract_ohlcv(bars)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "LOW_EXTRACTION_FAILED"


def test_extract_ohlcv_propagates_nonfinite_volume() -> None:
    bars = [
        make_bar(close=100.0, volume=float("nan")),
        make_bar(close=100.0, volume=1.0),
    ]
    result = detector_module._extract_ohlcv(bars)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "VOLUME_EXTRACTION_FAILED"


def test_atr_pct_for_series_propagates_calculate_atr_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "calculate_atr", lambda *_args, **_kwargs: Result.failed(ValueError("boom"), "X"))
    bars = make_platform_series(total_bars=120)
    result = detector_module._atr_pct_for_series(bars, ScannerConfig())
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ATR_FAILED"


def test_atr_pct_for_series_invalid_atr_pct_is_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "calculate_atr", lambda *_args, **_kwargs: Result.success(float("nan")))
    bars = make_platform_series(total_bars=120)
    result = detector_module._atr_pct_for_series(bars, ScannerConfig())
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_ATR_PCT"


def test_box_structure_for_window_propagates_detector_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detector_module,
        "detect_support_resistance",
        lambda *_args, **_kwargs: Result.failed(ValueError("boom"), "BOX_OHLCV_FAILED"),
    )
    bars = make_platform_series(total_bars=120)
    window_bars = bars[-30:]
    result = detector_module._box_structure_for_window(
        window_bars,
        ScannerConfig(),
        box_low_fallback=99.0,
        box_high_fallback=101.0,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "BOX_STRUCTURE_FAILED"


def test_target_and_invalidation_rejects_nonfinite_target() -> None:
    result = detector_module._target_and_invalidation(1.0, float("inf"))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_TARGET_LEVEL"


def test_volume_platform_previous_mean_nonfinite_is_failed() -> None:
    closes = [100.0] * 60
    prev_volumes = [float("nan")] * 30
    recent_volumes = [1_000_000.0] * 30
    bars = make_bars(closes, volume=prev_volumes + recent_volumes)

    result = detector_module.detect_volume_platform(bars, window=30, config=ScannerConfig())
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "PREVIOUS_OHLCV_FAILED"


def test_volume_platform_recent_mean_nonfinite_is_failed() -> None:
    closes = [100.0] * 60
    prev_volumes = [1_000_000.0] * 30
    recent_volumes = [float("nan")] * 30
    bars = make_bars(closes, volume=prev_volumes + recent_volumes)

    result = detector_module.detect_volume_platform(bars, window=30, config=ScannerConfig())
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "RECENT_OHLCV_FAILED"


def test_compute_price_platform_features_rejects_nonpositive_box_low() -> None:
    bars = [make_bar(close=100.0, volume=1.0, low=0.0, high=100.0) for _ in range(30)]
    result = detector_module.detect_price_platform(bars, window=30, config=ScannerConfig())
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_BOX_LOW"


def test_compute_price_platform_features_rejects_nonfinite_volatility(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = make_bars(np.linspace(100.0, 130.0, 30), volume=1_000_000.0, high_pct=0.0, low_pct=0.0)
    monkeypatch.setattr(detector_module, "_pct_change", lambda *_args, **_kwargs: Result.success(np.asarray([float("nan")])))
    result = detector_module.detect_price_platform(bars, window=30, config=ScannerConfig())
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_VOLATILITY"
