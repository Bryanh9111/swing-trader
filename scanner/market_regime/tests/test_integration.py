from __future__ import annotations

from typing import Any

import msgspec
import pytest

from common.interface import ResultStatus
from scanner.detector import detect_platform_candidate
from scanner.interface import ScannerConfig
from scanner.market_regime.config_loader import RegimeConfigLoader
from scanner.market_regime.detector import MarketRegimeDetector
from scanner.market_regime.interface import MarketRegime
from scanner.market_regime.tests.conftest import make_series_snapshot
from scanner.tests.conftest import make_platform_series


def _apply_scanner_overrides(base_config: ScannerConfig, overrides: dict[str, Any]) -> ScannerConfig:
    payload = msgspec.to_builtins(base_config)
    payload.update(overrides)
    return msgspec.convert(payload, type=ScannerConfig)


def test_market_regime_detection_and_config_loading_flow() -> None:
    closes = [100.0 + idx * 0.5 for idx in range(120)]
    spy_snapshot = make_series_snapshot("SPY", closes=closes)
    detector = MarketRegimeDetector()
    result = detector.detect({"SPY": spy_snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data.regime in {MarketRegime.BULL, MarketRegime.UNKNOWN}

    loader = RegimeConfigLoader()
    cfg_result = loader.load(result.data.regime)
    assert cfg_result.status is ResultStatus.SUCCESS
    regime_config = cfg_result.data
    assert regime_config.scanner["box_threshold"] >= 0.06


def test_regime_scanner_overrides_apply_expected_parameters() -> None:
    loader = RegimeConfigLoader()
    base = ScannerConfig()

    bull = loader.load("bull_market")
    assert bull.status is ResultStatus.SUCCESS
    bull_config = _apply_scanner_overrides(base, bull.data.scanner)
    assert pytest.approx(bull_config.box_threshold) == bull.data.scanner["box_threshold"]
    assert pytest.approx(bull_config.volume_increase_ratio) == bull.data.scanner["volume_increase_ratio"]

    bear = loader.load("bear_market")
    assert bear.status is ResultStatus.SUCCESS
    bear_config = _apply_scanner_overrides(base, bear.data.scanner)
    assert pytest.approx(bear_config.min_box_quality) == bear.data.scanner["min_box_quality"]
    assert pytest.approx(bear_config.volume_increase_ratio) == bear.data.scanner["volume_increase_ratio"]


def test_choppy_regime_breakthrough_confirmation_enforces_multi_day_signal() -> None:
    loader = RegimeConfigLoader()
    choppy = loader.load("choppy_market")
    assert choppy.status is ResultStatus.SUCCESS
    breakthrough_config = choppy.data.breakthrough

    bars = make_platform_series()
    # Disable use_breakthrough_filter to test breakthrough_config parameter separately.
    # The default ScannerConfig has use_breakthrough_filter=True which would filter
    # the candidate before breakthrough_config is even evaluated.
    config = ScannerConfig(use_breakthrough_filter=False)

    base_result = detect_platform_candidate("PLAT", bars, 30, config, detected_at=1)
    assert base_result.status is ResultStatus.SUCCESS
    assert base_result.data is not None

    strict_result = detect_platform_candidate(
        "PLAT",
        bars,
        30,
        config,
        detected_at=2,
        breakthrough_config=breakthrough_config,
    )
    assert strict_result.status is ResultStatus.SUCCESS
    assert strict_result.data is None
    assert strict_result.reason_code == "BREAKTHROUGH_NOT_CONFIRMED"
