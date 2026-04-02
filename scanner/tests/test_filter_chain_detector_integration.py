from __future__ import annotations

import math

from common.interface import ResultStatus
from scanner import ScannerConfig, detect_platform_candidate

from .conftest import make_bars, make_platform_series


def test_detect_platform_candidate_attaches_filter_chain_meta(platform_bars: list[object]) -> None:
    result = detect_platform_candidate("AAPL", platform_bars, 30, ScannerConfig(require_breakout_confirmation=False, require_breakout_volume_spike=False), detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    meta = result.data.meta
    assert isinstance(meta, dict)
    chain_meta = meta.get("filter_chain")
    assert isinstance(chain_meta, dict)
    assert chain_meta.get("logic") == "AND"
    filter_results = chain_meta.get("filter_results")
    assert isinstance(filter_results, dict)
    assert set(filter_results.keys()) >= {"price_platform", "volume_platform", "atr_range", "box_quality"}


def test_detect_platform_candidate_core_chain_preserves_feature_values(platform_bars: list[object]) -> None:
    result = detect_platform_candidate("AAPL", platform_bars, 30, ScannerConfig(require_breakout_confirmation=False, require_breakout_volume_spike=False), detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    candidate = result.data
    chain_meta = candidate.meta.get("filter_chain")
    assert isinstance(chain_meta, dict)
    filter_results = chain_meta.get("filter_results")
    assert isinstance(filter_results, dict)

    price_features = filter_results["price_platform"]["features"]
    volume_features = filter_results["volume_platform"]["features"]
    atr_features = filter_results["atr_range"]["features"]
    box_features = filter_results["box_quality"]["features"]

    assert candidate.features.box_range == price_features["box_range"]
    assert candidate.features.box_low == price_features["box_low"]
    assert candidate.features.box_high == price_features["box_high"]
    assert candidate.features.ma_diff == price_features["ma_diff"]
    assert candidate.features.volatility == price_features["volatility"]
    assert candidate.features.volume_change_ratio == volume_features["volume_change_ratio"]
    assert candidate.features.volume_stability == volume_features["volume_stability"]
    assert candidate.features.avg_dollar_volume == volume_features["avg_dollar_volume"]
    assert candidate.features.atr_pct == atr_features["atr_pct"]
    assert candidate.features.support_level == box_features["support"]
    assert candidate.features.resistance_level == box_features["resistance"]
    assert candidate.features.box_quality == box_features["box_quality"]


def test_optional_filters_disabled_by_default(platform_bars: list[object]) -> None:
    result = detect_platform_candidate("AAPL", platform_bars, 30, ScannerConfig(require_breakout_confirmation=False, require_breakout_volume_spike=False), detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    filter_results = result.data.meta["filter_chain"]["filter_results"]
    assert "low_position" not in filter_results
    assert "rapid_decline" not in filter_results
    assert "breakthrough_potential" in filter_results


def test_low_position_filter_can_be_enabled_and_pass() -> None:
    platform = make_platform_series(total_bars=120, base_price=80.0)
    prefix = make_bars([120.0] * 50, volume=2_000_000.0)
    bars: list[object] = prefix + platform

    config = ScannerConfig(use_low_position_filter=True, require_breakout_confirmation=False, require_breakout_volume_spike=False)
    result = detect_platform_candidate("LP", bars, 30, config, detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert "low_position" in result.data.meta["filter_chain"]["filter_results"]


def test_rapid_decline_filter_enabled_filters_out_platform_series(platform_bars: list[object]) -> None:
    config = ScannerConfig(use_rapid_decline_filter=True)
    result = detect_platform_candidate("RD", platform_bars, 30, config, detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "RAPID_DECLINE_FILTERED"


def test_all_filters_enabled_execute_and_report_reason_codes(platform_bars: list[object]) -> None:
    config = ScannerConfig(
        use_low_position_filter=True,
        use_rapid_decline_filter=True,
        use_breakthrough_filter=True,
    )
    result = detect_platform_candidate("ALL", platform_bars, 30, config, detected_at=1)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code in {"LOW_POSITION_FILTERED", "RAPID_DECLINE_FILTERED", "BREAKTHROUGH_FILTERED"}


def test_filter_chain_score_in_candidate_is_bounded(platform_bars: list[object]) -> None:
    result = detect_platform_candidate("AAPL", platform_bars, 30, ScannerConfig(require_breakout_confirmation=False, require_breakout_volume_spike=False), detected_at=1)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert 0.0 <= result.data.score <= 1.0
    assert math.isfinite(result.data.score)
