from __future__ import annotations

from typing import Any

import msgspec
import msgspec.json
import pytest

import scanner
from scanner import (
    CandidateSet,
    PlatformCandidate,
    PlatformFeatures,
    ScannerConfig,
    ScannerPlugin,
)
from plugins.interface import PluginCategory, PluginMetadata
from common.interface import Result

if not hasattr(msgspec.json, "DecodeError"):
    setattr(msgspec.json, "DecodeError", msgspec.DecodeError)
if not hasattr(msgspec.json, "EncodeError"):
    setattr(msgspec.json, "EncodeError", msgspec.EncodeError)


def test_scanner_package_reexports_and_all_exports_exist() -> None:
    assert isinstance(scanner.__all__, list)
    assert scanner.__version__
    for name in scanner.__all__:
        assert hasattr(scanner, name), f"scanner is missing export: {name}"


def test_scanner_config_defaults() -> None:
    config = ScannerConfig()
    assert config.windows == [20, 30, 60]
    assert config.window_weights == {20: 0.3, 30: 0.4, 60: 0.3}
    assert 0 < config.box_threshold < 1
    assert 0 < config.min_box_quality <= 1


def test_scanner_config_custom_values_and_msgspec_roundtrip() -> None:
    config = ScannerConfig(
        windows=[10, 15],
        box_threshold=0.12,
        ma_diff_threshold=0.03,
        volatility_threshold=0.04,
        volume_change_threshold=0.85,
        volume_stability_threshold=0.45,
        volume_increase_threshold=1.8,
        min_box_quality=0.7,
        min_atr_pct=0.005,
        max_atr_pct=0.06,
        min_dollar_volume=2_000_000,
        window_weights={10: 0.4, 15: 0.6},
    )
    encoded = msgspec.json.encode(config)
    decoded = msgspec.json.decode(encoded, type=ScannerConfig)
    assert decoded == config


def test_platform_features_all_fields_and_optional_fields() -> None:
    features = PlatformFeatures(
        box_range=0.05,
        box_low=95.0,
        box_high=100.0,
        ma_diff=0.01,
        volatility=0.02,
        atr_pct=0.02,
        volume_change_ratio=0.7,
        volume_stability=0.2,
        volume_increase_ratio=2.0,
        avg_dollar_volume=50_000_000.0,
        box_quality=0.9,
        support_level=95.0,
        resistance_level=100.0,
    )
    assert features.atr_pct == pytest.approx(0.02)
    assert features.volume_increase_ratio == pytest.approx(2.0)
    assert features.box_quality == pytest.approx(0.9)

    minimal = PlatformFeatures(
        box_range=0.05,
        box_low=95.0,
        box_high=100.0,
        ma_diff=0.01,
        volatility=0.02,
        volume_change_ratio=0.7,
        volume_stability=0.2,
        avg_dollar_volume=50_000_000.0,
    )
    assert minimal.atr_pct is None
    assert minimal.box_quality is None


def test_platform_candidate_schema_and_score_bounds() -> None:
    candidate = PlatformCandidate(
        symbol="AAPL",
        detected_at=123,
        window=30,
        score=0.85,
        features=PlatformFeatures(
            box_range=0.05,
            box_low=95.0,
            box_high=100.0,
            ma_diff=0.01,
            volatility=0.02,
            atr_pct=0.02,
            volume_change_ratio=0.7,
            volume_stability=0.2,
            avg_dollar_volume=50_000_000.0,
            box_quality=0.9,
            support_level=95.0,
            resistance_level=100.0,
        ),
        invalidation_level=94.05,
        target_level=101.0,
        reasons=["price_platform", "volume_platform"],
        meta={"source": "test"},
    )
    assert 0.0 <= candidate.score <= 1.0
    assert candidate.reasons
    encoded = msgspec.json.encode(candidate)
    decoded = msgspec.json.decode(encoded, type=PlatformCandidate)
    assert decoded == candidate


def test_candidate_set_empty_candidates_and_snapshot_metadata() -> None:
    snapshot = CandidateSet(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=123,
        candidates=[],
        total_scanned=0,
        total_detected=0,
        config_snapshot={},
        data_source="test",
        universe_source="test",
    )
    assert snapshot.candidates == []
    assert snapshot.total_detected == 0
    encoded = msgspec.json.encode(snapshot)
    decoded = msgspec.json.decode(encoded, type=CandidateSet)
    assert decoded == snapshot


def test_candidate_set_multiple_candidates() -> None:
    features = PlatformFeatures(
        box_range=0.05,
        box_low=95.0,
        box_high=100.0,
        ma_diff=0.01,
        volatility=0.02,
        volume_change_ratio=0.7,
        volume_stability=0.2,
        avg_dollar_volume=50_000_000.0,
    )
    candidates = [
        PlatformCandidate(
            symbol="AAPL",
            detected_at=1,
            window=20,
            score=0.8,
            features=features,
            invalidation_level=94.0,
            target_level=101.0,
            reasons=["x"],
        ),
        PlatformCandidate(
            symbol="MSFT",
            detected_at=2,
            window=30,
            score=0.9,
            features=features,
            invalidation_level=94.0,
            target_level=101.0,
            reasons=["y"],
        ),
    ]
    snapshot = CandidateSet(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=123,
        candidates=candidates,
        total_scanned=2,
        total_detected=2,
        config_snapshot=msgspec.structs.asdict(ScannerConfig()),
        data_source="test",
        universe_source="test",
    )
    assert snapshot.total_scanned == 2
    assert snapshot.total_detected == 2
    assert [c.symbol for c in snapshot.candidates] == ["AAPL", "MSFT"]


def test_scanner_plugin_protocol_is_runtime_checkable() -> None:
    class DummyScanner:
        metadata = PluginMetadata(
            name="dummy_scanner",
            version="0.0.0",
            category=PluginCategory.SCANNER,
            enabled=True,
        )

        def init(self, context: dict[str, Any] | None = None) -> Result[None]:  # type: ignore[override]
            raise NotImplementedError

        def validate_config(self, config: ScannerConfig) -> Result[ScannerConfig]:  # type: ignore[override]
            raise NotImplementedError

        def execute(self, payload: Any) -> Result[CandidateSet]:  # type: ignore[override]
            raise NotImplementedError

        def cleanup(self) -> Result[None]:  # type: ignore[override]
            raise NotImplementedError

    assert isinstance(DummyScanner(), ScannerPlugin)
