from __future__ import annotations

import pytest

from common.interface import Result, ResultStatus
from scanner import detector as detector_module
from scanner.interface import IndicatorScoringConfig, PlatformCandidate, PlatformFeatures, ScannerConfig


def _make_candidate(*, score: float, meta: dict | None = None) -> PlatformCandidate:
    features = PlatformFeatures(
        box_range=0.05,
        box_low=100.0,
        box_high=105.0,
        ma_diff=0.01,
        volatility=0.01,
        atr_pct=0.02,
        volume_change_ratio=0.6,
        volume_stability=0.1,
        avg_dollar_volume=1_000_000.0,
        box_quality=0.8,
        support_level=100.0,
        resistance_level=105.0,
    )
    return PlatformCandidate(
        symbol="TEST",
        detected_at=123,
        window=30,
        score=float(score),
        features=features,
        invalidation_level=99.0,
        target_level=110.0,
        reasons=[],
        meta=dict(meta or {}),
    )


def _stub_chain_candidate(monkeypatch: pytest.MonkeyPatch, candidate: PlatformCandidate) -> None:
    monkeypatch.setattr(
        detector_module,
        "_detect_platform_candidate_via_chain",
        lambda *args, **kwargs: Result.success(candidate),
    )


def test_indicator_scoring_disabled_no_change(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _make_candidate(score=0.8)
    _stub_chain_candidate(monkeypatch, candidate)

    import scanner.indicator_scoring as scoring

    monkeypatch.setattr(scoring, "compute_indicator_score", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))

    config = ScannerConfig(
        use_filter_chain_detector=True,
        indicator_scoring=IndicatorScoringConfig(enabled=False),
    )
    result = detector_module.detect_platform_candidate("TEST", [], 30, config, detected_at=123)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.score == pytest.approx(0.8)
    assert "indicator_scoring" not in result.data.meta


def test_indicator_scoring_multiply_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _make_candidate(score=0.8)
    _stub_chain_candidate(monkeypatch, candidate)

    import scanner.indicator_scoring as scoring

    monkeypatch.setattr(scoring, "compute_indicator_score", lambda *args, **kwargs: (0.5, {"ok": True}))

    config = ScannerConfig(
        use_filter_chain_detector=True,
        indicator_scoring=IndicatorScoringConfig(enabled=True, combination_mode="multiply"),
    )
    result = detector_module.detect_platform_candidate("TEST", [], 30, config, detected_at=123)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.score == pytest.approx(0.4)


def test_indicator_scoring_weighted_avg_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _make_candidate(score=0.8)
    _stub_chain_candidate(monkeypatch, candidate)

    import scanner.indicator_scoring as scoring

    monkeypatch.setattr(scoring, "compute_indicator_score", lambda *args, **kwargs: (0.5, {"ok": True}))

    config = ScannerConfig(
        use_filter_chain_detector=True,
        indicator_scoring=IndicatorScoringConfig(
            enabled=True,
            combination_mode="weighted_avg",
            base_weight=0.25,
        ),
    )
    result = detector_module.detect_platform_candidate("TEST", [], 30, config, detected_at=123)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.score == pytest.approx(0.8 * 0.25 + 0.5 * 0.75)


def test_indicator_scoring_min_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _make_candidate(score=0.8)
    _stub_chain_candidate(monkeypatch, candidate)

    import scanner.indicator_scoring as scoring

    monkeypatch.setattr(scoring, "compute_indicator_score", lambda *args, **kwargs: (0.5, {"ok": True}))

    config = ScannerConfig(
        use_filter_chain_detector=True,
        indicator_scoring=IndicatorScoringConfig(enabled=True, combination_mode="min"),
    )
    result = detector_module.detect_platform_candidate("TEST", [], 30, config, detected_at=123)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.score == pytest.approx(0.5)


def test_indicator_metadata_in_candidate_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _make_candidate(score=0.8, meta={"existing": 1})
    _stub_chain_candidate(monkeypatch, candidate)

    import scanner.indicator_scoring as scoring

    indicator_meta = {"details": {"rsi": {"score": 1.0}}}
    monkeypatch.setattr(scoring, "compute_indicator_score", lambda *args, **kwargs: (0.5, indicator_meta))

    config = ScannerConfig(
        use_filter_chain_detector=True,
        indicator_scoring=IndicatorScoringConfig(enabled=True, combination_mode="multiply"),
    )
    result = detector_module.detect_platform_candidate("TEST", [], 30, config, detected_at=123)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.meta["existing"] == 1
    assert result.data.meta["indicator_scoring"]["enabled"] is True
    assert result.data.meta["indicator_scoring"]["base_score"] == pytest.approx(0.8)
    assert result.data.meta["indicator_scoring"]["indicator_score"] == pytest.approx(0.5)
    assert result.data.meta["indicator_scoring"]["adjusted_score"] == pytest.approx(0.4)
    assert result.data.meta["indicator_scoring"]["combination_mode"] == "multiply"
    assert result.data.meta["indicator_scoring"]["details"] == indicator_meta

