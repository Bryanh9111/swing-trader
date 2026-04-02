from __future__ import annotations

import math

import pytest

from indicators.interface import MACDResult
from scanner.indicator_scoring import (
    _combine_scores_multiply,
    _combine_scores_weighted_avg,
    _score_atr,
    _score_bbands,
    _score_kdj,
    _score_macd,
    _score_rsi,
    compute_indicator_score,
)
from scanner.interface import IndicatorScoringConfig


def test_disabled_returns_neutral_score() -> None:
    score, meta = compute_indicator_score([], IndicatorScoringConfig(enabled=False))
    assert score == 1.0
    assert meta == {}


def test_rsi_scoring_optimal_range() -> None:
    assert _score_rsi(30.0, (30.0, 70.0)) == 1.0
    assert _score_rsi(50.0, (30.0, 70.0)) == 1.0
    assert _score_rsi(70.0, (30.0, 70.0)) == 1.0


def test_rsi_scoring_outside_range() -> None:
    # Below low: distance=(30-15)/30=0.5 => score=0.5
    assert _score_rsi(15.0, (30.0, 70.0)) == 0.5
    # Above high: distance=(85-70)/(100-70)=0.5 => score=0.5
    assert _score_rsi(85.0, (30.0, 70.0)) == 0.5


def test_macd_scoring_bullish() -> None:
    assert _score_macd(0.01, 0.0) == 1.0
    assert _score_macd(0.0, 0.0) == 1.0


def test_macd_scoring_bearish() -> None:
    assert _score_macd(-0.01, 0.0) == 0.5


def test_atr_scoring_optimal_range() -> None:
    assert _score_atr(0.02, (0.01, 0.03)) == 1.0
    # Below low: distance=(0.01-0.005)/0.01=0.5 => score=0.5
    assert _score_atr(0.005, (0.01, 0.03)) == 0.5


def test_rsi_scoring_edge_cases() -> None:
    # low <= 0 branch
    assert _score_rsi(-1.0, (0.0, 70.0)) == 0.0
    # high >= 100 branch
    assert _score_rsi(120.0, (30.0, 100.0)) == 0.0


def test_atr_scoring_edge_cases() -> None:
    # low <= 0 branch
    assert _score_atr(-0.1, (0.0, 0.03)) == 0.0
    # high <= 0 branch
    assert _score_atr(0.1, (-0.01, 0.0)) == 0.0


def test_bbands_scoring_thresholds() -> None:
    assert _score_bbands(0.0) == 1.0
    assert _score_bbands(0.2) == 1.0
    assert _score_bbands(0.5) == pytest.approx(0.5)
    assert _score_bbands(0.8) == 0.0
    assert _score_bbands(1.0) == 0.0


def test_kdj_scoring_oversold() -> None:
    """KDJ J <= oversold should return 1.0 (bullish)."""
    assert _score_kdj(0.0) == 1.0
    assert _score_kdj(10.0) == 1.0
    assert _score_kdj(20.0) == 1.0  # At threshold


def test_kdj_scoring_overbought() -> None:
    """KDJ J >= overbought should return 0.0 (bearish)."""
    assert _score_kdj(80.0) == 0.0  # At threshold
    assert _score_kdj(90.0) == 0.0
    assert _score_kdj(100.0) == 0.0
    assert _score_kdj(120.0) == 0.0  # J can exceed 100


def test_kdj_scoring_neutral() -> None:
    """KDJ J between thresholds should have linear interpolation."""
    # J=50 is midpoint between 20 and 80 => score = 0.5
    assert _score_kdj(50.0) == pytest.approx(0.5)
    # J=35 is 1/4 of the way => score = 0.75
    assert _score_kdj(35.0) == pytest.approx(0.75)
    # J=65 is 3/4 of the way => score = 0.25
    assert _score_kdj(65.0) == pytest.approx(0.25)


def test_kdj_scoring_custom_thresholds() -> None:
    """KDJ scoring with custom thresholds."""
    # Custom: oversold=10, overbought=90
    assert _score_kdj(10.0, oversold=10.0, overbought=90.0) == 1.0
    assert _score_kdj(90.0, oversold=10.0, overbought=90.0) == 0.0
    # J=50 is midpoint between 10 and 90 => score = 0.5
    assert _score_kdj(50.0, oversold=10.0, overbought=90.0) == pytest.approx(0.5)


def test_kdj_scoring_inverted_thresholds() -> None:
    """KDJ with inverted thresholds - early checks take precedence."""
    # With oversold=80, overbought=20:
    # j=50 <= oversold=80, so returns 1.0 (oversold branch)
    assert _score_kdj(50.0, oversold=80.0, overbought=20.0) == 1.0
    # j=10 <= oversold=80, so returns 1.0
    assert _score_kdj(10.0, oversold=80.0, overbought=20.0) == 1.0
    # j=90 > oversold=80 and j=90 >= overbought=20, so returns 0.0
    assert _score_kdj(90.0, oversold=80.0, overbought=20.0) == 0.0


def test_weight_combiners_return_neutral_without_positive_weights() -> None:
    assert _combine_scores_weighted_avg([("rsi", 0.1, 0.0)]) == 1.0
    assert _combine_scores_multiply([("rsi", 0.1, 0.0)]) == 1.0


def test_multiply_combiner_returns_zero_when_any_score_zero() -> None:
    assert _combine_scores_multiply([("rsi", 1.0, 0.5), ("macd", 0.0, 0.5)]) == 0.0


def test_combined_score_multiply_mode(monkeypatch) -> None:
    import scanner.indicator_scoring as scoring

    monkeypatch.setattr(scoring, "compute_rsi_last", lambda bars: 50.0)
    monkeypatch.setattr(scoring, "compute_macd_last", lambda bars: MACDResult(macd=0.0, signal=0.0, histogram=-0.1))
    monkeypatch.setattr(scoring, "compute_atr_last", lambda bars, percentage=True: 0.02)

    config = IndicatorScoringConfig(
        enabled=True,
        combination_mode="multiply",
        rsi_weight=0.3,
        macd_weight=0.4,
        atr_weight=0.3,
    )
    score, meta = compute_indicator_score([], config)

    # rsi=1.0, macd=0.5, atr=1.0 => weighted geometric mean
    expected = math.exp(0.3 * math.log(1.0) + 0.4 * math.log(0.5) + 0.3 * math.log(1.0))
    assert score == pytest.approx(expected)
    assert meta["combined"] == pytest.approx(expected)


def test_combined_score_weighted_avg_mode(monkeypatch) -> None:
    import scanner.indicator_scoring as scoring

    monkeypatch.setattr(scoring, "compute_rsi_last", lambda bars: 50.0)
    monkeypatch.setattr(scoring, "compute_macd_last", lambda bars: MACDResult(macd=0.0, signal=0.0, histogram=-0.1))
    monkeypatch.setattr(scoring, "compute_atr_last", lambda bars, percentage=True: 0.02)

    config = IndicatorScoringConfig(
        enabled=True,
        combination_mode="weighted_avg",
        rsi_weight=0.3,
        macd_weight=0.4,
        atr_weight=0.3,
    )
    score, _ = compute_indicator_score([], config)
    expected = 0.3 * 1.0 + 0.4 * 0.5 + 0.3 * 1.0
    assert score == expected


def test_graceful_degradation_missing_data(monkeypatch) -> None:
    import scanner.indicator_scoring as scoring

    monkeypatch.setattr(scoring, "compute_rsi_last", lambda bars: None)
    monkeypatch.setattr(scoring, "compute_macd_last", lambda bars: None)
    monkeypatch.setattr(scoring, "compute_atr_last", lambda bars, percentage=True: 0.02)

    config = IndicatorScoringConfig(enabled=True, combination_mode="weighted_avg")
    score, meta = compute_indicator_score([], config)
    assert score == 1.0
    assert meta["skipped"]["rsi"] == "insufficient_data"
    assert meta["skipped"]["macd"] == "insufficient_data"
    assert "atr" in meta["indicators"]


def test_graceful_degradation_all_missing_data_returns_neutral(monkeypatch) -> None:
    import scanner.indicator_scoring as scoring

    monkeypatch.setattr(scoring, "compute_rsi_last", lambda bars: None)
    monkeypatch.setattr(scoring, "compute_macd_last", lambda bars: None)
    monkeypatch.setattr(scoring, "compute_atr_last", lambda bars, percentage=True: None)

    config = IndicatorScoringConfig(enabled=True, combination_mode="multiply")
    score, meta = compute_indicator_score([], config)
    assert score == 1.0
    assert meta["skipped"]["atr"] == "insufficient_data"


def test_all_indicators_combined(monkeypatch) -> None:
    import scanner.indicator_scoring as scoring

    monkeypatch.setattr(scoring, "compute_rsi_last", lambda bars: 85.0)  # score 0.5 for (30,70)
    monkeypatch.setattr(scoring, "compute_macd_last", lambda bars: MACDResult(macd=0.0, signal=0.0, histogram=0.2))  # 1.0
    monkeypatch.setattr(scoring, "compute_atr_last", lambda bars, percentage=True: 0.06)  # score 0.0 for (0.01,0.03)

    config = IndicatorScoringConfig(
        enabled=True,
        combination_mode="weighted_avg",
        rsi_weight=0.3,
        macd_weight=0.4,
        atr_weight=0.3,
        rsi_optimal_range=(30.0, 70.0),
        atr_optimal_range=(0.01, 0.03),
        macd_histogram_threshold=0.0,
    )
    score, _ = compute_indicator_score([], config)
    expected = 0.3 * 0.5 + 0.4 * 1.0 + 0.3 * 0.0
    assert score == expected
