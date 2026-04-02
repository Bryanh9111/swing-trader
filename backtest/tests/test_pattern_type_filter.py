"""V28: Pattern-type filter tests.

Tests verify:
- _extract_intent_pattern_type() correctly parses shadow_score JSON
- Pattern-type filtering logic for BULL_C / BULL_U / CHOP / BEAR
"""
from __future__ import annotations

import json

import pytest

from backtest.orchestrator import BacktestOrchestrator
from strategy.interface import IntentType, TradeIntent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intent(
    *,
    symbol: str = "TEST",
    intent_id: str = "I-1",
    pattern_type: str | None = None,
    shadow_score_raw: str | None = None,
    metadata: dict[str, str] | None = "AUTO",
) -> TradeIntent:
    """Build a minimal TradeIntent with optional shadow_score metadata.

    If metadata=="AUTO" (default), auto-build from pattern_type/shadow_score_raw.
    If metadata is an explicit dict or None, use it directly.
    """
    if metadata == "AUTO":
        if shadow_score_raw is not None:
            meta = {"shadow_score": shadow_score_raw}
        elif pattern_type is not None:
            meta = {"shadow_score": json.dumps({"ss_pattern_type": pattern_type})}
        else:
            meta = None
    else:
        meta = metadata  # type: ignore[assignment]
    return TradeIntent(
        intent_id=intent_id,
        symbol=symbol,
        intent_type=IntentType.OPEN_LONG,
        quantity=1.0,
        created_at_ns=1,
        entry_price=100.0,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# A. _extract_intent_pattern_type tests
# ---------------------------------------------------------------------------


class TestExtractIntentPatternType:
    """Test _extract_intent_pattern_type static method."""

    def test_platform(self):
        intent = _make_intent(pattern_type="platform")
        assert BacktestOrchestrator._extract_intent_pattern_type(intent) == "platform"

    def test_trend_ma_crossover(self):
        intent = _make_intent(pattern_type="ma_crossover")
        assert BacktestOrchestrator._extract_intent_pattern_type(intent) == "ma_crossover"

    def test_missing_metadata_none(self):
        intent = _make_intent(metadata=None)
        assert BacktestOrchestrator._extract_intent_pattern_type(intent) == ""

    def test_missing_shadow_score_key(self):
        intent = _make_intent(metadata={"scanner_score": "0.85"})
        assert BacktestOrchestrator._extract_intent_pattern_type(intent) == ""

    def test_empty_shadow_score(self):
        intent = _make_intent(metadata={"shadow_score": ""})
        assert BacktestOrchestrator._extract_intent_pattern_type(intent) == ""

    def test_malformed_json(self):
        intent = _make_intent(shadow_score_raw="{bad json!!}")
        assert BacktestOrchestrator._extract_intent_pattern_type(intent) == ""

    def test_shadow_score_not_dict(self):
        intent = _make_intent(shadow_score_raw='"just_a_string"')
        assert BacktestOrchestrator._extract_intent_pattern_type(intent) == ""

    def test_no_ss_pattern_type_key(self):
        intent = _make_intent(shadow_score_raw=json.dumps({"ss_score": 0.8}))
        assert BacktestOrchestrator._extract_intent_pattern_type(intent) == ""


# ---------------------------------------------------------------------------
# B. Pattern-type filtering logic tests
# ---------------------------------------------------------------------------


def _apply_pattern_filter(
    intents: list[TradeIntent],
    is_confirmed: bool | None,
    regime_mode: str = "full",
) -> list[TradeIntent]:
    """Simulate the V28 pattern-type filter logic from orchestrator.

    This mirrors the exact filtering code in backtest/orchestrator.py lines 505-540.
    """
    if regime_mode == "none" or not intents or is_confirmed is None:
        return list(intents)

    effective_regime = "BULL_C" if is_confirmed else "BULL_U"
    filtered: list[TradeIntent] = []
    for ti in intents:
        pt = BacktestOrchestrator._extract_intent_pattern_type(ti)
        if effective_regime == "BULL_C":
            if pt in BacktestOrchestrator._TREND_PATTERN_TYPES or pt == "":
                filtered.append(ti)
        else:
            if pt == "platform" or pt == "":
                filtered.append(ti)
    return filtered


class TestPatternTypeFilter:
    """Test the V28 pattern-type filtering logic."""

    def test_bull_c_filters_platform(self):
        """BULL_C: platform intents removed, trend kept."""
        intents = [
            _make_intent(intent_id="T1", pattern_type="ma_crossover"),
            _make_intent(intent_id="T2", pattern_type="platform"),
            _make_intent(intent_id="T3", pattern_type="ma_crossover"),
            _make_intent(intent_id="T4", pattern_type="platform"),
        ]
        result = _apply_pattern_filter(intents, is_confirmed=True)
        symbols_kept = [i.intent_id for i in result]
        assert symbols_kept == ["T1", "T3"]

    def test_bull_u_filters_trend(self):
        """BULL_U: trend intents removed, platform kept."""
        intents = [
            _make_intent(intent_id="T1", pattern_type="ma_crossover"),
            _make_intent(intent_id="T2", pattern_type="platform"),
            _make_intent(intent_id="T3", pattern_type="ma_crossover"),
            _make_intent(intent_id="T4", pattern_type="platform"),
        ]
        result = _apply_pattern_filter(intents, is_confirmed=False)
        symbols_kept = [i.intent_id for i in result]
        assert symbols_kept == ["T2", "T4"]

    def test_unknown_type_kept_bull_c(self):
        """Unknown pattern_type ("") is kept in BULL_C."""
        intents = [
            _make_intent(intent_id="T1", pattern_type="ma_crossover"),
            _make_intent(intent_id="T2", metadata=None),  # unknown
        ]
        result = _apply_pattern_filter(intents, is_confirmed=True)
        assert len(result) == 2

    def test_unknown_type_kept_bull_u(self):
        """Unknown pattern_type ("") is kept in BULL_U."""
        intents = [
            _make_intent(intent_id="T1", pattern_type="platform"),
            _make_intent(intent_id="T2", metadata=None),  # unknown
        ]
        result = _apply_pattern_filter(intents, is_confirmed=False)
        assert len(result) == 2

    def test_chop_no_filter(self):
        """CHOP: is_confirmed is None → no filtering."""
        intents = [
            _make_intent(intent_id="T1", pattern_type="ma_crossover"),
            _make_intent(intent_id="T2", pattern_type="platform"),
        ]
        result = _apply_pattern_filter(intents, is_confirmed=None)
        assert len(result) == 2

    def test_bear_no_filter(self):
        """BEAR: is_confirmed is None → no filtering."""
        intents = [
            _make_intent(intent_id="T1", pattern_type="platform"),
            _make_intent(intent_id="T2", pattern_type="ma_crossover"),
        ]
        result = _apply_pattern_filter(intents, is_confirmed=None)
        assert len(result) == 2

    def test_regime_mode_none_no_filter(self):
        """regime_mode='none' → no filtering even if is_confirmed is set."""
        intents = [
            _make_intent(intent_id="T1", pattern_type="platform"),
        ]
        result = _apply_pattern_filter(intents, is_confirmed=True, regime_mode="none")
        assert len(result) == 1

    def test_empty_intents_no_error(self):
        """Empty intent list → empty result, no error."""
        result = _apply_pattern_filter([], is_confirmed=True)
        assert result == []

    def test_trend_type_kept_in_bull_c(self):
        """MA crossover trend type passes BULL_C filter."""
        intents = [
            _make_intent(intent_id="T1", pattern_type="ma_crossover"),
        ]
        result = _apply_pattern_filter(intents, is_confirmed=True)
        assert len(result) == 1


class TestTrendPatternTypesConstant:
    """Verify _TREND_PATTERN_TYPES is correctly defined."""

    def test_contains_expected_types(self):
        expected = {"ma_crossover"}
        assert BacktestOrchestrator._TREND_PATTERN_TYPES == expected

    def test_platform_not_in_trend(self):
        assert "platform" not in BacktestOrchestrator._TREND_PATTERN_TYPES


# ---------------------------------------------------------------------------
# C. V27.4 Candidate-level pattern filter tests
# ---------------------------------------------------------------------------


def _make_candidate(*, pattern_type: str | None = None, symbol: str = "TEST") -> dict:
    """Build a minimal candidate dict with optional shadow_score metadata."""
    candidate: dict = {"symbol": symbol}
    if pattern_type is not None:
        candidate["meta"] = {"shadow_score": {"ss_pattern_type": pattern_type}}
    return candidate


def _make_payload(candidates: list[dict]) -> dict:
    """Wrap candidates in the expected pipeline payload structure."""
    return {
        "candidates": {
            "candidates": candidates,
            "total_detected": len(candidates),
        },
    }


class TestExtractCandidatePatternType:
    """Test _extract_candidate_pattern_type static method."""

    def test_platform(self):
        c = _make_candidate(pattern_type="platform")
        assert BacktestOrchestrator._extract_candidate_pattern_type(c) == "platform"

    def test_trend(self):
        c = _make_candidate(pattern_type="ma_crossover")
        assert BacktestOrchestrator._extract_candidate_pattern_type(c) == "ma_crossover"

    def test_missing_meta(self):
        c = {"symbol": "X"}
        assert BacktestOrchestrator._extract_candidate_pattern_type(c) == ""

    def test_no_shadow_score(self):
        c = {"symbol": "X", "meta": {"other_key": 1}}
        assert BacktestOrchestrator._extract_candidate_pattern_type(c) == ""


class TestBuildCandidateFilter:
    """Test _build_candidate_filter for BULL_C / BULL_U regimes."""

    def test_bull_c_keeps_trend(self):
        """BULL_C: trend candidates kept, platform removed."""
        payload = _make_payload([
            _make_candidate(pattern_type="ma_crossover", symbol="A"),
            _make_candidate(pattern_type="platform", symbol="B"),
            _make_candidate(pattern_type="ma_crossover", symbol="C"),
        ])
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=True)
        result = f(payload)
        kept = result["candidates"]["candidates"]
        assert len(kept) == 2
        assert {c["symbol"] for c in kept} == {"A", "C"}

    def test_bull_c_keeps_unknown(self):
        """BULL_C: unknown (no pattern_type) candidates kept."""
        payload = _make_payload([
            _make_candidate(pattern_type="ma_crossover", symbol="A"),
            _make_candidate(symbol="B"),  # no pattern_type → ""
        ])
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=True)
        result = f(payload)
        kept = result["candidates"]["candidates"]
        assert len(kept) == 2

    def test_bull_u_keeps_platform(self):
        """BULL_U: platform candidates kept, trend removed."""
        payload = _make_payload([
            _make_candidate(pattern_type="ma_crossover", symbol="A"),
            _make_candidate(pattern_type="platform", symbol="B"),
            _make_candidate(pattern_type="ma_crossover", symbol="C"),
        ])
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=False)
        result = f(payload)
        kept = result["candidates"]["candidates"]
        assert len(kept) == 1
        assert kept[0]["symbol"] == "B"

    def test_bull_u_keeps_unknown(self):
        """BULL_U: unknown (no pattern_type) candidates kept."""
        payload = _make_payload([
            _make_candidate(pattern_type="platform", symbol="A"),
            _make_candidate(symbol="B"),  # no pattern_type → ""
        ])
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=False)
        result = f(payload)
        kept = result["candidates"]["candidates"]
        assert len(kept) == 2

    def test_preserves_payload_structure(self):
        """Filtered payload retains all top-level keys except modified candidates."""
        payload = _make_payload([_make_candidate(pattern_type="platform")])
        payload["extra_key"] = "preserve_me"
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=True)
        result = f(payload)
        assert result["extra_key"] == "preserve_me"
        assert "candidates" in result["candidates"]
        assert "total_detected" in result["candidates"]

    def test_updates_total_detected(self):
        """total_detected reflects post-filter count."""
        payload = _make_payload([
            _make_candidate(pattern_type="platform", symbol="A"),
            _make_candidate(pattern_type="ma_crossover", symbol="B"),
        ])
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=True)
        result = f(payload)
        assert result["candidates"]["total_detected"] == 1  # only trend kept

    def test_stats_recorded(self):
        """Filter function records cf_before and cf_kept attributes."""
        payload = _make_payload([
            _make_candidate(pattern_type="platform"),
            _make_candidate(pattern_type="ma_crossover"),
            _make_candidate(pattern_type="ma_crossover"),
        ])
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=True)
        f(payload)
        assert f.cf_before == 3
        assert f.cf_kept == 2  # trend only

    def test_empty_candidates(self):
        """Empty candidate list does not error."""
        payload = _make_payload([])
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=False)
        result = f(payload)
        assert result["candidates"]["candidates"] == []
        assert result["candidates"]["total_detected"] == 0
        assert f.cf_before == 0
        assert f.cf_kept == 0

    def test_missing_candidates_key(self):
        """Payload without 'candidates' key returns unchanged."""
        payload = {"other": "data"}
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=True)
        result = f(payload)
        assert result == {"other": "data"}

    def test_candidates_not_dict(self):
        """Non-dict 'candidates' value returns unchanged."""
        payload = {"candidates": "not_a_dict"}
        f = BacktestOrchestrator._build_candidate_filter(is_confirmed=True)
        result = f(payload)
        assert result["candidates"] == "not_a_dict"
