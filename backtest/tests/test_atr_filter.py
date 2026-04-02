"""V27.5: ATR% hard filter tests for trend candidates.

Tests verify:
- _extract_candidate_atr_pct() correctly parses shadow_score ATR% values
- ATR% filtering rejects out-of-range trend candidates
- Platform candidates are unaffected (fail-open)
- None ATR% values pass through (fail-open)
- BULL_C gets ATR filter only (no pattern filter)
- BULL_U gets pattern + ATR filter
- Backward compatibility when no ATR config is present
"""
from __future__ import annotations

import pytest

from backtest.orchestrator import BacktestOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    *,
    pattern_type: str | None = None,
    atr_pct: float | None = None,
    symbol: str = "TEST",
) -> dict:
    """Build a minimal candidate dict with optional shadow_score metadata."""
    candidate: dict = {"symbol": symbol}
    ss: dict = {}
    if pattern_type is not None:
        ss["ss_pattern_type"] = pattern_type
    if atr_pct is not None:
        ss["ss_atr_pct"] = atr_pct
    if ss:
        candidate["meta"] = {"shadow_score": ss}
    return candidate


def _make_payload(candidates: list[dict]) -> dict:
    """Wrap candidates in the expected pipeline payload structure."""
    return {
        "candidates": {
            "candidates": candidates,
            "total_detected": len(candidates),
        },
    }


# ---------------------------------------------------------------------------
# 1. _extract_candidate_atr_pct helper tests
# ---------------------------------------------------------------------------

class TestExtractCandidateAtrPct:
    """Test _extract_candidate_atr_pct static method."""

    def test_normal_float(self):
        c = _make_candidate(pattern_type="ma_crossover", atr_pct=0.055)
        assert BacktestOrchestrator._extract_candidate_atr_pct(c) == pytest.approx(0.055)

    def test_none_when_missing_meta(self):
        c = {"symbol": "X"}
        assert BacktestOrchestrator._extract_candidate_atr_pct(c) is None

    def test_none_when_missing_shadow_score(self):
        c = {"symbol": "X", "meta": {"other_key": 1}}
        assert BacktestOrchestrator._extract_candidate_atr_pct(c) is None

    def test_none_when_missing_atr_pct_key(self):
        c = {"symbol": "X", "meta": {"shadow_score": {"ss_pattern_type": "ma_crossover"}}}
        assert BacktestOrchestrator._extract_candidate_atr_pct(c) is None

    def test_string_coerced_to_float(self):
        c = {"symbol": "X", "meta": {"shadow_score": {"ss_atr_pct": "0.04"}}}
        assert BacktestOrchestrator._extract_candidate_atr_pct(c) == pytest.approx(0.04)

    def test_non_numeric_returns_none(self):
        c = {"symbol": "X", "meta": {"shadow_score": {"ss_atr_pct": "bad"}}}
        assert BacktestOrchestrator._extract_candidate_atr_pct(c) is None

    def test_meta_not_dict(self):
        c = {"symbol": "X", "meta": "not_a_dict"}
        assert BacktestOrchestrator._extract_candidate_atr_pct(c) is None

    def test_shadow_score_not_dict(self):
        c = {"symbol": "X", "meta": {"shadow_score": "not_a_dict"}}
        assert BacktestOrchestrator._extract_candidate_atr_pct(c) is None


# ---------------------------------------------------------------------------
# 2-5. ATR% filter core logic tests
# ---------------------------------------------------------------------------

class TestAtrFilterRejectsOutOfRange:
    """ATR% filter rejects trend candidates outside the configured range."""

    def test_rejects_below_min(self):
        """ATR% < min → trend candidate filtered out."""
        payload = _make_payload([
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.03, symbol="LOW"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.05, symbol="OK"),
        ])
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=True, pattern_filter=False,
            trend_atr_pct_min=0.04, trend_atr_pct_max=0.07,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        assert len(kept) == 1
        assert kept[0]["symbol"] == "OK"

    def test_rejects_above_max(self):
        """ATR% > max → trend candidate filtered out."""
        payload = _make_payload([
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.09, symbol="HIGH"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.05, symbol="OK"),
        ])
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=True, pattern_filter=False,
            trend_atr_pct_min=0.04, trend_atr_pct_max=0.07,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        assert len(kept) == 1
        assert kept[0]["symbol"] == "OK"

    def test_passes_in_range(self):
        """ATR% within [min, max] → trend candidate passes."""
        payload = _make_payload([
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.04, symbol="AT_MIN"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.055, symbol="MID"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.07, symbol="AT_MAX"),
        ])
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=True, pattern_filter=False,
            trend_atr_pct_min=0.04, trend_atr_pct_max=0.07,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        assert len(kept) == 3

    def test_passes_none_atr(self):
        """ss_atr_pct=None → candidate passes (fail-open)."""
        payload = _make_payload([
            _make_candidate(pattern_type="ma_crossover", symbol="NO_ATR"),  # no atr_pct
        ])
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=True, pattern_filter=False,
            trend_atr_pct_min=0.04, trend_atr_pct_max=0.07,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        assert len(kept) == 1
        assert kept[0]["symbol"] == "NO_ATR"

    def test_ignores_platform(self):
        """Platform candidates are NOT subject to ATR% filter."""
        payload = _make_payload([
            _make_candidate(pattern_type="platform", atr_pct=0.02, symbol="PLAT_LOW"),
            _make_candidate(pattern_type="platform", atr_pct=0.10, symbol="PLAT_HIGH"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.02, symbol="TREND_LOW"),
        ])
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=True, pattern_filter=False,
            trend_atr_pct_min=0.04, trend_atr_pct_max=0.07,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        symbols = {c["symbol"] for c in kept}
        # Both platforms pass; trend with low ATR filtered
        assert symbols == {"PLAT_LOW", "PLAT_HIGH"}


# ---------------------------------------------------------------------------
# 6-7. Regime-day interaction tests
# ---------------------------------------------------------------------------

class TestRegimeDayFilterCombination:
    """Test filter behavior on BULL_C vs BULL_U days."""

    def test_bull_c_gets_atr_filter_only(self):
        """BULL_C: no pattern filter, only ATR filter on trend candidates."""
        payload = _make_payload([
            _make_candidate(pattern_type="platform", atr_pct=0.03, symbol="PLAT"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.05, symbol="TREND_OK"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.02, symbol="TREND_LOW"),
        ])
        # BULL_C: is_confirmed=True, pattern_filter=False
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=True, pattern_filter=False,
            trend_atr_pct_min=0.04, trend_atr_pct_max=0.07,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        symbols = {c["symbol"] for c in kept}
        # Platform passes (no ATR filter), TREND_OK passes, TREND_LOW filtered
        assert symbols == {"PLAT", "TREND_OK"}

    def test_bull_u_gets_pattern_and_atr(self):
        """BULL_U: pattern filter + ATR filter applied."""
        payload = _make_payload([
            _make_candidate(pattern_type="platform", atr_pct=0.03, symbol="PLAT"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.05, symbol="TREND_OK"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.02, symbol="TREND_LOW"),
            _make_candidate(pattern_type="ma_crossover", atr_pct=0.05, symbol="TREND2"),
        ])
        # BULL_U: is_confirmed=False, pattern_filter=True
        # Pattern filter: only platform + unknown kept (trend removed)
        # ATR filter would apply to trend but they're already gone from pattern filter
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=False, pattern_filter=True,
            trend_atr_pct_min=0.04, trend_atr_pct_max=0.07,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        symbols = {c["symbol"] for c in kept}
        # BULL_U pattern filter removes all trend; only platform survives
        assert symbols == {"PLAT"}


# ---------------------------------------------------------------------------
# 8. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """No ATR config → V27.4.4 identical behavior."""

    def test_no_atr_config_no_filter_on_bull_c(self):
        """Without ATR config, BULL_C with pattern_filter=False passes everything."""
        payload = _make_payload([
            _make_candidate(pattern_type="platform", symbol="A"),
            _make_candidate(pattern_type="ma_crossover", symbol="B"),
            _make_candidate(pattern_type="ma_crossover", symbol="C"),
            _make_candidate(symbol="D"),  # unknown
        ])
        # No ATR params, no pattern filter = pass-through
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=True, pattern_filter=False,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        assert len(kept) == 4

    def test_no_atr_config_bull_u_pattern_only(self):
        """Without ATR config, BULL_U pattern filter works exactly as V27.4."""
        payload = _make_payload([
            _make_candidate(pattern_type="platform", symbol="A"),
            _make_candidate(pattern_type="ma_crossover", symbol="B"),
            _make_candidate(symbol="C"),  # unknown
        ])
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=False, pattern_filter=True,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        symbols = {c["symbol"] for c in kept}
        # Platform + unknown kept, trend removed
        assert symbols == {"A", "C"}

    def test_no_atr_config_bull_c_pattern_only(self):
        """Without ATR config, BULL_C pattern filter works exactly as V27.4."""
        payload = _make_payload([
            _make_candidate(pattern_type="platform", symbol="A"),
            _make_candidate(pattern_type="ma_crossover", symbol="B"),
            _make_candidate(symbol="C"),  # unknown
        ])
        f = BacktestOrchestrator._build_candidate_filter(
            is_confirmed=True, pattern_filter=True,
        )
        result = f(payload)
        kept = result["candidates"]["candidates"]
        symbols = {c["symbol"] for c in kept}
        # Trend + unknown kept, platform removed
        assert symbols == {"B", "C"}
