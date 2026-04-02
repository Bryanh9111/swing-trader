"""V27.7: ATR-aware Position Sizing — unit tests for _compute_atr_sizing_multiplier."""

from __future__ import annotations

import pytest

from backtest.orchestrator import BacktestOrchestrator


# --- Helper: shortcut to the static method under test ---
_compute = BacktestOrchestrator._compute_atr_sizing_multiplier

# Default config matching bull_market.yaml V27.7
_DEFAULT_CFG: dict = {
    "enabled": True,
    "buckets": [
        {"max_atr_pct": 0.03, "multiplier": 0.8},
        {"max_atr_pct": 0.05, "multiplier": 1.0},
        {"max_atr_pct": 0.08, "multiplier": 1.2},
    ],
}


class TestComputeAtrSizingMultiplier:
    """Unit tests for the bucket-matching logic."""

    def test_low_atr_gets_reduced(self):
        assert _compute(0.02, _DEFAULT_CFG) == 0.8

    def test_mid_atr_unchanged(self):
        assert _compute(0.04, _DEFAULT_CFG) == 1.0

    def test_high_atr_gets_boosted(self):
        assert _compute(0.06, _DEFAULT_CFG) == 1.2

    def test_boundary_exact_low(self):
        """ATR% exactly at bucket boundary (0.03) matches first bucket."""
        assert _compute(0.03, _DEFAULT_CFG) == 0.8

    def test_boundary_exact_mid(self):
        """ATR% exactly at 0.05 matches second bucket."""
        assert _compute(0.05, _DEFAULT_CFG) == 1.0

    def test_boundary_exact_high(self):
        """ATR% exactly at 0.08 matches third bucket."""
        assert _compute(0.08, _DEFAULT_CFG) == 1.2

    def test_above_all_buckets_fallback(self):
        """ATR% above all bucket max → fallback 1.0."""
        assert _compute(0.10, _DEFAULT_CFG) == 1.0

    def test_atr_none_fallback(self):
        """Missing ATR% → fallback 1.0."""
        assert _compute(None, _DEFAULT_CFG) == 1.0

    def test_config_none_fallback(self):
        """No config → fallback 1.0."""
        assert _compute(0.02, None) == 1.0

    def test_config_disabled_fallback(self):
        """Config exists but disabled → fallback 1.0."""
        cfg = {**_DEFAULT_CFG, "enabled": False}
        assert _compute(0.02, cfg) == 1.0

    def test_empty_buckets_fallback(self):
        """Config with empty buckets list → fallback 1.0."""
        cfg = {"enabled": True, "buckets": []}
        assert _compute(0.02, cfg) == 1.0

    def test_missing_buckets_key_fallback(self):
        """Config with no buckets key → fallback 1.0."""
        cfg = {"enabled": True}
        assert _compute(0.02, cfg) == 1.0

    def test_malformed_bucket_skipped(self):
        """Malformed bucket entry (not a dict) is skipped."""
        cfg = {
            "enabled": True,
            "buckets": [
                "not_a_dict",
                {"max_atr_pct": 0.05, "multiplier": 1.0},
            ],
        }
        assert _compute(0.02, cfg) == 1.0  # first bucket skipped, second matches

    def test_zero_atr(self):
        """ATR% of 0 matches first bucket."""
        assert _compute(0.0, _DEFAULT_CFG) == 0.8

    def test_very_small_atr(self):
        """Very small ATR% matches first bucket."""
        assert _compute(0.001, _DEFAULT_CFG) == 0.8

    def test_bucket_order_matters(self):
        """Buckets are matched in order — first match wins."""
        cfg = {
            "enabled": True,
            "buckets": [
                {"max_atr_pct": 0.05, "multiplier": 0.9},
                {"max_atr_pct": 0.03, "multiplier": 0.7},  # never reached for atr=0.02
            ],
        }
        assert _compute(0.02, cfg) == 0.9  # first bucket matches
