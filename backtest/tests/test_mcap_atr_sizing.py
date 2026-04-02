"""Tests for _extract_market_cap_map with static JSON fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repo root is on sys.path (same pattern as conftest.py)
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from backtest.orchestrator import BacktestOrchestrator


def _make_orchestrator(
    universe_equities: list | None = None,
    output_dir: str = "/tmp/_test_no_static",
) -> BacktestOrchestrator:
    """Create a minimal BacktestOrchestrator with mocked dependencies."""
    mock_eod = MagicMock()
    if universe_equities is not None:
        mock_eod.last_outputs_by_module = {
            "universe": {"equities": universe_equities}
        }
    else:
        mock_eod.last_outputs_by_module = {}

    orch = BacktestOrchestrator(
        eod_orchestrator=mock_eod,
        initial_capital=10000.0,
        quiet=True,
        output_dir=output_dir,
    )
    return orch


# ── _extract_market_cap_map: universe source ─────────────────────────────


class TestExtractMarketCapMap:
    def test_normal(self):
        """Normal universe data → correct mapping."""
        equities = [
            {"symbol": "AAPL", "market_cap": 3e12},
            {"symbol": "MSFT", "market_cap": 2.8e12},
            {"symbol": "XYZ", "market_cap": 5e9},
        ]
        orch = _make_orchestrator(equities)
        result = orch._extract_market_cap_map()
        assert result == {"AAPL": 3e12, "MSFT": 2.8e12, "XYZ": 5e9}

    def test_empty(self):
        """No universe data → empty dict."""
        orch = _make_orchestrator(None)
        result = orch._extract_market_cap_map()
        assert result == {}

    def test_missing_fields(self):
        """Partial/missing fields → skipped gracefully."""
        equities = [
            {"symbol": "AAPL"},  # no market_cap
            {"market_cap": 5e9},  # no symbol
            {"symbol": "GOOD", "market_cap": 3e9},
            {"symbol": "NEG", "market_cap": -100},  # negative
            {"symbol": "ZERO", "market_cap": 0},  # zero
            "not_a_dict",  # wrong type
        ]
        orch = _make_orchestrator(equities)
        result = orch._extract_market_cap_map()
        assert result == {"GOOD": 3e9}


# ── _extract_market_cap_map: static fallback ─────────────────────────────


class TestStaticMarketCapFallback:
    """Tests for _extract_market_cap_map static JSON fallback."""

    def _write_static_json(self, path: Path, data: dict) -> None:
        payload = {
            "fetched_at": "2026-02-10T00:00:00+00:00",
            "source": "polygon_ticker_details",
            "data": data,
        }
        path.write_text(json.dumps(payload))

    def test_fallback_from_static_when_universe_empty(self, tmp_path):
        """Universe empty + static file exists → reads from static."""
        orch = _make_orchestrator(None)
        orch._output_dir = str(tmp_path)
        self._write_static_json(tmp_path / "market_cap_static.json", {
            "AAPL": 3e12,
            "MSFT": 2.8e12,
        })
        result = orch._extract_market_cap_map()
        assert result == {"AAPL": 3e12, "MSFT": 2.8e12}

    def test_universe_overrides_static(self, tmp_path):
        """Universe has value for symbol → universe wins over static."""
        equities = [{"symbol": "AAPL", "market_cap": 3.5e12}]
        orch = _make_orchestrator(equities)
        orch._output_dir = str(tmp_path)
        self._write_static_json(tmp_path / "market_cap_static.json", {
            "AAPL": 3e12,       # should be overridden by universe
            "NEWCO": 1e9,       # only in static → should appear
        })
        result = orch._extract_market_cap_map()
        assert result["AAPL"] == pytest.approx(3.5e12)
        assert result["NEWCO"] == pytest.approx(1e9)

    def test_missing_static_file_degrades_gracefully(self):
        """Static file missing → returns only universe data (existing behavior)."""
        orch = _make_orchestrator(None)
        orch._output_dir = "/nonexistent/path"
        result = orch._extract_market_cap_map()
        assert result == {}
