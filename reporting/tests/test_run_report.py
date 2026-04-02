"""Tests for RunReport generation, serialisation, and Markdown rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reporting.run_report import (
    FillSummary,
    RunReport,
    RunReportGenerator,
    render_markdown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_metadata(
    *,
    run_id: str = "20260212-090000-PRE_MARKET_FULL_SCAN-abc123",
    run_type: str = "PRE_MARKET_FULL_SCAN",
    mode: str = "PAPER",
    system_version: str = "abc1234",
    start_time: int = 1_000_000_000_000_000_000,
    end_time: int = 1_000_012_300_000_000_000,
    status: str = "completed",
) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        run_type=run_type,
        mode=mode,
        system_version=system_version,
        start_time=start_time,
        end_time=end_time,
        status=status,
    )


def _make_outputs(
    *,
    intents: int = 2,
    fills: int = 1,
    blocks: int = 1,
    cash: float | None = 48200.0,
) -> dict:
    intent_groups = [{"symbol": f"SYM{i}"} for i in range(intents)]

    reports_list = []
    for i in range(fills):
        reports_list.append({
            "symbol": f"AAPL{i}",
            "side": "BUY",
            "status": "filled",
            "filled_qty": 50,
            "avg_fill_price": 185.20,
            "reason": "entry",
        })

    decisions = []
    for i in range(blocks):
        decisions.append({"action": "BLOCK", "reason": "max_exposure"})

    return {
        "market_regime.json": {"regime": "TREND", "confidence": 0.85},
        "intents.json": {"intent_groups": intent_groups},
        "risk_decisions.json": {"decisions": decisions},
        "execution_reports.json": {
            "reports": reports_list,
            "account_summary": {
                "cash": cash,
                "gross_exposure": 51800.0,
                "open_positions_count": 3,
                "realized_pnl_today": 120.0,
                "unrealized_pnl": -50.0,
            },
        },
    }


@pytest.fixture
def generator(tmp_path: Path) -> RunReportGenerator:
    return RunReportGenerator(base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Test: generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_full_report_fields(self, generator: RunReportGenerator):
        meta = _make_metadata()
        outputs = _make_outputs()
        report = generator.generate("run-1", meta, [], outputs, [])

        assert report.meta.run_id == "run-1"
        assert report.meta.mode == "PAPER"
        assert report.meta.run_type == "PRE_MARKET_FULL_SCAN"
        assert report.meta.status == "completed"
        assert report.meta.duration_ms > 0
        assert report.meta.git_sha == "abc1234"

    def test_market_regime(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], _make_outputs(), [])
        assert report.market.regime == "TREND"
        assert report.market.regime_confidence == 0.85

    def test_actions_counts(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], _make_outputs(intents=3, fills=2, blocks=2), [])
        assert report.actions.intents_generated == 3
        assert report.actions.fills_entry_count == 2
        assert report.actions.riskgate_blocks_count == 2
        assert report.actions.top_block_reasons == {"max_exposure": 2}

    def test_portfolio_extraction(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], _make_outputs(cash=48200.0), [])
        assert report.portfolio.cash == 48200.0
        assert report.portfolio.gross_exposure == 51800.0
        assert report.portfolio.open_positions_count == 3
        assert report.portfolio.realized_pnl_today == 120.0
        assert report.portfolio.unrealized_pnl == -50.0

    def test_fills_extraction(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], _make_outputs(fills=2), [])
        assert len(report.trades.fills) == 2
        assert report.trades.fills[0].symbol == "AAPL0"
        assert report.trades.fills[0].qty == 50

    def test_risk_all_none(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], _make_outputs(), [])
        assert report.risk.sl_count_5d is None
        assert report.risk.sl_count_10d is None
        assert report.risk.consecutive_losses is None

    def test_empty_outputs(self, generator: RunReportGenerator):
        """Empty run (no fills, no intents) should still generate valid report."""
        report = generator.generate("r", _make_metadata(), [], {}, [])
        assert report.meta.run_id == "r"
        assert report.actions.intents_generated == 0
        assert report.actions.fills_entry_count == 0
        assert len(report.trades.fills) == 0

    def test_missing_regime(self, generator: RunReportGenerator):
        outputs = _make_outputs()
        del outputs["market_regime.json"]
        report = generator.generate("r", _make_metadata(), [], outputs, [])
        assert report.market.regime == "unknown"
        assert report.market.regime_confidence is None


# ---------------------------------------------------------------------------
# Test: save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_creates_directory_and_files(self, generator: RunReportGenerator):
        meta = _make_metadata()
        outputs = _make_outputs()
        report = generator.generate("run-1", meta, [], outputs, [])
        result = generator.save(report)

        assert result.status.value == "success"
        out_dir = result.data
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "summary.md").exists()

    def test_json_is_valid(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], _make_outputs(), [])
        result = generator.save(report)
        json_path = result.data / "summary.json"
        data = json.loads(json_path.read_text())
        assert data["meta"]["run_id"] == "r"
        assert "market" in data
        assert "actions" in data

    def test_slot_mapping_eod(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(run_type="PRE_MARKET_FULL_SCAN"), [], _make_outputs(), [])
        result = generator.save(report)
        assert "run_0900" in str(result.data)

    def test_slot_mapping_intraday(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(run_type="INTRADAY_CHECK_1030"), [], _make_outputs(), [])
        result = generator.save(report)
        assert "run_1030" in str(result.data)


# ---------------------------------------------------------------------------
# Test: render_markdown()
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_contains_section_headers(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], _make_outputs(fills=1), [])
        md = render_markdown(report)
        assert "## Meta" in md
        assert "## Actions" in md
        assert "## Fills" in md
        assert "## Risk Diagnostics" in md
        assert "## Market" in md
        assert "## Portfolio" in md

    def test_contains_fill_table(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], _make_outputs(fills=1), [])
        md = render_markdown(report)
        assert "AAPL0" in md
        assert "BUY" in md
        assert "50" in md

    def test_no_fills_shows_placeholder(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], _make_outputs(fills=0), [])
        md = render_markdown(report)
        assert "(no fills)" in md

    def test_risk_v1_placeholder(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], {}, [])
        md = render_markdown(report)
        assert "V1" in md

    def test_empty_portfolio(self, generator: RunReportGenerator):
        report = generator.generate("r", _make_metadata(), [], {}, [])
        md = render_markdown(report)
        assert "no portfolio data" in md
