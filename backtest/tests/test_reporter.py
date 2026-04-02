from __future__ import annotations

import csv
from datetime import date

from backtest.portfolio_tracker import CompletedTrade
from backtest.reporter import BacktestReporter


def _trade(
    *,
    symbol: str = "AAPL",
    entry_date: date,
    entry_price: float = 100.0,
    exit_date: date,
    exit_price: float = 110.0,
    quantity: float = 1.0,
    realized_pnl: float,
    exit_reason: str = "TAKE_PROFIT",
    scanner_score: float | None = None,
) -> CompletedTrade:
    return CompletedTrade(
        symbol=symbol,
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=exit_date,
        exit_price=exit_price,
        quantity=quantity,
        realized_pnl=realized_pnl,
        exit_reason=exit_reason,
        scanner_score=scanner_score,
    )


def test_generate_stats_empty_trades() -> None:
    stats = BacktestReporter.generate_stats([], initial_capital=1000.0)
    assert stats.total_trades == 0
    assert stats.winning_trades == 0
    assert stats.losing_trades == 0
    assert stats.win_rate == 0.0
    assert stats.total_pnl == 0.0
    assert stats.total_pnl_pct == 0.0
    assert stats.avg_win == 0.0
    assert stats.avg_loss == 0.0
    assert stats.max_win == 0.0
    assert stats.max_loss == 0.0
    assert stats.avg_hold_days == 0.0


def test_generate_stats_all_wins_boundary() -> None:
    trades = [
        _trade(entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 3), realized_pnl=10.0),
        _trade(entry_date=date(2024, 1, 10), exit_date=date(2024, 1, 11), realized_pnl=20.0),
    ]
    stats = BacktestReporter.generate_stats(trades, initial_capital=100.0)
    assert stats.total_trades == 2
    assert stats.winning_trades == 2
    assert stats.losing_trades == 0
    assert stats.win_rate == 1.0
    assert stats.total_pnl == 30.0
    assert stats.total_pnl_pct == 0.3
    assert stats.avg_win == 15.0
    assert stats.avg_loss == 0.0
    assert stats.max_win == 20.0
    assert stats.max_loss == 0.0
    assert stats.avg_hold_days == 1.5


def test_generate_stats_all_losses_boundary() -> None:
    trades = [
        _trade(entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 2), realized_pnl=0.0),
        _trade(entry_date=date(2024, 1, 5), exit_date=date(2024, 1, 7), realized_pnl=-5.0),
    ]
    stats = BacktestReporter.generate_stats(trades, initial_capital=100.0)
    assert stats.total_trades == 2
    assert stats.winning_trades == 0
    assert stats.losing_trades == 2
    assert stats.win_rate == 0.0
    assert stats.total_pnl == -5.0
    assert stats.total_pnl_pct == -0.05
    assert stats.avg_win == 0.0
    assert stats.avg_loss == 2.5
    assert stats.max_win == 0.0
    assert stats.max_loss == 5.0
    assert stats.avg_hold_days == 1.5


def test_save_trades_csv(tmp_path) -> None:
    trades = [
        _trade(symbol="MSFT", entry_date=date(2024, 1, 10), exit_date=date(2024, 1, 12), realized_pnl=5.0),
        _trade(symbol="AAPL", entry_date=date(2024, 1, 1), exit_date=date(2024, 1, 3), realized_pnl=-2.0),
    ]
    output_path = tmp_path / "trades.csv"
    BacktestReporter.save_trades_csv(trades, str(output_path))

    with output_path.open("r", encoding="utf-8", newline="") as file:
        header = file.readline().strip()
        assert header == (
            "symbol,entry_date,entry_price,exit_date,exit_price,quantity,realized_pnl,"
            "hold_days,exit_reason,scanner_score,max_drawdown_pct,"
            "ss_version,ss_atr_pct,ss_box_quality,ss_volatility,ss_volume_chg_ratio,"
            "ss_rs_slope,ss_rs_lookback,ss_consolidation_days,ss_dist_to_support_pct,ss_pattern_type,ss_market_cap"
        )
        file.seek(0)
        reader = csv.DictReader(file)
        rows = list(reader)

    assert [row["symbol"] for row in rows] == ["AAPL", "MSFT"]
    assert rows[0]["entry_date"] == "2024-01-01"
    assert int(rows[0]["hold_days"]) == 2
    assert float(rows[0]["realized_pnl"]) == -2.0
    # shadow_scores columns present but empty when not set
    assert rows[0]["ss_atr_pct"] == ""
    assert rows[0]["ss_pattern_type"] == ""


def test_save_trades_csv_shadow_scores(tmp_path) -> None:
    """Shadow scores are flattened into 8 CSV columns."""
    shadow = {
        "ss_version": 1,
        "ss_atr_pct": 3.5,
        "ss_box_quality": 0.82,
        "ss_volatility": 0.04,
        "ss_volume_chg_ratio": 1.2,
        "ss_rs_slope": 0.015,
        "ss_rs_lookback": 20,
        "ss_consolidation_days": 25,
        "ss_dist_to_support_pct": 0.02,
        "ss_pattern_type": "platform",
    }
    trades = [
        CompletedTrade(
            symbol="TEST",
            entry_date=date(2024, 3, 1),
            entry_price=50.0,
            exit_date=date(2024, 3, 10),
            exit_price=55.0,
            quantity=10.0,
            realized_pnl=50.0,
            exit_reason="TAKE_PROFIT",
            shadow_scores=shadow,
        ),
    ]
    output_path = tmp_path / "shadow.csv"
    BacktestReporter.save_trades_csv(trades, str(output_path))

    with output_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert int(row["ss_version"]) == 1
    assert float(row["ss_atr_pct"]) == 3.5
    assert float(row["ss_box_quality"]) == 0.82
    assert float(row["ss_volatility"]) == 0.04
    assert float(row["ss_volume_chg_ratio"]) == 1.2
    assert float(row["ss_rs_slope"]) == 0.015
    assert int(row["ss_rs_lookback"]) == 20
    assert int(row["ss_consolidation_days"]) == 25
    assert float(row["ss_dist_to_support_pct"]) == 0.02
    assert row["ss_pattern_type"] == "platform"


def test_save_trades_csv_shadow_scores_partial_none(tmp_path) -> None:
    """Missing shadow_score keys output as empty string."""
    shadow = {
        "ss_version": 1,
        "ss_atr_pct": 2.0,
        "ss_box_quality": None,
        "ss_volatility": None,
        "ss_volume_chg_ratio": None,
        "ss_rs_slope": None,
        "ss_rs_lookback": 20,
        "ss_consolidation_days": 30,
        "ss_dist_to_support_pct": None,
        "ss_pattern_type": "ma_crossover",
    }
    trades = [
        CompletedTrade(
            symbol="TREND",
            entry_date=date(2024, 5, 1),
            entry_price=100.0,
            exit_date=date(2024, 5, 8),
            exit_price=95.0,
            quantity=5.0,
            realized_pnl=-25.0,
            exit_reason="STOP_LOSS",
            shadow_scores=shadow,
        ),
    ]
    output_path = tmp_path / "partial.csv"
    BacktestReporter.save_trades_csv(trades, str(output_path))

    with output_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    row = rows[0]
    assert float(row["ss_atr_pct"]) == 2.0
    assert row["ss_box_quality"] == ""
    assert row["ss_rs_slope"] == ""
    assert int(row["ss_consolidation_days"]) == 30
    assert row["ss_pattern_type"] == "ma_crossover"
