"""Reporting utilities for backtest results.

This module provides:
- :class:`BacktestStats` for summarized performance statistics.
- :class:`BacktestReporter` for generating stats and exporting completed trades to CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import csv

import msgspec
import structlog

from backtest.portfolio_tracker import CompletedTrade

logger: Final = structlog.get_logger(__name__).bind(component="backtest_reporter")


class BacktestStats(msgspec.Struct, frozen=True, kw_only=True):
    """Aggregate statistics for a backtest run."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_hold_days: float
    sharpe_ratio: float | None


class BacktestReporter:
    """Generate summary stats and export completed trades for a backtest."""

    @staticmethod
    def generate_stats(completed_trades: list[CompletedTrade], initial_capital: float) -> BacktestStats:
        """Compute aggregate backtest statistics.

        Args:
            completed_trades: List of closed trades to summarize.
            initial_capital: Starting cash used to compute ``total_pnl_pct``.

        Returns:
            A :class:`BacktestStats` instance containing computed metrics.

        Raises:
            ValueError: If ``initial_capital`` is negative.
        """

        if initial_capital < 0:
            raise ValueError("initial_capital must be non-negative")

        total_trades = len(completed_trades)
        pnls = [float(trade.realized_pnl) for trade in completed_trades]
        total_pnl = float(sum(pnls))
        total_pnl_pct = 0.0 if initial_capital == 0 else total_pnl / float(initial_capital)

        winning_pnls = [pnl for pnl in pnls if pnl > 0]
        losing_pnls = [pnl for pnl in pnls if pnl <= 0]

        winning_trades = len(winning_pnls)
        losing_trades = len(losing_pnls)
        win_rate = 0.0 if total_trades == 0 else winning_trades / total_trades

        avg_win = 0.0 if not winning_pnls else sum(winning_pnls) / len(winning_pnls)
        avg_loss = 0.0 if not losing_pnls else sum(abs(p) for p in losing_pnls) / len(losing_pnls)
        max_win = 0.0 if not winning_pnls else max(winning_pnls)
        max_loss = 0.0 if not losing_pnls else max(abs(p) for p in losing_pnls)

        hold_days = [(trade.exit_date - trade.entry_date).days for trade in completed_trades]
        avg_hold_days = 0.0 if not hold_days else sum(hold_days) / len(hold_days)

        stats = BacktestStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=float(win_rate),
            total_pnl=total_pnl,
            total_pnl_pct=float(total_pnl_pct),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            max_win=float(max_win),
            max_loss=float(max_loss),
            avg_hold_days=float(avg_hold_days),
            sharpe_ratio=None,
        )

        logger.info(
            "backtest_stats_generated",
            initial_capital=initial_capital,
            **{field: getattr(stats, field) for field in type(stats).__struct_fields__},
        )
        return stats

    @staticmethod
    def save_trades_csv(completed_trades: list[CompletedTrade], output_path: str) -> None:
        """Save completed trades to a CSV file.

        Args:
            completed_trades: Trades to export.
            output_path: Destination CSV file path.
        """

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        _shadow_keys = [
            "ss_version",
            "ss_atr_pct",
            "ss_box_quality",
            "ss_volatility",
            "ss_volume_chg_ratio",
            "ss_rs_slope",
            "ss_rs_lookback",
            "ss_consolidation_days",
            "ss_dist_to_support_pct",
            "ss_pattern_type",
            "ss_market_cap",
        ]

        fieldnames = [
            "symbol",
            "entry_date",
            "entry_price",
            "exit_date",
            "exit_price",
            "quantity",
            "realized_pnl",
            "hold_days",
            "exit_reason",
            "scanner_score",
            "max_drawdown_pct",
            *_shadow_keys,
        ]

        trades_sorted = sorted(completed_trades, key=lambda trade: trade.entry_date)

        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades_sorted:
                hold_days = (trade.exit_date - trade.entry_date).days
                ss = trade.shadow_scores or {}
                row = {
                    "symbol": trade.symbol,
                    "entry_date": trade.entry_date.isoformat(),
                    "entry_price": trade.entry_price,
                    "exit_date": trade.exit_date.isoformat(),
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "realized_pnl": trade.realized_pnl,
                    "hold_days": hold_days,
                    "exit_reason": trade.exit_reason,
                    "scanner_score": trade.scanner_score,
                    "max_drawdown_pct": trade.max_drawdown_pct,
                }
                for key in _shadow_keys:
                    val = ss.get(key)
                    row[key] = "" if val is None else val
                writer.writerow(row)

        logger.info("backtest_trades_csv_saved", output_path=str(path), trades=len(trades_sorted))
