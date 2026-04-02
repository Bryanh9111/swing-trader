"""Portfolio tracking for backtests.

This module provides:
- :class:`Position` as an immutable representation of an open position.
- :class:`CompletedTrade` as an immutable record of a closed trade.
- :class:`PortfolioTracker` for managing cash, positions, and realized PnL.
"""

from __future__ import annotations

from datetime import date
import math
from typing import Final

import msgspec
import structlog

logger: Final = structlog.get_logger(__name__).bind(component="portfolio_tracker")


class Position(msgspec.Struct, frozen=True, kw_only=True):
    """Immutable representation of an open position."""

    symbol: str
    entry_price: float
    quantity: float
    entry_date: date
    stop_loss_price: float
    take_profit_price: float
    intent_id: str
    partial_exits_completed: int = 0  # 0, 1, 2
    original_quantity: float | None = None  # Preserve original size after partial exits
    scanner_score: float | None = None
    shadow_scores: dict[str, object] | None = None

    @property
    def market_value(self) -> float:
        """Return the position's value at entry (cost basis)."""

        return self.entry_price * self.quantity

    def unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized PnL at ``current_price``.

        Args:
            current_price: Latest market price for ``symbol``.

        Returns:
            Unrealized profit/loss in currency units.
        """

        return (current_price - self.entry_price) * self.quantity


class CompletedTrade(msgspec.Struct, frozen=True, kw_only=True):
    """Immutable record of a completed trade."""

    symbol: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    quantity: float
    realized_pnl: float
    exit_reason: str  # STOP_LOSS | TAKE_PROFIT | TIME_STOP
    scanner_score: float | None = None
    max_drawdown_pct: float | None = None
    shadow_scores: dict[str, object] | None = None


class PortfolioTracker:
    """Track cash, open positions, and completed trades during a backtest."""

    def __init__(self, initial_capital: float = 2000.0) -> None:
        """Create a new portfolio tracker.

        Args:
            initial_capital: Starting cash balance.
        """

        if initial_capital < 0:
            raise ValueError("initial_capital must be non-negative")

        self.cash: float = float(initial_capital)
        self.positions: dict[str, Position] = {}
        self.completed_trades: list[CompletedTrade] = []

    def add_position(
        self,
        position: Position,
        max_budget: float | None = None,
        min_position_value: float = 50.0,
    ) -> Position:
        """Open a new position and deduct cash.

        If ``max_budget`` is provided, attempt to scale down the position size
        to fit within ``min(self.cash, max_budget)`` rather than rejecting
        immediately.

        Raises:
            ValueError: If there is insufficient cash or the symbol is already open.
        """

        cost = float(position.entry_price) * float(position.quantity)
        actual_position = position

        if max_budget is not None and max_budget > 0:
            budget_limit = min(float(self.cash), float(max_budget))
        else:
            budget_limit = float(self.cash)

        if cost > budget_limit:
            if max_budget is not None and max_budget > 0 and position.entry_price > 0:
                adjusted_qty = math.floor(budget_limit / float(position.entry_price))
                adjusted_cost = float(adjusted_qty) * float(position.entry_price)
                if adjusted_qty > 0 and adjusted_cost >= float(min_position_value):
                    actual_position = msgspec.structs.replace(position, quantity=float(adjusted_qty))
                    cost = adjusted_cost
                    logger.info(
                        "position_quantity_adjusted",
                        symbol=position.symbol,
                        intent_id=position.intent_id,
                        original_qty=position.quantity,
                        adjusted_qty=float(adjusted_qty),
                        max_budget=float(max_budget),
                        budget_limit=float(budget_limit),
                        adjusted_cost=float(adjusted_cost),
                    )
                else:
                    logger.info(
                        "position_rejected_insufficient_cash",
                        symbol=position.symbol,
                        cost=cost,
                        cash=self.cash,
                        intent_id=position.intent_id,
                        max_budget=float(max_budget),
                        budget_limit=float(budget_limit),
                        min_position_value=float(min_position_value),
                    )
                    raise ValueError("insufficient cash to open position")
            else:
                logger.info(
                    "position_rejected_insufficient_cash",
                    symbol=position.symbol,
                    cost=cost,
                    cash=self.cash,
                    intent_id=position.intent_id,
                )
                raise ValueError("insufficient cash to open position")

        if position.symbol in self.positions:
            raise ValueError(f"position already exists for symbol={position.symbol}")

        self.cash -= cost
        self.positions[actual_position.symbol] = actual_position
        logger.info(
            "position_opened",
            symbol=actual_position.symbol,
            entry_price=actual_position.entry_price,
            quantity=actual_position.quantity,
            entry_date=str(actual_position.entry_date),
            intent_id=actual_position.intent_id,
            cash=self.cash,
        )
        return actual_position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: date,
        exit_reason: str,
        max_drawdown_pct: float | None = None,
    ) -> float:
        """Close an existing position, add cash proceeds, and record a completed trade.

        Args:
            symbol: Position symbol to close.
            exit_price: Exit price for the position.
            exit_date: Exit date.
            exit_reason: One of ``STOP_LOSS``, ``TAKE_PROFIT``, ``TIME_STOP``.

        Returns:
            The realized PnL for the closed position.

        Raises:
            KeyError: If there is no open position for ``symbol``.
        """

        position = self.positions.pop(symbol)
        realized_pnl = (exit_price - position.entry_price) * position.quantity
        proceeds = exit_price * position.quantity
        self.cash += proceeds

        # Defensive check: TAKE_PROFIT should always result in positive PnL
        if exit_reason == "TAKE_PROFIT" and realized_pnl <= 0:
            logger.error(
                "TAKE_PROFIT_WITH_NEGATIVE_PNL_BUG",
                symbol=symbol,
                entry_price=position.entry_price,
                exit_price=exit_price,
                take_profit_price=position.take_profit_price,
                stop_loss_price=position.stop_loss_price,
                quantity=position.quantity,
                realized_pnl=realized_pnl,
                entry_date=str(position.entry_date),
                exit_date=str(exit_date),
                intent_id=position.intent_id,
            )
            # Reclassify as GAP_LOSS to prevent misleading statistics
            exit_reason = "GAP_LOSS"

        completed = CompletedTrade(
            symbol=position.symbol,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            quantity=position.quantity,
            realized_pnl=realized_pnl,
            exit_reason=exit_reason,
            scanner_score=position.scanner_score,
            max_drawdown_pct=max_drawdown_pct,
            shadow_scores=position.shadow_scores,
        )
        self.completed_trades.append(completed)

        logger.info(
            "position_closed",
            symbol=position.symbol,
            exit_price=exit_price,
            exit_date=str(exit_date),
            realized_pnl=realized_pnl,
            exit_reason=exit_reason,
            max_drawdown_pct=max_drawdown_pct,
            cash=self.cash,
        )

        return realized_pnl

    def reduce_position(
        self,
        symbol: str,
        exit_quantity: float,
        exit_price: float,
        exit_date: date,
        exit_reason: str,
        max_drawdown_pct: float | None = None,
    ) -> float:
        """Partially close an existing position and record a completed trade for the reduced lot.

        Args:
            symbol: Position symbol to reduce.
            exit_quantity: Quantity to close (must be less than current position quantity).
            exit_price: Exit price for the reduced quantity.
            exit_date: Exit date.
            exit_reason: Exit reason label (e.g., PARTIAL_TP_1).

        Returns:
            The realized PnL for the reduced quantity.

        Raises:
            KeyError: If there is no open position for ``symbol``.
            ValueError: If ``exit_quantity`` is not positive.
        """

        if exit_quantity <= 0:
            raise ValueError("exit_quantity must be positive")

        position = self.positions[symbol]
        if exit_quantity >= position.quantity:
            return self.close_position(symbol, exit_price, exit_date, exit_reason, max_drawdown_pct=max_drawdown_pct)

        realized_pnl = (exit_price - position.entry_price) * exit_quantity
        proceeds = exit_price * exit_quantity
        self.cash += proceeds

        partial_trade = CompletedTrade(
            symbol=symbol,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            quantity=exit_quantity,
            realized_pnl=realized_pnl,
            exit_reason=exit_reason,
            scanner_score=position.scanner_score,
            max_drawdown_pct=max_drawdown_pct,
            shadow_scores=position.shadow_scores,
        )
        self.completed_trades.append(partial_trade)

        remaining_qty = position.quantity - exit_quantity
        original_qty = position.original_quantity if position.original_quantity is not None else position.quantity
        updated = msgspec.structs.replace(
            position,
            quantity=float(remaining_qty),
            original_quantity=float(original_qty),
            partial_exits_completed=int(position.partial_exits_completed) + 1,
        )
        self.positions[symbol] = updated

        logger.info(
            "partial_position_closed",
            symbol=symbol,
            exit_quantity=exit_quantity,
            remaining_quantity=remaining_qty,
            exit_price=exit_price,
            exit_date=str(exit_date),
            exit_reason=exit_reason,
            partial_exits_completed=updated.partial_exits_completed,
            cash=self.cash,
        )
        return realized_pnl

    def get_total_equity(self) -> float:
        """Return total equity using cost-basis valuation for open positions."""

        return self.cash + sum(position.market_value for position in self.positions.values())

    def get_unrealized_pnl(self, market_prices: dict[str, float]) -> float:
        """Return total unrealized PnL across open positions.

        Args:
            market_prices: Mapping of ``symbol`` to current price.

        Returns:
            Sum of unrealized PnL for positions that have a corresponding market price.
        """

        total = 0.0
        for symbol, position in self.positions.items():
            price = market_prices.get(symbol)
            if price is None:
                continue
            total += position.unrealized_pnl(price)
        return total
