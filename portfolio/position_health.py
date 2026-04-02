"""Position health scoring for intelligent position rotation."""

from __future__ import annotations

from datetime import date

import msgspec

from backtest.portfolio_tracker import Position

__all__ = ["PositionHealthScore", "PositionHealthScorer", "PositionHealthConfig"]


class PositionHealthConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Position health scoring configuration."""

    # Weight config
    pnl_weight: float = 0.40  # PnL weight
    time_weight: float = 0.30  # Holding time weight
    trend_weight: float = 0.30  # Trend position weight

    # PnL scoring thresholds
    pnl_excellent_pct: float = 0.05  # Profit >5% = excellent
    pnl_poor_pct: float = -0.10  # Loss >10% = poor

    # Time scoring thresholds
    time_fresh_days: int = 5  # <5 days = fresh position
    time_mature_days: int = 15  # >15 days = mature position

    # Replacement thresholds
    min_score_to_keep: float = 0.30  # Below this score, position is replaceable
    new_opportunity_threshold: float = 0.95  # Score required for new opportunity


class PositionHealthScore(msgspec.Struct, frozen=True, kw_only=True):
    """Position health scoring result."""

    symbol: str
    health_score: float  # 0.0-1.0, composite health score
    pnl_score: float  # PnL score
    time_score: float  # Time score
    trend_score: float  # Trend position score
    unrealized_pnl_pct: float  # Unrealized PnL percentage
    days_held: int  # Days held
    reason: str  # Scoring reason


class PositionHealthScorer:
    """Position health scorer."""

    def __init__(self, config: PositionHealthConfig | None = None) -> None:
        self.config = config or PositionHealthConfig()

    def score_position(
        self,
        position: Position,
        current_price: float,
        current_date: date,
    ) -> PositionHealthScore:
        """Compute health score for a single position."""

        config = self.config

        unrealized_pnl = position.unrealized_pnl(current_price)
        if position.market_value > 0:
            unrealized_pnl_pct = unrealized_pnl / position.market_value
        else:
            unrealized_pnl_pct = 0.0

        if unrealized_pnl_pct >= config.pnl_excellent_pct:
            pnl_score = 1.0
        elif unrealized_pnl_pct <= config.pnl_poor_pct:
            pnl_score = 0.0
        else:
            range_size = config.pnl_excellent_pct - config.pnl_poor_pct
            pnl_score = (unrealized_pnl_pct - config.pnl_poor_pct) / range_size

        days_held = (current_date - position.entry_date).days
        if days_held <= config.time_fresh_days:
            time_score = 1.0
        elif days_held >= config.time_mature_days:
            time_score = 0.4
        else:
            range_days = config.time_mature_days - config.time_fresh_days
            progress = (days_held - config.time_fresh_days) / range_days
            time_score = 1.0 - (0.6 * progress)

        stop_loss = position.stop_loss_price
        take_profit = position.take_profit_price
        if take_profit > stop_loss:
            price_range = take_profit - stop_loss
            if price_range > 0:
                position_in_range = (current_price - stop_loss) / price_range
                trend_score = max(0.0, min(1.0, position_in_range))
            else:
                trend_score = 0.5
        else:
            trend_score = 0.5

        health_score = (
            pnl_score * config.pnl_weight
            + time_score * config.time_weight
            + trend_score * config.trend_weight
        )

        reasons: list[str] = []
        if pnl_score >= 0.8:
            reasons.append(f"profit {unrealized_pnl_pct * 100:.1f}%")
        elif pnl_score <= 0.3:
            reasons.append(f"loss {abs(unrealized_pnl_pct) * 100:.1f}%")

        if time_score >= 0.8:
            reasons.append(f"fresh {days_held}d")
        elif time_score <= 0.5:
            reasons.append(f"held {days_held}d")

        if trend_score <= 0.3:
            reasons.append("near stop-loss")
        elif trend_score >= 0.8:
            reasons.append("near take-profit")

        reason = "; ".join(reasons) if reasons else "normal"

        return PositionHealthScore(
            symbol=position.symbol,
            health_score=float(health_score),
            pnl_score=float(pnl_score),
            time_score=float(time_score),
            trend_score=float(trend_score),
            unrealized_pnl_pct=float(unrealized_pnl_pct),
            days_held=int(days_held),
            reason=reason,
        )

    def score_all_positions(
        self,
        positions: dict[str, Position],
        market_prices: dict[str, float],
        current_date: date,
    ) -> list[PositionHealthScore]:
        """Compute health scores for all positions, sorted ascending by health."""

        scores: list[PositionHealthScore] = []
        for symbol, position in positions.items():
            price = market_prices.get(symbol)
            if price is None:
                continue
            scores.append(self.score_position(position, price, current_date))

        scores.sort(key=lambda score: score.health_score)
        return scores

    def find_replaceable_positions(
        self,
        positions: dict[str, Position],
        market_prices: dict[str, float],
        current_date: date,
        min_score_to_keep: float | None = None,
    ) -> list[PositionHealthScore]:
        """Find replaceable positions (health score below threshold)."""

        threshold = (
            min_score_to_keep if min_score_to_keep is not None else self.config.min_score_to_keep
        )
        return [
            score
            for score in self.score_all_positions(positions, market_prices, current_date)
            if score.health_score < threshold
        ]

