"""Position rotation logic for replacing weak positions with better opportunities."""

from __future__ import annotations

from datetime import date
from typing import Final

import msgspec
import structlog

from backtest.portfolio_tracker import Position
from portfolio.position_health import (
    PositionHealthConfig,
    PositionHealthScore,
    PositionHealthScorer,
)

__all__ = ["RotationDecision", "PositionRotator", "RotationConfig"]

logger: Final = structlog.get_logger(__name__).bind(component="position_rotator")


class RotationConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Rotation configuration."""

    enabled: bool = True  # Enable rotation
    min_new_opportunity_score: float = 0.95  # Min score for new opportunity
    max_positions_to_rotate: int = 2  # Max rotations per day
    min_health_score_to_keep: float = 0.30  # Below this score, position is replaceable
    require_pnl_loss: bool = True  # Require replaced position to be in loss
    min_loss_pct_to_replace: float = -0.03  # Loss >3% required before replacement


class RotationDecision(msgspec.Struct, frozen=True, kw_only=True):
    """Rotation decision result."""

    should_rotate: bool  # Whether to rotate
    positions_to_close: list[str]  # Symbols to close
    reason: str  # Decision reason
    health_scores: dict[str, float]  # Health scores per position


class PositionRotator:
    """Intelligent position rotation decision maker.

    When positions are at capacity and a high-quality new opportunity appears,
    decides whether to replace existing weak positions.

    Decision flow:
    1. Check if position limit is reached
    2. Check if new opportunity score is high enough
    3. Evaluate existing position health
    4. Select weakest positions for replacement
    """

    def __init__(
        self,
        rotation_config: RotationConfig | None = None,
        health_config: PositionHealthConfig | None = None,
    ) -> None:
        self.rotation_config = rotation_config or RotationConfig()
        self.health_scorer = PositionHealthScorer(health_config)

    def evaluate_rotation(
        self,
        positions: dict[str, Position],
        market_prices: dict[str, float],
        current_date: date,
        new_opportunity_score: float,
        max_positions: int,
        num_new_opportunities: int = 1,
    ) -> RotationDecision:
        """Evaluate whether position rotation should occur.

        Args:
            positions: Current positions
            market_prices: Market prices
            current_date: Current date
            new_opportunity_score: Score of new opportunity (0.0-1.0)
            max_positions: Max position count
            num_new_opportunities: Number of slots needed

        Returns:
            RotationDecision: Rotation decision
        """

        config = self.rotation_config

        if not config.enabled:
            return RotationDecision(
                should_rotate=False,
                positions_to_close=[],
                reason="rotation_disabled",
                health_scores={},
            )

        current_count = len(positions)
        if current_count < max_positions:
            return RotationDecision(
                should_rotate=False,
                positions_to_close=[],
                reason="positions_not_full",
                health_scores={},
            )

        if new_opportunity_score < config.min_new_opportunity_score:
            return RotationDecision(
                should_rotate=False,
                positions_to_close=[],
                reason=(
                    "opportunity_score_too_low "
                    f"({new_opportunity_score:.2f} < {config.min_new_opportunity_score})"
                ),
                health_scores={},
            )

        all_scores = self.health_scorer.score_all_positions(positions, market_prices, current_date)
        health_scores = {score.symbol: score.health_score for score in all_scores}

        replaceable: list[PositionHealthScore] = []
        for score in all_scores:
            if score.health_score >= config.min_health_score_to_keep:
                continue

            if config.require_pnl_loss and score.unrealized_pnl_pct > config.min_loss_pct_to_replace:
                continue

            replaceable.append(score)

        if not replaceable:
            return RotationDecision(
                should_rotate=False,
                positions_to_close=[],
                reason="no_replaceable_positions",
                health_scores=health_scores,
            )

        num_to_rotate = min(
            len(replaceable),
            num_new_opportunities,
            config.max_positions_to_rotate,
        )
        positions_to_close = [score.symbol for score in replaceable[:num_to_rotate]]

        logger.info(
            "rotation_decision",
            should_rotate=True,
            positions_to_close=positions_to_close,
            new_opportunity_score=new_opportunity_score,
            health_scores=health_scores,
        )

        return RotationDecision(
            should_rotate=True,
            positions_to_close=positions_to_close,
            reason=(
                f"replacing {num_to_rotate} weak positions for high-score opportunity "
                f"({new_opportunity_score:.2f})"
            ),
            health_scores=health_scores,
        )

