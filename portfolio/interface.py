"""Portfolio management interfaces for adaptive position sizing."""

from __future__ import annotations

from enum import Enum
from typing import Any

import msgspec


class DynamicSizingTier(msgspec.Struct, frozen=True, kw_only=True):
    """Capital tier config - dynamically adjust position params by account size.

    Example tiers for a typical setup:
        Tier 1 (< $3000):  min=$150, max=$300, max_positions=10
        Tier 2 ($3000-$8000): min=$400, max=$800, max_positions=12
        Tier 3 (> $8000):  min=$600, max=$1500, max_positions=15
    """

    equity_threshold: float  # Min equity to activate this tier
    min_per_position: float  # Min dollar amount per position in this tier
    max_per_position: float  # Max dollar amount per position in this tier
    max_positions: int  # Max positions allowed in this tier

    # Dynamic trading frequency params
    max_new_positions_per_day: int  # Max new positions per day (~max_positions/3)
    max_positions_to_rotate: int  # Max rotations per day (~max_positions/5)
    min_score_threshold: float  # Min score threshold (0.87-0.92)


# Default tiered strategy
# V21: unified $0-$20k as (15, 5, 3, 0.90), $20k+ as (20, 6, 4, 0.88)
DEFAULT_SIZING_TIERS: tuple[DynamicSizingTier, ...] = (
    # Small account < $3000 (per-position amount scaled for small capital)
    DynamicSizingTier(
        equity_threshold=0,
        min_per_position=150,
        max_per_position=300,
        max_positions=15,
        max_new_positions_per_day=5,
        max_positions_to_rotate=3,
        min_score_threshold=0.90,
    ),
    # Medium account $3000-$8000
    DynamicSizingTier(
        equity_threshold=3000,
        min_per_position=400,
        max_per_position=800,
        max_positions=15,
        max_new_positions_per_day=5,
        max_positions_to_rotate=3,
        min_score_threshold=0.90,
    ),
    # Large account $8000-$20000
    DynamicSizingTier(
        equity_threshold=8000,
        min_per_position=600,
        max_per_position=1500,
        max_positions=15,
        max_new_positions_per_day=5,
        max_positions_to_rotate=3,
        min_score_threshold=0.90,
    ),
    # Extra-large account > $20000
    DynamicSizingTier(
        equity_threshold=20000,
        min_per_position=1000,
        max_per_position=2500,
        max_positions=20,
        max_new_positions_per_day=6,
        max_positions_to_rotate=4,
        min_score_threshold=0.88,
    ),
)


class SizingPolicy(str, Enum):
    PCT = "pct"  # Percentage mode (new)
    TIER_DOLLAR = "tier_dollar"  # Tier dollar mode (legacy)


class CapitalAllocationConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Capital allocation configuration."""

    # Max position count control
    max_positions: int | None = None  # None = dynamically computed
    max_exposure_pct: float = 0.85  # Max total exposure 85%

    # Per-position control (pct mode - used when use_dynamic_sizing=False)
    min_position_pct: float = 0.025  # Min position 2.5%
    max_position_pct: float = 0.10  # Max position 10%
    base_position_pct: float = 0.05  # Base position 5%

    # Dynamic sizing mode (auto-adjust by account size)
    use_dynamic_sizing: bool = True  # Enable dynamic position sizing
    sizing_tiers: tuple[DynamicSizingTier, ...] | None = None  # Custom tiers, None uses default

    # Position value calculation strategy
    sizing_policy: str = SizingPolicy.TIER_DOLLAR.value  # pct | tier_dollar
    target_position_pct: float = 0.06  # Target position pct (used when sizing_policy=pct)
    hard_max_position_value: float | None = None  # Optional: absolute max value safety cap
    hard_min_position_value: float | None = None  # Optional: absolute min value safety floor

    # Capital reserves
    cash_reserve_pct: float = 0.10  # Fixed reserve 10%
    opportunity_reserve_pct: float = 0.05  # Opportunity reserve 5%

    # Commission optimization
    min_position_value: float = 50.0  # Min position value $50
    max_commission_ratio: float = 0.02  # Max commission ratio 2%
    commission_per_trade: float = 1.0  # Commission per trade $1


class AllocationResult(msgspec.Struct, frozen=True, kw_only=True):
    """Capital allocation result."""

    available_capital: float  # Capital available for new positions
    reserved_capital: float  # Reserved capital
    max_new_positions: int  # Max new positions allowed
    per_position_budget: float  # Budget per new position
    current_exposure: float  # Current exposure
    current_position_count: int  # Current position count
    total_equity: float  # Total equity

    # Dynamic trading frequency params (for downstream strategy control)
    max_new_positions_per_day: int = 3  # Max new positions per day
    max_positions_to_rotate: int = 2  # Max rotations per day
    min_score_threshold: float = 0.90  # Min score threshold


CapitalAllocatorContext = dict[str, Any]
