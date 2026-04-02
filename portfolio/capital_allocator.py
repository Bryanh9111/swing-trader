"""Capital allocation algorithm for adaptive position sizing."""

from __future__ import annotations

import math
from typing import Final

from common.interface import Result

from portfolio.interface import (
    DEFAULT_SIZING_TIERS,
    AllocationResult,
    CapitalAllocationConfig,
    DynamicSizingTier,
    SizingPolicy,
)

__all__ = ["CapitalAllocator"]

_REASON_NEGATIVE_INPUTS: Final[str] = "NEGATIVE_INPUTS"
_REASON_INVALID_CONFIG: Final[str] = "INVALID_CONFIG"


def _is_pct(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= float(value) <= 1.0


def _validate_config(config: CapitalAllocationConfig) -> str | None:
    if config.max_positions is not None and config.max_positions <= 0:
        return "max_positions must be positive"

    sizing_policy = str(config.sizing_policy).strip().lower()
    if sizing_policy not in {SizingPolicy.PCT.value, SizingPolicy.TIER_DOLLAR.value}:
        return f"sizing_policy must be one of: {SizingPolicy.PCT.value}, {SizingPolicy.TIER_DOLLAR.value}"

    for name in (
        "max_exposure_pct",
        "min_position_pct",
        "max_position_pct",
        "base_position_pct",
        "cash_reserve_pct",
        "opportunity_reserve_pct",
    ):
        if not _is_pct(getattr(config, name)):
            return f"{name} must be in [0, 1]"

    if config.min_position_pct > config.base_position_pct:
        return "min_position_pct cannot exceed base_position_pct"

    if config.base_position_pct > config.max_position_pct:
        return "base_position_pct cannot exceed max_position_pct"

    if config.cash_reserve_pct + config.opportunity_reserve_pct >= 1.0:
        return "reserve percentages must sum to < 1"

    if config.min_position_value < 0:
        return "min_position_value must be non-negative"

    if config.max_commission_ratio <= 0 or not math.isfinite(config.max_commission_ratio):
        return "max_commission_ratio must be positive"

    if config.commission_per_trade < 0 or not math.isfinite(config.commission_per_trade):
        return "commission_per_trade must be non-negative"

    if sizing_policy == SizingPolicy.PCT.value and not _is_pct(float(config.target_position_pct)):
        return "target_position_pct must be in [0, 1] when sizing_policy='pct'"

    for name in ("hard_max_position_value", "hard_min_position_value"):
        value = getattr(config, name)
        if value is None:
            continue
        if value < 0 or not math.isfinite(value):
            return f"{name} must be non-negative and finite"

    if (
        config.hard_max_position_value is not None
        and config.hard_min_position_value is not None
        and config.hard_min_position_value > config.hard_max_position_value
    ):
        return "hard_min_position_value cannot exceed hard_max_position_value"

    return None


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _dynamic_max_positions(config: CapitalAllocationConfig) -> int:
    base = float(config.base_position_pct)
    if base <= 0:
        return 1
    return max(1, int(float(config.max_exposure_pct) / base))


def _find_sizing_tier(
    total_equity: float,
    tiers: tuple[DynamicSizingTier, ...],
) -> DynamicSizingTier:
    """Find the applicable sizing tier for the given total equity.

    Selects the tier with the highest equity_threshold <= total_equity.
    """

    applicable = tiers[0]  # Default to first tier
    for tier in tiers:
        if tier.equity_threshold <= total_equity:
            applicable = tier
        else:
            break
    return applicable


class CapitalAllocator:
    """Compute conservative budgets for opening new positions."""

    def __init__(self, config: CapitalAllocationConfig | None = None) -> None:
        self.config = config or CapitalAllocationConfig()

    def allocate(
        self,
        *,
        total_equity: float,
        current_exposure: float,
        current_position_count: int,
        context: dict[str, object] | None = None,
    ) -> Result[AllocationResult]:
        """Allocate capital for new positions.

        Args:
            total_equity: Total account equity (cash + position market value).
            current_exposure: Current total exposure/position value.
            current_position_count: Current number of open positions.
            context: Optional metadata (unused in Phase 1).
        """

        del context

        if total_equity < 0 or current_exposure < 0 or current_position_count < 0:
            return Result.failed(
                ValueError("inputs must be non-negative"),
                _REASON_NEGATIVE_INPUTS,
            )

        config_error = _validate_config(self.config)
        if config_error is not None:
            return Result.failed(ValueError(config_error), _REASON_INVALID_CONFIG)

        config = self.config
        total_equity_f = float(total_equity)
        current_exposure_f = float(current_exposure)

        reserved_capital = total_equity_f * (config.cash_reserve_pct + config.opportunity_reserve_pct)

        remaining_cash_after_reserves = max(total_equity_f - current_exposure_f - reserved_capital, 0.0)
        remaining_exposure_capacity = max(total_equity_f * config.max_exposure_pct - current_exposure_f, 0.0)
        available_capital = min(remaining_cash_after_reserves, remaining_exposure_capacity)

        commission_floor = config.commission_per_trade / config.max_commission_ratio

        sizing_policy = str(config.sizing_policy).strip().lower()
        if sizing_policy == SizingPolicy.PCT.value:
            # Pct mode - position value as equity percentage (behavioral params still from tier)
            target_value = total_equity_f * float(config.target_position_pct)
            min_value_floor = max(
                total_equity_f * config.min_position_pct,
                config.min_position_value,
                commission_floor,
            )
            max_value_cap = total_equity_f * config.max_position_pct

            # Optional safety caps
            if config.hard_max_position_value is not None:
                max_value_cap = min(max_value_cap, float(config.hard_max_position_value))
            if config.hard_min_position_value is not None:
                min_value_floor = max(min_value_floor, float(config.hard_min_position_value))

            tiers = config.sizing_tiers if config.sizing_tiers is not None else DEFAULT_SIZING_TIERS
            tier = _find_sizing_tier(total_equity_f, tiers)
            max_new_positions_per_day = int(tier.max_new_positions_per_day)
            max_positions_to_rotate = int(tier.max_positions_to_rotate)
            min_score_threshold = float(tier.min_score_threshold)
            max_positions_limit = int(tier.max_positions) if config.max_positions is None else int(config.max_positions)

            base_value = float(target_value)
        else:
            # tier_dollar (legacy): keep existing use_dynamic_sizing branch unchanged
            if config.use_dynamic_sizing:
                tiers = config.sizing_tiers if config.sizing_tiers is not None else DEFAULT_SIZING_TIERS
                tier = _find_sizing_tier(total_equity_f, tiers)

                max_new_positions_per_day = int(tier.max_new_positions_per_day)
                max_positions_to_rotate = int(tier.max_positions_to_rotate)
                min_score_threshold = float(tier.min_score_threshold)

                # Use tier's absolute dollar amounts instead of percentages
                min_value_floor = max(
                    tier.min_per_position,
                    config.min_position_value,
                    commission_floor,
                )
                max_value_cap = float(tier.max_per_position)

                # Use tier's max_positions
                max_positions_limit = int(tier.max_positions)
                base_value = (float(tier.min_per_position) + float(tier.max_per_position)) / 2.0
            else:
                # Original percentage mode
                max_new_positions_per_day = 3
                max_positions_to_rotate = 2
                min_score_threshold = 0.90

                min_value_floor = max(
                    total_equity_f * config.min_position_pct,
                    config.min_position_value,
                    commission_floor,
                )
                max_value_cap = total_equity_f * config.max_position_pct
                max_positions_limit = (
                    config.max_positions if config.max_positions is not None else _dynamic_max_positions(config)
                )
                base_value = total_equity_f * config.base_position_pct

        # Effective minimum position size: floor + cap checks.
        if max_value_cap > 0 and min_value_floor > max_value_cap:
            return Result.failed(
                ValueError("minimum position value exceeds maximum position cap"),
                _REASON_INVALID_CONFIG,
            )

        target_value = _clamp(base_value, lower=min_value_floor, upper=max_value_cap)

        max_new_positions_by_count = max(max_positions_limit - int(current_position_count), 0)

        max_new_positions_by_capital = (
            int(available_capital // min_value_floor) if min_value_floor > 0 else 0
        )
        max_new_positions = min(max_new_positions_by_count, max_new_positions_by_capital)

        per_position_budget = 0.0
        if max_new_positions > 0:
            per_position_budget = min(target_value, available_capital / float(max_new_positions))

            # If the computed per-position budget dips under the minimum floor,
            # reduce the position count to maintain the minimum.
            if per_position_budget < min_value_floor and min_value_floor > 0:
                per_position_budget = float(min_value_floor)
                max_new_positions = int(available_capital // per_position_budget)

            per_position_budget = _clamp(per_position_budget, lower=0.0, upper=max_value_cap)

            # Reconcile after clamping; ensure budget never exceeds available capital.
            if per_position_budget > 0:
                max_new_positions = min(max_new_positions, int(available_capital // per_position_budget))
                if max_new_positions <= 0:
                    per_position_budget = 0.0

        result = AllocationResult(
            available_capital=float(available_capital),
            reserved_capital=float(reserved_capital),
            max_new_positions=int(max_new_positions),
            per_position_budget=float(per_position_budget),
            current_exposure=float(current_exposure_f),
            current_position_count=int(current_position_count),
            total_equity=float(total_equity_f),
            max_new_positions_per_day=int(max_new_positions_per_day),
            max_positions_to_rotate=int(max_positions_to_rotate),
            min_score_threshold=float(min_score_threshold),
        )
        return Result.success(result)
