from __future__ import annotations

import pytest

from common.interface import ResultStatus
from portfolio.capital_allocator import CapitalAllocator
from portfolio.interface import (
    DEFAULT_SIZING_TIERS,
    CapitalAllocationConfig,
    DynamicSizingTier,
)


def test_allocate_computes_reserved_available_and_budgets() -> None:
    allocator = CapitalAllocator(CapitalAllocationConfig(use_dynamic_sizing=False))

    res = allocator.allocate(total_equity=1000.0, current_exposure=400.0, current_position_count=4)

    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.reserved_capital == pytest.approx(150.0)
    assert res.data.available_capital == pytest.approx(450.0)
    assert res.data.max_new_positions == 9
    assert res.data.per_position_budget == pytest.approx(50.0)
    assert res.data.max_new_positions_per_day == 3
    assert res.data.max_positions_to_rotate == 2
    assert res.data.min_score_threshold == pytest.approx(0.90)


def test_allocate_respects_explicit_max_positions() -> None:
    allocator = CapitalAllocator(
        CapitalAllocationConfig(
            max_positions=3,
            base_position_pct=0.05,
            min_position_pct=0.025,
            max_position_pct=0.10,
            max_exposure_pct=0.90,
            use_dynamic_sizing=False,
        )
    )

    res = allocator.allocate(total_equity=1000.0, current_exposure=100.0, current_position_count=2)

    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.max_new_positions == 1
    assert res.data.per_position_budget > 0


def test_allocate_returns_zero_when_no_capacity_remaining() -> None:
    allocator = CapitalAllocator()

    res = allocator.allocate(total_equity=1000.0, current_exposure=900.0, current_position_count=10)

    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.available_capital == pytest.approx(0.0)
    assert res.data.max_new_positions == 0
    assert res.data.per_position_budget == pytest.approx(0.0)


def test_allocate_rejects_negative_inputs() -> None:
    allocator = CapitalAllocator()

    res = allocator.allocate(total_equity=-1.0, current_exposure=0.0, current_position_count=0)
    assert res.status is ResultStatus.FAILED
    assert res.reason_code == "NEGATIVE_INPUTS"


def test_allocate_rejects_infeasible_commission_constraints() -> None:
    allocator = CapitalAllocator(
        CapitalAllocationConfig(
            max_position_pct=0.10,
            min_position_value=50.0,
            commission_per_trade=5.0,
            max_commission_ratio=0.02,  # requires >= $250 position to keep ratio <= 2%
            use_dynamic_sizing=False,
        )
    )

    res = allocator.allocate(total_equity=1000.0, current_exposure=0.0, current_position_count=0)
    assert res.status is ResultStatus.FAILED
    assert res.reason_code == "INVALID_CONFIG"


def test_dynamic_sizing_tier_selection() -> None:
    allocator = CapitalAllocator(CapitalAllocationConfig(use_dynamic_sizing=True, sizing_tiers=DEFAULT_SIZING_TIERS))

    res_1k = allocator.allocate(total_equity=1000.0, current_exposure=0.0, current_position_count=0)
    assert res_1k.status is ResultStatus.SUCCESS
    assert res_1k.data is not None
    assert DEFAULT_SIZING_TIERS[0].min_per_position <= res_1k.data.per_position_budget <= DEFAULT_SIZING_TIERS[0].max_per_position
    assert res_1k.data.per_position_budget < DEFAULT_SIZING_TIERS[1].min_per_position
    assert res_1k.data.max_new_positions_per_day == DEFAULT_SIZING_TIERS[0].max_new_positions_per_day
    assert res_1k.data.max_positions_to_rotate == DEFAULT_SIZING_TIERS[0].max_positions_to_rotate
    assert res_1k.data.min_score_threshold == pytest.approx(DEFAULT_SIZING_TIERS[0].min_score_threshold)

    res_5k = allocator.allocate(total_equity=5000.0, current_exposure=0.0, current_position_count=0)
    assert res_5k.status is ResultStatus.SUCCESS
    assert res_5k.data is not None
    assert DEFAULT_SIZING_TIERS[1].min_per_position <= res_5k.data.per_position_budget <= DEFAULT_SIZING_TIERS[1].max_per_position
    assert res_5k.data.per_position_budget < DEFAULT_SIZING_TIERS[2].min_per_position
    assert res_5k.data.max_new_positions_per_day == DEFAULT_SIZING_TIERS[1].max_new_positions_per_day
    assert res_5k.data.max_positions_to_rotate == DEFAULT_SIZING_TIERS[1].max_positions_to_rotate
    assert res_5k.data.min_score_threshold == pytest.approx(DEFAULT_SIZING_TIERS[1].min_score_threshold)

    res_10k = allocator.allocate(total_equity=10000.0, current_exposure=0.0, current_position_count=0)
    assert res_10k.status is ResultStatus.SUCCESS
    assert res_10k.data is not None
    assert DEFAULT_SIZING_TIERS[2].min_per_position <= res_10k.data.per_position_budget <= DEFAULT_SIZING_TIERS[2].max_per_position
    assert res_10k.data.per_position_budget < DEFAULT_SIZING_TIERS[3].min_per_position
    assert res_10k.data.max_new_positions_per_day == DEFAULT_SIZING_TIERS[2].max_new_positions_per_day
    assert res_10k.data.max_positions_to_rotate == DEFAULT_SIZING_TIERS[2].max_positions_to_rotate
    assert res_10k.data.min_score_threshold == pytest.approx(DEFAULT_SIZING_TIERS[2].min_score_threshold)


def test_dynamic_sizing_respects_tier_limits() -> None:
    allocator = CapitalAllocator(CapitalAllocationConfig(use_dynamic_sizing=True, sizing_tiers=DEFAULT_SIZING_TIERS))

    res = allocator.allocate(total_equity=2000.0, current_exposure=0.0, current_position_count=0)
    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None

    tier0 = DEFAULT_SIZING_TIERS[0]
    assert tier0.min_per_position <= res.data.per_position_budget <= tier0.max_per_position
    assert res.data.max_new_positions <= tier0.max_positions
    assert res.data.max_new_positions_per_day == tier0.max_new_positions_per_day
    assert res.data.max_positions_to_rotate == tier0.max_positions_to_rotate
    assert res.data.min_score_threshold == pytest.approx(tier0.min_score_threshold)


def test_dynamic_sizing_custom_tiers() -> None:
    custom_tiers = (
        DynamicSizingTier(
            equity_threshold=0,
            min_per_position=100,
            max_per_position=200,
            max_positions=5,
            max_new_positions_per_day=1,
            max_positions_to_rotate=1,
            min_score_threshold=0.92,
        ),
        DynamicSizingTier(
            equity_threshold=5000,
            min_per_position=300,
            max_per_position=600,
            max_positions=10,
            max_new_positions_per_day=3,
            max_positions_to_rotate=2,
            min_score_threshold=0.88,
        ),
    )

    allocator = CapitalAllocator(CapitalAllocationConfig(use_dynamic_sizing=True, sizing_tiers=custom_tiers))

    res_2k = allocator.allocate(total_equity=2000.0, current_exposure=0.0, current_position_count=0)
    assert res_2k.status is ResultStatus.SUCCESS
    assert res_2k.data is not None
    assert custom_tiers[0].min_per_position <= res_2k.data.per_position_budget <= custom_tiers[0].max_per_position
    assert res_2k.data.max_new_positions <= custom_tiers[0].max_positions
    assert res_2k.data.max_new_positions_per_day == custom_tiers[0].max_new_positions_per_day
    assert res_2k.data.max_positions_to_rotate == custom_tiers[0].max_positions_to_rotate
    assert res_2k.data.min_score_threshold == pytest.approx(custom_tiers[0].min_score_threshold)

    res_6k = allocator.allocate(total_equity=6000.0, current_exposure=0.0, current_position_count=0)
    assert res_6k.status is ResultStatus.SUCCESS
    assert res_6k.data is not None
    assert custom_tiers[1].min_per_position <= res_6k.data.per_position_budget <= custom_tiers[1].max_per_position
    assert res_6k.data.max_new_positions <= custom_tiers[1].max_positions
    assert res_6k.data.max_new_positions_per_day == custom_tiers[1].max_new_positions_per_day
    assert res_6k.data.max_positions_to_rotate == custom_tiers[1].max_positions_to_rotate
    assert res_6k.data.min_score_threshold == pytest.approx(custom_tiers[1].min_score_threshold)


def test_pct_sizing_anchors_position_value_to_equity_pct() -> None:
    allocator = CapitalAllocator(
        CapitalAllocationConfig(
            sizing_policy="pct",
            target_position_pct=0.06,
            max_positions=10,  # override tier max_positions; behavior params still from tier
            sizing_tiers=DEFAULT_SIZING_TIERS,
        )
    )

    res = allocator.allocate(total_equity=10000.0, current_exposure=0.0, current_position_count=0)

    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.per_position_budget == pytest.approx(600.0)  # 10k * 6%
    assert res.data.max_new_positions == 10

    tier = DEFAULT_SIZING_TIERS[2]  # $8000-$20000
    assert res.data.max_new_positions_per_day == tier.max_new_positions_per_day
    assert res.data.max_positions_to_rotate == tier.max_positions_to_rotate
    assert res.data.min_score_threshold == pytest.approx(tier.min_score_threshold)


def test_pct_sizing_respects_hard_value_safety_valves() -> None:
    allocator = CapitalAllocator(
        CapitalAllocationConfig(
            sizing_policy="pct",
            target_position_pct=0.06,
            max_positions=5,
            hard_min_position_value=400.0,
            hard_max_position_value=500.0,
            sizing_tiers=DEFAULT_SIZING_TIERS,
        )
    )

    res = allocator.allocate(total_equity=10000.0, current_exposure=0.0, current_position_count=0)

    assert res.status is ResultStatus.SUCCESS
    assert res.data is not None
    assert res.data.per_position_budget == pytest.approx(500.0)
