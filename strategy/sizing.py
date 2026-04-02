"""Position sizing algorithms for the Phase 3.2 Strategy Engine.

This module implements the *Position Sizing* responsibility in the Strategy
Engine boundary (Phase 3.2). A position sizer determines the share quantity for
an entry intent given:

- The intended entry price (and optionally stop-loss price).
- The current account equity.
- Strategy-level risk configuration (``strategy.interface.StrategyEngineConfig``).
- Event Guard trading constraints (``event_guard.interface.TradeConstraints``).

The Strategy Engine consumes the output via ``PositionSizerProtocol`` and
expects deterministic, conservative behaviour with explicit error signaling via
``common.interface.Result[T]``.

Implemented sizing patterns:

Pattern A: Fixed-Percent Sizing (baseline)
    Allocate a fixed fraction of equity to a position, then convert value to
    shares. This is a common baseline and is also used as a fallback for other
    patterns when required inputs are missing.

Pattern 3: Fixed-Risk Sizing (extracted from NautilusTrader)
    Compute the distance to stop-loss and size such that the dollar risk equals
    a fixed fraction of equity. This module implements a Phase 1–style
    simplification which ignores exchange rates, commission models, and
    instrument-specific rounding rules.

Pattern 4: Volatility-Scaled Sizing (simplified from ibkr/rob_port)
    Scale the risk allocation by volatility (proxy: ATR). In Phase 1, market
    data integration is not yet available here; the implementation degrades to
    fixed-percent sizing until ATR becomes available from
    ``data.interface.PriceSeriesSnapshot``.

Rounding:
    All algorithms floor the final share quantity to whole shares (``int``),
    aligning with typical US equity trading constraints for share quantities in
    early phases. Fractional shares and instrument batch sizes can be added in a
    later phase.
"""

from __future__ import annotations

import math
from typing import Final

from strategy.interface import PositionSizerProtocol, Result, StrategyEngineConfig

from data.interface import PriceSeriesSnapshot
from event_guard.interface import TradeConstraints
from portfolio.interface import AllocationResult

__all__ = [
    "FixedPercentSizer",
    "FixedRiskSizer",
    "VolatilityScaledSizer",
    "QualityScaledSizer",
    "AdaptivePositionSizer",
    "create_position_sizer",
]

_REASON_NEGATIVE_INPUTS: Final[str] = "NEGATIVE_INPUTS"
_REASON_ZERO_RISK_DISTANCE: Final[str] = "ZERO_RISK_DISTANCE"
_REASON_ZERO_SHARES: Final[str] = "ZERO_SHARES"
_REASON_COMMISSION_TOO_HIGH: Final[str] = "COMMISSION_TOO_HIGH"


def _apply_max_position_caps(
    *,
    position_size: float,
    entry_price: float,
    account_equity: float,
    config: StrategyEngineConfig,
    constraints: TradeConstraints | None,
) -> float:
    """Apply strategy and Event Guard hard caps to a computed share quantity."""

    adjusted_size = float(position_size)

    # Strategy-level hard cap: max fraction of equity in a single position.
    max_position_value = float(account_equity) * float(config.max_position_pct)
    if max_position_value >= 0:
        max_shares = max_position_value / float(entry_price)
        adjusted_size = min(adjusted_size, max_shares)

    # Event Guard cap: absolute shares cap (policy-defined).
    if constraints is not None and constraints.max_position_size is not None:
        cap = float(constraints.max_position_size)
        if cap >= 0:
            adjusted_size = min(adjusted_size, cap)

    return adjusted_size


def _floor_to_whole_shares(position_size: float) -> float:
    """Floor size to whole shares using Python int truncation."""

    return float(int(position_size))


class FixedPercentSizer:
    """Fixed-percent position sizing.

    Pattern: baseline fixed-percent sizing (foundational).

    Algorithm:
        ``position_value = account_equity * config.position_size_pct``
        ``position_size = position_value / entry_price``
        ``position_size = min(position_size, equity_cap_shares, constraints_cap)``
        ``position_size = floor(position_size)`` (whole shares)

    Args:
        symbol: Stock symbol (unused, kept for protocol consistency).
        entry_price: Intended entry price.
        stop_loss_price: Stop price (unused in this sizer).
        account_equity: Account equity in quote currency.
        config: Strategy Engine config controlling size percentages and caps.
        constraints: Optional Event Guard constraints (max_position_size).

    Returns:
        Result[float]: Successful result holds whole-share quantity as float.

    Errors:
        - NEGATIVE_INPUTS: entry_price <= 0 or account_equity <= 0.
        - ZERO_SHARES: computed share quantity floors to 0.
    """

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        config: StrategyEngineConfig,
        constraints: TradeConstraints | None = None,
        market_data: PriceSeriesSnapshot | None = None,
    ) -> Result[float]:
        if entry_price <= 0 or account_equity <= 0:
            return Result.failed(
                ValueError("entry_price, account_equity must be positive"),
                _REASON_NEGATIVE_INPUTS,
            )

        position_value = float(account_equity) * float(config.position_size_pct)
        position_size = position_value / float(entry_price)
        position_size = _apply_max_position_caps(
            position_size=position_size,
            entry_price=entry_price,
            account_equity=account_equity,
            config=config,
            constraints=constraints,
        )
        position_size = _floor_to_whole_shares(position_size)

        if position_size <= 0:
            return Result.failed(
                ValueError("calculated position size is zero"),
                _REASON_ZERO_SHARES,
            )

        return Result.success(position_size)


class FixedRiskSizer:
    """Fixed-risk position sizing based on stop-loss distance.

    Extracted pattern source:
        ``nautilus_trader/risk/sizing.pyx`` (Fixed-Risk position sizing).

    Reference algorithm (NautilusTrader; simplified narrative):
        - Convert price distance to "risk points" in instrument increments.
        - Convert desired equity risk to cash risk after commissions.
        - Compute size so that (risk points * point value * size) ~= risk cash.
        - Apply hard limits and instrument batch rounding rules.

    Phase 1 / AST simplification:
        - Ignore exchange rates and commission models.
        - Use raw price distance as risk distance (no tick size handling yet).
        - Round down to whole shares.

    Simplified algorithm:
        ``risk_points = abs(entry_price - stop_loss_price)``
        ``risk_money = account_equity * config.risk_per_trade_pct``
        ``position_size = risk_money / risk_points``
        ``position_size = min(position_size, equity_cap_shares, constraints_cap)``
        ``position_size = floor(position_size)`` (whole shares)

    Args:
        symbol: Stock symbol (unused, kept for protocol consistency).
        entry_price: Intended entry price.
        stop_loss_price: Intended stop-loss price.
        account_equity: Account equity in quote currency.
        config: Strategy Engine config controlling risk percentage and caps.
        constraints: Optional Event Guard constraints (max_position_size).

    Returns:
        Result[float]: Successful result holds whole-share quantity as float.

    Errors:
        - NEGATIVE_INPUTS: entry_price <= 0 or stop_loss_price <= 0 or account_equity <= 0.
        - ZERO_RISK_DISTANCE: entry_price == stop_loss_price.
        - ZERO_SHARES: computed share quantity floors to 0.
    """

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        config: StrategyEngineConfig,
        constraints: TradeConstraints | None = None,
        market_data: PriceSeriesSnapshot | None = None,
    ) -> Result[float]:
        if entry_price <= 0 or stop_loss_price <= 0 or account_equity <= 0:
            return Result.failed(
                ValueError("entry_price, stop_loss_price, account_equity must be positive"),
                _REASON_NEGATIVE_INPUTS,
            )

        risk_points = abs(float(entry_price) - float(stop_loss_price))
        if risk_points == 0:
            return Result.failed(
                ValueError("entry_price cannot equal stop_loss_price"),
                _REASON_ZERO_RISK_DISTANCE,
            )

        risk_money = float(account_equity) * float(config.risk_per_trade_pct)
        position_size = risk_money / float(risk_points)
        position_size = _apply_max_position_caps(
            position_size=position_size,
            entry_price=entry_price,
            account_equity=account_equity,
            config=config,
            constraints=constraints,
        )
        position_size = _floor_to_whole_shares(position_size)

        if position_size <= 0:
            return Result.failed(
                ValueError("calculated position size is zero"),
                _REASON_ZERO_SHARES,
            )

        return Result.success(position_size)


class VolatilityScaledSizer:
    """Volatility-scaled position sizing using ATR% when available.

    Extracted pattern source:
        ``ibkr/rob_port`` chapter 2 (volatility targeting / risk scaling).

    Reference algorithm (simplified from the source material):
        ``target_risk = capital * risk_target``
        ``contract_risk = price * volatility_pct``
        ``num_contracts = target_risk / contract_risk``

    ATR-based algorithm (AST):
        - Use ``market_data.atr_pct`` as a volatility proxy (ATR / price).
        - Risk distance: ``risk_distance = entry_price * atr_pct * config.atr_multiplier``
        - Target cash risk: ``risk_money = account_equity * config.risk_per_trade_pct``
        - Size: ``shares = floor(risk_money / risk_distance)``
        - Apply strategy and Event Guard caps.

    Fallback algorithm:
        If ``market_data`` or ``atr_pct`` is unavailable/invalid, degrade to the
        fixed-percent sizing baseline:
        ``position_value = account_equity * config.position_size_pct``
        ``position_size = position_value / entry_price``
        ``position_size = min(position_size, equity_cap_shares, constraints_cap)``
        ``position_size = floor(position_size)`` (whole shares)

    Args:
        symbol: Stock symbol (unused in Phase 1 implementation).
        entry_price: Intended entry price.
        stop_loss_price: Stop price (unused in Phase 1 implementation).
        account_equity: Account equity in quote currency.
        config: Strategy Engine config controlling sizing and caps.
        constraints: Optional Event Guard constraints (max_position_size).

    Returns:
        Result[float]: Successful result holds whole-share quantity as float.

    Errors:
        - NEGATIVE_INPUTS: entry_price <= 0 or account_equity <= 0.
        - ZERO_SHARES: computed share quantity floors to 0.
    """

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        config: StrategyEngineConfig,
        constraints: TradeConstraints | None = None,
        market_data: PriceSeriesSnapshot | None = None,
    ) -> Result[float]:
        if entry_price <= 0 or account_equity <= 0:
            return Result.failed(
                ValueError("entry_price, account_equity must be positive"),
                _REASON_NEGATIVE_INPUTS,
            )

        if market_data is not None:
            atr_pct = getattr(market_data, "atr_pct", None)
            if atr_pct is not None:
                try:
                    atr_pct_value = float(atr_pct)
                except Exception:  # noqa: BLE001
                    atr_pct_value = 0.0
                if math.isfinite(atr_pct_value) and atr_pct_value > 0:
                    risk_distance = (
                        float(entry_price) * float(atr_pct_value) * float(config.atr_multiplier)
                    )
                    if math.isfinite(risk_distance) and risk_distance > 0:
                        risk_money = float(account_equity) * float(config.risk_per_trade_pct)
                        position_size = risk_money / float(risk_distance)
                        position_size = _apply_max_position_caps(
                            position_size=position_size,
                            entry_price=entry_price,
                            account_equity=account_equity,
                            config=config,
                            constraints=constraints,
                        )
                        position_size = _floor_to_whole_shares(position_size)

                        if position_size <= 0:
                            return Result.failed(
                                ValueError("calculated position size is zero"),
                                _REASON_ZERO_SHARES,
                            )

                        return Result.success(position_size)

        # Fallback: fixed-percent sizing.
        position_value = float(account_equity) * float(config.position_size_pct)
        position_size = position_value / float(entry_price)
        position_size = _apply_max_position_caps(
            position_size=position_size,
            entry_price=entry_price,
            account_equity=account_equity,
            config=config,
            constraints=constraints,
        )
        position_size = _floor_to_whole_shares(position_size)

        if position_size <= 0:
            return Result.failed(
                ValueError("calculated position size is zero"),
                _REASON_ZERO_SHARES,
            )

        return Result.success(position_size)


class QualityScaledSizer:
    """Quality-scaled position sizing based on signal score.

    Pattern: Adaptive position sizing that scales allocation by signal quality.

    Algorithm:
        1. Extract score from market_data.quality_flags.get("score", 0.90)
        2. Classify quality tier based on score thresholds
        3. Calculate position_value = equity * base_position_pct * tier_multiplier
        4. Apply min_position_pct floor
        5. Apply max_position_pct and constraints caps
        6. Check commission_ratio (skip if > max_commission_ratio)
        7. Return floor(shares)

    Quality Tiers:
        - Excellent (score >= threshold_excellent): multiplier from config
        - Good (threshold_good <= score < threshold_excellent): multiplier from config
        - Acceptable (threshold_acceptable <= score < threshold_good): multiplier from config
        - Skip (score < threshold_acceptable): return 0 shares

    Args:
        symbol: Stock symbol (unused, kept for protocol consistency).
        entry_price: Intended entry price.
        stop_loss_price: Stop price (unused in this sizer).
        account_equity: Account equity in quote currency.
        config: Strategy Engine config controlling quality tiers and caps.
        constraints: Optional Event Guard constraints (max_position_size).
        market_data: PriceSeriesSnapshot with quality_flags containing "score".

    Returns:
        Result[float]: Successful result holds whole-share quantity as float.

    Errors:
        - NEGATIVE_INPUTS: entry_price <= 0 or account_equity <= 0.
        - ZERO_SHARES: computed share quantity floors to 0 or below quality threshold.
        - COMMISSION_TOO_HIGH: commission ratio exceeds max_commission_ratio.
    """

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        config: StrategyEngineConfig,
        constraints: TradeConstraints | None = None,
        market_data: PriceSeriesSnapshot | None = None,
    ) -> Result[float]:
        if entry_price <= 0 or account_equity <= 0:
            return Result.failed(
                ValueError("entry_price, account_equity must be positive"),
                _REASON_NEGATIVE_INPUTS,
            )

        score = 0.90
        if market_data is not None:
            try:
                quality_flags = getattr(market_data, "quality_flags", {})
                score = float(quality_flags.get("score", 0.90))
            except Exception:  # noqa: BLE001
                score = 0.90

        if score < float(config.min_score_threshold):
            return Result.failed(
                ValueError(
                    f"score {score} below min_score_threshold {config.min_score_threshold}"
                ),
                _REASON_ZERO_SHARES,
            )

        if score >= float(config.quality_threshold_excellent):
            multiplier = float(config.quality_multiplier_excellent)
        elif score >= float(config.quality_threshold_good):
            multiplier = float(config.quality_multiplier_good)
        elif score >= float(config.quality_threshold_acceptable):
            multiplier = float(config.quality_multiplier_acceptable)
        else:
            return Result.failed(
                ValueError(f"score {score} below acceptable threshold"),
                _REASON_ZERO_SHARES,
            )

        base_value = float(account_equity) * float(config.base_position_pct)
        quality_adjusted_value = base_value * multiplier

        min_value = float(account_equity) * float(config.min_position_pct)
        if quality_adjusted_value < min_value:
            quality_adjusted_value = min_value

        position_size = quality_adjusted_value / float(entry_price)
        position_size = _apply_max_position_caps(
            position_size=position_size,
            entry_price=entry_price,
            account_equity=account_equity,
            config=config,
            constraints=constraints,
        )
        position_size = _floor_to_whole_shares(position_size)

        if position_size <= 0:
            return Result.failed(
                ValueError("calculated position size is zero"),
                _REASON_ZERO_SHARES,
            )

        commission = float(config.transaction_costs.commission_per_trade)
        position_value = position_size * float(entry_price)
        if position_value > 0:
            commission_ratio = commission / position_value
            max_ratio = float(config.max_commission_ratio)
            if commission_ratio > max_ratio:
                return Result.failed(
                    ValueError(
                        f"commission ratio {commission_ratio:.4f} exceeds max {max_ratio}"
                    ),
                    _REASON_COMMISSION_TOO_HIGH,
                )

        return Result.success(position_size)


class AdaptivePositionSizer:
    """Adaptive position sizing combining budget, ATR risk, quality, and commission checks.

    Phase 2 requirements:
        - Budget cap: base_position_pct or allocation_context.per_position_budget.
        - ATR risk sizing (when market_data.atr_pct is available).
        - Optional quality scaling (reusing QualityScaledSizer tier logic).
        - Commission ratio rejection.
        - Apply hard caps and floor to whole shares.
    """

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        config: StrategyEngineConfig,
        constraints: TradeConstraints | None = None,
        market_data: PriceSeriesSnapshot | None = None,
        allocation_context: AllocationResult | None = None,
    ) -> Result[float]:
        if entry_price <= 0 or account_equity <= 0:
            return Result.failed(
                ValueError("entry_price, account_equity must be positive"),
                _REASON_NEGATIVE_INPUTS,
            )

        # 1) Budget upper bound.
        if allocation_context is not None:
            budget = float(allocation_context.per_position_budget)
        else:
            budget = float(account_equity) * float(config.base_position_pct)

        if not math.isfinite(budget) or budget <= 0:
            return Result.failed(
                ValueError("calculated budget must be positive"),
                _REASON_ZERO_SHARES,
            )

        # 2) ATR-based risk sizing (optional).
        risk_based_shares = float("inf")
        if market_data is not None:
            atr_pct = getattr(market_data, "atr_pct", None)
            if atr_pct is not None:
                try:
                    atr_pct_value = float(atr_pct)
                except Exception:  # noqa: BLE001
                    atr_pct_value = 0.0
                if math.isfinite(atr_pct_value) and atr_pct_value > 0:
                    risk_distance = (
                        float(entry_price) * float(atr_pct_value) * float(config.atr_multiplier)
                    )
                    if math.isfinite(risk_distance) and risk_distance > 0:
                        risk_money = float(account_equity) * float(config.risk_per_trade_pct)
                        risk_based_shares = risk_money / float(risk_distance)

        # 3) Budget-based sizing.
        budget_based_shares = budget / float(entry_price)

        # 4) Pick the tighter constraint.
        position_size = min(float(risk_based_shares), float(budget_based_shares))

        # 5) Quality scaling (optional).
        if bool(config.quality_scaling_enabled) and market_data is not None:
            score = 0.90
            try:
                quality_flags = getattr(market_data, "quality_flags", {})
                score = float(quality_flags.get("score", 0.90))
            except Exception:  # noqa: BLE001
                score = 0.90

            if score < float(config.min_score_threshold):
                return Result.failed(
                    ValueError(
                        f"score {score} below min_score_threshold {config.min_score_threshold}"
                    ),
                    _REASON_ZERO_SHARES,
                )

            if score >= float(config.quality_threshold_excellent):
                multiplier = float(config.quality_multiplier_excellent)
            elif score >= float(config.quality_threshold_good):
                multiplier = float(config.quality_multiplier_good)
            elif score >= float(config.quality_threshold_acceptable):
                multiplier = float(config.quality_multiplier_acceptable)
            else:
                return Result.failed(
                    ValueError(f"score {score} below acceptable threshold"),
                    _REASON_ZERO_SHARES,
                )

            position_size *= multiplier

        # 6) Commission check.
        commission = float(config.transaction_costs.commission_per_trade)
        position_value = float(position_size) * float(entry_price)
        if position_value > 0:
            commission_ratio = commission / position_value
            max_ratio = float(config.max_commission_ratio)
            if commission_ratio > max_ratio:
                return Result.failed(
                    ValueError(
                        f"commission ratio {commission_ratio:.4f} exceeds max {max_ratio}"
                    ),
                    _REASON_COMMISSION_TOO_HIGH,
                )

        # 7) Apply caps + rounding.
        position_size = _apply_max_position_caps(
            position_size=position_size,
            entry_price=entry_price,
            account_equity=account_equity,
            config=config,
            constraints=constraints,
        )
        position_size = _floor_to_whole_shares(position_size)

        if position_size <= 0:
            return Result.failed(
                ValueError("calculated position size is zero"),
                _REASON_ZERO_SHARES,
            )

        return Result.success(position_size)


POSITION_SIZERS: Final[dict[str, type[PositionSizerProtocol]]] = {
    "fixed_percent": FixedPercentSizer,
    "fixed_risk": FixedRiskSizer,
    "volatility_scaled": VolatilityScaledSizer,
    "quality_scaled": QualityScaledSizer,
    "adaptive": AdaptivePositionSizer,
}


def create_position_sizer(config: StrategyEngineConfig) -> PositionSizerProtocol:
    """Create a position sizer instance from Strategy Engine configuration.

    Args:
        config: Strategy Engine configuration.

    Returns:
        PositionSizerProtocol: Concrete position sizing implementation.

    Raises:
        ValueError: When ``config.position_sizer`` is unknown.
    """

    sizer_type = str(config.position_sizer).strip().lower()
    factory = POSITION_SIZERS.get(sizer_type)
    if factory is not None:
        return factory()

    expected = "', '".join(sorted(POSITION_SIZERS.keys()))
    raise ValueError(
        f"Unknown position_sizer type: {config.position_sizer}. Expected: '{expected}'"
    )
