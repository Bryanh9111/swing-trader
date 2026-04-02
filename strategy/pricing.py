"""Phase 3.2 Strategy Engine pricing policies (entry/stop/target).

This module implements the Strategy Engine "price policy" boundary
(``PricePolicyProtocol``): turn a market snapshot into entry, stop-loss, and
take-profit prices, returning ``Result[float]`` with stable reason codes.

Implemented policies:
- Pattern 5: ATR bracket (from NautilusTrader ``ema_cross_bracket.py``).
- Pattern 6: Trailing stop (simplified from NautilusTrader ``trailing.rs``).
- Ladder entry: staged entries spaced by pct; SL/TP reuse ATR bracket.

Phase 1 constraints:
- Uses latest close as a proxy for market price (no bid/ask).
- ATR uses ``market_data.atr_pct`` if present, otherwise a simple fallback ATR.
"""

from __future__ import annotations

import structlog

from strategy.interface import PricePolicyProtocol, Result, StrategyEngineConfig
from strategy.interface import TransactionCostConfig

from data.interface import PriceSeriesSnapshot

__all__ = [
    "ATRBracketPolicy",
    "TrailingStopPolicy",
    "LadderEntryPolicy",
    "create_price_policy",
]

logger = structlog.get_logger(__name__).bind(component="strategy_pricing")


class ATRBracketPolicy:
    """ATR-based stop-loss/take-profit bracket policy (Pattern 5).

    Source: NautilusTrader ``examples/strategies/ema_cross_bracket.py``.

    Algorithm:
    - bracket_distance = entry_price * atr_pct * config.atr_multiplier
    - LONG:  sl = entry - dist, tp = entry + dist
    - SHORT: sl = entry + dist, tp = entry - dist

    Phase 1: entry=latest close, atr_pct from ``market_data.atr_pct`` or fallback.
    """

    def calculate_entry_price(
        self,
        symbol: str,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Calculate entry price (market proxy) using the latest close.

        Phase 1: Use latest close as entry (simulating a market order).

        Args:
            symbol: Stock/instrument symbol.
            side: ``"LONG"`` or ``"SHORT"`` (any other value is treated as SHORT).
            config: Strategy engine configuration (unused here).
            market_data: Market data snapshot providing ``bars``.

        Returns:
            Result[float]: Entry price on success.

        Backtest Assumptions & Limitations:
            **ASSUMPTION**: Able to execute at EOD close price on the same day.

            **REALITY**: In live trading, execution typically occurs at:
            - Next trading day's market open (T+1)
            - Next trading day's limit order fill (T+1)
            - Potentially with overnight gap risk

            **IMPLICATION**: Backtest may overestimate performance if:
            - Frequent overnight gaps occur (earnings, news, macro events)
            - Market opens significantly different from previous close
            - Gap direction is systematically unfavorable to entries

            **EXAMPLE**:
            - Scanner detects setup on 2024-01-02 EOD
            - Uses close=$100.00 (+ 0.15% costs) = $100.15 as entry
            - Actual fill on 2024-01-03 open might be $101.50 (1.35% gap up)
            - Backtest underestimates actual entry cost by $1.35/share

            **MITIGATION OPTIONS** (not currently implemented):
            1. Add overnight gap buffer: entry = close * (1 + gap_buffer + tx_costs)
            2. Use next-day open as entry (requires T+1 data in backtest)
            3. Apply statistical gap penalty based on historical gap distribution

            **CURRENT DECISION**: Accept this limitation for simplicity in Phase 1.
            Results should be interpreted as "best-case" estimates assuming no gaps.
        """

        if not market_data.bars:
            return Result.failed(
                ValueError("market_data.bars is empty"),
                "EMPTY_MARKET_DATA",
            )

        latest_bar = market_data.bars[-1]
        tx_costs: TransactionCostConfig = config.transaction_costs

        # For LONG entry: apply spread + slippage (increases entry price).
        #
        # Note: Commission is per-trade, not per-share, so we don't add it here.
        # It will be accounted for in PnL calculations by the execution layer.
        if side == "LONG":
            spread_slippage_factor = 1.0 + tx_costs.spread_pct + tx_costs.slippage_pct
            entry_price_with_costs = float(latest_bar.close) * spread_slippage_factor
            logger.debug(
                "Applied transaction costs to entry price",
                symbol=symbol,
                base_price=float(latest_bar.close),
                adjusted_price=entry_price_with_costs,
                spread_pct=tx_costs.spread_pct,
                slippage_pct=tx_costs.slippage_pct,
            )
            entry_price = entry_price_with_costs
        else:
            entry_price = float(latest_bar.close)

        if entry_price <= 0:
            return Result.failed(
                ValueError("invalid entry price (<=0)"),
                "INVALID_ENTRY_PRICE",
            )

        return Result.success(float(entry_price))

    def _get_atr_pct(self, market_data: PriceSeriesSnapshot, entry_price: float) -> float | None:
        atr_pct = getattr(market_data, "atr_pct", None)
        if atr_pct is not None:
            try:
                atr_pct_value = float(atr_pct)
            except Exception:  # noqa: BLE001
                atr_pct_value = 0.0
            if atr_pct_value > 0:
                return atr_pct_value
        if market_data.bars:
            from indicators.interface import compute_atr_last

            return compute_atr_last(
                market_data.bars,
                period=14,
                percentage=True,
                reference_price=entry_price,
            )
        return None

    def _get_dynamic_tp_sl_ratio(
        self,
        config: StrategyEngineConfig,
        atr_pct: float | None,
    ) -> float:
        """Dynamically calculate tp_sl_ratio based on volatility (ATR%).

        Args:
            config: Strategy engine configuration.
            atr_pct: ATR as percentage of price (e.g., 0.03 = 3%).

        Returns:
            Adjusted tp_sl_ratio based on volatility tier.
        """

        if not getattr(config, "dynamic_tp_sl_enabled", True):
            return float(config.tp_sl_ratio)

        if atr_pct is None:
            return float(config.tp_sl_ratio)

        low_threshold = float(getattr(config, "low_volatility_threshold", 0.02))
        high_threshold = float(getattr(config, "high_volatility_threshold", 0.04))
        base_ratio = float(config.tp_sl_ratio)
        low_vol_ratio = float(getattr(config, "tp_sl_ratio_low_vol", base_ratio))
        high_vol_ratio = float(getattr(config, "tp_sl_ratio_high_vol", base_ratio))

        if atr_pct < low_threshold:
            volatility_tier = "LOW"
            selected_ratio = low_vol_ratio
        elif atr_pct > high_threshold:
            volatility_tier = "HIGH"
            selected_ratio = high_vol_ratio
        else:
            volatility_tier = "NORMAL"
            selected_ratio = base_ratio

        logger.debug(
            "Dynamic tp_sl_ratio selected",
            atr_pct=f"{atr_pct:.4f}",
            volatility_tier=volatility_tier,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            selected_ratio=selected_ratio,
        )

        return selected_ratio

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Calculate stop-loss price based on ATR bracket distance with regime constraint.

        Algorithm:
        - ATR-based: sl_pct = atr_pct * atr_multiplier
        - Apply regime constraint: final_sl_pct = min(atr_sl_pct, config.stop_loss_pct)
        - LONG: sl_price = entry_price * (1 - final_sl_pct)
        """

        atr_pct = self._get_atr_pct(market_data, float(entry_price))

        if atr_pct is not None and atr_pct > 0:
            atr_sl_pct = float(atr_pct) * float(config.atr_multiplier)
        else:
            atr_sl_pct = float(config.stop_loss_pct)
            logger.warning(
                "ATR unavailable, using regime stop_loss_pct as fallback",
                symbol=symbol,
                fallback_sl_pct=atr_sl_pct,
            )

        max_sl_pct = float(config.stop_loss_pct)
        final_sl_pct = min(atr_sl_pct, max_sl_pct)
        if final_sl_pct <= 0:
            return Result.failed(
                ValueError("invalid stop loss pct (<=0)"),
                "INVALID_STOP_LOSS",
            )

        logger.debug(
            "Stop loss calculation with regime constraint",
            symbol=symbol,
            atr_sl_pct=atr_sl_pct,
            max_sl_pct=max_sl_pct,
            final_sl_pct=final_sl_pct,
        )

        tx_costs: TransactionCostConfig = config.transaction_costs

        if side == "LONG":
            base_sl_price = float(entry_price) * (1.0 - final_sl_pct)
            spread_slippage_factor = 1.0 - tx_costs.spread_pct - tx_costs.slippage_pct
            sl_price = base_sl_price * spread_slippage_factor
        else:  # SHORT
            sl_price = float(entry_price) * (1.0 + final_sl_pct)

        if sl_price <= 0:
            return Result.failed(
                ValueError("invalid stop loss price (<=0)"),
                "INVALID_STOP_LOSS",
            )

        return Result.success(float(sl_price))

    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Calculate take-profit price based on regime parameters.

        Algorithm:
        - If use_fixed_tp: tp_pct = config.take_profit_pct
        - Else: tp_pct = final_sl_pct * tp_sl_ratio (based on the realized SL distance)
        """

        atr_pct = self._get_atr_pct(market_data, float(entry_price))

        if atr_pct is not None and atr_pct > 0:
            atr_sl_pct = float(atr_pct) * float(config.atr_multiplier)
        else:
            atr_sl_pct = float(config.stop_loss_pct)

        max_sl_pct = float(config.stop_loss_pct)
        final_sl_pct = min(atr_sl_pct, max_sl_pct)
        if final_sl_pct <= 0:
            return Result.failed(
                ValueError("invalid stop loss pct (<=0)"),
                "INVALID_TAKE_PROFIT",
            )

        use_fixed_tp = bool(getattr(config, "use_fixed_tp", False))
        if use_fixed_tp:
            final_tp_pct = float(config.take_profit_pct)
            dynamic_ratio: float | None = None
        else:
            dynamic_ratio = self._get_dynamic_tp_sl_ratio(config, atr_pct)
            final_tp_pct = final_sl_pct * dynamic_ratio
        if final_tp_pct <= 0:
            return Result.failed(
                ValueError("invalid take profit pct (<=0)"),
                "INVALID_TAKE_PROFIT",
            )

        # Determine volatility tier for logging
        low_threshold = float(getattr(config, "low_volatility_threshold", 0.015))
        high_threshold = float(getattr(config, "high_volatility_threshold", 0.03))
        if atr_pct is None:
            volatility_tier = "UNKNOWN"
        elif atr_pct < low_threshold:
            volatility_tier = "LOW"
        elif atr_pct > high_threshold:
            volatility_tier = "HIGH"
        else:
            volatility_tier = "NORMAL"

        logger.info(
            "Take profit calculation",
            symbol=symbol,
            atr_pct=f"{atr_pct:.4f}" if atr_pct else "N/A",
            volatility_tier=volatility_tier,
            dynamic_ratio=dynamic_ratio,
            final_sl_pct=f"{final_sl_pct:.4f}",
            final_tp_pct=f"{final_tp_pct:.4f}",
        )

        tx_costs: TransactionCostConfig = config.transaction_costs

        if side == "LONG":
            base_tp_price = float(entry_price) * (1.0 + final_tp_pct)
            spread_slippage_factor = 1.0 - tx_costs.spread_pct - tx_costs.slippage_pct
            tp_price = base_tp_price * spread_slippage_factor
        else:  # SHORT
            tp_price = float(entry_price) * (1.0 - final_tp_pct)

        if tp_price <= 0:
            return Result.failed(
                ValueError("invalid take profit price (<=0)"),
                "INVALID_TAKE_PROFIT",
            )

        return Result.success(float(tp_price))


class TrailingStopPolicy:
    """Trailing stop policy (Pattern 6, Phase 1 simplified).

    Source: simplified from NautilusTrader ``crates/execution/src/trailing.rs``.

    Phase 1:
    - Only computes an initial trailing stop (no stateful updates).
    - Offset is a PRICE pct: ``config.trailing_offset_pct``.
    - Uses latest close as proxy for bid/ask; TP is fixed (non-trailing).
    """

    def calculate_entry_price(
        self,
        symbol: str,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Calculate entry price using the latest close (market proxy)."""

        if not market_data.bars:
            return Result.failed(
                ValueError("market_data.bars is empty"),
                "EMPTY_MARKET_DATA",
            )

        latest_bar = market_data.bars[-1]
        entry_price = float(latest_bar.close)

        if entry_price <= 0:
            return Result.failed(
                ValueError("invalid entry price (<=0)"),
                "INVALID_ENTRY_PRICE",
            )

        return Result.success(float(entry_price))

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Calculate initial trailing stop price.

        Algorithm (Phase 1 simplified):
        - LONG: ``sl = entry_price * (1 - trailing_offset_pct)``
        - SHORT: ``sl = entry_price * (1 + trailing_offset_pct)``
        """

        trailing_offset_pct = float(config.trailing_offset_pct)

        if side == "LONG":
            sl_price = float(entry_price) * (1.0 - trailing_offset_pct)
        else:  # SHORT
            sl_price = float(entry_price) * (1.0 + trailing_offset_pct)

        if sl_price <= 0:
            return Result.failed(
                ValueError("invalid stop loss price (<=0)"),
                "INVALID_STOP_LOSS",
            )

        return Result.success(float(sl_price))

    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Calculate a fixed take-profit price (non-trailing).

        Algorithm (Phase 1 simplified):
        - LONG: ``tp = entry_price * (1 + 2 * trailing_offset_pct)``
        - SHORT: ``tp = entry_price * (1 - 2 * trailing_offset_pct)``
        """

        trailing_offset_pct = float(config.trailing_offset_pct)

        if side == "LONG":
            tp_price = float(entry_price) * (1.0 + 2.0 * trailing_offset_pct)
        else:  # SHORT
            tp_price = float(entry_price) * (1.0 - 2.0 * trailing_offset_pct)

        if tp_price <= 0:
            return Result.failed(
                ValueError("invalid take profit price (<=0)"),
                "INVALID_TAKE_PROFIT",
            )

        return Result.success(float(tp_price))


class LadderEntryPolicy:
    """Ladder entry policy (Phase 1 simplified).

    Entry is staged across ladder levels spaced by ``config.ladder_spacing_pct``.
    SL/TP reuse ATR bracket logic (Pattern 5). Phase 1 uses latest close as base.
    """

    def calculate_entry_price(
        self,
        symbol: str,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
        ladder_level: int = 0,
    ) -> Result[float]:
        """Calculate ladder entry price for a specific ladder level.

        Algorithm:
        - Level 0: ``entry = latest_close``
        - LONG:
            ``entry = latest_close * (1 - ladder_level * ladder_spacing_pct)``
        - SHORT:
            ``entry = latest_close * (1 + ladder_level * ladder_spacing_pct)``

        Args:
            symbol: Stock/instrument symbol.
            side: ``"LONG"`` or ``"SHORT"`` (any other value is treated as SHORT).
            config: Strategy engine configuration (uses ``ladder_spacing_pct``).
            market_data: Market data snapshot providing ``bars``.
            ladder_level: 0-based ladder index.

        Returns:
            Result[float]: Entry price for the given ladder level.
        """

        if not market_data.bars:
            return Result.failed(
                ValueError("market_data.bars is empty"),
                "EMPTY_MARKET_DATA",
            )

        latest_bar = market_data.bars[-1]
        base_price = float(latest_bar.close)

        ladder_offset = float(ladder_level) * float(config.ladder_spacing_pct)

        if side == "LONG":
            entry_price = base_price * (1.0 - ladder_offset)
        else:  # SHORT
            entry_price = base_price * (1.0 + ladder_offset)

        if entry_price <= 0:
            return Result.failed(
                ValueError("invalid entry price (<=0)"),
                "INVALID_ENTRY_PRICE",
            )

        return Result.success(float(entry_price))

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Calculate stop-loss price using ATR bracket logic (Pattern 5)."""

        return ATRBracketPolicy().calculate_stop_loss(
            symbol,
            entry_price,
            side,
            config,
            market_data,
        )

    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Calculate take-profit price using ATR bracket logic (Pattern 5)."""

        return ATRBracketPolicy().calculate_take_profit(
            symbol,
            entry_price,
            side,
            config,
            market_data,
        )


def create_price_policy(config: StrategyEngineConfig) -> PricePolicyProtocol:
    """Create a price policy instance based on Strategy Engine configuration.

    Args:
        config: Strategy engine configuration.

    Returns:
        PricePolicyProtocol: Concrete price policy implementation.

    Raises:
        ValueError: If ``config.price_policy`` is unknown.
    """

    policy_type = str(config.price_policy)

    if policy_type == "atr_bracket":
        return ATRBracketPolicy()
    if policy_type == "trailing_stop":
        return TrailingStopPolicy()
    if policy_type == "ladder_entry":
        return LadderEntryPolicy()
    raise ValueError(
        f"Unknown price_policy type: {policy_type}. "
        "Expected: 'atr_bracket', 'trailing_stop', 'ladder_entry'"
    )
