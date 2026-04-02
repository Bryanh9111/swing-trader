"""Capital usage check for Risk Gate (Phase 3.3 capital limit enforcement).

This check enforces the max_capital_usage constraint from PortfolioRiskConfig,
ensuring the system respects user-defined capital limits even when broker
buying power is higher.
"""

from __future__ import annotations

from typing import Any, Mapping

from common.interface import Result
from risk_gate.interface import (
    CheckResult,
    CheckStatus,
    RiskCheckContext,
    RiskCheckProtocol,
    RiskGateConfig,
)
from strategy.interface import TradeIntent


class CapitalUsageCheck(RiskCheckProtocol):
    """Check if order respects max_capital_usage limit.

    Logic:
    1. Extract buying_power and account_equity from context
    2. Calculate allowed_capital based on config:
       - If max_capital_usage is None: allowed = buying_power (no limit)
       - If capital_usage_type="absolute": allowed = min(buying_power, max_capital_usage)
       - If capital_usage_type="percentage": allowed = min(buying_power, account_equity * max_capital_usage)
    3. Calculate proposed capital usage:
       - Sum of market_value of all existing positions (from context.positions)
       - Plus notional value of this new order intent
    4. FAIL if proposed_usage > allowed_capital with reason code "CAPITAL_LIMIT_EXCEEDED"
    """

    _CAPITAL_LIMIT_EXCEEDED = "CAPITAL_LIMIT_EXCEEDED"
    _MISSING_ACCOUNT_DATA = "MISSING_ACCOUNT_DATA"

    def check(
        self,
        intent: TradeIntent,
        context: RiskCheckContext,
        config: RiskGateConfig,
    ) -> Result[CheckResult]:
        """Evaluate capital usage check for the given intent.

        Args:
            intent: Strategy Engine trade intent to evaluate.
            context: Point-in-time risk context (equity, positions, buying_power).
            config: Risk Gate configuration thresholds.

        Returns:
            Result[CheckResult]: PASS if within capital limit, FAIL if exceeds limit.
        """

        max_capital_usage = config.portfolio.max_capital_usage
        capital_usage_type = str(config.portfolio.capital_usage_type or "absolute").lower()

        if max_capital_usage is None:
            return Result.success(
                CheckResult(
                    status=CheckStatus.PASS,
                    reason_codes=[],
                    details={"max_capital_usage": None, "note": "No capital limit configured"},
                )
            )

        account_equity = float(getattr(context, "account_equity", 0.0) or 0.0)

        buying_power: float | None = None
        if isinstance(context.portfolio_snapshot, Mapping):
            buying_power = self._coerce_float(context.portfolio_snapshot.get("buying_power"))

        if buying_power is None or buying_power <= 0:
            buying_power = account_equity

        if capital_usage_type == "percentage":
            allowed_capital = min(buying_power, account_equity * float(max_capital_usage))
        else:
            allowed_capital = min(buying_power, float(max_capital_usage))

        current_usage = 0.0
        for symbol, position in (context.positions or {}).items():
            market_value = self._extract_market_value(symbol=str(symbol), position=position, context=context)
            if market_value is None:
                continue
            current_usage += abs(float(market_value))

        order_notional = self._estimate_order_notional(intent, context)
        proposed_usage = current_usage + order_notional

        if proposed_usage > allowed_capital:
            return Result.success(
                CheckResult(
                    status=CheckStatus.FAIL,
                    reason_codes=[self._CAPITAL_LIMIT_EXCEEDED],
                    details={
                        "allowed_capital": float(allowed_capital),
                        "current_usage": float(current_usage),
                        "order_notional": float(order_notional),
                        "proposed_usage": float(proposed_usage),
                        "max_capital_usage": float(max_capital_usage),
                        "capital_usage_type": capital_usage_type,
                        "buying_power": float(buying_power),
                        "account_equity": float(account_equity),
                    },
                )
            )

        return Result.success(
            CheckResult(
                status=CheckStatus.PASS,
                reason_codes=[],
                details={
                    "allowed_capital": float(allowed_capital),
                    "current_usage": float(current_usage),
                    "order_notional": float(order_notional),
                    "proposed_usage": float(proposed_usage),
                    "remaining_capital": float(allowed_capital - proposed_usage),
                },
            )
        )

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_market_value(
        cls,
        *,
        symbol: str,
        position: Any,
        context: RiskCheckContext,
    ) -> float | None:
        if position is None:
            return None

        if isinstance(position, Mapping):
            market_value = cls._coerce_float(position.get("market_value"))
            if market_value is not None:
                return market_value
            qty = cls._coerce_float(position.get("quantity") or position.get("qty") or position.get("shares"))
            if qty is None:
                return None
            ref_price = cls._reference_price(context, symbol)
            if ref_price is None:
                return None
            return float(qty) * float(ref_price)

        if isinstance(position, (int, float)):
            ref_price = cls._reference_price(context, symbol)
            if ref_price is None:
                return None
            return float(position) * float(ref_price)

        market_value = cls._coerce_float(getattr(position, "market_value", None))
        if market_value is not None:
            return market_value
        qty = cls._coerce_float(getattr(position, "quantity", None))
        if qty is None:
            return None
        ref_price = cls._reference_price(context, symbol)
        if ref_price is None:
            return None
        return float(qty) * float(ref_price)

    @staticmethod
    def _reference_price(context: RiskCheckContext, symbol: str) -> float | None:
        md = (context.market_data or {}).get(str(symbol))
        if md is None:
            return None
        try:
            bars = list(getattr(md, "bars", None) or [])
        except Exception:
            return None
        if not bars:
            return None
        close = getattr(bars[-1], "close", None)
        try:
            value = float(close)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    def _estimate_order_notional(self, intent: TradeIntent, context: RiskCheckContext) -> float:
        """Estimate notional value of the order from intent.

        Args:
            intent: Trade intent.

        Returns:
            Estimated notional value in USD.
        """

        quantity = getattr(intent, "quantity", None)
        price = (
            getattr(intent, "entry_price", None)
            or getattr(intent, "stop_loss_price", None)
            or getattr(intent, "take_profit_price", None)
        )

        if quantity is None and hasattr(intent, "metadata") and getattr(intent, "metadata") is not None:
            try:
                quantity = intent.metadata.get("quantity")  # type: ignore[union-attr]
            except Exception:
                quantity = None
        if price is None and hasattr(intent, "metadata") and getattr(intent, "metadata") is not None:
            try:
                price = intent.metadata.get("price")  # type: ignore[union-attr]
            except Exception:
                price = None

        if price is None:
            price = self._reference_price(context, str(getattr(intent, "symbol", "")))

        if quantity is None or price is None:
            return 0.0

        try:
            return abs(float(quantity) * float(price))
        except (TypeError, ValueError):
            return 0.0

