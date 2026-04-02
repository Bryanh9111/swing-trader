"""Risk Gate check implementations (Phase 3.3).

This module contains the default set of Risk Checks applied by the Risk Gate.
Checks are grouped by the required context, but all implementations live in a
single file to keep early-phase wiring simple.

Design principles (see ``docs/sessions/session-2025-12-19-risk-gate.md``):
    - Deterministic: a check must be a pure function of (intent, context, config)
      plus any explicitly documented internal counters for run-scoped operational
      limits (OrderCount/RateLimit).
    - Fail-closed by default: missing critical inputs should return FAIL rather
      than passing on uncertainty.
    - Machine-readable reason codes: checks emit stable reason codes so the Risk
      Gate can aggregate decisions deterministically.

Reason codes used by this module:
    Portfolio-level:
        - ``PORTFOLIO.MAX_LEVERAGE``
        - ``PORTFOLIO.DRAWDOWN_LIMIT``
        - ``PORTFOLIO.DAILY_LOSS_LIMIT`` (also triggers Safe Mode)
        - ``PORTFOLIO.CONCENTRATION_LIMIT``
    Symbol-level:
        - ``SYMBOL.MAX_POSITION``
        - ``SYMBOL.PRICE_BAND``
        - ``SYMBOL.VOLATILITY_HALT`` (also triggers Safe Mode)
        - ``SYMBOL.REDUCE_ONLY_VIOLATION``
    Operational-level:
        - ``OPS.ORDER_COUNT_LIMIT``
        - ``OPS.RATE_LIMIT``

Fail-closed missing data:
    - ``MISSING_MARKET_DATA`` when reference prices/returns are unavailable.
    - ``MISSING_ACCOUNT`` when equity/portfolio metrics are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import fabs, isfinite, sqrt
from typing import Any, Iterable, Mapping, Sequence

from common.interface import Result
from data.interface import PriceSeriesSnapshot
from risk_gate.interface import CheckResult, CheckStatus, RiskCheckContext, RiskCheckProtocol, RiskGateConfig
from strategy.interface import IntentType, TradeIntent

__all__ = [
    "LeverageCheck",
    "DrawdownCheck",
    "DailyLossCheck",
    "ConcentrationCheck",
    "MaxPositionCheck",
    "PriceBandCheck",
    "VolatilityHaltCheck",
    "ReduceOnlyCheck",
    "OrderCountCheck",
    "check_daily_new_position_limit",
    "RateLimitCheck",
]


def _pass(*, details: dict[str, Any] | None = None) -> Result[CheckResult]:
    return Result.success(CheckResult(status=CheckStatus.PASS, reason_codes=[], details=details or {}))


def _fail(
    reason_codes: list[str],
    *,
    details: dict[str, Any] | None = None,
    safe_mode_trigger: bool = False,
) -> Result[CheckResult]:
    payload = dict(details or {})
    if safe_mode_trigger:
        payload["safe_mode_trigger"] = True
    return Result.success(CheckResult(status=CheckStatus.FAIL, reason_codes=list(reason_codes), details=payload))


def _get_from_obj_or_mapping(obj: Any, key: str) -> Any | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _get_float(obj: Any, *keys: str) -> float | None:
    for key in keys:
        value = _get_from_obj_or_mapping(obj, key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_position_quantity(position: Any) -> float | None:
    """Best-effort extraction of a signed position quantity.

    The repo intentionally keeps ``Position`` as ``Any`` in early phases. This
    helper supports a small set of common shapes:
        - numeric (float/int): treated as signed quantity
        - mappings/objects with keys/attrs: ``quantity``, ``qty``, ``size``, ``shares``
    """

    if position is None:
        return None
    if isinstance(position, (int, float)):
        return float(position)

    for key in ("quantity", "qty", "size", "shares"):
        value = _get_from_obj_or_mapping(position, key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _latest_close(snapshot: PriceSeriesSnapshot) -> float | None:
    try:
        bars = list(snapshot.bars or [])
    except Exception:  # pragma: no cover - defensive for early-phase adapters
        return None
    if not bars:
        return None
    close = getattr(bars[-1], "close", None)
    if close is None:
        return None
    try:
        value = float(close)
    except (TypeError, ValueError):
        return None
    if value <= 0 or not isfinite(value):
        return None
    return value


def _get_symbol_market_snapshot(context: RiskCheckContext, symbol: str) -> PriceSeriesSnapshot | None:
    value = (context.market_data or {}).get(str(symbol))
    if value is None or not isinstance(value, PriceSeriesSnapshot):
        return None
    return value


def _get_symbol_reference_price(context: RiskCheckContext, symbol: str) -> float | None:
    md = _get_symbol_market_snapshot(context, symbol)
    if md is None:
        return None
    return _latest_close(md)


def _intent_score(intent: TradeIntent, market_data: Mapping[str, PriceSeriesSnapshot] | None = None) -> float:
    metadata = getattr(intent, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("scanner_score", "score"):
            raw = metadata.get(key)
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass

    if market_data is None:
        return 0.0

    snapshot = market_data.get(str(intent.symbol))
    if not isinstance(snapshot, PriceSeriesSnapshot):
        return 0.0

    quality_flags = getattr(snapshot, "quality_flags", None)
    if not isinstance(quality_flags, Mapping):
        return 0.0

    raw = quality_flags.get("score")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _intent_price(intent: TradeIntent) -> float | None:
    """Return the relevant limit/trigger price for a TradeIntent.

    - Entry/close intents: ``entry_price`` (None indicates market order).
    - Stop loss: ``stop_loss_price``.
    - Take profit: ``take_profit_price``.
    """

    if intent.intent_type in (IntentType.STOP_LOSS,):
        return intent.stop_loss_price
    if intent.intent_type in (IntentType.TAKE_PROFIT,):
        return intent.take_profit_price
    return intent.entry_price


def _infer_delta_quantity(intent: TradeIntent, current_qty: float | None) -> float | None:
    qty = float(intent.quantity)
    if qty < 0:
        # Strategy contract expects quantity to be non-negative.
        return None

    it = intent.intent_type
    if it == IntentType.OPEN_LONG:
        return qty
    if it == IntentType.OPEN_SHORT:
        return -qty
    if it == IntentType.CLOSE_LONG:
        return -qty
    if it == IntentType.CLOSE_SHORT:
        return qty
    if it in (IntentType.CANCEL_PENDING,):
        return 0.0

    if it in (IntentType.REDUCE_POSITION, IntentType.STOP_LOSS, IntentType.TAKE_PROFIT):
        if current_qty is None:
            return None
        if current_qty == 0:
            return 0.0
        sign = 1.0 if current_qty > 0 else -1.0
        reduce_amount = min(fabs(current_qty), qty)
        return -sign * reduce_amount

    # Unknown/unsupported intent types should be blocked conservatively.
    return None


def _projected_quantity(intent: TradeIntent, current_qty: float | None) -> float | None:
    base = 0.0 if current_qty is None else float(current_qty)
    delta = _infer_delta_quantity(intent, current_qty)
    if delta is None:
        return None
    return base + float(delta)


def _compute_gross_exposure(context: RiskCheckContext) -> tuple[float | None, list[str]]:
    """Return (gross_exposure, missing_symbols).

    Exposure is computed as sum(abs(qty) * last_price) for all positions with
    non-zero quantities. If a required price is missing, the symbol is added to
    missing_symbols and exposure returns None.
    """

    gross_from_snapshot = _get_float(context.portfolio_snapshot, "gross_exposure", "exposure", "gross_notional")
    if gross_from_snapshot is not None and gross_from_snapshot >= 0 and isfinite(gross_from_snapshot):
        return gross_from_snapshot, []

    missing: list[str] = []
    total = 0.0
    for symbol, pos in (context.positions or {}).items():
        qty = _extract_position_quantity(pos)
        if qty is None:
            missing.append(str(symbol))
            continue
        if qty == 0:
            continue
        price = _get_symbol_reference_price(context, str(symbol))
        if price is None:
            missing.append(str(symbol))
            continue
        total += fabs(qty) * float(price)

    if missing:
        return None, missing
    return total, []


def _symbol_exposure(symbol: str, qty: float, ref_price: float) -> float:
    return fabs(float(qty)) * float(ref_price)


def _portfolio_metrics(context: RiskCheckContext) -> tuple[float | None, dict[str, Any]]:
    equity = float(context.account_equity)
    if not isfinite(equity) or equity <= 0:
        return None, {"account_equity": context.account_equity}
    return equity, {"account_equity": equity}


class LeverageCheck(RiskCheckProtocol):
    """Portfolio-level leverage cap: ``gross_exposure / account_equity``.

    Algorithm:
        1. Compute current gross exposure across the portfolio.
        2. Project the intent's symbol position change and recompute the
           projected gross exposure for the portfolio.
        3. Compute ``leverage = projected_gross_exposure / equity`` and fail
           if it exceeds ``config.portfolio.max_leverage``.

    Fail-closed behavior:
        - Missing equity => FAIL ``MISSING_ACCOUNT``.
        - Missing market data required for exposure => FAIL ``MISSING_MARKET_DATA``.
        - Unknown intent semantics => FAIL ``MISSING_ACCOUNT``.
    """

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        if getattr(intent, "reduce_only", False):
            return _pass(details={"check": "LeverageCheck", "symbol": str(intent.symbol), "skipped": True, "reason": "reduce_only"})

        equity, base_details = _portfolio_metrics(context)
        if equity is None:
            return _fail(["MISSING_ACCOUNT"], details={"check": "LeverageCheck", **base_details})

        gross, missing = _compute_gross_exposure(context)
        if gross is None:
            return _fail(
                ["MISSING_MARKET_DATA"],
                details={
                    "check": "LeverageCheck",
                    **base_details,
                    "missing_symbols": missing,
                },
            )

        symbol = str(intent.symbol)
        current_qty = _extract_position_quantity((context.positions or {}).get(symbol))
        projected_qty = _projected_quantity(intent, current_qty)
        if projected_qty is None:
            return _fail(
                ["MISSING_ACCOUNT"],
                details={
                    "check": "LeverageCheck",
                    **base_details,
                    "symbol": symbol,
                    "reason": "cannot_infer_projected_quantity",
                },
            )

        ref_price = _get_symbol_reference_price(context, symbol)
        if ref_price is None:
            return _fail(
                ["MISSING_MARKET_DATA"],
                details={"check": "LeverageCheck", **base_details, "symbol": symbol},
            )

        current_exposure = _symbol_exposure(symbol, current_qty or 0.0, ref_price)
        projected_exposure = _symbol_exposure(symbol, projected_qty, ref_price)
        projected_gross = float(gross) - float(current_exposure) + float(projected_exposure)
        leverage = projected_gross / float(equity)

        threshold = float(config.portfolio.max_leverage)
        details = {
            "check": "LeverageCheck",
            **base_details,
            "symbol": symbol,
            "gross_exposure_before": float(gross),
            "gross_exposure_after": float(projected_gross),
            "equity": float(equity),
            "leverage": float(leverage),
            "max_leverage": threshold,
        }

        if leverage > threshold:
            return _fail(["PORTFOLIO.MAX_LEVERAGE"], details=details)
        return _pass(details=details)


class DrawdownCheck(RiskCheckProtocol):
    """Portfolio-level peak-to-trough drawdown cap.

    Algorithm:
        - ``dd = (peak_equity - current_equity) / peak_equity``.
        - Fail when ``dd > config.portfolio.max_drawdown_pct``.

    ``peak_equity`` is expected to be maintained by the orchestrator and passed
    in via ``context.portfolio_snapshot`` (e.g., ``{"peak_equity": 12345.0}``).

    Fail-closed behavior:
        - Missing/invalid peak equity => FAIL ``MISSING_ACCOUNT``.
        - Missing/invalid equity => FAIL ``MISSING_ACCOUNT``.
    """

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        equity, base_details = _portfolio_metrics(context)
        if equity is None:
            return _fail(["MISSING_ACCOUNT"], details={"check": "DrawdownCheck", **base_details})

        peak = _get_float(context.portfolio_snapshot, "peak_equity", "equity_peak", "peak")
        if peak is None or not isfinite(peak) or peak <= 0:
            return _fail(
                ["MISSING_ACCOUNT"],
                details={
                    "check": "DrawdownCheck",
                    **base_details,
                    "reason": "missing_peak_equity",
                },
            )

        dd = (float(peak) - float(equity)) / float(peak)
        threshold = float(config.portfolio.max_drawdown_pct)
        details = {
            "check": "DrawdownCheck",
            **base_details,
            "peak_equity": float(peak),
            "equity": float(equity),
            "drawdown_pct": float(dd),
            "max_drawdown_pct": threshold,
        }
        if dd > threshold:
            return _fail(["PORTFOLIO.DRAWDOWN_LIMIT"], details=details)
        return _pass(details=details)


class DailyLossCheck(RiskCheckProtocol):
    """Portfolio-level daily loss cap relative to day-start equity.

    Algorithm:
        - ``daily_loss = (day_start_equity - current_equity) / day_start_equity``
        - Fail when ``daily_loss > config.portfolio.max_daily_loss_pct``.

    ``day_start_equity`` is expected to be maintained by the orchestrator and
    passed in via ``context.portfolio_snapshot`` (e.g.,
    ``{"day_start_equity": 12345.0}``).

    Safe Mode:
        - When the limit is breached, the check sets ``details["safe_mode_trigger"]=True``
          so the Risk Gate can emit a DOWNGRADE decision.

    Fail-closed behavior:
        - Missing/invalid day-start equity => FAIL ``MISSING_ACCOUNT``.
        - Missing/invalid equity => FAIL ``MISSING_ACCOUNT``.
    """

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        equity, base_details = _portfolio_metrics(context)
        if equity is None:
            return _fail(["MISSING_ACCOUNT"], details={"check": "DailyLossCheck", **base_details})

        day_start = _get_float(context.portfolio_snapshot, "day_start_equity", "equity_day_start", "start_equity")
        if day_start is None or not isfinite(day_start) or day_start <= 0:
            return _fail(
                ["MISSING_ACCOUNT"],
                details={
                    "check": "DailyLossCheck",
                    **base_details,
                    "reason": "missing_day_start_equity",
                },
            )

        daily_loss = (float(day_start) - float(equity)) / float(day_start)
        threshold = float(config.portfolio.max_daily_loss_pct)
        details = {
            "check": "DailyLossCheck",
            **base_details,
            "day_start_equity": float(day_start),
            "equity": float(equity),
            "daily_loss_pct": float(daily_loss),
            "max_daily_loss_pct": threshold,
        }
        if daily_loss > threshold:
            return _fail(["PORTFOLIO.DAILY_LOSS_LIMIT"], details=details, safe_mode_trigger=True)
        return _pass(details=details)


class ConcentrationCheck(RiskCheckProtocol):
    """Portfolio-level single-symbol concentration cap.

    Algorithm:
        - Compute projected post-intent symbol exposure:
          ``symbol_exposure = abs(projected_qty) * ref_price``.
        - Compute concentration fraction: ``symbol_exposure / equity``.
        - Fail when fraction exceeds ``config.portfolio.max_concentration_pct``.

    Fail-closed behavior:
        - Missing equity => FAIL ``MISSING_ACCOUNT``.
        - Missing market data for the intent symbol => FAIL ``MISSING_MARKET_DATA``.
        - Unknown intent semantics => FAIL ``MISSING_ACCOUNT``.
    """

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        if getattr(intent, "reduce_only", False):
            return _pass(details={"check": "ConcentrationCheck", "symbol": str(intent.symbol), "skipped": True, "reason": "reduce_only"})

        equity, base_details = _portfolio_metrics(context)
        if equity is None:
            return _fail(["MISSING_ACCOUNT"], details={"check": "ConcentrationCheck", **base_details})

        symbol = str(intent.symbol)
        ref_price = _get_symbol_reference_price(context, symbol)
        if ref_price is None:
            return _fail(
                ["MISSING_MARKET_DATA"],
                details={"check": "ConcentrationCheck", **base_details, "symbol": symbol},
            )

        current_qty = _extract_position_quantity((context.positions or {}).get(symbol))
        projected_qty = _projected_quantity(intent, current_qty)
        if projected_qty is None:
            return _fail(
                ["MISSING_ACCOUNT"],
                details={
                    "check": "ConcentrationCheck",
                    **base_details,
                    "symbol": symbol,
                    "reason": "cannot_infer_projected_quantity",
                },
            )

        symbol_exposure = _symbol_exposure(symbol, projected_qty, ref_price)
        concentration = symbol_exposure / float(equity)
        threshold = float(config.portfolio.max_concentration_pct)
        details = {
            "check": "ConcentrationCheck",
            **base_details,
            "symbol": symbol,
            "equity": float(equity),
            "ref_price": float(ref_price),
            "projected_qty": float(projected_qty),
            "symbol_exposure": float(symbol_exposure),
            "concentration_pct": float(concentration),
            "max_concentration_pct": threshold,
        }
        if concentration > threshold:
            return _fail(["PORTFOLIO.CONCENTRATION_LIMIT"], details=details)
        return _pass(details=details)


class MaxPositionCheck(RiskCheckProtocol):
    """Symbol-level maximum absolute position size cap.

    Algorithm:
        - Project the post-intent position quantity for ``intent.symbol``.
        - Fail when ``abs(projected_qty) > config.symbol.max_position_size``.

    Fail-closed behavior:
        - If projected quantity cannot be inferred => FAIL ``MISSING_ACCOUNT``.
    """

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        symbol = str(intent.symbol)
        current_qty = _extract_position_quantity((context.positions or {}).get(symbol))
        projected_qty = _projected_quantity(intent, current_qty)
        if projected_qty is None:
            return _fail(
                ["MISSING_ACCOUNT"],
                details={"check": "MaxPositionCheck", "symbol": symbol, "reason": "cannot_infer_projected_quantity"},
            )

        threshold = float(config.symbol.max_position_size)
        details = {
            "check": "MaxPositionCheck",
            "symbol": symbol,
            "current_qty": float(current_qty or 0.0),
            "projected_qty": float(projected_qty),
            "max_position_size": threshold,
        }
        if fabs(projected_qty) > threshold:
            return _fail(["SYMBOL.MAX_POSITION"], details=details)
        return _pass(details=details)


class PriceBandCheck(RiskCheckProtocol):
    """Symbol-level price band check (basis points around last close).

    Algorithm:
        - Choose reference price ``ref`` from market data (latest close).
        - Choose intended price from the intent:
            - Entry/close intents: ``entry_price`` (market orders are skipped).
            - Stop loss: ``stop_loss_price``.
            - Take profit: ``take_profit_price``.
        - Compute deviation in bps:
            ``dev_bps = abs(price - ref) / ref * 10000``.
        - Fail when ``dev_bps > config.symbol.max_price_band_bps``.

    Fail-closed behavior:
        - Missing market data => FAIL ``MISSING_MARKET_DATA``.
        - Missing required intent price for non-market intents => FAIL ``CHECK_FAILED``.
    """

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        symbol = str(intent.symbol)
        ref = _get_symbol_reference_price(context, symbol)
        if ref is None:
            return _fail(["MISSING_MARKET_DATA"], details={"check": "PriceBandCheck", "symbol": symbol})

        price = _intent_price(intent)
        if price is None:
            # Market orders (entry_price=None) are allowed; protective legs must specify a price.
            if intent.intent_type in (IntentType.OPEN_LONG, IntentType.OPEN_SHORT, IntentType.CLOSE_LONG, IntentType.CLOSE_SHORT):
                return _pass(details={"check": "PriceBandCheck", "symbol": symbol, "skipped": "market_order"})
            return _fail(
                ["CHECK_FAILED"],
                details={"check": "PriceBandCheck", "symbol": symbol, "reason": "missing_intent_price"},
            )

        if float(ref) <= 0:
            return _fail(
                ["MISSING_MARKET_DATA"],
                details={"check": "PriceBandCheck", "symbol": symbol, "reason": "invalid_ref_price"},
            )

        dev_bps = fabs(float(price) - float(ref)) / float(ref) * 10_000.0
        threshold = float(config.symbol.max_price_band_bps)

        # Dynamic price band calculation (ATR-based adaptive threshold)
        effective_threshold = threshold  # Default to base threshold (floor)
        dynamic_price_band_details: dict[str, object] | None = None

        if config.symbol.use_dynamic_price_band:
            # Attempt to calculate ATR-based dynamic threshold
            md = context.market_data.get(symbol)
            if md is not None and hasattr(md, "bars") and md.bars:
                try:
                    from indicators.interface import compute_atr_last

                    # Calculate ATR as percentage of reference price
                    atr_pct = compute_atr_last(
                        md.bars,
                        period=config.symbol.atr_period,
                        percentage=True,
                        reference_price=ref,
                    )

                    if atr_pct is not None:
                        # Dynamic threshold = max(base_threshold, ATR% * multiplier * 10000)
                        dynamic_bps = float(atr_pct) * float(config.symbol.atr_multiplier) * 10_000.0
                        effective_threshold = max(threshold, dynamic_bps)
                        dynamic_price_band_details = {
                            "enabled": True,
                            "atr_pct": float(atr_pct),
                            "atr_multiplier": float(config.symbol.atr_multiplier),
                            "atr_period": config.symbol.atr_period,
                            "dynamic_bps": float(dynamic_bps),
                            "effective_bps": float(effective_threshold),
                            "base_bps": threshold,
                        }
                    else:
                        # ATR calculation failed (insufficient data), fall back to base threshold
                        dynamic_price_band_details = {
                            "enabled": True,
                            "atr_calculation_failed": True,
                            "fallback_to_base": True,
                            "effective_bps": threshold,
                        }
                except Exception as exc:  # noqa: BLE001
                    dynamic_price_band_details = {
                        "enabled": True,
                        "atr_calculation_failed": True,
                        "error": str(exc),
                        "fallback_to_base": True,
                        "effective_bps": threshold,
                    }
            else:
                # Market data insufficient for ATR calculation, fall back to base threshold
                dynamic_price_band_details = {
                    "enabled": True,
                    "insufficient_bars": True,
                    "fallback_to_base": True,
                    "effective_bps": threshold,
                }

        details = {
            "check": "PriceBandCheck",
            "symbol": symbol,
            "intent_type": intent.intent_type.value,
            "price": float(price),
            "ref_price": float(ref),
            "dev_bps": float(dev_bps),
            "max_price_band_bps": float(effective_threshold),
        }
        if dynamic_price_band_details is not None:
            details["dynamic_price_band"] = dynamic_price_band_details

        if dev_bps > effective_threshold:
            return _fail(["SYMBOL.PRICE_BAND"], details=details)
        return _pass(details=details)


def _returns_from_closes(closes: Iterable[float]) -> list[float]:
    values = list(closes)
    out: list[float] = []
    for i in range(1, len(values)):
        prev = float(values[i - 1])
        cur = float(values[i])
        if prev <= 0:
            return []
        out.append((cur - prev) / prev)
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / float(len(values))
    var = sum((x - mean) ** 2 for x in values) / float(len(values))
    return mean, sqrt(var)


class VolatilityHaltCheck(RiskCheckProtocol):
    """Symbol-level volatility halt via returns z-score.

    Algorithm:
        - Extract recent closes from market data bars.
        - Compute returns series and z-score for the most recent return:
          ``z = (r_last - mean) / std``.
        - Fail when ``abs(z) > config.symbol.max_volatility_z_score``.

    Safe Mode:
        - When breached, this check sets ``details["safe_mode_trigger"]=True``.

    Fail-closed behavior:
        - Missing market data or insufficient bars => FAIL ``MISSING_MARKET_DATA``.
    """

    def __init__(self, *, lookback_returns: int = 20) -> None:
        self._lookback_returns = int(lookback_returns)

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        symbol = str(intent.symbol)
        md = _get_symbol_market_snapshot(context, symbol)
        if md is None:
            return _fail(["MISSING_MARKET_DATA"], details={"check": "VolatilityHaltCheck", "symbol": symbol})

        bars = list(md.bars or [])
        min_bars = self._lookback_returns + 1
        if len(bars) < min_bars:
            return _fail(
                ["MISSING_MARKET_DATA"],
                details={
                    "check": "VolatilityHaltCheck",
                    "symbol": symbol,
                    "reason": "insufficient_bars",
                    "bars": len(bars),
                    "min_bars": min_bars,
                },
            )

        closes = [float(getattr(b, "close")) for b in bars[-min_bars:]]
        returns = _returns_from_closes(closes)
        if len(returns) < self._lookback_returns:
            return _fail(
                ["MISSING_MARKET_DATA"],
                details={
                    "check": "VolatilityHaltCheck",
                    "symbol": symbol,
                    "reason": "invalid_returns",
                    "bars": len(bars),
                },
            )

        r_last = float(returns[-1])
        mean, std = _mean_std(returns[:-1] or returns)
        if std == 0.0:
            z = float("inf") if r_last != mean else 0.0
        else:
            z = (r_last - mean) / std

        threshold = float(config.symbol.max_volatility_z_score)
        details = {
            "check": "VolatilityHaltCheck",
            "symbol": symbol,
            "intent_type": intent.intent_type.value,
            "lookback_returns": self._lookback_returns,
            "r_last": float(r_last),
            "mean": float(mean),
            "std": float(std),
            "z_score": float(z),
            "max_volatility_z_score": threshold,
        }

        if fabs(float(z)) > threshold:
            return _fail(["SYMBOL.VOLATILITY_HALT"], details=details, safe_mode_trigger=True)
        return _pass(details=details)


class ReduceOnlyCheck(RiskCheckProtocol):
    """Symbol-level reduce-only correctness check.

    Contract:
        When ``intent.reduce_only=True``, the intent must not increase the
        absolute position size for ``intent.symbol``.

    Algorithm:
        - Project the post-intent position quantity.
        - Require ``abs(projected_qty) <= abs(current_qty)``.

    Fail-closed behavior:
        - If reduce-only is set but current position is missing/unknown => FAIL.
        - If projected quantity cannot be inferred => FAIL.
    """

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        if not bool(intent.reduce_only):
            return _pass(details={"check": "ReduceOnlyCheck", "symbol": str(intent.symbol), "skipped": True})

        symbol = str(intent.symbol)
        parent_id = getattr(intent, "parent_intent_id", None)
        if parent_id:
            intents_obj = _get_from_obj_or_mapping(context, "intents")
            parent: TradeIntent | None = None
            if isinstance(intents_obj, Mapping):
                candidate = intents_obj.get(str(parent_id))
                if candidate is not None:
                    parent = candidate  # type: ignore[assignment]
            elif isinstance(intents_obj, Iterable) and not isinstance(intents_obj, (str, bytes)):
                for candidate in intents_obj:
                    if _get_from_obj_or_mapping(candidate, "intent_id") == str(parent_id):
                        parent = candidate  # type: ignore[assignment]
                        break

            if parent is not None:
                parent_type = _get_from_obj_or_mapping(parent, "intent_type")
                parent_reduce_only = bool(_get_from_obj_or_mapping(parent, "reduce_only"))
                if (not parent_reduce_only) and parent_type in (IntentType.OPEN_LONG, IntentType.OPEN_SHORT):
                    return _pass(
                        details={
                            "check": "ReduceOnlyCheck",
                            "symbol": symbol,
                            "intent_type": intent.intent_type.value,
                            "skipped": "bracket_parent_opens_position",
                            "parent_intent_id": str(parent_id),
                            "parent_intent_type": getattr(parent_type, "value", str(parent_type)),
                        }
                    )
        current_qty = _extract_position_quantity((context.positions or {}).get(symbol))
        if current_qty is None:
            return _fail(
                ["SYMBOL.REDUCE_ONLY_VIOLATION"],
                details={"check": "ReduceOnlyCheck", "symbol": symbol, "reason": "missing_current_position"},
            )

        projected_qty = _projected_quantity(intent, current_qty)
        if projected_qty is None:
            return _fail(
                ["SYMBOL.REDUCE_ONLY_VIOLATION"],
                details={"check": "ReduceOnlyCheck", "symbol": symbol, "reason": "cannot_infer_projected_quantity"},
            )

        details = {
            "check": "ReduceOnlyCheck",
            "symbol": symbol,
            "intent_type": intent.intent_type.value,
            "current_qty": float(current_qty),
            "projected_qty": float(projected_qty),
        }
        if fabs(projected_qty) > fabs(current_qty) + 1e-9:
            return _fail(["SYMBOL.REDUCE_ONLY_VIOLATION"], details=details)
        return _pass(details=details)


@dataclass(slots=True)
class OrderCountCheck(RiskCheckProtocol):
    """Operational-level max orders per run (run-scoped counter).

    This check is stateful by design: it counts unique intent ids passed through
    it during an evaluation run to enforce a run-level cap.

    Algorithm:
        - Maintain a set of seen intent ids.
        - For each new intent id, increment counter.
        - Fail when ``count > config.operational.max_orders_per_run``.
    """

    initial_count: int = 0
    _seen: set[str] = field(default_factory=set, init=False)
    _count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._count = int(self.initial_count)

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        intent_id = str(intent.intent_id)
        if intent_id not in self._seen:
            self._seen.add(intent_id)
            self._count += 1

        limit = int(config.operational.max_orders_per_run)
        details = {
            "check": "OrderCountCheck",
            "intent_id": intent_id,
            "count": int(self._count),
            "max_orders_per_run": limit,
        }
        if self._count > limit:
            return _fail(["OPS.ORDER_COUNT_LIMIT"], details=details)
        return _pass(details=details)


def check_daily_new_position_limit(
    intents: Sequence[TradeIntent],
    config: RiskGateConfig,
    *,
    market_data: Mapping[str, PriceSeriesSnapshot] | None = None,
) -> CheckResult:
    """Check daily new position limit and identify top N symbols by score.

    Notes:
        The Risk Gate uses per-intent checks for ALLOW/BLOCK decisions, but this
        helper operates on a batch to support run-level filtering/downgrading.

    Logic:
        - Count OPEN_LONG/OPEN_SHORT intents (new positions)
        - If count <= limit: PASS
        - If count > limit:
          - Sort new-position intents by score (descending)
          - Keep top N unique symbols
          - Return FAIL with kept/rejected symbols
    """

    new_position_intents = [
        intent
        for intent in list(intents or [])
        if intent.intent_type in (IntentType.OPEN_LONG, IntentType.OPEN_SHORT) and not bool(intent.reduce_only)
    ]

    count = len(new_position_intents)
    limit = int(config.operational.max_new_positions_per_day)

    if count <= limit:
        return CheckResult(
            status=CheckStatus.PASS,
            reason_codes=[],
            details={
                "new_position_count": count,
                "max_new_positions_per_day": limit,
            },
        )

    ranked = sorted(
        new_position_intents,
        key=lambda intent: (-_intent_score(intent, market_data), str(intent.symbol), str(intent.intent_id)),
    )

    kept_symbols: list[str] = []
    seen: set[str] = set()
    for intent in ranked:
        symbol = str(intent.symbol)
        if symbol in seen:
            continue
        seen.add(symbol)
        kept_symbols.append(symbol)
        if len(kept_symbols) >= limit:
            break

    rejected_symbols: list[str] = []
    for intent in ranked:
        symbol = str(intent.symbol)
        if symbol in seen:
            continue
        seen.add(symbol)
        rejected_symbols.append(symbol)

    return CheckResult(
        status=CheckStatus.FAIL,
        reason_codes=["DAILY_NEW_POSITION_LIMIT_EXCEEDED"],
        details={
            "new_position_count": count,
            "max_new_positions_per_day": limit,
            "rejected_count": len(rejected_symbols),
            "kept_symbols": kept_symbols,
            "rejected_symbols": rejected_symbols,
        },
    )


@dataclass(slots=True)
class RateLimitCheck(RiskCheckProtocol):
    """Operational-level submit rate limit (simplified total-count implementation).

    The full implementation would enforce a sliding time window (e.g., max N
    submits per second). Per requirements for Phase 3.3, this simplified
    version skips time window maintenance and checks only a total cap.

    Algorithm:
        - Maintain a set of seen intent ids.
        - For each new intent id, increment counter.
        - Fail when ``count > rate_limit`` (defaults to ``config.operational.max_order_count``).
    """

    initial_count: int = 0
    rate_limit: int | None = None
    _seen: set[str] = field(default_factory=set, init=False)
    _count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._count = int(self.initial_count)

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        intent_id = str(intent.intent_id)
        if intent_id not in self._seen:
            self._seen.add(intent_id)
            self._count += 1

        limit = int(self.rate_limit) if self.rate_limit is not None else int(config.operational.max_order_count)
        details = {
            "check": "RateLimitCheck",
            "intent_id": intent_id,
            "count": int(self._count),
            "rate_limit": int(limit),
        }
        if self._count > limit:
            return _fail(["OPS.RATE_LIMIT"], details=details)
        return _pass(details=details)
