"""Phase 3.2 Strategy Engine core intent generation logic.
Converts scanner candidates into deterministic trade intent snapshots while
integrating Event Guard constraints and degrading into protective-only mode
when critical inputs are missing.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any

import msgspec

from common.interface import Result, ResultStatus
from data.interface import PriceSeriesSnapshot
from indicators.interface import compute_macd_last, compute_rsi_last
from event_guard.interface import TradeConstraints
from portfolio.interface import AllocationResult
from scanner.interface import CandidateSet
from strategy.interface import (
    IntentGroup,
    IntentType,
    OrderIntentSet,
    PositionSizerProtocol,
    PricePolicyProtocol,
    StrategyEngineConfig,
    TradeIntent,
)

__all__ = ["generate_intents"]
_SCHEMA_VERSION = "3.2.0"
_DEFAULT_STRATEGY_ID = "PLATFORM_STRATEGY"
_PROTECTIVE_STRATEGY_ID = "PROTECTIVE"
_DEFAULT_SIDE = "LONG"


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def _generate_intent_id(
    strategy_id: str,
    symbol: str,
    bar_ts_ns: int,
    intent_type: IntentType,
    ladder_level: int | None = None,
) -> str:
    """Generate a deterministic intent id for replay stability.

    Args:
        strategy_id: Stable identifier for the producing strategy.
        symbol: Instrument identifier (normalized).
        bar_ts_ns: Reference timestamp, typically the latest bar timestamp.
        intent_type: Intent classification.
        ladder_level: Optional ladder index for staged entries.
    Returns:
        str: Deterministic id in the form
        ``O-<datetime>-<strategy>-<symbol>-<intent_type>-<ladder>-<hash>``.
    """

    dt = datetime.fromtimestamp(bar_ts_ns / 1e9, tz=UTC)
    dt_str = dt.strftime("%Y%m%dT%H%M%S")

    ladder_value = 0 if ladder_level is None else int(ladder_level)
    hash_input = f"{strategy_id}|{symbol}|{int(bar_ts_ns)}|{intent_type.value}|{ladder_value}"
    hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:8]

    ladder_str = f"-{int(ladder_level)}" if ladder_level is not None else ""
    return f"O-{dt_str}-{strategy_id}-{symbol}-{intent_type.value}{ladder_str}-{hash_hex}"


def _build_intent_group(
    symbol: str,
    entry_intent: TradeIntent,
    sl_intent: TradeIntent,
    tp_intent: TradeIntent,
    created_at_ns: int,
) -> IntentGroup:
    """Construct an atomic bracket group (entry + stop-loss + take-profit).
    Returns:
        IntentGroup: Group with correct OTO/OUO linkage semantics.
    """

    entry_with_links = msgspec.structs.replace(
        entry_intent,
        linked_intent_ids=[sl_intent.intent_id, tp_intent.intent_id],
        contingency_type="OTO",
    )
    sl_with_parent = msgspec.structs.replace(
        sl_intent,
        parent_intent_id=entry_intent.intent_id,
        reduce_only=True,
        contingency_type="OUO",
    )
    tp_with_parent = msgspec.structs.replace(
        tp_intent,
        parent_intent_id=entry_intent.intent_id,
        reduce_only=True,
        contingency_type="OUO",
    )

    group_id = f"G-{entry_intent.intent_id}"
    return IntentGroup(
        group_id=group_id,
        symbol=symbol,
        intents=[entry_with_links, sl_with_parent, tp_with_parent],
        created_at_ns=int(created_at_ns),
        contingency_type="OUO",
    )


def _apply_constraints(
    base_quantity: float,
    constraints: TradeConstraints | None,
    intent_type: IntentType,
) -> tuple[float, list[str]]:
    """Apply Event Guard constraints to a quantity (Pattern 7 subset)."""

    if constraints is None:
        return float(base_quantity), []

    reason_codes: list[str] = []
    adjusted_qty = float(base_quantity)

    if constraints.max_position_size is not None:
        cap = float(constraints.max_position_size)
        if cap >= 0:
            adjusted_qty = min(adjusted_qty, cap)
            if adjusted_qty < base_quantity:
                reason_codes.append("CONSTRAINT_MAX_POSITION_SIZE")
    if intent_type in (IntentType.OPEN_LONG, IntentType.OPEN_SHORT) and not constraints.can_open_new:
        reason_codes.append("CONSTRAINT_NO_NEW_POSITIONS")
        return 0.0, reason_codes
    if (
        intent_type in (IntentType.CLOSE_LONG, IntentType.CLOSE_SHORT, IntentType.REDUCE_POSITION)
        and not constraints.can_decrease
    ):
        reason_codes.append("CONSTRAINT_NO_DECREASE")
        return 0.0, reason_codes

    return adjusted_qty, reason_codes


def _check_indicator_confirmation(
    symbol: str,
    market_data: PriceSeriesSnapshot,
    config: StrategyEngineConfig,
) -> tuple[bool, list[str]]:
    """Check if indicators confirm entry (RSI/MACD checks).

    Returns:
        tuple[bool, list[str]]: (passed, reason_codes)
        - passed=True: all enabled checks passed
        - passed=False: at least one check failed
        - reason_codes: list of failure reasons (empty if passed)
    """

    if not config.indicators.enabled:
        return True, []

    reason_codes: list[str] = []

    if config.indicators.rsi_enabled:
        rsi_value = compute_rsi_last(market_data.bars, period=config.indicators.rsi_period)
        if rsi_value is None:
            pass
        elif rsi_value > config.indicators.rsi_overbought:
            reason_codes.append("RSI_OVERBOUGHT")

    if config.indicators.macd_enabled:
        macd_result = compute_macd_last(
            market_data.bars,
            fast=config.indicators.macd_fast,
            slow=config.indicators.macd_slow,
            signal=config.indicators.macd_signal,
        )
        if macd_result is None:
            pass
        elif config.indicators.macd_require_bullish and macd_result.histogram <= 0:
            reason_codes.append("MACD_NOT_BULLISH")

    passed = len(reason_codes) == 0
    return passed, reason_codes


def _generate_protective_intents(
    symbols: list[str],
    reason_code: str,
    created_at_ns: int,
    *,
    system_version: str,
) -> OrderIntentSet:
    """Generate protective-only intents when inputs are missing/uncertain."""

    intent_groups: list[IntentGroup] = []
    unique_symbols: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        norm = _normalize_symbol(symbol)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        unique_symbols.append(norm)

    for symbol in unique_symbols:
        cancel_intent = TradeIntent(
            intent_id=_generate_intent_id(
                _PROTECTIVE_STRATEGY_ID,
                symbol,
                int(created_at_ns),
                IntentType.CANCEL_PENDING,
            ),
            symbol=symbol,
            intent_type=IntentType.CANCEL_PENDING,
            quantity=0.0,
            created_at_ns=int(created_at_ns),
            reduce_only=True,
            reason_codes=[reason_code, "DEGRADED_PROTECTIVE_MODE"],
        )
        reduce_intent = TradeIntent(
            intent_id=_generate_intent_id(
                _PROTECTIVE_STRATEGY_ID,
                symbol,
                int(created_at_ns),
                IntentType.REDUCE_POSITION,
            ),
            symbol=symbol,
            intent_type=IntentType.REDUCE_POSITION,
            quantity=0.0,
            created_at_ns=int(created_at_ns),
            reduce_only=True,
            reason_codes=[reason_code, "DEGRADED_PROTECTIVE_MODE"],
        )
        stop_intent = TradeIntent(
            intent_id=_generate_intent_id(
                _PROTECTIVE_STRATEGY_ID,
                symbol,
                int(created_at_ns),
                IntentType.STOP_LOSS,
            ),
            symbol=symbol,
            intent_type=IntentType.STOP_LOSS,
            quantity=0.0,
            created_at_ns=int(created_at_ns),
            reduce_only=True,
            reason_codes=[reason_code, "DEGRADED_PROTECTIVE_MODE"],
        )
        group = IntentGroup(
            group_id=f"G-PROTECTIVE-{symbol}",
            symbol=symbol,
            intents=[cancel_intent, reduce_intent, stop_intent],
            created_at_ns=int(created_at_ns),
            contingency_type="OUO",
        )
        intent_groups.append(group)
    return OrderIntentSet(
        schema_version=_SCHEMA_VERSION,
        system_version=system_version,
        asof_timestamp=int(created_at_ns),
        intent_groups=intent_groups,
        constraints_applied={},
        source_candidates=unique_symbols,
    )


def generate_intents(
    candidates: CandidateSet,
    constraints: dict[str, TradeConstraints],
    market_data: dict[str, PriceSeriesSnapshot],
    account_equity: float,
    config: StrategyEngineConfig,
    position_sizer: PositionSizerProtocol,
    price_policy: PricePolicyProtocol,
    current_time_ns: int | None = None,
    allocation_context: AllocationResult | None = None,
) -> Result[OrderIntentSet]:
    """Generate trading intents from scanner candidates (Phase 3.2).
    Args:
        candidates: Scanner output snapshot (:class:`scanner.interface.CandidateSet`).
        constraints: Event Guard constraints keyed by symbol.
        market_data: Market snapshots keyed by symbol (:class:`data.interface.PriceSeriesSnapshot`).
        account_equity: Current account equity in quote currency.
        config: Strategy Engine configuration.
        position_sizer: Implementation of :class:`strategy.interface.PositionSizerProtocol`.
        price_policy: Implementation of :class:`strategy.interface.PricePolicyProtocol`.
        current_time_ns: Optional timestamp override for tests; defaults to ``time.time_ns()``.
    Returns:
        Result[OrderIntentSet]:
            - ``SUCCESS``: normal intent-set output (may be empty).
            - ``DEGRADED``: protective-only intent set when inputs are missing.
            - ``FAILED``: unexpected exception or unrecoverable error.
    Degradation strategy:
        - If ``candidates.candidates`` is empty -> ``SUCCESS`` with empty ``intent_groups``.
        - If any candidate symbol has missing/empty market data -> ``DEGRADED`` with protective intents.
        - If constraints forbid all new positions -> ``SUCCESS`` with empty ``intent_groups``.
    """
    try:
        now_ns = int(time.time_ns() if current_time_ns is None else current_time_ns)
        equity = float(account_equity)
        if equity < 0:
            raise ValueError("account_equity must be >= 0")
        raw_candidates = list(candidates.candidates or [])
        if not raw_candidates:
            return Result.success(
                OrderIntentSet(
                    schema_version=_SCHEMA_VERSION,
                    system_version=candidates.system_version,
                    asof_timestamp=now_ns,
                    intent_groups=[],
                    constraints_applied={},
                    source_candidates=[],
                )
            )
        best_by_symbol: dict[str, Any] = {}
        for candidate in raw_candidates:
            symbol = _normalize_symbol(candidate.symbol)
            if not symbol:
                continue
            current = best_by_symbol.get(symbol)
            if current is None:
                best_by_symbol[symbol] = candidate
                continue
            if (candidate.score, candidate.detected_at) > (current.score, current.detected_at):
                best_by_symbol[symbol] = candidate
        best_candidates = sorted(
            best_by_symbol.values(),
            key=lambda c: (-float(c.score), _normalize_symbol(c.symbol), int(c.detected_at)),
        )
        if allocation_context is not None:
            max_new_positions = max(int(getattr(allocation_context, "max_new_positions", 0)), 0)
            best_candidates = best_candidates[:max_new_positions]
        source_symbols = list(best_by_symbol.keys())
        missing_symbols: list[str] = []
        for candidate in best_candidates:
            symbol = _normalize_symbol(candidate.symbol)
            if not symbol:
                continue
            snapshot = market_data.get(symbol)
            if snapshot is None:
                missing_symbols.append(symbol)
                continue
            try:
                has_bars = bool(list(snapshot.bars))
            except Exception:  # noqa: BLE001 - tolerate schema drift
                has_bars = False
            if not has_bars:
                missing_symbols.append(symbol)
        if missing_symbols:
            protective = _generate_protective_intents(
                missing_symbols,
                "MISSING_MARKET_DATA",
                now_ns,
                system_version=candidates.system_version,
            )
            return Result.degraded(protective, RuntimeError("missing market data"), "MISSING_MARKET_DATA")
        intent_groups: list[IntentGroup] = []
        constraints_applied: dict[str, dict[str, Any]] = {}
        degradation_events: list[dict[str, Any]] = []
        for candidate in best_candidates:
            symbol = _normalize_symbol(candidate.symbol)
            if not symbol:
                continue
            symbol_constraints = constraints.get(symbol)
            if symbol_constraints is not None:
                constraints_applied[symbol] = msgspec.to_builtins(symbol_constraints)
                if not symbol_constraints.can_open_new:
                    continue
                try:
                    windows = list(symbol_constraints.no_trade_windows)
                except Exception:  # noqa: BLE001
                    windows = []
                if windows and any(int(start) <= now_ns <= int(end) for start, end in windows):
                    continue
            snapshot = market_data[symbol]
            indicator_passed, indicator_reasons = _check_indicator_confirmation(
                symbol=symbol,
                market_data=snapshot,
                config=config,
            )
            if not indicator_passed:
                degradation_events.append(
                    {
                        "symbol": symbol,
                        "reason_codes": indicator_reasons,
                        "event_type": "INDICATOR_FILTER_BLOCK",
                        "timestamp_ns": int(time.time_ns()),
                    }
                )
                continue
            try:
                bars = list(snapshot.bars)
                bar_ts_ns = int(bars[-1].timestamp) if bars else now_ns
            except Exception:  # noqa: BLE001 - tolerate schema drift
                bar_ts_ns = now_ns
            # --- Pricing ---
            try:
                entry_res = price_policy.calculate_entry_price(
                    symbol,
                    _DEFAULT_SIDE,
                    config,
                    snapshot,
                )
            except Exception as exc:  # noqa: BLE001
                return Result.failed(exc, "PRICE_POLICY_ENTRY_EXCEPTION")
            if entry_res.status is ResultStatus.FAILED:
                continue
            entry_price = float(entry_res.data)
            if entry_price <= 0:
                continue
            entry_levels: list[tuple[int | None, float]] = [(None, entry_price)]
            if config.entry_strategy == "ladder" and int(config.ladder_levels) > 0:
                levels = int(config.ladder_levels)
                spacing = max(0.0, float(config.ladder_spacing_pct))
                entry_levels = [
                    (level, entry_price * (1.0 - spacing * float(level))) for level in range(levels)
                ]
            # For sizing: compute a total quantity budget using the base entry/stop.
            try:
                sl_res_base = price_policy.calculate_stop_loss(
                    symbol,
                    entry_price,
                    _DEFAULT_SIDE,
                    config,
                    snapshot,
                )
            except Exception as exc:  # noqa: BLE001
                return Result.failed(exc, "PRICE_POLICY_STOP_EXCEPTION")
            if sl_res_base.status is ResultStatus.FAILED:
                continue
            base_stop = float(sl_res_base.data)
            if base_stop <= 0:
                continue
            # Pass candidate score to quality_flags for quality-scaled sizing
                if hasattr(candidate, "score"):
                    try:
                        # Convert snapshot to dict, inject score, convert back
                        snapshot_dict = msgspec.to_builtins(snapshot)
                        if "quality_flags" not in snapshot_dict:
                            snapshot_dict["quality_flags"] = {}
                        snapshot_dict["quality_flags"]["score"] = float(candidate.score)
                        snapshot = msgspec.convert(snapshot_dict, type=snapshot.__class__)  # type: ignore[arg-type]
                    except Exception:  # noqa: BLE001
                        # Use original snapshot if conversion fails
                        pass
            try:
                sizing_kwargs: dict[str, Any] = {
                    "constraints": symbol_constraints,
                    "market_data": snapshot,
                }
                if allocation_context is not None:
                    sizing_kwargs["allocation_context"] = allocation_context
                size_res = position_sizer.calculate_size(
                    symbol,
                    entry_price,
                    base_stop,
                    equity,
                    config,
                    **sizing_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                return Result.failed(exc, "POSITION_SIZER_EXCEPTION")
            if size_res.status is ResultStatus.FAILED:
                continue
            total_qty = float(size_res.data)
            if total_qty <= 0:
                continue
            total_qty, sizing_reason = _apply_constraints(
                total_qty, symbol_constraints, IntentType.OPEN_LONG
            )
            if total_qty <= 0:
                continue
            per_leg_qty = total_qty
            if len(entry_levels) > 1:
                per_leg_qty = total_qty / float(len(entry_levels))
            if per_leg_qty <= 0:
                continue
            for ladder_level, level_entry_price in entry_levels:
                if level_entry_price <= 0:
                    continue
                try:
                    sl_res = price_policy.calculate_stop_loss(
                        symbol,
                        level_entry_price,
                        _DEFAULT_SIDE,
                        config,
                        snapshot,
                    )
                    tp_res = price_policy.calculate_take_profit(
                        symbol,
                        level_entry_price,
                        _DEFAULT_SIDE,
                        config,
                        snapshot,
                    )
                except Exception as exc:  # noqa: BLE001
                    return Result.failed(exc, "PRICE_POLICY_BRACKET_EXCEPTION")
                if sl_res.status is ResultStatus.FAILED or tp_res.status is ResultStatus.FAILED:
                    continue
                stop_loss_price = float(sl_res.data)
                take_profit_price = float(tp_res.data)
                if stop_loss_price <= 0 or take_profit_price <= 0:
                    continue
                intent_reason_codes = ["SCAN_SIGNAL", *sizing_reason]
                candidate_reason_codes: list[str] = []
                try:
                    candidate_reason_codes = list(candidate.reasons or [])
                except Exception:  # noqa: BLE001
                    candidate_reason_codes = []
                if candidate_reason_codes:
                    intent_reason_codes.extend([f"SCAN_{code}" for code in candidate_reason_codes])
                if ladder_level is not None:
                    intent_reason_codes.append(f"LADDER_LEVEL_{int(ladder_level)}")
                # Extract atr_pct from candidate features for position sizing gate
                atr_pct_value = ""
                if hasattr(candidate, "features"):
                    features = getattr(candidate, "features", None)
                    if features is not None:
                        atr_pct = getattr(features, "atr_pct", None)
                        if atr_pct is not None:
                            atr_pct_value = str(atr_pct)

                # Shadow score: transparent pass-through as JSON string
                _cand_meta = getattr(candidate, "meta", None)
                _shadow_raw = ""
                if isinstance(_cand_meta, dict):
                    _ss = _cand_meta.get("shadow_score")
                    if isinstance(_ss, dict):
                        try:
                            _shadow_raw = json.dumps(_ss)
                        except (TypeError, ValueError):
                            pass

                metadata: dict[str, str] = {
                    "scanner_window": str(getattr(candidate, "window", "")),
                    "scanner_score": str(getattr(candidate, "score", "")),
                    "atr_pct": atr_pct_value,
                    "shadow_score": _shadow_raw,
                }
                entry_intent = TradeIntent(
                    intent_id=_generate_intent_id(
                        _DEFAULT_STRATEGY_ID,
                        symbol,
                        bar_ts_ns,
                        IntentType.OPEN_LONG,
                        ladder_level=ladder_level,
                    ),
                    symbol=symbol,
                    intent_type=IntentType.OPEN_LONG,
                    quantity=float(per_leg_qty),
                    created_at_ns=now_ns,
                    entry_price=float(level_entry_price),
                    ladder_level=ladder_level,
                    reason_codes=intent_reason_codes,
                    metadata=metadata,
                )
                sl_intent = TradeIntent(
                    intent_id=_generate_intent_id(
                        _DEFAULT_STRATEGY_ID,
                        symbol,
                        bar_ts_ns,
                        IntentType.STOP_LOSS,
                        ladder_level=ladder_level,
                    ),
                    symbol=symbol,
                    intent_type=IntentType.STOP_LOSS,
                    quantity=float(per_leg_qty),
                    created_at_ns=now_ns,
                    stop_loss_price=float(stop_loss_price),
                    ladder_level=ladder_level,
                    reason_codes=sizing_reason,
                    metadata=metadata,
                )
                tp_intent = TradeIntent(
                    intent_id=_generate_intent_id(
                        _DEFAULT_STRATEGY_ID,
                        symbol,
                        bar_ts_ns,
                        IntentType.TAKE_PROFIT,
                        ladder_level=ladder_level,
                    ),
                    symbol=symbol,
                    intent_type=IntentType.TAKE_PROFIT,
                    quantity=float(per_leg_qty),
                    created_at_ns=now_ns,
                    take_profit_price=float(take_profit_price),
                    ladder_level=ladder_level,
                    reason_codes=sizing_reason,
                    metadata=metadata,
                )
                intent_groups.append(_build_intent_group(symbol, entry_intent, sl_intent, tp_intent, now_ns))
        return Result.success(
            OrderIntentSet(
                schema_version=_SCHEMA_VERSION,
                system_version=candidates.system_version,
                asof_timestamp=now_ns,
                intent_groups=intent_groups,
                constraints_applied=constraints_applied,
                source_candidates=source_symbols,
                degradation_events=degradation_events,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return Result.failed(exc, "STRATEGY_ENGINE_EXCEPTION")
