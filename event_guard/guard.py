"""Core Event Guard policy logic (risk flags + trading constraints).

This module translates upcoming :class:`~event_guard.interface.MarketEvent`
entries into deterministic, conservative trading constraints.

The Event Guard operates in two phases:

1) :func:`generate_risk_flags` summarises upcoming events into compact
   :class:`~event_guard.interface.RiskFlags` (presence of events, sizing
   multiplier, and a best-effort "no trade until" timestamp).
2) :func:`generate_trade_constraints` expands those flags into concrete
   :class:`~event_guard.interface.TradeConstraints` that downstream execution
   components can enforce.

Notes
-----
- All timestamps are UTC nanosecond epoch integers.
- "Trading day" vs "calendar day" differences are ignored for now; blackout
  windows are computed using calendar days (24h increments).
- When event data is missing or malformed and ``config.use_conservative_defaults``
  is enabled, the guard degrades into conservative constraints:
  * Cannot open new positions
  * Cannot increase existing positions
  * Can decrease / close positions
  * max position multiplier = 0.5 (represented via ``TradeConstraints.max_position_size``)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog

from .interface import (
    EventGuardConfig,
    EventSnapshot,
    EventType,
    MarketEvent,
    RiskFlags,
    RiskLevel,
    TimestampNs,
    TradeConstraints,
)

__all__ = [
    "generate_risk_flags",
    "generate_trade_constraints",
    "apply_event_guard",
]


_NS_PER_SECOND = 1_000_000_000
_NS_PER_DAY = 86_400 * _NS_PER_SECOND


_RISK_SEVERITY: dict[RiskLevel, int] = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}

_RISK_TO_MAX_POSITION_MULTIPLIER: dict[RiskLevel, float] = {
    RiskLevel.NONE: 1.0,
    RiskLevel.LOW: 0.9,
    RiskLevel.MEDIUM: 0.75,
    RiskLevel.HIGH: 0.5,
    RiskLevel.CRITICAL: 0.0,
}


@dataclass(frozen=True, slots=True)
class _RiskPolicy:
    can_open_new: bool
    can_increase: bool
    can_decrease: bool
    max_position_multiplier: float


_RISK_TO_POLICY: dict[RiskLevel, _RiskPolicy] = {
    RiskLevel.NONE: _RiskPolicy(True, True, True, 1.0),
    RiskLevel.LOW: _RiskPolicy(True, True, True, _RISK_TO_MAX_POSITION_MULTIPLIER[RiskLevel.LOW]),
    RiskLevel.MEDIUM: _RiskPolicy(
        False, True, True, _RISK_TO_MAX_POSITION_MULTIPLIER[RiskLevel.MEDIUM]
    ),
    RiskLevel.HIGH: _RiskPolicy(
        False, False, True, _RISK_TO_MAX_POSITION_MULTIPLIER[RiskLevel.HIGH]
    ),
    RiskLevel.CRITICAL: _RiskPolicy(
        False, False, True, _RISK_TO_MAX_POSITION_MULTIPLIER[RiskLevel.CRITICAL]
    ),
}


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def _validate_timestamp_ns(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def _datetime_to_ns(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.astimezone(UTC).timestamp() * _NS_PER_SECOND)


def _ns_to_datetime_utc(ts_ns: int) -> datetime:
    if ts_ns <= 0:
        raise ValueError("ts_ns must be > 0")
    return datetime.fromtimestamp(ts_ns / _NS_PER_SECOND, tz=UTC)


def _add_days_ns(ts_ns: int, days: int) -> int:
    dt = _ns_to_datetime_utc(ts_ns)
    return _datetime_to_ns(dt + timedelta(days=int(days)))


def _lookahead_days(config: EventGuardConfig) -> int:
    """Compute a deterministic event lookahead horizon in days.

    The config does not currently expose a dedicated lookahead window. We
    derive a conservative horizon from the blackout windows so that
    "upcoming" captures the events that can impact constraints.
    """

    earnings_span = int(config.earnings_blackout_days_before) + int(config.earnings_blackout_days_after) + 1
    lockup_span = int(config.lockup_blackout_days) * 2 + 1
    split_span = int(config.split_blackout_days) + 1
    return max(7, earnings_span, lockup_span, split_span)


def _most_severe_risk(events: list[MarketEvent]) -> RiskLevel:
    if not events:
        return RiskLevel.NONE
    return max(events, key=lambda e: _RISK_SEVERITY.get(e.risk_level, 0)).risk_level


def _nearest_high_risk_event(events: list[MarketEvent]) -> MarketEvent | None:
    """Return the nearest HIGH/CRITICAL event; fallback to nearest event."""

    if not events:
        return None

    high_risk = [event for event in events if event.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
    candidates = high_risk or events
    return min(candidates, key=lambda e: (e.event_date, _RISK_SEVERITY.get(e.risk_level, 0)))


def _event_blackout_window(event: MarketEvent, config: EventGuardConfig) -> tuple[int, int] | None:
    """Return (start_ns, end_ns) for an event blackout window."""

    ts_ns = _validate_timestamp_ns(event.event_date)
    if ts_ns is None:
        return None

    if event.event_type == EventType.EARNINGS:
        start = _add_days_ns(ts_ns, -int(config.earnings_blackout_days_before))
        end = _add_days_ns(ts_ns, int(config.earnings_blackout_days_after))
        return (start, end)

    if event.event_type in (EventType.STOCK_SPLIT, EventType.REVERSE_SPLIT):
        start = ts_ns
        end = _add_days_ns(ts_ns, int(config.split_blackout_days))
        return (start, end)

    if event.event_type == EventType.LOCKUP_EXPIRATION:
        days = int(config.lockup_blackout_days)
        start = _add_days_ns(ts_ns, -days)
        end = _add_days_ns(ts_ns, days)
        return (start, end)

    if event.event_type == EventType.DIVIDEND:
        start = ts_ns
        end = _add_days_ns(ts_ns, 1)
        return (start, end)

    if event.event_type == EventType.OFFERING:
        # Not yet parameterised in config; keep a conservative short window.
        start = ts_ns
        end = _add_days_ns(ts_ns, 3)
        return (start, end)

    # OTHER: small generic cool-off window.
    start = ts_ns
    end = _add_days_ns(ts_ns, 1)
    return (start, end)


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort, dedupe, and merge overlapping/adjacent windows."""

    normalized = {(int(start), int(end)) for start, end in windows if start <= end}
    sorted_windows = sorted(normalized, key=lambda w: (w[0], w[1]))
    if not sorted_windows:
        return []

    merged: list[tuple[int, int]] = []
    current_start, current_end = sorted_windows[0]
    for start, end in sorted_windows[1:]:
        if start <= current_end + 1:  # merge overlap/adjacent
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _conservative_risk_flags(symbol: str, *, reason: str) -> RiskFlags:
    return RiskFlags(
        symbol=symbol,
        has_upcoming_event=True,
        events=[],
        no_trade_until=None,
        max_position_multiplier=0.5,
        reason=reason,
    )


def _conservative_constraints(symbol: str, *, reason_code: str) -> TradeConstraints:
    return TradeConstraints(
        symbol=symbol,
        can_open_new=False,
        can_increase=False,
        can_decrease=True,
        max_position_size=0.5,
        no_trade_windows=[],
        reason_codes=[reason_code],
    )


def generate_risk_flags(
    symbol: str,
    events: list[MarketEvent],
    config: EventGuardConfig,
    current_time_ns: int,
) -> RiskFlags:
    """Generate risk flags for a symbol based on upcoming events.

    This function is intentionally pure (no I/O). It filters the provided
    events down to "upcoming" entries within a derived lookahead horizon and
    computes:

    - ``has_upcoming_event``: whether any relevant event is upcoming
    - ``no_trade_until``: latest end timestamp among computed blackout windows
    - ``max_position_multiplier``: most restrictive multiplier implied by the
      upcoming events' risk levels

    Args:
        symbol: Target symbol/ticker.
        events: Candidate events (may include invalid entries; processed best-effort).
        config: Event Guard configuration (blackout window sizes, conservative toggle).
        current_time_ns: Current timestamp in UTC nanoseconds.

    Returns:
        A :class:`~event_guard.interface.RiskFlags` instance.
    """

    logger = structlog.get_logger(__name__).bind(component="event_guard", symbol=symbol)
    norm_symbol = _normalize_symbol(symbol)

    now_ns = _validate_timestamp_ns(current_time_ns)
    if now_ns is None:
        now_ns = time.time_ns()
        logger.warning("event_guard.invalid_current_time", current_time_ns=current_time_ns, used_time_ns=now_ns)

    lookahead_ns = now_ns + _lookahead_days(config) * _NS_PER_DAY
    upcoming: list[MarketEvent] = []
    invalid_events = 0

    for event in list(events or []):
        try:
            if _normalize_symbol(event.symbol) != norm_symbol:
                continue
            ts_ns = _validate_timestamp_ns(event.event_date)
            if ts_ns is None:
                invalid_events += 1
                continue
            if ts_ns < now_ns or ts_ns > lookahead_ns:
                continue
            upcoming.append(event)
        except Exception:  # noqa: BLE001 - best-effort filtering.
            invalid_events += 1

    upcoming.sort(key=lambda e: (e.event_date, e.event_type.value))

    if not upcoming:
        if invalid_events and config.use_conservative_defaults:
            logger.warning("event_guard.degraded_invalid_events", invalid_events=invalid_events)
            return _conservative_risk_flags(norm_symbol, reason="CONSERVATIVE_DEFAULTS_INVALID_EVENT_DATA")

        return RiskFlags(
            symbol=norm_symbol,
            has_upcoming_event=False,
            events=[],
            no_trade_until=None,
            max_position_multiplier=1.0,
            reason=None,
        )

    windows: list[tuple[int, int]] = []
    for event in upcoming:
        window = _event_blackout_window(event, config)
        if window is not None:
            windows.append(window)

    no_trade_until = max((end for _, end in windows), default=None)
    max_position_multiplier = min(
        (_RISK_TO_MAX_POSITION_MULTIPLIER.get(e.risk_level, 1.0) for e in upcoming),
        default=1.0,
    )
    most_severe = _most_severe_risk(upcoming)
    nearest = _nearest_high_risk_event(upcoming) or upcoming[0]

    logger.info(
        "event_guard.risk_flags_generated",
        upcoming_events=len(upcoming),
        nearest_event_type=nearest.event_type.value,
        nearest_event_date=nearest.event_date,
        most_severe_risk_level=most_severe.value,
        no_trade_until=no_trade_until,
        max_position_multiplier=max_position_multiplier,
    )

    return RiskFlags(
        symbol=norm_symbol,
        has_upcoming_event=True,
        events=upcoming,
        no_trade_until=no_trade_until,
        max_position_multiplier=float(max_position_multiplier),
        reason=f"UPCOMING_{nearest.event_type.value}_{most_severe.value}",
    )


def generate_trade_constraints(
    risk_flags: RiskFlags,
    config: EventGuardConfig,
) -> TradeConstraints:
    """Generate trading constraints from risk flags.

    The generated constraints are the strictest combination of:
    - risk-level policy mapping (open/increase/decrease permissions)
    - blackout windows around each upcoming event
    - max position multiplier derived from risk levels

    Note: ``TradeConstraints`` currently does not expose a dedicated position
    multiplier field. This implementation stores the multiplier in
    ``TradeConstraints.max_position_size`` when it is < 1.0 as a
    best-effort/forward-compatible representation.

    Args:
        risk_flags: Risk flags for the symbol.
        config: Event Guard configuration.

    Returns:
        A :class:`~event_guard.interface.TradeConstraints` instance.
    """

    symbol = _normalize_symbol(risk_flags.symbol)
    logger = structlog.get_logger(__name__).bind(component="event_guard", symbol=symbol)

    if not risk_flags.has_upcoming_event and not risk_flags.events:
        return TradeConstraints(symbol=symbol)

    if risk_flags.reason and risk_flags.reason.startswith("CONSERVATIVE_DEFAULTS"):
        logger.warning("event_guard.conservative_trade_constraints", reason=risk_flags.reason)
        return _conservative_constraints(symbol, reason_code=risk_flags.reason)

    most_severe = _most_severe_risk(list(risk_flags.events))
    policy = _RISK_TO_POLICY.get(most_severe, _RISK_TO_POLICY[RiskLevel.MEDIUM])

    windows: list[tuple[int, int]] = []
    reason_codes: set[str] = set()

    for event in list(risk_flags.events):
        reason_codes.add(f"EVENT_TYPE_{event.event_type.value}")
        reason_codes.add(f"EVENT_RISK_{event.risk_level.value}")

        window = _event_blackout_window(event, config)
        if window is not None:
            windows.append(window)
            if event.event_type == EventType.EARNINGS:
                reason_codes.add("EARNINGS_BLACKOUT")
            elif event.event_type in (EventType.STOCK_SPLIT, EventType.REVERSE_SPLIT):
                reason_codes.add("SPLIT_BLACKOUT")
            elif event.event_type == EventType.LOCKUP_EXPIRATION:
                reason_codes.add("LOCKUP_BLACKOUT")
            elif event.event_type == EventType.DIVIDEND:
                reason_codes.add("DIVIDEND_BLACKOUT")
            elif event.event_type == EventType.OFFERING:
                reason_codes.add("OFFERING_BLACKOUT")
            else:
                reason_codes.add("EVENT_BLACKOUT")

    merged_windows = _merge_windows(windows)
    max_multiplier = float(risk_flags.max_position_multiplier)
    if max_multiplier <= 0:
        max_multiplier = 0.0

    # Combine risk-level policy multiplier and per-event multiplier, taking the most conservative.
    policy_multiplier = float(policy.max_position_multiplier)
    effective_multiplier = min(policy_multiplier, max_multiplier)
    if effective_multiplier < 1.0:
        reason_codes.add("MAX_POSITION_REDUCED")

    constraints = TradeConstraints(
        symbol=symbol,
        can_open_new=bool(policy.can_open_new),
        can_increase=bool(policy.can_increase),
        can_decrease=bool(policy.can_decrease),
        max_position_size=effective_multiplier if effective_multiplier < 1.0 else None,
        no_trade_windows=merged_windows,
        reason_codes=sorted(reason_codes),
    )

    logger.info(
        "event_guard.trade_constraints_generated",
        most_severe_risk_level=most_severe.value,
        can_open_new=constraints.can_open_new,
        can_increase=constraints.can_increase,
        can_decrease=constraints.can_decrease,
        max_position_multiplier=effective_multiplier,
        windows=len(constraints.no_trade_windows),
        reason_codes=constraints.reason_codes,
    )

    return constraints


def apply_event_guard(
    symbols: list[str],
    event_snapshot: EventSnapshot,
    config: EventGuardConfig,
    current_time_ns: int | None = None,
) -> dict[str, TradeConstraints]:
    """Apply event guard logic to multiple symbols, returning constraints map.

    Args:
        symbols: Symbols to evaluate.
        event_snapshot: Snapshot of normalized events (may be partial coverage).
        config: Event Guard configuration.
        current_time_ns: Optional current time override in UTC nanoseconds.

    Returns:
        Mapping ``{symbol: TradeConstraints}`` for each input symbol.
    """

    logger = structlog.get_logger(__name__).bind(component="event_guard", source=event_snapshot.source)

    now_ns = _validate_timestamp_ns(current_time_ns) if current_time_ns is not None else time.time_ns()
    if now_ns is None:
        now_ns = time.time_ns()
        logger.warning("event_guard.invalid_current_time", current_time_ns=current_time_ns, used_time_ns=now_ns)

    normalized_symbols = [_normalize_symbol(sym) for sym in (symbols or []) if str(sym).strip()]
    events_by_symbol: dict[str, list[MarketEvent]] = {sym: [] for sym in normalized_symbols}
    for event in list(event_snapshot.events or []):
        sym = _normalize_symbol(event.symbol)
        if sym in events_by_symbol:
            events_by_symbol[sym].append(event)

    covered = {_normalize_symbol(sym) for sym in (event_snapshot.symbols_covered or [])}

    constraints_by_symbol: dict[str, TradeConstraints] = {}
    for sym in normalized_symbols:
        symbol_logger = logger.bind(symbol=sym)
        symbol_events = events_by_symbol.get(sym, [])

        if config.use_conservative_defaults and sym not in covered:
            symbol_logger.warning(
                "event_guard.symbol_not_covered",
                symbols_covered=len(covered),
                applied="CONSERVATIVE_DEFAULTS",
            )
            constraints_by_symbol[sym] = _conservative_constraints(sym, reason_code="CONSERVATIVE_DEFAULTS_MISSING_EVENT_DATA")
            continue

        try:
            risk_flags = generate_risk_flags(sym, symbol_events, config, now_ns)
            constraints = generate_trade_constraints(risk_flags, config)
        except Exception as exc:  # noqa: BLE001 - safety net for pipeline.
            if config.use_conservative_defaults:
                symbol_logger.exception("event_guard.exception_degraded", error=str(exc))
                constraints_by_symbol[sym] = _conservative_constraints(sym, reason_code="CONSERVATIVE_DEFAULTS_EXCEPTION")
                continue
            symbol_logger.exception("event_guard.exception_failed", error=str(exc))
            constraints_by_symbol[sym] = TradeConstraints(symbol=sym, reason_codes=["EVENT_GUARD_EXCEPTION"])
            continue

        constraints_by_symbol[sym] = constraints

    logger.info(
        "event_guard.applied",
        symbols=len(normalized_symbols),
        snapshot_events=len(event_snapshot.events),
        constrained=sum(1 for c in constraints_by_symbol.values() if (not c.can_open_new) or c.no_trade_windows or c.max_position_size),
    )

    return constraints_by_symbol
