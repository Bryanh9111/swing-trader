"""Event Guard interface definitions for event-driven trading constraints.

This module defines the Phase 3.1 Event Guard schemas for Automated Swing
Trader (AST). The Event Guard is responsible for acquiring upcoming
market-moving events (earnings, offerings, lockups, splits/dividends) and
translating them into conservative trading constraints.

Design goals (per ``docs/ROADMAP.md`` Phase 3.1):
- Event data acquisition interface for upstream sources.
- Deterministic, immutable schemas built on ``msgspec.Struct``.
- Conservative defaults: when data is missing/uncertain, callers should
  restrict trading rather than assume safety.

Cross-module contracts (Phase 1–2.3):
- All persisted payloads include ``schema_version``, ``system_version``, and
  ``asof_timestamp`` metadata (see ``journal.interface.SnapshotBase``).
- Classification fields use ``Enum`` types.
- Timestamps are ``int`` nanoseconds since Unix epoch (UTC).
- Operational boundaries use ``common.interface.Result[T]`` for all
  command/query operations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, TypeAlias, runtime_checkable

import msgspec

from common.interface import Result
from journal.interface import SnapshotBase

__all__ = [
    "TimestampNs",
    "EventType",
    "RiskLevel",
    "MarketEvent",
    "RiskFlags",
    "TradeConstraints",
    "EventGuardConfig",
    "EventSnapshot",
    "EventGuardOutput",
    "EventDataSourceAdapter",
]

TimestampNs: TypeAlias = int


class EventType(str, Enum):
    """Enumerate supported market event categories.

    The values in this enum are intentionally broad and source-agnostic so
    multiple providers (Polygon, FMP, manual overrides) can normalize into a
    stable set of event categories.
    """

    EARNINGS = "EARNINGS"
    DIVIDEND = "DIVIDEND"
    STOCK_SPLIT = "STOCK_SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    OFFERING = "OFFERING"
    LOCKUP_EXPIRATION = "LOCKUP_EXPIRATION"
    OTHER = "OTHER"


class RiskLevel(str, Enum):
    """Risk severity levels for events/constraints.

    This is inspired by the ``NewsImpact`` levels commonly used in event-driven
    trading systems, extended with a stricter ``CRITICAL`` level that signals
    "block trading" behaviour.
    """

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MarketEvent(msgspec.Struct, frozen=True, kw_only=True):
    """Canonical representation of a market event affecting a single symbol.

    Attributes:
        symbol: Instrument identifier (e.g., ``"AAPL"``).
        event_type: Normalized event category.
        event_date: Event timestamp in nanoseconds since Unix epoch (UTC).
        risk_level: Severity rating used to derive trading constraints.
        source: Source identifier (e.g., ``"polygon"``, ``"fmp"``, ``"manual"``).
        metadata: Optional key/value tags such as ``category``, ``url``,
            ``exchange``, ``currency``, or ``provider_event_id``.
    """

    symbol: str
    event_type: EventType
    event_date: TimestampNs
    risk_level: RiskLevel
    source: str
    metadata: dict[str, str] | None = msgspec.field(default=None)


class RiskFlags(msgspec.Struct, frozen=True, kw_only=True):
    """Risk flags derived for a symbol based on upcoming events.

    This structure captures a compact summary that can be attached to downstream
    decision logs or candidate evaluation reports.

    Attributes:
        symbol: Instrument identifier.
        has_upcoming_event: Whether any upcoming relevant event exists.
        events: Relevant events used to compute the flags.
        no_trade_until: Optional timestamp until which trading is forbidden.
        max_position_multiplier: Upper bound multiplier for position sizing;
            values < 1.0 indicate size reduction.
        reason: Optional human-readable explanation for the applied flags.
    """

    symbol: str
    has_upcoming_event: bool = msgspec.field(default=False)
    events: list[MarketEvent] = msgspec.field(default_factory=list)
    no_trade_until: TimestampNs | None = msgspec.field(default=None)
    max_position_multiplier: float = msgspec.field(default=1.0)
    reason: str | None = msgspec.field(default=None)


class TradeConstraints(msgspec.Struct, frozen=True, kw_only=True):
    """Trading constraints for a symbol derived from events and policy rules.

    Attributes:
        symbol: Instrument identifier.
        can_open_new: Whether opening new positions is allowed.
        can_increase: Whether adding to existing positions is allowed.
        can_decrease: Whether reducing an existing position is allowed.
        max_position_size: Optional absolute position size cap (policy-defined units).
        no_trade_windows: Forbidden trading windows expressed as
            ``[(start_ns, end_ns), ...]`` in UTC nanoseconds.
        reason_codes: Machine-readable reason codes (e.g.,
            ``"EVENT_WINDOW_RESTRICTION"``, ``"EARNINGS_BLACKOUT"``).
    """

    symbol: str
    can_open_new: bool = msgspec.field(default=True)
    can_increase: bool = msgspec.field(default=True)
    can_decrease: bool = msgspec.field(default=True)
    max_position_size: float | None = msgspec.field(default=None)
    no_trade_windows: list[tuple[TimestampNs, TimestampNs]] = msgspec.field(default_factory=list)
    reason_codes: list[str] = msgspec.field(default_factory=list)


class EventGuardConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration controlling event acquisition and constraint generation.

    Attributes:
        primary_source: Primary provider identifier (e.g., ``"polygon"``, ``"fmp"``).
        fallback_source: Optional fallback provider identifier, used only at run
            boundaries (see "No Mixed Data Sources" contract).
        manual_events_file: Optional path to a CSV/JSON file providing manual
            overrides or supplemental events.
        earnings_blackout_days_before: Days before earnings during which trading
            should be restricted.
        earnings_blackout_days_after: Days after earnings during which trading
            should be restricted.
        split_blackout_days: Cooling-off days after split/reverse split events.
        lockup_blackout_days: Cooling-off window (before/after) around lockup
            expiration events.
        use_conservative_defaults: When event data is unavailable or incomplete,
            instruct callers to degrade into conservative constraints.
        sources: Optional per-source settings for the event acquisition layer
            (e.g. enabling/disabling providers and tuning request concurrency).
    """

    primary_source: str
    fallback_source: str | None = msgspec.field(default=None)
    manual_events_file: str | None = msgspec.field(default=None)

    earnings_blackout_days_before: int = msgspec.field(default=2)
    earnings_blackout_days_after: int = msgspec.field(default=1)
    split_blackout_days: int = msgspec.field(default=5)
    lockup_blackout_days: int = msgspec.field(default=3)

    use_conservative_defaults: bool = msgspec.field(default=True)
    sources: dict[str, Any] = msgspec.field(default_factory=dict)


class EventSnapshot(SnapshotBase, frozen=True, kw_only=True):
    """Versioned snapshot of acquired/normalized events for a symbol set.

    ``EventSnapshot`` is intended for journaling and replay. It embeds the
    standard snapshot metadata envelope from ``journal.interface.SnapshotBase``.

    Attributes:
        schema_version: See ``journal.interface.SnapshotBase``.
        system_version: See ``journal.interface.SnapshotBase``.
        asof_timestamp: See ``journal.interface.SnapshotBase``.
        events: All acquired/normalized events in the snapshot.
        source: Source identifier (e.g., ``"polygon"``, ``"fmp"``, ``"manual"``).
        symbols_covered: Symbols for which event acquisition was attempted.
    """

    events: list[MarketEvent] = msgspec.field(default_factory=list)
    source: str
    symbols_covered: list[str] = msgspec.field(default_factory=list)


class EventGuardOutput(SnapshotBase, frozen=True, kw_only=True):
    """Event Guard plugin output snapshot for journaling and replay.

    This snapshot wraps the complete Event Guard output including candidate
    pass-through and generated trading constraints. It embeds the standard
    snapshot metadata envelope from ``journal.interface.SnapshotBase``.

    Attributes:
        schema_version: See ``journal.interface.SnapshotBase``.
        system_version: See ``journal.interface.SnapshotBase``.
        asof_timestamp: See ``journal.interface.SnapshotBase``.
        candidates: Serialized CandidateSet from upstream scanner.
        constraints: Symbol-to-TradeConstraints mapping (serialized).
    """

    SCHEMA_VERSION = "1.1.0"

    candidates: dict[str, Any] = msgspec.field(default_factory=dict)
    constraints: dict[str, Any] = msgspec.field(default_factory=dict)


@runtime_checkable
class EventDataSourceAdapter(Protocol):
    """Protocol for event data source adapters (earnings/offerings/lockups).

    Implementations should normalize raw provider payloads into
    :class:`~event_guard.interface.MarketEvent` entries and return them inside a
    journal-friendly :class:`~event_guard.interface.EventSnapshot`.
    """

    def fetch_events(
        self,
        symbols: list[str],
        start_ns: TimestampNs,
        end_ns: TimestampNs,
    ) -> Result[EventSnapshot]:
        """Fetch and normalize events for symbols within a time range."""

    def get_source_name(self) -> str:
        """Return a stable identifier for the data source (e.g., ``\"polygon\"``)."""
