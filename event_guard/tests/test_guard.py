from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import event_guard.guard as guard_module
from event_guard.interface import (
    EventGuardConfig,
    EventSnapshot,
    EventType,
    MarketEvent,
    RiskFlags,
    RiskLevel,
)


def _dt_to_ns(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.astimezone(UTC).timestamp() * 1_000_000_000)


def _make_event(
    *,
    symbol: str,
    event_type: EventType,
    event_date_ns: int,
    risk_level: RiskLevel,
    source: str = "test",
) -> MarketEvent:
    return MarketEvent(
        symbol=symbol,
        event_type=event_type,
        event_date=event_date_ns,
        risk_level=risk_level,
        source=source,
    )


@pytest.fixture
def config() -> EventGuardConfig:
    return EventGuardConfig(primary_source="polygon")


@pytest.fixture
def base_now_ns() -> int:
    return _dt_to_ns(datetime(2025, 1, 1, tzinfo=UTC))


def test_generate_risk_flags_no_events_returns_no_risk(config: EventGuardConfig, base_now_ns: int) -> None:
    """Return clean RiskFlags when no events exist for the symbol."""

    flags = guard_module.generate_risk_flags("aapl", [], config, current_time_ns=base_now_ns)
    assert flags.symbol == "AAPL"
    assert flags.has_upcoming_event is False
    assert flags.events == []
    assert flags.no_trade_until is None
    assert flags.max_position_multiplier == pytest.approx(1.0)
    assert flags.reason is None


@pytest.mark.parametrize(
    ("event_type", "risk_level", "expected_multiplier"),
    [
        (EventType.EARNINGS, RiskLevel.HIGH, 0.5),
        (EventType.STOCK_SPLIT, RiskLevel.HIGH, 0.5),
        (EventType.DIVIDEND, RiskLevel.LOW, 0.9),
    ],
)
def test_generate_risk_flags_single_event_sets_no_trade_until_and_multiplier(
    config: EventGuardConfig,
    base_now_ns: int,
    event_type: EventType,
    risk_level: RiskLevel,
    expected_multiplier: float,
) -> None:
    """Compute risk flags from a single upcoming event."""

    event_date = base_now_ns + guard_module._NS_PER_DAY
    event = _make_event(
        symbol="AAPL",
        event_type=event_type,
        event_date_ns=event_date,
        risk_level=risk_level,
    )
    flags = guard_module.generate_risk_flags("AAPL", [event], config, current_time_ns=base_now_ns)

    assert flags.has_upcoming_event is True
    assert flags.events == [event]
    assert flags.no_trade_until is not None and flags.no_trade_until >= event_date
    assert flags.max_position_multiplier == pytest.approx(expected_multiplier)
    assert flags.reason is not None and flags.reason.startswith("UPCOMING_")


def test_generate_risk_flags_multiple_events_choose_most_conservative(config: EventGuardConfig, base_now_ns: int) -> None:
    """Pick the strictest multiplier among multiple upcoming events."""

    dividend = _make_event(
        symbol="AAPL",
        event_type=EventType.DIVIDEND,
        event_date_ns=base_now_ns + guard_module._NS_PER_DAY,
        risk_level=RiskLevel.LOW,
    )
    earnings = _make_event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date_ns=base_now_ns + 3 * guard_module._NS_PER_DAY,
        risk_level=RiskLevel.HIGH,
    )
    flags = guard_module.generate_risk_flags("AAPL", [earnings, dividend], config, current_time_ns=base_now_ns)

    assert flags.events == [dividend, earnings]
    assert flags.max_position_multiplier == pytest.approx(0.5)
    assert flags.reason == "UPCOMING_EARNINGS_HIGH"


def test_generate_risk_flags_filters_expired_and_outside_lookahead(config: EventGuardConfig, base_now_ns: int) -> None:
    """Ignore events in the past and beyond the derived lookahead horizon."""

    expired = _make_event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date_ns=base_now_ns - guard_module._NS_PER_DAY,
        risk_level=RiskLevel.HIGH,
    )
    beyond_lookahead = _make_event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date_ns=base_now_ns + 8 * guard_module._NS_PER_DAY,
        risk_level=RiskLevel.HIGH,
    )

    flags = guard_module.generate_risk_flags(
        "AAPL",
        [expired, beyond_lookahead],
        config,
        current_time_ns=base_now_ns,
    )
    assert flags.has_upcoming_event is False
    assert flags.events == []


def test_generate_risk_flags_invalid_events_trigger_conservative_defaults(base_now_ns: int) -> None:
    """Degrade to conservative defaults when events are invalid and enabled."""

    config = EventGuardConfig(primary_source="polygon", use_conservative_defaults=True)
    invalid = _make_event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date_ns=0,
        risk_level=RiskLevel.HIGH,
    )

    flags = guard_module.generate_risk_flags("aapl", [invalid], config, current_time_ns=base_now_ns)
    assert flags.symbol == "AAPL"
    assert flags.has_upcoming_event is True
    assert flags.events == []
    assert flags.max_position_multiplier == pytest.approx(0.5)
    assert flags.reason == "CONSERVATIVE_DEFAULTS_INVALID_EVENT_DATA"


def test_generate_risk_flags_invalid_current_time_uses_time_time_ns(monkeypatch) -> None:
    """Use time.time_ns when current_time_ns is invalid."""

    fixed_now = _dt_to_ns(datetime(2025, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(guard_module.time, "time_ns", lambda: fixed_now)

    config = EventGuardConfig(primary_source="polygon")
    event = _make_event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date_ns=fixed_now + guard_module._NS_PER_DAY,
        risk_level=RiskLevel.HIGH,
    )
    flags = guard_module.generate_risk_flags("AAPL", [event], config, current_time_ns=-1)
    assert flags.has_upcoming_event is True
    assert flags.events == [event]


@pytest.mark.parametrize(
    ("risk_level", "expected_open", "expected_increase", "expected_decrease", "expected_max_position"),
    [
        (RiskLevel.NONE, True, True, True, None),
        (RiskLevel.LOW, True, True, True, 0.9),
        (RiskLevel.MEDIUM, False, True, True, 0.75),
        (RiskLevel.HIGH, False, False, True, 0.5),
        (RiskLevel.CRITICAL, False, False, True, 0.0),
    ],
)
def test_generate_trade_constraints_risk_level_mapping(
    config: EventGuardConfig,
    base_now_ns: int,
    risk_level: RiskLevel,
    expected_open: bool,
    expected_increase: bool,
    expected_decrease: bool,
    expected_max_position: float | None,
) -> None:
    """Map risk levels into policies and position sizing constraints."""

    event = _make_event(
        symbol="AAPL",
        event_type=EventType.OTHER,
        event_date_ns=base_now_ns + guard_module._NS_PER_DAY,
        risk_level=risk_level,
    )
    flags = RiskFlags(symbol="AAPL", has_upcoming_event=True, events=[event], max_position_multiplier=1.0, reason="x")
    constraints = guard_module.generate_trade_constraints(flags, config)

    assert constraints.symbol == "AAPL"
    assert constraints.can_open_new is expected_open
    assert constraints.can_increase is expected_increase
    assert constraints.can_decrease is expected_decrease
    assert constraints.max_position_size == expected_max_position
    assert constraints.no_trade_windows
    assert "EVENT_TYPE_OTHER" in constraints.reason_codes
    assert f"EVENT_RISK_{risk_level.value}" in constraints.reason_codes


def test_generate_trade_constraints_merges_overlapping_and_adjacent_windows(
    config: EventGuardConfig, base_now_ns: int
) -> None:
    """Merge overlapping/adjacent no_trade_windows across multiple events."""

    # earnings window: [-2d, +1d] around event_date => [day 8, day 11]
    earnings_dt = datetime.fromtimestamp(base_now_ns / 1_000_000_000, tz=UTC) + timedelta(days=10)
    earnings_ns = _dt_to_ns(earnings_dt)
    earnings = _make_event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date_ns=earnings_ns,
        risk_level=RiskLevel.HIGH,
    )

    # dividend window: [ts, ts+1d], set ts to end of earnings window (adjacent/overlap) => merge.
    dividend = _make_event(
        symbol="AAPL",
        event_type=EventType.DIVIDEND,
        event_date_ns=guard_module._add_days_ns(earnings_ns, config.earnings_blackout_days_after),
        risk_level=RiskLevel.LOW,
    )

    flags = RiskFlags(
        symbol="AAPL",
        has_upcoming_event=True,
        events=[earnings, dividend],
        max_position_multiplier=0.5,
        reason="UPCOMING_EARNINGS_HIGH",
    )
    constraints = guard_module.generate_trade_constraints(flags, config)

    assert len(constraints.no_trade_windows) == 1
    start, end = constraints.no_trade_windows[0]
    assert start <= earnings_ns <= end
    assert end >= dividend.event_date
    assert "EARNINGS_BLACKOUT" in constraints.reason_codes
    assert "DIVIDEND_BLACKOUT" in constraints.reason_codes


def test_generate_trade_constraints_conservative_reason_maps_to_conservative_constraints(config: EventGuardConfig) -> None:
    """Generate conservative constraints when risk_flags indicate conservative defaults."""

    flags = RiskFlags(
        symbol="AAPL",
        has_upcoming_event=True,
        events=[],
        max_position_multiplier=0.5,
        reason="CONSERVATIVE_DEFAULTS_INVALID_EVENT_DATA",
    )
    constraints = guard_module.generate_trade_constraints(flags, config)

    assert constraints.can_open_new is False
    assert constraints.can_increase is False
    assert constraints.can_decrease is True
    assert constraints.max_position_size == pytest.approx(0.5)
    assert constraints.reason_codes == ["CONSERVATIVE_DEFAULTS_INVALID_EVENT_DATA"]


def test_apply_event_guard_multi_symbol_batch_processing(config: EventGuardConfig, base_now_ns: int) -> None:
    """Apply guard for multiple symbols and aggregate snapshot events per symbol."""

    earnings = _make_event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date_ns=base_now_ns + guard_module._NS_PER_DAY,
        risk_level=RiskLevel.HIGH,
    )
    tsla_earnings = _make_event(
        symbol="TSLA",
        event_type=EventType.EARNINGS,
        event_date_ns=base_now_ns + guard_module._NS_PER_DAY,
        risk_level=RiskLevel.HIGH,
    )
    snapshot = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=base_now_ns,
        events=[earnings, tsla_earnings],
        source="polygon",
        symbols_covered=["AAPL", "MSFT"],
    )

    constraints = guard_module.apply_event_guard(["aapl", "MSFT "], snapshot, config, current_time_ns=base_now_ns)
    assert set(constraints.keys()) == {"AAPL", "MSFT"}
    assert constraints["AAPL"].can_open_new is False
    assert constraints["AAPL"].no_trade_windows
    assert constraints["MSFT"].can_open_new is True
    assert constraints["MSFT"].no_trade_windows == []


def test_apply_event_guard_missing_symbol_covered_degrades_to_conservative_defaults(base_now_ns: int) -> None:
    """Degrade conservatively when a symbol was not covered by the event snapshot."""

    config = EventGuardConfig(primary_source="polygon", use_conservative_defaults=True)
    snapshot = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=base_now_ns,
        events=[],
        source="polygon",
        symbols_covered=["AAPL"],
    )

    constraints = guard_module.apply_event_guard(["AAPL", "MSFT"], snapshot, config, current_time_ns=base_now_ns)
    assert constraints["MSFT"].can_open_new is False
    assert constraints["MSFT"].reason_codes == ["CONSERVATIVE_DEFAULTS_MISSING_EVENT_DATA"]


def test_apply_event_guard_current_time_invalid_uses_time_time_ns(monkeypatch, base_now_ns: int) -> None:
    """Use time.time_ns when apply_event_guard receives invalid current_time_ns."""

    monkeypatch.setattr(guard_module.time, "time_ns", lambda: base_now_ns)
    config = EventGuardConfig(primary_source="polygon")
    snapshot = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=base_now_ns,
        events=[],
        source="polygon",
        symbols_covered=["AAPL"],
    )
    constraints = guard_module.apply_event_guard(["AAPL"], snapshot, config, current_time_ns=0)
    assert constraints["AAPL"].symbol == "AAPL"


def test_apply_event_guard_exception_path_degrades_when_enabled(monkeypatch, base_now_ns: int) -> None:
    """Degrade to conservative constraints when an exception occurs and enabled."""

    config = EventGuardConfig(primary_source="polygon", use_conservative_defaults=True)
    snapshot = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=base_now_ns,
        events=[],
        source="polygon",
        symbols_covered=["AAPL"],
    )

    monkeypatch.setattr(guard_module, "generate_risk_flags", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    constraints = guard_module.apply_event_guard(["AAPL"], snapshot, config, current_time_ns=base_now_ns)
    assert constraints["AAPL"].reason_codes == ["CONSERVATIVE_DEFAULTS_EXCEPTION"]


def test_apply_event_guard_exception_path_returns_soft_failure_when_conservative_disabled(
    monkeypatch, base_now_ns: int
) -> None:
    """Return a non-conservative exception marker when conservative defaults are disabled."""

    config = EventGuardConfig(primary_source="polygon", use_conservative_defaults=False)
    snapshot = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=base_now_ns,
        events=[],
        source="polygon",
        symbols_covered=["AAPL"],
    )

    monkeypatch.setattr(guard_module, "generate_risk_flags", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    constraints = guard_module.apply_event_guard(["AAPL"], snapshot, config, current_time_ns=base_now_ns)
    assert constraints["AAPL"].reason_codes == ["EVENT_GUARD_EXCEPTION"]


def test_guard_private_helpers_cover_branch_edges(config: EventGuardConfig, base_now_ns: int) -> None:
    """Cover helper edge cases (timestamp validation, blackout windows, window merge)."""

    assert guard_module._validate_timestamp_ns(None) is None
    assert guard_module._validate_timestamp_ns("x") is None  # type: ignore[arg-type]

    naive = datetime(2025, 1, 1, 0, 0, 0)
    assert guard_module._datetime_to_ns(naive) == _dt_to_ns(naive.replace(tzinfo=UTC))

    with pytest.raises(ValueError):
        guard_module._ns_to_datetime_utc(0)

    assert guard_module._most_severe_risk([]) is RiskLevel.NONE
    assert guard_module._nearest_high_risk_event([]) is None

    invalid = _make_event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date_ns=0,
        risk_level=RiskLevel.HIGH,
    )
    assert guard_module._event_blackout_window(invalid, config) is None

    lockup_event = _make_event(
        symbol="AAPL",
        event_type=EventType.LOCKUP_EXPIRATION,
        event_date_ns=base_now_ns + 10 * guard_module._NS_PER_DAY,
        risk_level=RiskLevel.MEDIUM,
    )
    lockup_window = guard_module._event_blackout_window(lockup_event, config)
    assert lockup_window is not None
    assert lockup_window[0] < lockup_window[1]

    offering_event = _make_event(
        symbol="AAPL",
        event_type=EventType.OFFERING,
        event_date_ns=base_now_ns + 11 * guard_module._NS_PER_DAY,
        risk_level=RiskLevel.MEDIUM,
    )
    offering_window = guard_module._event_blackout_window(offering_event, config)
    assert offering_window is not None
    assert offering_window[1] - offering_window[0] >= 3 * guard_module._NS_PER_DAY

    assert guard_module._merge_windows([]) == []
    assert guard_module._merge_windows([(1, 2), (4, 5)]) == [(1, 2), (4, 5)]


def test_generate_risk_flags_skips_other_symbol_and_counts_exceptions(base_now_ns: int) -> None:
    """Skip events for other symbols and treat malformed entries as invalid events."""

    config = EventGuardConfig(primary_source="polygon", use_conservative_defaults=True)
    other_symbol = _make_event(
        symbol="MSFT",
        event_type=EventType.EARNINGS,
        event_date_ns=base_now_ns + guard_module._NS_PER_DAY,
        risk_level=RiskLevel.HIGH,
    )
    flags = guard_module.generate_risk_flags("AAPL", [other_symbol, object()], config, current_time_ns=base_now_ns)  # type: ignore[list-item]
    assert flags.reason == "CONSERVATIVE_DEFAULTS_INVALID_EVENT_DATA"


def test_generate_trade_constraints_reason_codes_for_specific_event_types_and_negative_multiplier(
    config: EventGuardConfig, base_now_ns: int
) -> None:
    """Emit event-type reason codes and clamp negative multipliers to 0."""

    split = _make_event(
        symbol="AAPL",
        event_type=EventType.STOCK_SPLIT,
        event_date_ns=base_now_ns + guard_module._NS_PER_DAY,
        risk_level=RiskLevel.LOW,
    )
    lockup = _make_event(
        symbol="AAPL",
        event_type=EventType.LOCKUP_EXPIRATION,
        event_date_ns=base_now_ns + 2 * guard_module._NS_PER_DAY,
        risk_level=RiskLevel.LOW,
    )
    offering = _make_event(
        symbol="AAPL",
        event_type=EventType.OFFERING,
        event_date_ns=base_now_ns + 3 * guard_module._NS_PER_DAY,
        risk_level=RiskLevel.LOW,
    )
    flags = RiskFlags(
        symbol="AAPL",
        has_upcoming_event=True,
        events=[split, lockup, offering],
        max_position_multiplier=-1.0,
        reason="UPCOMING_STOCK_SPLIT_LOW",
    )
    constraints = guard_module.generate_trade_constraints(flags, config)

    assert constraints.max_position_size == 0.0
    assert "SPLIT_BLACKOUT" in constraints.reason_codes
    assert "LOCKUP_BLACKOUT" in constraints.reason_codes
    assert "OFFERING_BLACKOUT" in constraints.reason_codes
