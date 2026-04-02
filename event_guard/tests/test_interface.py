from __future__ import annotations

from typing import Any

import msgspec
import msgspec.json
import pytest

from event_guard.interface import (
    EventGuardConfig,
    EventSnapshot,
    EventType,
    MarketEvent,
    RiskFlags,
    RiskLevel,
    TradeConstraints,
)
from journal.interface import SnapshotBase


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (EventType.EARNINGS, "EARNINGS"),
        (EventType.DIVIDEND, "DIVIDEND"),
        (EventType.STOCK_SPLIT, "STOCK_SPLIT"),
        (EventType.REVERSE_SPLIT, "REVERSE_SPLIT"),
        (EventType.OFFERING, "OFFERING"),
        (EventType.LOCKUP_EXPIRATION, "LOCKUP_EXPIRATION"),
        (EventType.OTHER, "OTHER"),
    ],
)
def test_event_type_enum_values(member: EventType, value: str) -> None:
    """Ensure EventType enum values are stable and complete."""

    assert member.value == value
    assert EventType(value) is member


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (RiskLevel.NONE, "NONE"),
        (RiskLevel.LOW, "LOW"),
        (RiskLevel.MEDIUM, "MEDIUM"),
        (RiskLevel.HIGH, "HIGH"),
        (RiskLevel.CRITICAL, "CRITICAL"),
    ],
)
def test_risk_level_enum_values(member: RiskLevel, value: str) -> None:
    """Ensure RiskLevel enum values are stable and complete."""

    assert member.value == value
    assert RiskLevel(value) is member


def test_market_event_schema_required_fields_validation() -> None:
    """Validate required fields are enforced for MarketEvent via msgspec.convert."""

    payload: dict[str, Any] = {
        "symbol": "AAPL",
        "event_type": "EARNINGS",
        "event_date": 1_700_000_000_000_000_000,
        "risk_level": "HIGH",
        "source": "polygon",
    }
    event = msgspec.convert(payload, type=MarketEvent)
    assert event.symbol == "AAPL"

    for missing in ("symbol", "event_type", "event_date", "risk_level", "source"):
        bad = dict(payload)
        bad.pop(missing)
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert(bad, type=MarketEvent)


def test_market_event_schema_optional_field_defaults() -> None:
    """Ensure optional fields default correctly when omitted."""

    event = MarketEvent(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date=1_700_000_000_000_000_000,
        risk_level=RiskLevel.HIGH,
        source="polygon",
    )
    assert event.metadata is None


def test_market_event_msgspec_roundtrip_encode_decode() -> None:
    """Ensure MarketEvent serializes/deserializes via msgspec JSON roundtrip."""

    event = MarketEvent(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date=1_700_000_000_000_000_000,
        risk_level=RiskLevel.HIGH,
        source="polygon",
        metadata={"provider_event_id": "123"},
    )
    encoded = msgspec.json.encode(event)
    decoded = msgspec.json.decode(encoded, type=MarketEvent)
    assert decoded == event


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"symbol": "AAPL", "event_type": "INVALID", "event_date": 1, "risk_level": "HIGH", "source": "x"}, "event_type"),
        ({"symbol": "AAPL", "event_type": "EARNINGS", "event_date": 1, "risk_level": "INVALID", "source": "x"}, "risk_level"),
        ({"symbol": "AAPL", "event_type": "EARNINGS", "event_date": "not-a-ts", "risk_level": "HIGH", "source": "x"}, "event_date"),
    ],
)
def test_market_event_schema_rejects_invalid_enum_or_types(payload: dict[str, Any], field: str) -> None:
    """Reject invalid event_type/risk_level values and invalid timestamp types."""

    with pytest.raises(msgspec.ValidationError) as excinfo:
        msgspec.convert(payload, type=MarketEvent)
    assert field in str(excinfo.value)


def test_risk_flags_schema_defaults_and_full_instance() -> None:
    """Validate RiskFlags defaults and a full instance construction."""

    minimal = RiskFlags(symbol="AAPL")
    assert minimal.has_upcoming_event is False
    assert minimal.events == []
    assert minimal.no_trade_until is None
    assert minimal.max_position_multiplier == pytest.approx(1.0)
    assert minimal.reason is None

    full = RiskFlags(
        symbol="AAPL",
        has_upcoming_event=True,
        events=[
            MarketEvent(
                symbol="AAPL",
                event_type=EventType.EARNINGS,
                event_date=1_700_000_000_000_000_000,
                risk_level=RiskLevel.HIGH,
                source="polygon",
            )
        ],
        no_trade_until=1_700_000_000_000_000_000,
        max_position_multiplier=0.5,
        reason="UPCOMING_EARNINGS_HIGH",
    )
    encoded = msgspec.json.encode(full)
    decoded = msgspec.json.decode(encoded, type=RiskFlags)
    assert decoded == full


def test_trade_constraints_schema_defaults_and_full_instance() -> None:
    """Validate TradeConstraints defaults and a full instance construction."""

    minimal = TradeConstraints(symbol="AAPL")
    assert minimal.can_open_new is True
    assert minimal.can_increase is True
    assert minimal.can_decrease is True
    assert minimal.max_position_size is None
    assert minimal.no_trade_windows == []
    assert minimal.reason_codes == []

    full = TradeConstraints(
        symbol="AAPL",
        can_open_new=False,
        can_increase=False,
        can_decrease=True,
        max_position_size=0.5,
        no_trade_windows=[(1, 2), (10, 20)],
        reason_codes=["EARNINGS_BLACKOUT"],
    )
    encoded = msgspec.json.encode(full)
    decoded = msgspec.json.decode(encoded, type=TradeConstraints)
    assert decoded == full


def test_event_guard_config_schema_defaults_and_custom_values_roundtrip() -> None:
    """Validate EventGuardConfig defaults and custom values with msgspec JSON roundtrip."""

    config = EventGuardConfig(primary_source="polygon")
    assert config.primary_source == "polygon"
    assert config.fallback_source is None
    assert config.manual_events_file is None
    assert config.earnings_blackout_days_before == 2
    assert config.earnings_blackout_days_after == 1
    assert config.split_blackout_days == 5
    assert config.lockup_blackout_days == 3
    assert config.use_conservative_defaults is True
    assert config.sources == {}

    custom = EventGuardConfig(
        primary_source="polygon",
        fallback_source="manual",
        manual_events_file="/tmp/events.csv",
        earnings_blackout_days_before=0,
        earnings_blackout_days_after=0,
        split_blackout_days=1,
        lockup_blackout_days=2,
        use_conservative_defaults=False,
    )
    encoded = msgspec.json.encode(custom)
    decoded = msgspec.json.decode(encoded, type=EventGuardConfig)
    assert decoded == custom


def test_event_snapshot_schema_inherits_snapshot_base_and_validates() -> None:
    """Validate EventSnapshot embeds SnapshotBase envelope and msgspec conversion."""

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "system_version": "deadbeef",
        "asof_timestamp": 1_700_000_000_000_000_000,
        "events": [
            {
                "symbol": "aapl",
                "event_type": "EARNINGS",
                "event_date": 1_700_000_000_000_000_000,
                "risk_level": "HIGH",
                "source": "polygon",
                "metadata": {"provider_event_id": "123"},
            }
        ],
        "source": "polygon",
        "symbols_covered": ["AAPL", "MSFT"],
    }
    snapshot = msgspec.convert(payload, type=EventSnapshot)
    assert isinstance(snapshot, SnapshotBase)
    assert snapshot.source == "polygon"
    assert snapshot.symbols_covered == ["AAPL", "MSFT"]
    assert snapshot.events[0].symbol == "aapl"

    minimal = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        source="polygon",
    )
    assert minimal.events == []
    assert minimal.symbols_covered == []


def test_event_guard_structs_are_frozen() -> None:
    """Ensure msgspec frozen structs reject field mutation."""

    event = MarketEvent(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date=1,
        risk_level=RiskLevel.HIGH,
        source="polygon",
    )
    with pytest.raises(AttributeError):
        event.symbol = "MSFT"  # type: ignore[misc]
