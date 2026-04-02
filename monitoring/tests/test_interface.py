from __future__ import annotations

import msgspec
import msgspec.json

from monitoring.interface import Alert, AlertLevel, MetricSnapshot, SLOSnapshot, SLOTarget

if not hasattr(msgspec.json, "DecodeError"):
    setattr(msgspec.json, "DecodeError", msgspec.DecodeError)
if not hasattr(msgspec.json, "EncodeError"):
    setattr(msgspec.json, "EncodeError", msgspec.EncodeError)


def test_metric_snapshot_creation_and_serialization() -> None:
    snapshot = MetricSnapshot(
        run_id="run-1",
        module_name="scanner",
        timestamp=100,
        metrics={"count": 1.0},
        labels={"status": "ok"},
    )

    payload = msgspec.json.encode(snapshot)
    decoded = msgspec.json.decode(payload, type=MetricSnapshot)

    assert decoded == snapshot
    assert decoded.metrics["count"] == 1.0


def test_alert_creation_with_all_fields() -> None:
    alert = Alert(
        alert_id="alert-1",
        level=AlertLevel.CRITICAL,
        title="Title",
        message="Message",
        timestamp=123,
        run_id="run-1",
        module_name="scanner",
        metadata={"exc": "boom", "retry": 1},
    )

    payload = msgspec.json.encode(alert)
    decoded = msgspec.json.decode(payload, type=Alert)
    assert decoded == alert
    assert decoded.level is AlertLevel.CRITICAL


def test_alert_level_enum() -> None:
    assert AlertLevel.INFO.value == "INFO"
    assert AlertLevel.WARNING.value == "WARNING"
    assert AlertLevel.ERROR.value == "ERROR"
    assert AlertLevel.CRITICAL.value == "CRITICAL"


def test_slo_target_and_snapshot() -> None:
    target = SLOTarget(
        name="latency",
        description="Latency threshold",
        target_value=250.0,
        comparison="<=",
        window_seconds=60,
    )
    snapshot = SLOSnapshot(
        slo_name=target.name,
        current_value=200.0,
        target_value=target.target_value,
        is_met=True,
        window_start=1,
        window_end=2,
        sample_count=3,
    )

    encoded_target = msgspec.json.encode(target)
    decoded_target = msgspec.json.decode(encoded_target, type=SLOTarget)
    assert decoded_target == target

    encoded_snapshot = msgspec.json.encode(snapshot)
    decoded_snapshot = msgspec.json.decode(encoded_snapshot, type=SLOSnapshot)
    assert decoded_snapshot == snapshot


def test_msgspec_encoding_decoding_roundtrip() -> None:
    class Payload(msgspec.Struct, frozen=True):
        snapshot: MetricSnapshot
        alert: Alert

    payload = Payload(
        snapshot=MetricSnapshot(
            run_id="run-1",
            module_name="scanner",
            timestamp=100,
            metrics={"foo": 1.23},
            labels={"env": "test"},
        ),
        alert=Alert(
            alert_id="alert-1",
            level=AlertLevel.INFO,
            title="hello",
            message="world",
            timestamp=200,
            run_id=None,
            module_name=None,
            metadata={},
        ),
    )

    encoded = msgspec.json.encode(payload)
    decoded = msgspec.json.decode(encoded, type=Payload)
    assert decoded == payload
