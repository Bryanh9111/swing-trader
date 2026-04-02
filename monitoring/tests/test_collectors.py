from __future__ import annotations

import logging

import pytest

from monitoring.collectors import InMemoryAlertEmitter, InMemoryMetricsCollector, InMemorySLOTracker
from monitoring.interface import Alert, AlertLevel, SLOTarget


def test_in_memory_metrics_collector_record_and_get_metrics(
    monkeypatch: pytest.MonkeyPatch,
    logger_mock,
) -> None:
    times = iter([100, 200])
    monkeypatch.setattr("monitoring.collectors.time.time_ns", lambda: next(times))

    collector = InMemoryMetricsCollector(logger=logger_mock)
    collector.record_metric(
        run_id="run-1",
        module_name="scanner",
        metric_name="count",
        value=1.0,
        labels={"status": "ok"},
    )
    collector.record_metric(
        run_id="run-1",
        module_name="scanner",
        metric_name="latency_ms",
        value=10.5,
        labels={"status": "ok"},
    )

    snapshots = collector.get_metrics("run-1", "scanner")
    assert len(snapshots) == 2
    assert snapshots[0].timestamp == 100
    assert snapshots[0].metrics == {"count": 1.0}
    assert snapshots[1].timestamp == 200
    assert snapshots[1].metrics == {"latency_ms": 10.5}


def test_in_memory_metrics_collector_empty_for_unknown_run(logger_mock) -> None:
    collector = InMemoryMetricsCollector(logger=logger_mock)
    assert collector.get_metrics("missing-run", "scanner") == []


def test_in_memory_metrics_collector_non_structlog_logger_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([100])
    monkeypatch.setattr("monitoring.collectors.time.time_ns", lambda: next(times))

    logger = logging.getLogger("monitoring.tests.metrics")
    collector = InMemoryMetricsCollector(logger=logger)
    collector.record_metric(
        run_id="run-1",
        module_name="scanner",
        metric_name="count",
        value=1.0,
        labels={},
    )
    assert collector.get_metrics("run-1", "scanner")[0].timestamp == 100


def test_in_memory_alert_emitter_emit_alert_and_get_alerts(logger_mock) -> None:
    emitter = InMemoryAlertEmitter(logger=logger_mock)
    alert = Alert(
        alert_id="a1",
        level=AlertLevel.WARNING,
        title="warn",
        message="warn-msg",
        timestamp=1,
        run_id="run-1",
        module_name="scanner",
        metadata={},
    )
    emitter.emit_alert(alert)

    alerts = emitter.get_alerts("run-1")
    assert alerts == [alert]


def test_in_memory_alert_emitter_filter_by_run_id(logger_mock) -> None:
    emitter = InMemoryAlertEmitter(logger=logger_mock)
    emitter.emit_alert(
        Alert(
            alert_id="a1",
            level=AlertLevel.INFO,
            title="info",
            message="info-msg",
            timestamp=1,
            run_id="run-1",
            module_name="scanner",
            metadata={},
        )
    )
    emitter.emit_alert(
        Alert(
            alert_id="a2",
            level=AlertLevel.ERROR,
            title="error",
            message="error-msg",
            timestamp=2,
            run_id="run-2",
            module_name="scanner",
            metadata={},
        )
    )

    assert [alert.alert_id for alert in emitter.get_alerts("run-1")] == ["a1"]
    assert [alert.alert_id for alert in emitter.get_alerts("run-2")] == ["a2"]


def test_in_memory_alert_emitter_different_levels_non_structlog_branch(caplog) -> None:
    logger = logging.getLogger("monitoring.tests.alerts")
    emitter = InMemoryAlertEmitter(logger=logger)

    with caplog.at_level(logging.DEBUG, logger="monitoring.tests.alerts"):
        emitter.emit_alert(
            Alert(
                alert_id="a1",
                level=AlertLevel.INFO,
                title="info",
                message="info-msg",
                timestamp=1,
                run_id="run-1",
                module_name="scanner",
                metadata={},
            )
        )
        emitter.emit_alert(
            Alert(
                alert_id="a2",
                level=AlertLevel.CRITICAL,
                title="crit",
                message="crit-msg",
                timestamp=2,
                run_id="run-1",
                module_name="scanner",
                metadata={},
            )
        )

    assert any("Alert emitted" in record.message for record in caplog.records)


def test_in_memory_slo_tracker_track_and_check_slo_met(
    monkeypatch: pytest.MonkeyPatch,
    sample_slo_target: SLOTarget,
    logger_mock,
) -> None:
    times = iter(
        [
            500_000_000,  # old sample
            1_000_000_000,  # sample 1
            1_500_000_000,  # sample 2
            2_000_000_000,  # check window_end
        ]
    )
    monkeypatch.setattr("monitoring.collectors.time.time_ns", lambda: next(times))

    tracker = InMemorySLOTracker([sample_slo_target], logger=logger_mock)
    tracker.track_slo(sample_slo_target.name, 0.1)
    tracker.track_slo(sample_slo_target.name, 1.0)
    tracker.track_slo(sample_slo_target.name, 1.0)

    snapshot = tracker.check_slo(sample_slo_target.name)
    assert snapshot.sample_count == 2
    assert snapshot.current_value == pytest.approx(1.0)
    assert snapshot.is_met is True
    assert snapshot.window_start == 1_000_000_000
    assert snapshot.window_end == 2_000_000_000


def test_in_memory_slo_tracker_not_met_and_le_comparison(
    monkeypatch: pytest.MonkeyPatch,
    logger_mock,
) -> None:
    target = SLOTarget(
        name="latency",
        description="Latency over window",
        target_value=0.5,
        comparison="<=",
        window_seconds=1,
    )
    times = iter([1_000_000_000, 1_500_000_000, 2_000_000_000])
    monkeypatch.setattr("monitoring.collectors.time.time_ns", lambda: next(times))

    tracker = InMemorySLOTracker([target], logger=logger_mock)
    tracker.track_slo(target.name, 1.0)
    tracker.track_slo(target.name, 0.4)

    snapshot = tracker.check_slo(target.name)
    assert snapshot.sample_count == 2
    assert snapshot.current_value == pytest.approx(0.7)
    assert snapshot.is_met is False


def test_in_memory_slo_tracker_unknown_target_raises(logger_mock) -> None:
    tracker = InMemorySLOTracker(logger=logger_mock)
    with pytest.raises(KeyError):
        tracker.track_slo("missing", 1.0)
    with pytest.raises(KeyError):
        tracker.check_slo("missing")


def test_in_memory_slo_tracker_non_structlog_logger_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = SLOTarget(
        name="ratio",
        description="ratio",
        target_value=0.5,
        comparison=">=",
        window_seconds=1,
    )
    times = iter([1_000_000_000, 2_000_000_000])
    monkeypatch.setattr("monitoring.collectors.time.time_ns", lambda: next(times))

    logger = logging.getLogger("monitoring.tests.slo")
    tracker = InMemorySLOTracker([target], logger=logger)
    tracker.track_slo(target.name, 0.6)
    snapshot = tracker.check_slo(target.name)
    assert snapshot.is_met is True

