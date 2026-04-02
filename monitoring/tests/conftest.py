from __future__ import annotations

from unittest.mock import Mock

import pytest

from monitoring.interface import Alert, AlertLevel, MetricSnapshot, SLOTarget


@pytest.fixture
def run_id() -> str:
    return "run-123"


@pytest.fixture
def module_name() -> str:
    return "scanner"


@pytest.fixture
def logger_mock() -> Mock:
    logger = Mock(name="logger")
    logger.bind.return_value = logger
    return logger


@pytest.fixture
def sample_metric_snapshot(run_id: str, module_name: str) -> MetricSnapshot:
    return MetricSnapshot(
        run_id=run_id,
        module_name=module_name,
        timestamp=123,
        metrics={"latency_ms": 10.0},
        labels={"status": "ok"},
    )


@pytest.fixture
def sample_alert(run_id: str, module_name: str) -> Alert:
    return Alert(
        alert_id="alert-1",
        level=AlertLevel.WARNING,
        title="Test alert",
        message="Something happened",
        timestamp=456,
        run_id=run_id,
        module_name=module_name,
        metadata={"threshold": 0.9},
    )


@pytest.fixture
def sample_slo_target() -> SLOTarget:
    return SLOTarget(
        name="success_rate",
        description="Success rate over the last window",
        target_value=0.9,
        comparison=">=",
        window_seconds=1,
    )

