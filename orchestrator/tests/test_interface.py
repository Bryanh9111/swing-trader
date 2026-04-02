from __future__ import annotations

from dataclasses import FrozenInstanceError

import structlog

import pytest

from common.events import InMemoryEventBus
from common.interface import ResultStatus
from journal.interface import OperatingMode, RunType
from orchestrator.interface import ModuleResult, OrchestratorContext, RunSummary


def test_run_summary_fields() -> None:
    summary = RunSummary(
        candidates_count=0,
        failures=0,
        status=ResultStatus.SUCCESS,
        duration_ms=125,
    )

    assert summary.candidates_count == 0
    assert summary.failures == 0
    assert summary.status is ResultStatus.SUCCESS
    assert summary.duration_ms == 125


def test_module_result_defaults() -> None:
    result = ModuleResult(module_name="test-module", status=ResultStatus.SUCCESS)

    assert result.output_data is None
    assert result.error is None


def test_orchestrator_context_is_frozen() -> None:
    logger = structlog.get_logger("tests.orchestrator").bind()
    event_bus = InMemoryEventBus()

    context = OrchestratorContext(
        run_id="run-001",
        run_type=RunType.PRE_MARKET_FULL_SCAN,
        mode=OperatingMode.DRY_RUN,
        config={"mode": "DRY_RUN"},
        logger=logger,
        event_bus=event_bus,
    )

    with pytest.raises(FrozenInstanceError):
        context.run_id = "other-run"  # type: ignore[misc]

