from __future__ import annotations

from common.interface import ResultStatus
from journal.interface import OperatingMode, RunMetadata, RunType
from orchestrator.interface import ModuleResult
from orchestrator.summary import generate_summary


def _metadata(status: str, start: int = 0, end: int | None = None) -> RunMetadata:
    return RunMetadata(
        run_id="run-123",
        run_type=RunType.PRE_MARKET_FULL_SCAN,
        mode=OperatingMode.DRY_RUN,
        system_version="test",
        start_time=start,
        end_time=end,
        status=status,
    )


def test_generate_summary_success() -> None:
    metadata = _metadata(status="completed", start=0, end=2_000_000)
    module_results = [
        ModuleResult(module_name="universe", status=ResultStatus.SUCCESS, output_data={"symbols": []}),
        ModuleResult(module_name="data", status=ResultStatus.SUCCESS, output_data={"bars": {}}),
        ModuleResult(module_name="scanner", status=ResultStatus.SUCCESS, output_data=[]),
    ]

    summary = generate_summary(metadata, module_results)

    assert summary.status is ResultStatus.SUCCESS
    assert summary.failures == 0
    assert summary.candidates_count == 0
    assert summary.duration_ms == 2


def test_generate_summary_failure_metadata_status_drives_result() -> None:
    metadata = _metadata(status="failed", start=0, end=5_000_000)
    module_results = [
        ModuleResult(module_name="universe", status=ResultStatus.SUCCESS),
        ModuleResult(module_name="data", status=ResultStatus.SUCCESS),
        ModuleResult(module_name="scanner", status=ResultStatus.SUCCESS, output_data=[{"symbol": "ABC"}]),
    ]

    summary = generate_summary(metadata, module_results)

    assert summary.status is ResultStatus.FAILED
    assert summary.candidates_count == 1


def test_generate_summary_degraded_from_module() -> None:
    metadata = _metadata(status="completed", start=0, end=1_500_000)
    module_results = [
        ModuleResult(module_name="universe", status=ResultStatus.DEGRADED),
        ModuleResult(module_name="data", status=ResultStatus.SUCCESS),
        ModuleResult(module_name="scanner", status=ResultStatus.SUCCESS, output_data=[]),
    ]

    summary = generate_summary(metadata, module_results)

    assert summary.status is ResultStatus.DEGRADED
    assert summary.duration_ms == 1

