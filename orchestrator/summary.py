"""Run summary generation utilities for orchestrator executions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from common.interface import ResultStatus
from journal.interface import RunMetadata

from .interface import ModuleResult, RunSummary

__all__ = ["generate_summary"]


def generate_summary(metadata: RunMetadata, module_results: Sequence[ModuleResult]) -> RunSummary:
    """Aggregate module outcomes into a ``RunSummary``."""

    failures = sum(1 for result in module_results if result.status is ResultStatus.FAILED)
    has_degraded_module = any(result.status is ResultStatus.DEGRADED for result in module_results)

    candidates_count = _extract_candidates(module_results)
    execution_reports_count = _extract_execution_reports(module_results)
    duration_ms = _calculate_duration(metadata)

    aggregate_status = _resolve_status(metadata.status, failures > 0, has_degraded_module)

    return RunSummary(
        candidates_count=candidates_count,
        execution_reports_count=execution_reports_count,
        failures=failures,
        status=aggregate_status,
        duration_ms=duration_ms,
    )


def _extract_candidates(module_results: Sequence[ModuleResult]) -> int:
    """Return the candidate count inferred from scanner module outputs."""

    for result in module_results:
        if "scanner" not in result.module_name.lower():
            continue
        if isinstance(result.output_data, list):
            return len(result.output_data)
        if isinstance(result.output_data, dict):
            candidates = result.output_data.get("candidates")
            if isinstance(candidates, list):
                return len(candidates)
            total_detected = result.output_data.get("total_detected")
            if isinstance(total_detected, int):
                return total_detected
    return 0


def _extract_execution_reports(module_results: Sequence[ModuleResult]) -> int:
    """Return execution report count inferred from execution module outputs."""

    for result in module_results:
        if "execution" not in result.module_name.lower():
            continue
        payload = result.output_data
        if not isinstance(payload, Mapping):
            continue
        reports = payload.get("reports")
        if isinstance(reports, list):
            return len(reports)
    return 0


def _calculate_duration(metadata: RunMetadata) -> int:
    """Return the runtime duration in milliseconds."""

    if metadata.end_time is None:
        return 0

    elapsed_ns = metadata.end_time - metadata.start_time
    if elapsed_ns <= 0:
        return 0

    return int(elapsed_ns // 1_000_000)


def _resolve_status(metadata_status: str, has_failures: bool, has_degraded_module: bool) -> ResultStatus:
    """Determine the aggregate status respecting metadata annotations."""

    normalised_status = metadata_status.strip().lower()
    if normalised_status == "failed":
        return ResultStatus.FAILED
    if normalised_status == "degraded":
        return ResultStatus.DEGRADED

    if has_failures:
        return ResultStatus.FAILED
    if has_degraded_module:
        return ResultStatus.DEGRADED
    return ResultStatus.SUCCESS
