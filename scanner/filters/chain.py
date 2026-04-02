"""FilterChain manager for composing and executing scanner filters.

Phase 1 architecture reference:
`../References-DoNotLinkToAnyProjectsHere/Scanner Reference/platform-stocks-selection/docs/refactoring_plan.md`

The chain aggregates per-filter :class:`~scanner.filters.base.FilterResult`
instances into a single :class:`ChainResult`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TYPE_CHECKING

import msgspec

from common.interface import Result, ResultStatus
from scanner.interface import ScannerConfig

from .base import FilterProtocol, FilterResult
from .stats import FilterStats, get_global_filter_stats

if TYPE_CHECKING:
    from data.interface import PriceBar
else:  # pragma: no cover - type-only import may not exist in early phases.
    PriceBar = Any  # type: ignore[assignment]

__all__ = [
    "ChainResult",
    "FilterChain",
]


class ChainResult(msgspec.Struct, frozen=True, kw_only=True):
    """Aggregated outcome for a filter chain execution."""

    passed: bool
    filter_results: dict[str, FilterResult] = msgspec.field(default_factory=dict)
    combined_score: float = 0.0
    reasons: list[str] = msgspec.field(default_factory=list)


class FilterChain:
    """Manage and execute a sequence of filters with AND/OR semantics."""

    _INVALID_LOGIC_REASON = "CHAIN_INVALID_LOGIC"
    _DUPLICATE_FILTER_REASON = "CHAIN_DUPLICATE_FILTER_NAME"
    _FILTER_FAILED_REASON = "CHAIN_FILTER_FAILED"

    def __init__(
        self,
        *,
        filters: Sequence[FilterProtocol] | None = None,
        logic: Literal["AND", "OR"] = "AND",
        stats: FilterStats | None = None,
    ) -> None:
        self.filters: list[FilterProtocol] = list(filters or [])
        self.logic: Literal["AND", "OR"] = logic
        self._stats: FilterStats = stats or get_global_filter_stats()

    def get_stats(self) -> FilterStats:
        return self._stats

    def add_filter(self, filter: FilterProtocol) -> None:
        """Append a filter to the chain, rejecting duplicate filter names."""

        name = getattr(filter, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Filter must define a non-empty 'name' class attribute.")

        if any(existing.name == name for existing in self.filters):
            raise ValueError(f"Duplicate filter name: {name!r}")

        self.filters.append(filter)

    def get_enabled_filters(self) -> list[FilterProtocol]:
        """Return the enabled filters in order."""

        return [filter for filter in self.filters if getattr(filter, "enabled", True)]

    def execute(
        self,
        bars: Sequence[PriceBar],
        config: ScannerConfig,
        *,
        context: dict[str, Any] | None = None,
    ) -> Result[ChainResult]:
        """Execute enabled filters and return a combined ChainResult."""

        if self.logic not in ("AND", "OR"):
            return Result.failed(
                ValueError(f"Invalid chain logic: {self.logic!r}"),
                self._INVALID_LOGIC_REASON,
            )

        self._stats.record_candidate()

        filter_results: dict[str, FilterResult] = {}
        reasons: list[str] = []
        scores: list[float] = []
        enabled_seen = 0
        chain_passed = True if self.logic == "AND" else False

        for filter in self.filters:
            name = filter.name
            if name in filter_results:
                return Result.failed(
                    ValueError(f"Duplicate filter name during execution: {name!r}"),
                    self._DUPLICATE_FILTER_REASON,
                )

            if not getattr(filter, "enabled", True):
                filter_result = FilterResult(
                    passed=True,
                    reason="FILTER_DISABLED",
                    score=1.0,
                    metadata={"enabled": False},
                )
                filter_results[name] = filter_result
                reasons.append(f"{name}: {filter_result.reason}")
                self._stats.record_filter(name=name, passed=True)
                continue

            enabled_seen += 1
            result = filter.apply(bars, config, context=context)
            if result.status is ResultStatus.FAILED:
                self._stats.record_filter(name=name, passed=False)
                partial = ChainResult(
                    passed=False,
                    filter_results=dict(filter_results),
                    combined_score=_combine_scores(scores),
                    reasons=list(reasons),
                )
                return Result.degraded(
                    partial,
                    result.error or ValueError("Filter failed"),
                    self._FILTER_FAILED_REASON,
                )

            filter_result = result.data
            if filter_result is None:
                self._stats.record_filter(name=name, passed=False)
                partial = ChainResult(
                    passed=False,
                    filter_results=dict(filter_results),
                    combined_score=_combine_scores(scores),
                    reasons=list(reasons),
                )
                return Result.degraded(
                    partial,
                    ValueError("Filter returned no data"),
                    self._FILTER_FAILED_REASON,
                )

            filter_result = FilterResult(
                passed=filter_result.passed,
                reason=filter_result.reason,
                score=filter_result.score,
                features=dict(filter_result.features),
                metadata={
                    **dict(filter_result.metadata),
                    "result_status": result.status.name.lower(),
                    "reason_code": result.reason_code,
                },
            )

            filter_results[name] = filter_result
            reasons.append(f"{name}: {filter_result.reason}")
            scores.append(_clamp_score(filter_result.score))
            self._stats.record_filter(name=name, passed=bool(filter_result.passed))
            _record_key_metrics(self._stats, filter_name=name, filter_result=filter_result, config=config)

            if self.logic == "AND":
                if not filter_result.passed:
                    chain_passed = False
            else:  # OR
                if filter_result.passed:
                    chain_passed = True

        if enabled_seen == 0:
            return Result.success(
                ChainResult(
                    passed=True,
                    filter_results=filter_results,
                    combined_score=1.0,
                    reasons=reasons or ["NO_FILTERS_ENABLED"],
                ),
                reason_code="NO_FILTERS_ENABLED",
            )

        return Result.success(
            ChainResult(
                passed=chain_passed,
                filter_results=filter_results,
                combined_score=_combine_scores(scores),
                reasons=reasons,
            )
        )


def _record_key_metrics(
    stats: FilterStats,
    *,
    filter_name: str,
    filter_result: FilterResult,
    config: ScannerConfig,
) -> None:
    if filter_name == "box_quality":
        stats.record_metric(
            name="box_quality",
            value=filter_result.features.get("box_quality"),
            passed=bool(filter_result.passed),
            threshold=float(getattr(config, "min_box_quality", 0.0) or 0.0),
        )
        return

    if filter_name == "volume_platform":
        volume_stability = filter_result.features.get("volume_stability")
        threshold = float(getattr(config, "volume_stability_threshold", 0.0) or 0.0)
        try:
            passed = float(volume_stability) >= threshold
        except Exception:  # noqa: BLE001
            passed = False
        stats.record_metric(
            name="volume_stability",
            value=volume_stability,
            passed=passed,
            threshold=threshold,
        )


def _clamp_score(score: float) -> float:
    try:
        value = float(score)
    except Exception:  # noqa: BLE001 - tolerate mis-typed scores.
        return 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def _combine_scores(scores: Sequence[float]) -> float:
    if not scores:
        return 1.0
    return float(sum(scores) / len(scores))
