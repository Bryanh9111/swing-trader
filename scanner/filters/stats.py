"""Filter statistics utilities for debugging and backtest summaries.

The scanner filter pipeline can be hard to tune without visibility into which
filters reject symbols and how key metrics distribute across the scanned
universe. This module provides a lightweight, in-memory collector that can be
shared across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:  # noqa: BLE001 - tolerate schema drift / mixed types.
        return None
    if out != out:  # NaN
        return None
    if out in (float("inf"), float("-inf")):
        return None
    return out


@dataclass(slots=True)
class FilterCounter:
    passed: int = 0
    failed: int = 0

    def record(self, passed: bool) -> None:
        if passed:
            self.passed += 1
        else:
            self.failed += 1


@dataclass(slots=True)
class MetricCounter:
    """Streaming metric summary with optional threshold pass/fail counts."""

    count: int = 0
    sum_value: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    passed: int = 0
    failed: int = 0
    last_threshold: float | None = None
    histogram_bins: int = 10
    histogram: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.histogram:
            self.histogram = [0 for _ in range(max(1, int(self.histogram_bins)))]

    def record(self, value: float, *, passed: bool | None = None, threshold: float | None = None) -> None:
        self.count += 1
        self.sum_value += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

        if threshold is not None:
            self.last_threshold = float(threshold)
        if passed is not None:
            if passed:
                self.passed += 1
            else:
                self.failed += 1

        # Histogram focuses on [0, 1] bounded metrics; out-of-range values get clamped.
        bins = max(1, int(self.histogram_bins))
        idx = int(max(0.0, min(0.999999, value)) * bins)
        idx = max(0, min(bins - 1, idx))
        if idx >= len(self.histogram):
            self.histogram.extend([0 for _ in range(idx - len(self.histogram) + 1)])
        self.histogram[idx] += 1

    @property
    def avg_value(self) -> float:
        if self.count <= 0:
            return 0.0
        return float(self.sum_value / self.count)


class FilterStats:
    """Collect aggregate stats for filter outcomes and key metric distributions."""

    def __init__(self) -> None:
        self.total_candidates_scanned: int = 0
        self.filters: dict[str, FilterCounter] = {}
        self.metrics: dict[str, MetricCounter] = {}

    def record_candidate(self) -> None:
        self.total_candidates_scanned += 1

    def record_filter(self, *, name: str, passed: bool) -> None:
        counter = self.filters.get(name)
        if counter is None:
            counter = FilterCounter()
            self.filters[name] = counter
        counter.record(passed)

    def record_metric(
        self,
        *,
        name: str,
        value: Any,
        passed: bool | None = None,
        threshold: float | None = None,
        histogram_bins: int = 10,
    ) -> None:
        value_f = _as_finite_float(value)
        if value_f is None:
            return
        counter = self.metrics.get(name)
        if counter is None:
            counter = MetricCounter(histogram_bins=int(histogram_bins))
            self.metrics[name] = counter
        counter.record(value_f, passed=passed, threshold=threshold)

    def merge(self, other: FilterStats) -> None:
        self.total_candidates_scanned += int(other.total_candidates_scanned)
        for name, counter in other.filters.items():
            target = self.filters.get(name)
            if target is None:
                self.filters[name] = FilterCounter(passed=counter.passed, failed=counter.failed)
            else:
                target.passed += counter.passed
                target.failed += counter.failed
        for name, counter in other.metrics.items():
            target = self.metrics.get(name)
            if target is None:
                target = MetricCounter(histogram_bins=int(counter.histogram_bins))
                self.metrics[name] = target
            target.count += counter.count
            target.sum_value += counter.sum_value
            target.min_value = min(target.min_value, counter.min_value)
            target.max_value = max(target.max_value, counter.max_value)
            target.passed += counter.passed
            target.failed += counter.failed
            if counter.last_threshold is not None:
                target.last_threshold = counter.last_threshold
            if len(target.histogram) < len(counter.histogram):
                target.histogram.extend([0 for _ in range(len(counter.histogram) - len(target.histogram))])
            for idx, value in enumerate(counter.histogram):
                target.histogram[idx] += int(value)

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append("=== Filter Statistics ===")
        lines.append(f"Total candidates scanned: {self.total_candidates_scanned}")

        # Key metrics first (stable ordering for backtest logs).
        for metric_name in ("box_quality", "volume_stability"):
            metric = self.metrics.get(metric_name)
            if metric is None or metric.count == 0:
                continue
            lines.append(
                f"{metric_name} filter: passed={metric.passed}, failed={metric.failed}, avg_value={metric.avg_value:.3f}"
            )

        # Then other per-filter pass/fail counts.
        for filter_name in sorted(self.filters):
            if filter_name == "box_quality":
                continue
            counter = self.filters[filter_name]
            lines.append(f"{filter_name}: passed={counter.passed}, failed={counter.failed}")

        return lines

    def summary_text(self) -> str:
        return "\n".join(self.summary_lines())


_GLOBAL_FILTER_STATS: FilterStats | None = None
_GLOBAL_FILTER_STATS_LOCK = Lock()


def get_global_filter_stats() -> FilterStats:
    global _GLOBAL_FILTER_STATS  # noqa: PLW0603 - module-level singleton.
    if _GLOBAL_FILTER_STATS is None:
        with _GLOBAL_FILTER_STATS_LOCK:
            if _GLOBAL_FILTER_STATS is None:
                _GLOBAL_FILTER_STATS = FilterStats()
    return _GLOBAL_FILTER_STATS


def reset_global_filter_stats() -> None:
    global _GLOBAL_FILTER_STATS  # noqa: PLW0603 - module-level singleton.
    with _GLOBAL_FILTER_STATS_LOCK:
        _GLOBAL_FILTER_STATS = FilterStats()
