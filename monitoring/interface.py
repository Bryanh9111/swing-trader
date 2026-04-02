"""Interface definitions for Phase 2.4 monitoring and alerting.

This module defines the canonical schema contracts and protocol boundaries for
basic monitoring within Automated Swing Trader (AST). The intent is to provide
typed, deterministic payloads (via ``msgspec.Struct``) that can be collected,
persisted, and queried by downstream monitoring implementations without
coupling callers to a specific backend.

The interfaces mirror established patterns in the codebase:

- ``common.interface.Result[T]`` for structured outcomes (used by concrete
  monitoring implementations elsewhere).
- ``journal.interface.SnapshotBase`` style for immutable, versionable payloads.
- ``orchestrator.interface.RunSummary`` for run-level aggregation conventions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import msgspec

__all__ = [
    "MetricSnapshot",
    "AlertLevel",
    "Alert",
    "SLOTarget",
    "SLOSnapshot",
    "MetricsCollector",
    "AlertEmitter",
    "SLOTracker",
]


class MetricSnapshot(msgspec.Struct, kw_only=True, frozen=True):
    """Point-in-time snapshot of collected metrics for a module invocation.

    Attributes:
        run_id: Correlation identifier for the orchestrator execution.
        module_name: Module emitting the metrics (e.g., ``"scanner"``).
        timestamp: Nanosecond Unix epoch timestamp for when the snapshot was
            produced.
        metrics: Key-value map of numeric metrics. Metric names should remain
            stable across releases to support longitudinal analysis.
        labels: Context labels attached to the snapshot (e.g., ``status``,
            ``run_type``, ``mode``).
    """

    run_id: str
    module_name: str
    timestamp: int
    metrics: dict[str, float]
    labels: dict[str, str]


class AlertLevel(str, Enum):
    """Enumerate the supported alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Alert(msgspec.Struct, kw_only=True, frozen=True):
    """Structured alert emitted by monitoring components.

    Alerts are designed to be routed to logs, persisted, or sent to external
    destinations (email/Slack/etc.) by an implementation of :class:`AlertEmitter`.

    Attributes:
        alert_id: Unique identifier for the alert instance.
        level: Severity of the alert.
        title: Short, human-friendly summary suitable for dashboards.
        message: Full alert body including remediation context.
        timestamp: Unix epoch timestamp (nanoseconds recommended) describing
            when the alert was emitted.
        run_id: Optional run correlation identifier if the alert is tied to an
            orchestrator execution.
        module_name: Optional module name if the alert is tied to a specific
            subsystem.
        metadata: Arbitrary implementation-defined fields (e.g., exception
            type, thresholds, payload identifiers).
    """

    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: int
    run_id: str | None
    module_name: str | None
    metadata: dict[str, Any]


class SLOTarget(msgspec.Struct, kw_only=True, frozen=True):
    """Service Level Objective (SLO) target definition.

    Attributes:
        name: Stable SLO identifier (e.g., ``"scanner_success_rate"``).
        description: Human readable description of what the SLO represents.
        target_value: Numeric target threshold such as ``0.95`` for 95%.
        comparison: Comparator describing how to evaluate the SLO threshold.
        window_seconds: Time window (in seconds) used when aggregating samples.
    """

    name: str
    description: str
    target_value: float
    comparison: Literal[">=", "<="]
    window_seconds: int


class SLOSnapshot(msgspec.Struct, kw_only=True, frozen=True):
    """Computed SLO state over the evaluation window.

    Attributes:
        slo_name: SLO identifier being evaluated.
        current_value: Aggregated value computed over the window.
        target_value: Target value for the SLO at evaluation time.
        is_met: ``True`` when ``current_value`` satisfies the SLO comparator.
        window_start: Inclusive window start timestamp for the aggregation.
        window_end: Inclusive window end timestamp for the aggregation.
        sample_count: Count of samples used in the evaluation window.
    """

    slo_name: str
    current_value: float
    target_value: float
    is_met: bool
    window_start: int
    window_end: int
    sample_count: int


@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol contract for recording and querying metrics snapshots."""

    def record_metric(
        self,
        run_id: str,
        module_name: str,
        metric_name: str,
        value: float,
        labels: Mapping[str, str],
    ) -> None:
        """Record a single metric sample for the specified run/module."""

    def get_metrics(self, run_id: str, module_name: str) -> list[MetricSnapshot]:
        """Return recorded metric snapshots for ``run_id`` and ``module_name``."""


@runtime_checkable
class AlertEmitter(Protocol):
    """Protocol contract for emitting and querying alerts."""

    def emit_alert(self, alert: Alert) -> None:
        """Emit ``alert`` to the configured alert destination(s)."""

    def get_alerts(self, run_id: str) -> list[Alert]:
        """Return emitted alerts correlated to ``run_id``."""


@runtime_checkable
class SLOTracker(Protocol):
    """Protocol contract for tracking SLO samples and computing compliance."""

    def track_slo(self, slo_name: str, value: float) -> None:
        """Record a sample ``value`` for the named SLO."""

    def check_slo(self, slo_name: str) -> SLOSnapshot:
        """Compute and return a snapshot for the named SLO."""

