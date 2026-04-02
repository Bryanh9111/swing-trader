"""In-memory monitoring implementations for Phase 2.4.

This module provides simple in-memory implementations of the monitoring
protocols declared in :mod:`monitoring.interface`.

The implementations are intentionally lightweight:

- They require no external services or persistence.
- They are suitable for unit testing, local development, and reference usage.
- They are not designed for concurrency-heavy workloads (thread safety is not
  required for Phase 2.4).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, DefaultDict, Iterable, Mapping

from common.interface import BoundLogger

from .interface import (
    Alert,
    AlertEmitter,
    AlertLevel,
    MetricSnapshot,
    MetricsCollector,
    SLOSnapshot,
    SLOTarget,
    SLOTracker,
)

__all__ = ["InMemoryMetricsCollector", "InMemoryAlertEmitter", "InMemorySLOTracker"]


class InMemoryMetricsCollector(MetricsCollector):
    """In-memory metrics collector.

    Metrics are stored as point-in-time :class:`~monitoring.interface.MetricSnapshot`
    objects keyed by ``(run_id, module_name)``.

    Notes
    -----
    Each call to :meth:`record_metric` produces a new snapshot containing only
    the recorded metric. This keeps the implementation deterministic and
    avoids guessing aggregation semantics at collection time.
    """

    def __init__(self, *, logger: logging.Logger | BoundLogger | None = None) -> None:
        """Create a new in-memory metrics collector.

        Parameters
        ----------
        logger:
            Optional logger used to record debug/diagnostic output. Accepts a
            stdlib ``logging.Logger`` or a structlog ``BoundLogger``.
        """

        self._logger = logger or logging.getLogger(__name__)
        self._metrics: DefaultDict[tuple[str, str], list[MetricSnapshot]] = defaultdict(list)

    def record_metric(
        self,
        run_id: str,
        module_name: str,
        metric_name: str,
        value: float,
        labels: Mapping[str, str],
    ) -> None:
        """Record a metric sample as a new snapshot."""

        snapshot = MetricSnapshot(
            run_id=run_id,
            module_name=module_name,
            timestamp=time.time_ns(),
            metrics={metric_name: float(value)},
            labels=dict(labels),
        )
        self._metrics[(run_id, module_name)].append(snapshot)
        self._log_debug(
            "Recorded metric",
            run_id=run_id,
            module_name=module_name,
            metric_name=metric_name,
            value=value,
        )

    def get_metrics(self, run_id: str, module_name: str) -> list[MetricSnapshot]:
        """Return snapshots recorded for ``run_id`` and ``module_name``."""

        return list(self._metrics.get((run_id, module_name), []))

    def _log_debug(self, message: str, **kwargs: Any) -> None:
        if hasattr(self._logger, "bind"):
            self._logger.debug(message, **kwargs)
            return

        if kwargs:
            formatted = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
            message = f"{message} | {formatted}"
        self._logger.debug(message)


class InMemoryAlertEmitter(AlertEmitter):
    """In-memory alert emitter.

    Alerts are appended to an in-memory list. Each emitted alert is also logged
    via the provided logger to make alerts visible during development and
    tests.
    """

    def __init__(self, *, logger: logging.Logger | BoundLogger | None = None) -> None:
        """Create a new in-memory alert emitter.

        Parameters
        ----------
        logger:
            Optional logger used to report emitted alerts. Accepts a stdlib
            ``logging.Logger`` or a structlog ``BoundLogger``.
        """

        self._logger = logger or logging.getLogger(__name__)
        self._alerts: list[Alert] = []

    def emit_alert(self, alert: Alert) -> None:
        """Emit ``alert`` by recording it in memory and logging it."""

        self._alerts.append(alert)
        self._log_alert(alert)

    def get_alerts(self, run_id: str) -> list[Alert]:
        """Return emitted alerts correlated to ``run_id``."""

        return [alert for alert in self._alerts if alert.run_id == run_id]

    def _log_alert(self, alert: Alert) -> None:
        payload = {
            "alert_id": alert.alert_id,
            "title": alert.title,
            "message": alert.message,
            "run_id": alert.run_id,
            "module_name": alert.module_name,
            "metadata": alert.metadata,
        }

        if hasattr(self._logger, "bind"):
            if alert.level is AlertLevel.INFO:
                self._logger.info("Alert emitted", level=alert.level.value, **payload)
            elif alert.level is AlertLevel.WARNING:
                self._logger.warning("Alert emitted", level=alert.level.value, **payload)
            else:
                self._logger.error("Alert emitted", level=alert.level.value, **payload)
            return

        level = getattr(self._logger, alert.level.value.lower(), None)
        if not callable(level):
            level = self._logger.error

        formatted = ", ".join(f"{key}={value!r}" for key, value in payload.items())
        level(f"Alert emitted ({alert.level.value}) | {formatted}")


class InMemorySLOTracker(SLOTracker):
    """In-memory SLO tracker.

    The tracker keeps:

    - A registry of :class:`~monitoring.interface.SLOTarget` definitions by
      name.
    - A time-series of ``(timestamp_ns, value)`` samples for each SLO.

    Evaluation semantics
    --------------------
    :meth:`check_slo` computes the arithmetic mean of samples within the target
    window and compares it against the target threshold.
    """

    def __init__(
        self,
        targets: Iterable[SLOTarget] | None = None,
        *,
        logger: logging.Logger | BoundLogger | None = None,
    ) -> None:
        """Create a new in-memory SLO tracker.

        Parameters
        ----------
        targets:
            Optional iterable of target definitions to register at
            construction time.
        logger:
            Optional logger used to record evaluation diagnostics. Accepts a
            stdlib ``logging.Logger`` or a structlog ``BoundLogger``.
        """

        self._logger = logger or logging.getLogger(__name__)
        self._targets: dict[str, SLOTarget] = {}
        self._samples: DefaultDict[str, list[tuple[int, float]]] = defaultdict(list)

        if targets is not None:
            for target in targets:
                self.add_target(target)

    def add_target(self, target: SLOTarget) -> None:
        """Register or replace a target definition."""

        self._targets[target.name] = target

    def track_slo(self, slo_name: str, value: float) -> None:
        """Record a sample for ``slo_name`` at ``time.time_ns()``."""

        if slo_name not in self._targets:
            raise KeyError(f"Unknown SLO target '{slo_name}'. Register it first.")

        timestamp = time.time_ns()
        self._samples[slo_name].append((timestamp, float(value)))
        self._log_debug("Tracked SLO sample", slo_name=slo_name, timestamp=timestamp, value=value)

    def check_slo(self, slo_name: str) -> SLOSnapshot:
        """Compute SLO compliance for ``slo_name`` over its configured window."""

        target = self._targets.get(slo_name)
        if target is None:
            raise KeyError(f"Unknown SLO target '{slo_name}'. Register it first.")

        window_end = time.time_ns()
        window_start = window_end - int(target.window_seconds) * 1_000_000_000

        samples = [
            value
            for (timestamp, value) in self._samples.get(slo_name, [])
            if window_start <= timestamp <= window_end
        ]

        sample_count = len(samples)
        current_value = sum(samples) / sample_count if sample_count else 0.0

        if target.comparison == ">=":
            is_met = current_value >= target.target_value
        else:
            is_met = current_value <= target.target_value

        snapshot = SLOSnapshot(
            slo_name=slo_name,
            current_value=float(current_value),
            target_value=float(target.target_value),
            is_met=bool(is_met),
            window_start=int(window_start),
            window_end=int(window_end),
            sample_count=int(sample_count),
        )
        self._log_debug(
            "Computed SLO snapshot",
            slo_name=slo_name,
            current_value=current_value,
            target_value=target.target_value,
            comparison=target.comparison,
            is_met=is_met,
            sample_count=sample_count,
        )
        return snapshot

    def _log_debug(self, message: str, **kwargs: Any) -> None:
        if hasattr(self._logger, "bind"):
            self._logger.debug(message, **kwargs)
            return

        if kwargs:
            formatted = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
            message = f"{message} | {formatted}"
        self._logger.debug(message)
