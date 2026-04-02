"""Monitoring and alerting subsystem for Automated Swing Trader.

This package defines the Phase 2.4 monitoring contracts (metrics, alerts, SLOs)
and will host concrete implementations for collecting, persisting, and
reporting runtime health signals.
"""

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
from .collectors import InMemoryAlertEmitter, InMemoryMetricsCollector, InMemorySLOTracker

__all__ = [
    "Alert",
    "AlertEmitter",
    "AlertLevel",
    "InMemoryAlertEmitter",
    "InMemoryMetricsCollector",
    "InMemorySLOTracker",
    "MetricSnapshot",
    "MetricsCollector",
    "SLOSnapshot",
    "SLOTarget",
    "SLOTracker",
]
