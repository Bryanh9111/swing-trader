"""Base filter contracts for the scanner filtering pipeline.

This module introduces a Phase 1-style filter architecture (Strategy + Protocol)
inspired by the reference project's refactoring plan:
`../References-DoNotLinkToAnyProjectsHere/Scanner Reference/platform-stocks-selection/docs/refactoring_plan.md`.

Filters are small, independently testable units that accept a bar series plus a
validated :class:`~scanner.interface.ScannerConfig`, returning a structured
:class:`~common.interface.Result` containing a :class:`FilterResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, TYPE_CHECKING, runtime_checkable

import msgspec

from common.interface import Result, ResultStatus
from scanner.interface import ScannerConfig

if TYPE_CHECKING:
    from data.interface import PriceBar
else:  # pragma: no cover - type-only import may not exist in early phases.
    PriceBar = Any  # type: ignore[assignment]

__all__ = [
    "FilterResult",
    "FilterProtocol",
    "BaseFilter",
]


class FilterResult(msgspec.Struct, frozen=True, kw_only=True):
    """Structured outcome for a single filter evaluation.

    Attributes:
        passed: Whether the filter passed.
        reason: Human-readable reason explaining pass/fail.
        score: Score in [0, 1]. When a filter is purely boolean, returning 1.0
            for pass and 0.0 for fail is recommended.
        features: Detected features for downstream consumption.
        metadata: Debug metadata (intermediate values, thresholds, etc.).
    """

    passed: bool
    reason: str
    score: float = 1.0
    features: dict[str, Any] = msgspec.field(default_factory=dict)
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


@runtime_checkable
class FilterProtocol(Protocol):
    """Runtime-checkable protocol for scanner filters."""

    name: ClassVar[str]
    enabled: bool

    def apply(
        self,
        bars: Sequence[PriceBar],
        config: ScannerConfig,
        *,
        context: dict[str, Any] | None = None,
    ) -> Result[FilterResult]:
        """Run the filter and return a structured :class:`~common.interface.Result`."""


class BaseFilter(ABC, FilterProtocol):
    """Abstract base class implementing the filter protocol ergonomics.

    Subclasses should override :attr:`name` and implement :meth:`_apply_filter`.
    """

    name: ClassVar[str] = "base_filter"

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def apply(
        self,
        bars: Sequence[PriceBar],
        config: ScannerConfig,
        *,
        context: dict[str, Any] | None = None,
    ) -> Result[FilterResult]:
        """Apply the filter, honoring ``enabled`` and normalizing exceptions."""

        if not self.enabled:
            return Result.success(
                FilterResult(
                    passed=True,
                    reason="FILTER_DISABLED",
                    score=1.0,
                    metadata={"enabled": False},
                ),
                reason_code="FILTER_DISABLED",
            )

        try:
            result = self._apply_filter(bars, config)
        except Exception as exc:  # noqa: BLE001 - normalize plugin/filter failures.
            return Result.failed(exc, "FILTER_EXCEPTION")

        if result.status is ResultStatus.FAILED:
            return result

        if result.data is None:
            return Result.failed(ValueError("Filter returned no data"), "FILTER_EMPTY_RESULT")

        return result

    @abstractmethod
    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        """Implement the filter logic and return a FilterResult wrapped in Result."""
