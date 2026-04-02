"""Scanner filter architecture (Phase 1).

This package provides a small, composable filtering layer that can be applied
after core detection (or as part of scanning) to accept/reject candidates while
capturing a consistent, debuggable rationale.

Design goals
------------
- Strategy pattern: each filter is a standalone class implementing
  :class:`~scanner.filters.base.FilterProtocol`.
- Composite pattern: :class:`~scanner.filters.chain.FilterChain` composes
  multiple filters using AND/OR logic and aggregates per-filter outputs.
- Structured results: all operations return :class:`~common.interface.Result`
  and use :class:`~scanner.filters.base.FilterResult` / :class:`ChainResult`
  schemas for deterministic serialization and observability.

Reference
---------
This design is adapted from the Phase 1 "filter base class + filter manager" plan in:
`../References-DoNotLinkToAnyProjectsHere/Scanner Reference/platform-stocks-selection/docs/refactoring_plan.md`

Usage examples
--------------

Create a custom filter:

.. code-block:: python

    from collections.abc import Sequence
    from scanner.filters import BaseFilter, FilterResult
    from common.interface import Result
    from scanner.interface import ScannerConfig

    class MinBarsFilter(BaseFilter):
        name = "min_bars"

        def __init__(self, *, min_bars: int = 60, enabled: bool = True) -> None:
            super().__init__(enabled=enabled)
            self._min_bars = min_bars

        def _apply_filter(self, bars: Sequence[object], config: ScannerConfig) -> Result[FilterResult]:
            passed = len(bars) >= self._min_bars
            return Result.success(
                FilterResult(
                    passed=passed,
                    reason="OK" if passed else "INSUFFICIENT_BARS",
                    score=1.0 if passed else 0.0,
                    features={"bars": len(bars), "min_bars": self._min_bars},
                )
            )

Compose filters into a chain:

.. code-block:: python

    from scanner.filters import FilterChain

    chain = FilterChain()
    chain.add_filter(MinBarsFilter(min_bars=60))
    chain_result = chain.execute(bars, ScannerConfig())
"""

from .base import BaseFilter, FilterProtocol, FilterResult
from .adx_entry_filter import ADXEntryFilter
from .atr_filter import ATRFilter
from .box_quality_filter import BoxQualityFilter
from .chain import ChainResult, FilterChain
from .breakthrough_filter import BreakthroughFilter
from .pullback_confirmation_filter import PullbackConfirmationFilter
from .low_position_filter import LowPositionFilter
from .price_platform_filter import PricePlatformFilter
from .rapid_decline_filter import RapidDeclineFilter
from .volume_platform_filter import VolumePlatformFilter
from .market_cap_filter import MarketCapFilter
from .ma200_trend_filter import MA200TrendFilter
from .liquidity_filter import LiquidityFilter

__all__ = [
    "FilterProtocol",
    "BaseFilter",
    "FilterResult",
    "FilterChain",
    "ChainResult",
    "PricePlatformFilter",
    "VolumePlatformFilter",
    "ATRFilter",
    "ADXEntryFilter",
    "BoxQualityFilter",
    "LowPositionFilter",
    "RapidDeclineFilter",
    "BreakthroughFilter",
    "PullbackConfirmationFilter",
    "MarketCapFilter",
    "MA200TrendFilter",
    "LiquidityFilter",
]
