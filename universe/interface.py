"""Universe Builder interface definitions for US equity universe construction.

This module defines schemas and protocols for the Universe Builder plugin,
which filters US equities based on exchange, price, liquidity, and market cap
thresholds to produce a tradable universe snapshot. The structures follow the
patterns established in common/interface.py and journal/interface.py.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import msgspec

from common.interface import Result
from journal.interface import SnapshotBase
from plugins.interface import PluginBase

__all__ = [
    "EquityInfo",
    "UniverseSnapshot",
    "UniverseFilterCriteria",
    "UniverseBuilderPlugin",
]


class EquityInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Reference data for a single US-listed equity.

    This schema captures the minimal per-symbol fields required for filtering
    and downstream scan workflows.

    Attributes:
        symbol: Equity ticker symbol (e.g., ``"AAPL"``).
        exchange: Listing exchange identifier (e.g., ``"NASDAQ"``).
        price: Current or last close price.
        avg_dollar_volume_20d: 20-day average dollar volume, used as a liquidity proxy.
        market_cap: Market capitalization in USD.
        is_otc: Whether the symbol is traded over-the-counter.
        is_halted: Whether the symbol is currently halted.
        sector: Optional sector label (e.g., ``"Technology"``).

    Example:
        >>> EquityInfo(
        ...     symbol="AAPL",
        ...     exchange="NASDAQ",
        ...     price=195.0,
        ...     avg_dollar_volume_20d=12_500_000_000.0,
        ...     market_cap=3_000_000_000_000.0,
        ...     is_otc=False,
        ...     is_halted=False,
        ...     sector="Technology",
        ... )
        EquityInfo(symbol='AAPL', exchange='NASDAQ', price=195.0, avg_dollar_volume_20d=12500000000.0, market_cap=3000000000000.0, is_otc=False, is_halted=False, sector='Technology')
    """

    symbol: str
    exchange: str
    name: str | None = None
    price: float | None = None
    avg_dollar_volume_20d: float | None = None
    market_cap: float | None = None
    is_otc: bool = False
    is_halted: bool = False
    sector: str | None = None
    asset_type: str | None = None


class UniverseFilterCriteria(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for filtering a tradable US equity universe.

    The criteria is consumed by universe builder plugins and can be embedded
    (as builtins) into the resulting ``UniverseSnapshot.filter_criteria`` to
    support reproducible backfills and audits.

    Attributes:
        exchanges: Allowed exchanges (e.g., ``["NYSE", "NASDAQ", "AMEX"]``).
        min_price: Minimum equity price.
        max_price: Optional maximum equity price.
        min_avg_dollar_volume_20d: Minimum 20-day average dollar volume.
        min_market_cap: Optional market cap floor.
        exclude_otc: Exclude over-the-counter symbols when ``True``.
        exclude_halted: Exclude halted symbols when ``True``.
        exclude_etfs: Exclude ETF symbols when ``True`` (default).
        max_results: Optional cap on the number of equities returned.
        enable_details_enrich: Enable Polygon per-ticker details enrichment.
        details_enrich_top_k: Max symbols to enrich via details API.
        details_enrich_multiplier: Multiplier for ``max_results`` when selecting enrich candidates.

    Example:
        >>> UniverseFilterCriteria(
        ...     exchanges=["NYSE", "NASDAQ", "AMEX"],
        ...     min_price=5.0,
        ...     max_price=None,
        ...     min_avg_dollar_volume_20d=10_000_000.0,
        ...     min_market_cap=1_000_000_000.0,
        ...     exclude_otc=True,
        ...     exclude_halted=True,
        ...     exclude_etfs=True,
        ...     max_results=3000,
        ... )
        UniverseFilterCriteria(exchanges=['NYSE', 'NASDAQ', 'AMEX'], min_price=5.0, max_price=None, min_avg_dollar_volume_20d=10000000.0, min_market_cap=1000000000.0, exclude_otc=True, exclude_halted=True, exclude_etfs=True, max_results=3000)
    """

    exchanges: list[str]
    min_price: float
    max_price: float | None
    min_avg_dollar_volume_20d: float
    min_market_cap: float | None
    exclude_otc: bool
    exclude_halted: bool
    exclude_etfs: bool = True
    max_results: int | None
    enable_details_enrich: bool = True
    details_enrich_top_k: int = 500
    details_enrich_multiplier: float = 1.5


class UniverseSnapshot(SnapshotBase, frozen=True, kw_only=True):
    """Versioned snapshot of a filtered tradable universe.

    ``UniverseSnapshot`` extends ``SnapshotBase`` with universe-specific
    metadata and the filtered list of equities.

    Attributes:
        schema_version: See ``journal.interface.SnapshotBase``.
        system_version: See ``journal.interface.SnapshotBase``.
        asof_timestamp: See ``journal.interface.SnapshotBase``.
        source: Data source identifier such as ``"polygon"``, ``"fmp"``, or ``"cache"``.
        equities: The filtered equity list.
        filter_criteria: Builtins snapshot of the filtering configuration used.
        total_candidates: Count of symbols considered before filtering.
        total_filtered: Count of symbols returned after filtering.

    Example:
        >>> criteria = UniverseFilterCriteria(
        ...     exchanges=["NYSE", "NASDAQ", "AMEX"],
        ...     min_price=5.0,
        ...     max_price=None,
        ...     min_avg_dollar_volume_20d=10_000_000.0,
        ...     min_market_cap=None,
        ...     exclude_otc=True,
        ...     exclude_halted=True,
        ...     max_results=None,
        ... )
        >>> UniverseSnapshot(
        ...     schema_version="1.0.0",
        ...     system_version="abc1234",
        ...     asof_timestamp=1_700_000_000_000_000_000,
        ...     source="cache",
        ...     equities=[
        ...         EquityInfo(
        ...             symbol="AAPL",
        ...             exchange="NASDAQ",
        ...             price=195.0,
        ...             avg_dollar_volume_20d=12_500_000_000.0,
        ...             market_cap=3_000_000_000_000.0,
        ...             is_otc=False,
        ...             is_halted=False,
        ...             sector="Technology",
        ...         )
        ...     ],
        ...     filter_criteria=msgspec.to_builtins(criteria),
        ...     total_candidates=8000,
        ...     total_filtered=3500,
        ... )
        UniverseSnapshot(schema_version='1.0.0', system_version='abc1234', asof_timestamp=1700000000000000000, source='cache', equities=[EquityInfo(symbol='AAPL', exchange='NASDAQ', price=195.0, avg_dollar_volume_20d=12500000000.0, market_cap=3000000000000.0, is_otc=False, is_halted=False, sector='Technology')], filter_criteria={'exchanges': ['NYSE', 'NASDAQ', 'AMEX'], 'min_price': 5.0, 'max_price': None, 'min_avg_dollar_volume_20d': 10000000.0, 'min_market_cap': None, 'exclude_otc': True, 'exclude_halted': True, 'max_results': None}, total_candidates=8000, total_filtered=3500)
    """

    source: str
    equities: list[EquityInfo]
    filter_criteria: dict[str, Any]
    total_candidates: int
    total_filtered: int


@runtime_checkable
class UniverseBuilderPlugin(
    PluginBase[UniverseFilterCriteria, None, UniverseSnapshot],
    Protocol,
):
    """Protocol contract for universe builder plugins.

    Universe builder plugins construct a US equity universe based on an
    internal configuration (``UniverseFilterCriteria``), returning a
    ``UniverseSnapshot`` suitable for journaling.

    The plugin lifecycle follows the standard ``PluginBase`` contract.

    Example:
        >>> import msgspec
        >>> from plugins.interface import PluginMetadata, PluginCategory
        >>> from universe.interface import (
        ...     UniverseBuilderPlugin,
        ...     UniverseFilterCriteria,
        ...     UniverseSnapshot,
        ... )
        >>>
        >>> class MyUniverseBuilder:
        ...     metadata = PluginMetadata(
        ...         name="my_universe_builder",
        ...         version="1.0.0",
        ...         category=PluginCategory.SCANNER,
        ...         description="Example universe builder plugin.",
        ...     )
        ...
        ...     def __init__(self, criteria: UniverseFilterCriteria) -> None:
        ...         self._criteria = criteria
        ...
        ...     def init(self, context: dict[str, Any] | None = None) -> Result[None]:
        ...         return Result.success(None)
        ...
        ...     def validate_config(self, config: UniverseFilterCriteria) -> Result[UniverseFilterCriteria]:
        ...         return Result.success(config)
        ...
        ...     def execute(self, payload: None) -> Result[UniverseSnapshot]:
        ...         snapshot = UniverseSnapshot(
        ...             schema_version="1.0.0",
        ...             system_version="abc1234",
        ...             asof_timestamp=1_700_000_000_000_000_000,
        ...             source="cache",
        ...             equities=[],
        ...             filter_criteria=msgspec.to_builtins(self._criteria),
        ...             total_candidates=0,
        ...             total_filtered=0,
        ...         )
        ...         return Result.success(snapshot)
        ...
        ...     def cleanup(self) -> Result[None]:
        ...         return Result.success(None)
        >>>
        >>> plugin: UniverseBuilderPlugin = MyUniverseBuilder(
        ...     UniverseFilterCriteria(
        ...         exchanges=["NYSE", "NASDAQ", "AMEX"],
        ...         min_price=5.0,
        ...         max_price=None,
        ...         min_avg_dollar_volume_20d=10_000_000.0,
        ...         min_market_cap=None,
        ...         exclude_otc=True,
        ...         exclude_halted=True,
        ...         max_results=None,
        ...     )
        ... )
    """

    def init(self, context: dict[str, Any] | None = None) -> Result[None]:
        """Initialise the plugin and acquire any resources needed."""

    def validate_config(self, config: UniverseFilterCriteria) -> Result[UniverseFilterCriteria]:
        """Validate and optionally normalise universe filter criteria."""

    def execute(self, payload: None) -> Result[UniverseSnapshot]:
        """Build and return a versioned ``UniverseSnapshot`` result."""

    def cleanup(self) -> Result[None]:
        """Release resources held by the plugin implementation."""
