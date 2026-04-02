from __future__ import annotations

from typing import Any

import msgspec
import pytest

from universe.builder import UniverseBuilder
from universe.interface import EquityInfo, UniverseBuilderPlugin, UniverseFilterCriteria, UniverseSnapshot


def test_equity_info_creation_with_all_fields() -> None:
    equity = EquityInfo(
        symbol="AAPL",
        exchange="NASDAQ",
        price=195.0,
        avg_dollar_volume_20d=12_500_000_000.0,
        market_cap=3_000_000_000_000.0,
        is_otc=False,
        is_halted=False,
        sector="Technology",
    )

    assert equity.symbol == "AAPL"
    assert equity.exchange == "NASDAQ"
    assert equity.price == 195.0
    assert equity.avg_dollar_volume_20d > 0
    assert equity.market_cap > 0
    assert equity.is_otc is False
    assert equity.is_halted is False
    assert equity.sector == "Technology"


def test_equity_info_validation_rejects_invalid_types() -> None:
    payload: dict[str, Any] = {
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "price": "195.0",
        "avg_dollar_volume_20d": 12_500_000_000.0,
        "market_cap": 3_000_000_000_000.0,
        "is_otc": False,
        "is_halted": False,
        "sector": "Technology",
    }

    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(payload, type=EquityInfo)


def test_universe_filter_criteria_validation() -> None:
    criteria = UniverseFilterCriteria(
        exchanges=["NYSE", "NASDAQ"],
        min_price=5.0,
        max_price=None,
        min_avg_dollar_volume_20d=10_000_000.0,
        min_market_cap=1_000_000_000.0,
        exclude_otc=True,
        exclude_halted=True,
        max_results=2500,
    )

    assert criteria.exchanges == ["NYSE", "NASDAQ"]
    assert criteria.max_price is None
    assert criteria.exclude_otc is True
    assert criteria.max_results == 2500


def test_universe_snapshot_with_nested_equities_list() -> None:
    criteria = UniverseFilterCriteria(
        exchanges=["NYSE", "NASDAQ"],
        min_price=5.0,
        max_price=None,
        min_avg_dollar_volume_20d=10_000_000.0,
        min_market_cap=None,
        exclude_otc=True,
        exclude_halted=True,
        max_results=None,
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "system_version": "test",
        "asof_timestamp": 1_700_000_000_000_000_000,
        "source": "cache",
        "equities": [
            {
                "symbol": "AAPL",
                "exchange": "NASDAQ",
                "price": 195.0,
                "avg_dollar_volume_20d": 12_500_000_000.0,
                "market_cap": 3_000_000_000_000.0,
                "is_otc": False,
                "is_halted": False,
                "sector": "Technology",
            }
        ],
        "filter_criteria": msgspec.to_builtins(criteria),
        "total_candidates": 1,
        "total_filtered": 1,
    }

    snapshot = msgspec.convert(payload, type=UniverseSnapshot)

    assert snapshot.source == "cache"
    assert snapshot.total_candidates == 1
    assert snapshot.total_filtered == 1
    assert len(snapshot.equities) == 1
    assert isinstance(snapshot.equities[0], EquityInfo)


def test_universe_builder_plugin_protocol_compliance_runtime_checkable() -> None:
    criteria = UniverseFilterCriteria(
        exchanges=["NYSE", "NASDAQ"],
        min_price=5.0,
        max_price=None,
        min_avg_dollar_volume_20d=10_000_000.0,
        min_market_cap=None,
        exclude_otc=True,
        exclude_halted=True,
        max_results=None,
    )

    builder = UniverseBuilder(criteria)

    assert isinstance(builder, UniverseBuilderPlugin)
    assert not isinstance(object(), UniverseBuilderPlugin)

