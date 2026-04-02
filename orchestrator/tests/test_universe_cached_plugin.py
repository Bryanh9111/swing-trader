from __future__ import annotations

from pathlib import Path

import msgspec

from common.interface import ResultStatus
from orchestrator.plugins import UniverseCachedPlugin
from universe.interface import EquityInfo, UniverseSnapshot


def test_universe_cached_plugin_loads_universe_snapshot(tmp_path: Path) -> None:
    snapshot = UniverseSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1_700_000_000_000_000_000,
        source="cache_file",
        equities=[
            EquityInfo(
                symbol="AAPL",
                exchange="NASDAQ",
                price=195.0,
                avg_dollar_volume_20d=12_500_000_000.0,
                market_cap=3_000_000_000_000.0,
                is_otc=False,
                is_halted=False,
                sector="Technology",
            )
        ],
        filter_criteria={},
        total_candidates=1,
        total_filtered=1,
    )

    cache_path = tmp_path / "universe.json"
    cache_path.write_bytes(msgspec.json.encode(msgspec.to_builtins(snapshot)))

    plugin = UniverseCachedPlugin(config={"cache_path": str(cache_path)})
    validated = plugin.validate_config(plugin.config)
    assert validated.status is ResultStatus.SUCCESS
    init_result = plugin.init({})
    assert init_result.status is ResultStatus.SUCCESS

    result = plugin.execute({})
    assert result.status is ResultStatus.SUCCESS
    assert isinstance(result.data, dict)
    assert len(result.data.get("equities") or []) == 1


def test_universe_cached_plugin_loads_equities_cache_entry(tmp_path: Path) -> None:
    cache_payload = {
        "saved_at_ns": 1_700_000_000_000_000_000,
        "equities": [
            msgspec.to_builtins(
                EquityInfo(
                    symbol="MSFT",
                    exchange="NASDAQ",
                    price=410.0,
                    avg_dollar_volume_20d=9_500_000_000.0,
                    market_cap=3_100_000_000_000.0,
                    is_otc=False,
                    is_halted=False,
                    sector="Technology",
                )
            )
        ],
    }

    cache_path = tmp_path / "yfinance_universe.json"
    cache_path.write_bytes(msgspec.json.encode(cache_payload))

    plugin = UniverseCachedPlugin(config={"cache_path": str(cache_path)})
    validated = plugin.validate_config(plugin.config)
    assert validated.status is ResultStatus.SUCCESS
    init_result = plugin.init({})
    assert init_result.status is ResultStatus.SUCCESS

    result = plugin.execute({})
    assert result.status is ResultStatus.SUCCESS
    assert isinstance(result.data, dict)
    assert result.data.get("source") == "cache_file"
    assert len(result.data.get("equities") or []) == 1

