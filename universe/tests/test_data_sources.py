from __future__ import annotations

import time

import msgspec

from common.interface import ResultStatus
from universe.interface import EquityInfo
from universe.data_sources import (
    fetch_cached_universe,
    fetch_fmp_universe,
    fetch_polygon_ticker_details,
    fetch_polygon_universe,
)


def test_fetch_polygon_universe_missing_api_key_fails(monkeypatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    result = fetch_polygon_universe()

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "MISSING_API_KEY"


def test_fetch_polygon_universe_paginates_and_maps_fields(monkeypatch) -> None:
    import json
    from dataclasses import dataclass
    from typing import Any

    import requests

    import universe.data_sources as ds

    @dataclass(slots=True)
    class _StubResponse:
        status_code: int
        content: bytes
        text: str = ""

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    page1 = {
        "results": [
            {"ticker": "aapl", "name": "Apple Inc.", "primary_exchange": "XNAS"},
            {"ticker": "msft", "name": "Microsoft", "primary_exchange": "XNAS"},
        ],
        "next_url": "https://api.polygon.io/v3/reference/tickers?cursor=abc",
    }
    page2 = {
        "results": [
            {"ticker": "brk.b", "name": "Berkshire Hathaway", "primary_exchange": "XNYS"},
        ],
        "next_url": None,
    }

    seen: list[str] = []

    def fake_get(url: str, params: dict[str, Any] | None, timeout: float) -> _StubResponse:
        seen.append(url)
        if url.endswith("/v3/reference/tickers"):
            assert params is not None
            assert params["active"] == "true"
            assert params["market"] == "stocks"
            assert params["limit"] == "1000"
            assert params["apiKey"] == "test-key"
            return _StubResponse(status_code=200, content=json.dumps(page1).encode("utf-8"), text="OK")
        assert "cursor=abc" in url
        assert "apiKey=test-key" in url
        assert params is None
        return _StubResponse(status_code=200, content=json.dumps(page2).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)

    result = fetch_polygon_universe()

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert [e.symbol for e in result.data] == ["AAPL", "MSFT", "BRK.B"]
    assert result.data[0].name == "Apple Inc."
    assert result.data[0].exchange == "XNAS"
    assert result.data[0].price is None
    assert result.data[0].market_cap is None
    assert result.data[0].avg_dollar_volume_20d is None
    assert seen == [
        "https://api.polygon.io/v3/reference/tickers",
        "https://api.polygon.io/v3/reference/tickers?cursor=abc&apiKey=test-key",
    ]


def test_fetch_fmp_universe_not_implemented_fails() -> None:
    result = fetch_fmp_universe()

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NOT_IMPLEMENTED"


def test_fetch_cached_universe_uses_fresh_cache(monkeypatch, tmp_path) -> None:
    import universe.data_sources as ds

    cache_path = tmp_path / "yfinance_universe.json"
    monkeypatch.setattr(ds, "_CACHE_PATH", cache_path)
    monkeypatch.setattr(ds, "_CACHE_TTL_SECONDS", 86400)

    equities = [
        EquityInfo(
            symbol="AAPL",
            exchange="NASDAQ",
            price=195.0,
            avg_dollar_volume_20d=10_000_000_000.0,
            market_cap=3_000_000_000_000.0,
            is_otc=False,
            is_halted=False,
            sector="Technology",
        )
    ]
    payload = msgspec.json.encode({"saved_at_ns": time.time_ns(), "equities": [msgspec.to_builtins(equities[0])]})
    cache_path.write_bytes(payload)

    result = fetch_cached_universe()
    assert result.status is ResultStatus.SUCCESS
    assert result.data == equities


def test_fetch_cached_universe_refreshes_stale_cache(monkeypatch, tmp_path) -> None:
    import universe.data_sources as ds

    cache_path = tmp_path / "yfinance_universe.json"
    monkeypatch.setattr(ds, "_CACHE_PATH", cache_path)
    monkeypatch.setattr(ds, "_CACHE_TTL_SECONDS", 1)

    stale_equities = [
        EquityInfo(
            symbol="MSFT",
            exchange="NASDAQ",
            price=350.0,
            avg_dollar_volume_20d=9_000_000_000.0,
            market_cap=2_500_000_000_000.0,
            is_otc=False,
            is_halted=False,
            sector="Technology",
        )
    ]
    stale_payload = msgspec.json.encode(
        {"saved_at_ns": time.time_ns() - 10_000_000_000, "equities": [msgspec.to_builtins(stale_equities[0])]}
    )
    cache_path.write_bytes(stale_payload)

    new_equities = [
        EquityInfo(
            symbol="NVDA",
            exchange="NASDAQ",
            price=450.0,
            avg_dollar_volume_20d=12_000_000_000.0,
            market_cap=1_500_000_000_000.0,
            is_otc=False,
            is_halted=False,
            sector="Technology",
        )
    ]

    def _fake_fetch() -> ds._FetchOutcome:
        return ds._FetchOutcome(status="success", equities=new_equities)

    monkeypatch.setattr(ds, "_fetch_yfinance_universe", _fake_fetch)

    result = fetch_cached_universe()
    assert result.status is ResultStatus.SUCCESS
    assert result.data == new_equities

    decoded = msgspec.json.decode(cache_path.read_bytes())
    assert decoded["equities"][0]["symbol"] == "NVDA"


def test_fetch_cached_universe_network_failure_returns_failed_when_no_cache(monkeypatch, tmp_path) -> None:
    import universe.data_sources as ds

    cache_path = tmp_path / "yfinance_universe.json"
    monkeypatch.setattr(ds, "_CACHE_PATH", cache_path)
    monkeypatch.setattr(ds, "_CACHE_TTL_SECONDS", 1)

    def _fake_fetch() -> ds._FetchOutcome:
        return ds._FetchOutcome(status="failed", equities=[], error=RuntimeError("network down"))

    monkeypatch.setattr(ds, "_fetch_yfinance_universe", _fake_fetch)

    result = fetch_cached_universe()
    assert result.status is ResultStatus.FAILED


def test_fetch_cached_universe_network_failure_returns_degraded_with_stale_cache(monkeypatch, tmp_path) -> None:
    import universe.data_sources as ds

    cache_path = tmp_path / "yfinance_universe.json"
    monkeypatch.setattr(ds, "_CACHE_PATH", cache_path)
    monkeypatch.setattr(ds, "_CACHE_TTL_SECONDS", 1)

    stale_equities = [
        EquityInfo(
            symbol="AAPL",
            exchange="NASDAQ",
            price=195.0,
            avg_dollar_volume_20d=10_000_000_000.0,
            market_cap=3_000_000_000_000.0,
            is_otc=False,
            is_halted=False,
            sector="Technology",
        )
    ]
    stale_payload = msgspec.json.encode(
        {"saved_at_ns": time.time_ns() - 10_000_000_000, "equities": [msgspec.to_builtins(stale_equities[0])]}
    )
    cache_path.write_bytes(stale_payload)

    def _fake_fetch() -> ds._FetchOutcome:
        return ds._FetchOutcome(status="failed", equities=[], error=RuntimeError("network down"))

    monkeypatch.setattr(ds, "_fetch_yfinance_universe", _fake_fetch)

    result = fetch_cached_universe()
    assert result.status is ResultStatus.DEGRADED
    assert result.data == stale_equities


def test_fetch_polygon_ticker_details_success(monkeypatch) -> None:
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    def _response(symbol: str):
        payload = {"results": {"market_cap": 123.0, "sector": f"Sector-{symbol}"}}
        return MagicMock(status_code=200, json=lambda: payload)

    with (
        patch("data.rate_limiter.TokenBucketRateLimiter") as limiter_cls,
        patch("requests.get") as mock_get,
    ):
        limiter = MagicMock()
        limiter.acquire.return_value = True
        limiter_cls.return_value = limiter

        mock_get.side_effect = lambda url, params, timeout: _response(url.rsplit("/", 1)[-1])

        result = fetch_polygon_ticker_details(["aapl", " msft "], max_workers=1, timeout_seconds=7.0)

        assert result == {
            "AAPL": {"market_cap": 123.0, "sector": "Sector-AAPL"},
            "MSFT": {"market_cap": 123.0, "sector": "Sector-MSFT"},
        }

        assert limiter.acquire.call_count == 2
        limiter.acquire.assert_any_call(timeout_seconds=7.0)
        assert mock_get.call_count == 2
        mock_get.assert_any_call(
            "https://api.polygon.io/v3/reference/tickers/AAPL",
            params={"apiKey": "test-key"},
            timeout=7.0,
        )


def test_fetch_polygon_ticker_details_partial_success(monkeypatch) -> None:
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    def _fake_get(url: str, params: dict[str, str], timeout: float):
        symbol = url.rsplit("/", 1)[-1]
        if symbol == "AAPL":
            return MagicMock(status_code=200, json=lambda: {"results": {"market_cap": 123.0, "sector": "Tech"}})
        return MagicMock(status_code=200, json=lambda: {"results": {"sector": "Finance"}})

    with (
        patch("data.rate_limiter.TokenBucketRateLimiter") as limiter_cls,
        patch("requests.get", side_effect=_fake_get) as mock_get,
    ):
        limiter = MagicMock()
        limiter.acquire.return_value = True
        limiter_cls.return_value = limiter

        result = fetch_polygon_ticker_details(["AAPL", "MSFT"], max_workers=1, timeout_seconds=5.0)

        assert result["AAPL"]["market_cap"] == 123.0
        assert result["AAPL"]["sector"] == "Tech"
        assert result["MSFT"]["market_cap"] is None
        assert result["MSFT"]["sector"] == "Finance"
        assert mock_get.call_count == 2


def test_fetch_polygon_ticker_details_api_failure(monkeypatch) -> None:
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    with (
        patch("data.rate_limiter.TokenBucketRateLimiter") as limiter_cls,
        patch("requests.get", return_value=MagicMock(status_code=500, json=lambda: {"results": {}})) as mock_get,
    ):
        limiter = MagicMock()
        limiter.acquire.return_value = True
        limiter_cls.return_value = limiter

        result = fetch_polygon_ticker_details(["AAPL"], max_workers=1, timeout_seconds=5.0)

        assert result == {}
        assert limiter.acquire.call_count == 1
        assert mock_get.call_count == 1


def test_fetch_polygon_ticker_details_rate_limiting(monkeypatch) -> None:
    import threading
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    started = threading.Event()
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def _fake_get(url: str, params: dict[str, str], timeout: float):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            started.set()
        # Keep the thread busy briefly to expose concurrency.
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return MagicMock(status_code=200, json=lambda: {"results": {"market_cap": 1.0}})

    with (
        patch("data.rate_limiter.TokenBucketRateLimiter") as limiter_cls,
        patch("requests.get", side_effect=_fake_get) as mock_get,
    ):
        limiter = MagicMock()
        limiter.acquire.return_value = True
        limiter_cls.return_value = limiter

        result = fetch_polygon_ticker_details([f"SYM{i}" for i in range(10)], max_workers=3, timeout_seconds=5.0)

        assert started.is_set()
        assert len(result) == 10
        assert limiter.acquire.call_count == 10
        assert mock_get.call_count == 10
        assert max_in_flight <= 3


def test_fetch_polygon_ticker_details_timeout(monkeypatch) -> None:
    from unittest.mock import MagicMock, patch

    import requests

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    with (
        patch("data.rate_limiter.TokenBucketRateLimiter") as limiter_cls,
        patch("requests.get", side_effect=requests.exceptions.Timeout("timeout")) as mock_get,
    ):
        limiter = MagicMock()
        limiter.acquire.return_value = True
        limiter_cls.return_value = limiter

        result = fetch_polygon_ticker_details(["AAPL"], max_workers=1, timeout_seconds=1.0)

        assert result == {}
        assert limiter.acquire.call_count == 1
        assert mock_get.call_count == 1


def test_fetch_polygon_ticker_details_empty_input(monkeypatch) -> None:
    from unittest.mock import patch

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    with (
        patch("data.rate_limiter.TokenBucketRateLimiter") as limiter_cls,
        patch("requests.get") as mock_get,
    ):
        result = fetch_polygon_ticker_details([], max_workers=1)
        assert result == {}
        assert limiter_cls.call_count == 0
        assert mock_get.call_count == 0
