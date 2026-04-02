from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pytest
import requests

from common.interface import ResultStatus
from data.polygon_adapter import PolygonDataAdapter
from data.interface import PriceBar
from data.rate_limiter import TokenBucketRateLimiter


@dataclass(slots=True)
class _StubResponse:
    status_code: int
    content: bytes
    text: str = ""


class _StubLimiter:
    def __init__(self, *, acquire_result: bool = True, call_order: list[str] | None = None) -> None:
        self.acquire_result = acquire_result
        self.calls: list[float | None] = []
        self.call_order = call_order

    def acquire(self, timeout_seconds: float | None = None) -> bool:
        self.calls.append(timeout_seconds)
        if self.call_order is not None:
            self.call_order.append("acquire")
        return self.acquire_result


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _bar(dt: datetime, *, open_: float = 1.0, high: float = 2.0, low: float = 0.5, close: float = 1.5) -> PriceBar:
    return PriceBar(
        timestamp=_ms(dt) * 1_000_000,
        open=float(open_),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=10,
    )


def test_detect_missing_dates_finds_midwindow_gaps() -> None:
    bars = [
        _bar(datetime(2025, 1, 1, tzinfo=UTC)),
        _bar(datetime(2025, 1, 5, tzinfo=UTC)),
    ]
    missing = PolygonDataAdapter._detect_missing_dates(
        bars,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 10),
        max_gap_days=5,
    )
    assert missing == [date(2025, 1, 2), date(2025, 1, 3)]


def test_detect_missing_dates_excludes_weekends() -> None:
    bars = [
        _bar(datetime(2025, 1, 3, tzinfo=UTC)),  # Fri
        _bar(datetime(2025, 1, 6, tzinfo=UTC)),  # Mon
    ]
    missing = PolygonDataAdapter._detect_missing_dates(
        bars,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 10),
        max_gap_days=5,
    )
    assert missing == []


def test_detect_missing_dates_ignores_large_gaps() -> None:
    bars = [
        _bar(datetime(2025, 1, 1, tzinfo=UTC)),
        _bar(datetime(2025, 1, 10, tzinfo=UTC)),
    ]
    missing = PolygonDataAdapter._detect_missing_dates(
        bars,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 10),
        max_gap_days=5,
    )
    assert missing == []


def test_detect_missing_dates_respects_window_boundaries() -> None:
    bars = [
        _bar(datetime(2024, 12, 31, tzinfo=UTC)),
        _bar(datetime(2025, 1, 5, tzinfo=UTC)),
    ]
    missing = PolygonDataAdapter._detect_missing_dates(
        bars,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 10),
        max_gap_days=5,
    )
    assert missing == [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]


def test_detect_missing_dates_empty_bars() -> None:
    missing = PolygonDataAdapter._detect_missing_dates(
        [],
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 10),
        max_gap_days=5,
    )
    assert missing == []


def test_polygon_initialization_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    adapter = PolygonDataAdapter(rate_limiter=TokenBucketRateLimiter(calls_per_minute=60, bucket_size=1))
    assert adapter.get_source_name() == "polygon"


def test_fetch_price_series_missing_api_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    adapter = PolygonDataAdapter(api_key=None, rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "MISSING_API_KEY"


def test_fetch_price_series_invalid_range_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-02", "2025-01-01")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_RANGE"


def test_fetch_price_series_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    limiter = _StubLimiter()

    payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            {"t": _ms(datetime(2025, 1, 2, tzinfo=UTC)), "o": 2, "h": 3, "l": 1.5, "c": 2.5, "v": 12},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        assert "api.polygon.io" in url
        assert params["apiKey"] == "test-key"
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=limiter)
    result = adapter.fetch_price_series("aapl", "2025-01-01", "2025-01-02")

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.symbol == "AAPL"
    assert result.data.source == "polygon"
    assert len(result.data.bars) == 2
    assert result.data.quality_flags["availability"] is True
    assert result.data.quality_flags["completeness"] is True
    assert result.data.quality_flags["freshness"] is True
    assert "polygon_latency_ms" in result.data.quality_flags
    assert limiter.calls and limiter.calls[0] == 65.0


def test_fetch_price_series_rate_limiting_applied_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    call_order: list[str] = []
    limiter = _StubLimiter(call_order=call_order)

    payload = {"status": "OK", "results": []}

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        call_order.append("requests.get")
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=limiter)
    adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert call_order[:2] == ["acquire", "requests.get"]


@pytest.mark.parametrize("status_code", [404, 500])
def test_fetch_price_series_http_error_handling(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    limiter = _StubLimiter()

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=status_code, content=b"{}", text="nope")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=limiter)
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "HTTP_ERROR"


def test_fetch_price_series_network_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NETWORK_ERROR"


def test_fetch_price_series_decode_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=b"not-json", text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "DECODE_ERROR"


def test_fetch_price_series_api_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {"status": "ERROR", "message": "bad key"}

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "API_ERROR"


def test_fetch_price_series_unexpected_results_type_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {"status": "OK", "results": {"unexpected": True}}

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "API_ERROR"


def test_fetch_price_series_rate_limit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter(acquire_result=False))
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "RATE_LIMIT"


def test_fetch_price_series_empty_response_returns_failed_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {"status": "OK", "results": []}

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NO_DATA"


def test_fetch_price_series_quality_degraded_when_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            {"t": _ms(datetime(2025, 1, 2, tzinfo=UTC)), "o": "bad"},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", date(2025, 1, 1), date(2025, 1, 2))
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "QUALITY_DEGRADED"
    assert result.data is not None
    assert result.data.quality_flags["completeness"] is False


def test_fetch_price_series_quality_degraded_when_non_mapping_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {
        "status": "OK",
        "results": [
            123,
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-01")
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.quality_flags["completeness"] is False


def test_fetch_price_series_quality_degraded_when_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2020, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2020-01-01", "2020-01-10")
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.quality_flags["freshness"] is False


def test_fetch_price_series_fills_midwindow_gap_with_open_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    range_payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            {"t": _ms(datetime(2025, 1, 5, tzinfo=UTC)), "o": 3, "h": 4, "l": 2.5, "c": 3.5, "v": 11},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        if "/v2/aggs/ticker/" in url and "/range/1/day/" in url:
            return _StubResponse(status_code=200, content=json.dumps(range_payload).encode("utf-8"), text="OK")

        if "/v1/open-close/" in url:
            requested_date = url.split("/")[-1]
            if requested_date == "2025-01-02":
                oc_payload = {
                    "status": "OK",
                    "open": 2,
                    "high": 3,
                    "low": 1.5,
                    "close": 2.5,
                    "volume": 12,
                }
                return _StubResponse(status_code=200, content=json.dumps(oc_payload).encode("utf-8"), text="OK")
            return _StubResponse(status_code=404, content=b"{}", text="not found")

        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-05")

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert [datetime.fromtimestamp(b.timestamp / 1_000_000_000, tz=UTC).date() for b in result.data.bars] == [
        date(2025, 1, 1),
        date(2025, 1, 2),
        date(2025, 1, 5),
    ]
    assert result.data.quality_flags["midgap_fill_attempted"] is True
    assert result.data.quality_flags["midgap_fill_count"] == 1
    assert "open_close_status" in result.data.quality_flags
    assert "open_close_latency_ms" in result.data.quality_flags


def test_fetch_price_series_skips_large_midwindow_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    open_close_calls = 0

    range_payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            {"t": _ms(datetime(2025, 1, 10, tzinfo=UTC)), "o": 3, "h": 4, "l": 2.5, "c": 3.5, "v": 11},
        ],
    }
    prev_payload = {
        "status": "OK",
        "results": {
            "t": _ms(datetime(2025, 1, 10, tzinfo=UTC)),
            "o": 3,
            "h": 4,
            "l": 2.5,
            "c": 3.5,
            "v": 11,
        },
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        nonlocal open_close_calls
        if "/v2/aggs/ticker/" in url and "/range/1/day/" in url:
            return _StubResponse(status_code=200, content=json.dumps(range_payload).encode("utf-8"), text="OK")
        if url.endswith("/prev"):
            return _StubResponse(status_code=200, content=json.dumps(prev_payload).encode("utf-8"), text="OK")
        if "/v1/open-close/" in url:
            open_close_calls += 1
            return _StubResponse(status_code=500, content=b"{}", text="nope")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-15")

    assert open_close_calls == 0
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "QUALITY_DEGRADED"
    assert result.data is not None
    assert result.data.quality_flags["midgap_fill_attempted"] is False


def test_fetch_price_series_handles_open_close_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    range_payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            {"t": _ms(datetime(2025, 1, 3, tzinfo=UTC)), "o": 3, "h": 4, "l": 2.5, "c": 3.5, "v": 11},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        if "/v2/aggs/ticker/" in url and "/range/1/day/" in url:
            return _StubResponse(status_code=200, content=json.dumps(range_payload).encode("utf-8"), text="OK")
        if "/v1/open-close/" in url:
            return _StubResponse(status_code=500, content=b"{}", text="nope")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-03")

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert len(result.data.bars) == 2
    assert result.data.quality_flags["midgap_fill_attempted"] is True
    assert result.data.quality_flags["midgap_fill_failed"] >= 1


def test_fetch_price_series_reason_code_midgap_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    range_payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            {"t": _ms(datetime(2025, 1, 5, tzinfo=UTC)), "o": 3, "h": 4, "l": 2.5, "c": 3.5, "v": 11},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        if "/v2/aggs/ticker/" in url and "/range/1/day/" in url:
            return _StubResponse(status_code=200, content=json.dumps(range_payload).encode("utf-8"), text="OK")
        if url.endswith("/prev"):
            return _StubResponse(status_code=500, content=b"{}", text="nope")
        if "/v1/open-close/" in url:
            requested_date = url.split("/")[-1]
            if requested_date == "2025-01-02":
                oc_payload = {
                    "status": "OK",
                    "open": 2,
                    "high": 3,
                    "low": 1.5,
                    "close": 2.5,
                    "volume": 12,
                }
                return _StubResponse(status_code=200, content=json.dumps(oc_payload).encode("utf-8"), text="OK")
            return _StubResponse(status_code=404, content=b"{}", text="not found")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-10")

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "QUALITY_DEGRADED_MIDGAP_USED"


def test_fetch_price_series_reason_code_prev_midgap_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    range_payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            {"t": _ms(datetime(2025, 1, 5, tzinfo=UTC)), "o": 3, "h": 4, "l": 2.5, "c": 3.5, "v": 11},
        ],
    }
    prev_payload = {
        "status": "OK",
        "results": {
            "t": _ms(datetime(2025, 1, 15, tzinfo=UTC)),
            "o": 8,
            "h": 9,
            "l": 7,
            "c": 8.5,
            "v": 25,
        },
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        if "/v2/aggs/ticker/" in url and "/range/1/day/" in url:
            return _StubResponse(status_code=200, content=json.dumps(range_payload).encode("utf-8"), text="OK")
        if url.endswith("/prev"):
            return _StubResponse(status_code=200, content=json.dumps(prev_payload).encode("utf-8"), text="OK")
        if "/v1/open-close/" in url:
            requested_date = url.split("/")[-1]
            if requested_date == "2025-01-02":
                oc_payload = {
                    "status": "OK",
                    "open": 2,
                    "high": 3,
                    "low": 1.5,
                    "close": 2.5,
                    "volume": 12,
                }
                return _StubResponse(status_code=200, content=json.dumps(oc_payload).encode("utf-8"), text="OK")
            return _StubResponse(status_code=404, content=b"{}", text="not found")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-20")

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "QUALITY_DEGRADED_PREV_MIDGAP_USED"
    assert result.data is not None
    assert result.data.quality_flags["gapfill_prev_used"] is True
    assert result.data.quality_flags["midgap_fill_count"] == 1


def test_assess_completeness_rejects_invalid_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {
        "status": "OK",
        "results": [
            {"t": 0, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "1970-01-01", "1970-01-01")
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.quality_flags["completeness"] is False


def test_assess_completeness_rejects_negative_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": -100},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-01")
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.quality_flags["completeness"] is False


def test_assess_completeness_rejects_nan_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {
        "status": "OK",
        "results": [
            {
                "t": _ms(datetime(2025, 1, 1, tzinfo=UTC)),
                "o": 1,
                "h": 2,
                "l": 0.5,
                "c": "nan",
                "v": 10,
            },
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-01")
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.quality_flags["completeness"] is False


def test_assess_completeness_rejects_high_less_than_low(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 10, "l": 15, "c": 1.5, "v": 10},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-01")
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.quality_flags["completeness"] is False


def test_assess_completeness_accepts_valid_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        return _StubResponse(status_code=200, content=json.dumps(payload).encode("utf-8"), text="OK")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-01")
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.quality_flags["completeness"] is True


def test_fetch_price_series_uses_prev_gapfill_when_recent_empty_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    from data import polygon_adapter as polygon_module

    class _FakeDate(date):
        @classmethod
        def today(cls) -> date:  # type: ignore[override]
            return date(2025, 1, 10)

    monkeypatch.setattr(polygon_module, "date", _FakeDate)

    range_payload = {"status": "OK", "results": []}
    prev_payload = {
        "status": "OK",
        "results": {
            "t": _ms(datetime(2025, 1, 5, tzinfo=UTC)),
            "o": 8,
            "h": 9,
            "l": 7,
            "c": 8.5,
            "v": 25,
        },
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        if "/v2/aggs/ticker/" in url and "/range/1/day/" in url:
            return _StubResponse(status_code=200, content=json.dumps(range_payload).encode("utf-8"), text="OK")
        if url.endswith("/prev"):
            return _StubResponse(status_code=200, content=json.dumps(prev_payload).encode("utf-8"), text="OK")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_StubLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-10")

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "QUALITY_DEGRADED_PREV_USED"
    assert result.data is not None
    assert result.data.quality_flags["gapfill_prev_used"] is True


def test_fetch_price_series_midgap_fill_open_close_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    class _SequenceLimiter:
        def __init__(self) -> None:
            self.calls: list[float | None] = []
            self._results = iter([True, False])

        def acquire(self, timeout_seconds: float | None = None) -> bool:
            self.calls.append(timeout_seconds)
            return next(self._results, True)

    range_payload = {
        "status": "OK",
        "results": [
            {"t": _ms(datetime(2025, 1, 1, tzinfo=UTC)), "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            {"t": _ms(datetime(2025, 1, 3, tzinfo=UTC)), "o": 3, "h": 4, "l": 2.5, "c": 3.5, "v": 11},
        ],
    }

    def fake_get(url: str, params: dict[str, Any], timeout: float) -> _StubResponse:
        if "/v2/aggs/ticker/" in url and "/range/1/day/" in url:
            return _StubResponse(status_code=200, content=json.dumps(range_payload).encode("utf-8"), text="OK")
        raise AssertionError(f"requests.get should not be called for URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    adapter = PolygonDataAdapter(rate_limiter=_SequenceLimiter())
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-03")

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.quality_flags["midgap_fill_attempted"] is True
    assert result.data.quality_flags["midgap_fill_failed"] >= 1
    assert result.data.quality_flags["open_close_status"] == "RATE_LIMIT"
