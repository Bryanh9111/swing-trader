from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import requests

import event_guard.sources as sources_module
from common.interface import Result, ResultStatus
from event_guard.interface import EventGuardConfig, EventSnapshot, EventType, MarketEvent, RiskLevel


def _utc_midnight_ns(yyyy_mm_dd: str) -> int:
    dt = datetime.strptime(yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


@dataclass(slots=True)
class _StubEarning:
    ticker: str | None
    date: str | None
    time: str | None = None
    fiscal_period: str | None = None
    fiscal_year: str | None = None
    importance: str | None = None
    date_status: str | None = None
    company_name: str | None = None
    currency: str | None = None
    benzinga_id: str | None = None


@dataclass(slots=True)
class _StubDividend:
    ticker: str | None
    ex_dividend_date: str | None
    pay_date: str | None = None
    record_date: str | None = None
    declaration_date: str | None = None
    frequency: str | None = None
    cash_amount: str | None = None
    dividend_type: str | None = None
    currency: str | None = None


@dataclass(slots=True)
class _StubSplit:
    ticker: str | None
    execution_date: str | None
    split_from: int | None = None
    split_to: int | None = None


class _StubPolygonClient:
    def __init__(self, earnings: list[_StubEarning], dividends: list[_StubDividend], splits: list[_StubSplit]) -> None:
        self.earnings = earnings
        self.dividends = dividends
        self.splits = splits
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_benzinga_earnings(self, **kwargs: Any) -> list[_StubEarning]:
        self.calls.append(("earnings", kwargs))
        return list(self.earnings)

    def list_dividends(self, **kwargs: Any) -> list[_StubDividend]:
        self.calls.append(("dividends", kwargs))
        return list(self.dividends)

    def list_splits(self, **kwargs: Any) -> list[_StubSplit]:
        self.calls.append(("splits", kwargs))
        return list(self.splits)


def test_polygon_event_source_fetch_events_happy_path_normalizes_events(monkeypatch) -> None:
    """Fetch Polygon earnings/dividends/splits and normalize into MarketEvent list."""

    start_ns = _utc_midnight_ns("2025-01-01")
    end_ns = _utc_midnight_ns("2025-01-10")
    in_range = "2025-01-02"
    out_of_range = "2025-02-01"

    client = _StubPolygonClient(
        earnings=[
            _StubEarning(ticker="aapl", date=in_range, time="amc", benzinga_id="123"),
            _StubEarning(ticker=None, date=in_range),
            _StubEarning(ticker="msft", date=out_of_range),
        ],
        dividends=[
            _StubDividend(ticker="AAPL", ex_dividend_date=in_range, cash_amount="0.25", currency="USD"),
        ],
        splits=[
            _StubSplit(ticker="AAPL", execution_date=in_range, split_from=1, split_to=4),
            _StubSplit(ticker="AAPL", execution_date=in_range, split_from=10, split_to=1),
        ],
    )

    monkeypatch.setattr(sources_module, "HAS_MASSIVE_SDK", True)
    monkeypatch.setattr(sources_module.PolygonEventSource, "_build_client", lambda _self: client)

    source = sources_module.PolygonEventSource(api_key="test-key")
    result = source.fetch_events([" aapl "], start_ns, end_ns)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.source == "polygon"
    assert result.data.symbols_covered == ["AAPL"]

    events = result.data.events
    assert [e.symbol for e in events] == ["AAPL", "AAPL", "AAPL", "AAPL"]
    assert {e.event_type for e in events} == {
        EventType.EARNINGS,
        EventType.DIVIDEND,
        EventType.STOCK_SPLIT,
        EventType.REVERSE_SPLIT,
    }
    earnings = next(e for e in events if e.event_type is EventType.EARNINGS)
    assert earnings.risk_level is RiskLevel.HIGH
    assert earnings.metadata is not None and earnings.metadata.get("provider_event_id") == "123"

    dividend = next(e for e in events if e.event_type is EventType.DIVIDEND)
    assert dividend.risk_level is RiskLevel.LOW
    assert dividend.metadata is not None and dividend.metadata.get("cash_amount") == "0.25"

    reverse_split = next(e for e in events if e.event_type is EventType.REVERSE_SPLIT)
    assert reverse_split.risk_level is RiskLevel.HIGH
    assert reverse_split.metadata is not None and reverse_split.metadata["split_from"] == "10"

    assert any(name == "earnings" for name, _ in client.calls)
    assert any(name == "dividends" for name, _ in client.calls)
    assert any(name == "splits" for name, _ in client.calls)


def test_polygon_event_source_fetch_events_filters_empty_symbols_without_api_key(monkeypatch) -> None:
    """Return success with empty snapshot when no symbols provided."""

    monkeypatch.setattr(sources_module, "HAS_MASSIVE_SDK", True)
    monkeypatch.setattr(sources_module.PolygonEventSource, "_build_client", lambda _self: None)

    source = sources_module.PolygonEventSource(api_key="")
    result = source.fetch_events(["", "   "], start_ns=1, end_ns=2)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.symbols_covered == []
    assert result.data.events == []


def test_polygon_event_source_fetch_events_missing_api_key_fails(monkeypatch) -> None:
    """Reject Polygon calls when api_key is missing."""

    monkeypatch.setattr(sources_module, "HAS_MASSIVE_SDK", True)
    monkeypatch.setattr(sources_module.PolygonEventSource, "_build_client", lambda _self: object())

    source = sources_module.PolygonEventSource(api_key="")
    result = source.fetch_events(["AAPL"], start_ns=1, end_ns=2)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "MISSING_API_KEY"


def test_polygon_event_source_fetch_events_invalid_range_fails(monkeypatch) -> None:
    """Reject invalid time ranges (start_ns > end_ns)."""

    monkeypatch.setattr(sources_module, "HAS_MASSIVE_SDK", True)
    monkeypatch.setattr(sources_module.PolygonEventSource, "_build_client", lambda _self: object())

    source = sources_module.PolygonEventSource(api_key="key")
    result = source.fetch_events(["AAPL"], start_ns=10, end_ns=1)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_RANGE"


def test_polygon_event_source_fetch_events_not_available_fails(monkeypatch) -> None:
    """Degrade to NOT_AVAILABLE when Massive SDK client is unavailable."""

    monkeypatch.setattr(sources_module, "HAS_MASSIVE_SDK", True)
    monkeypatch.setattr(sources_module.PolygonEventSource, "_build_client", lambda _self: None)

    source = sources_module.PolygonEventSource(api_key="key")
    result = source.fetch_events(["AAPL"], start_ns=1, end_ns=2)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NOT_AVAILABLE"


@pytest.mark.parametrize(
    ("exc_name", "expected_reason"),
    [("AuthError", "MISSING_API_KEY"), ("BadResponse", "HTTP_ERROR"), ("RuntimeError", "HTTP_ERROR")],
)
def test_polygon_event_source_fetch_events_sdk_errors_map_to_reason_codes(
    monkeypatch, exc_name: str, expected_reason: str
) -> None:
    """Map SDK exceptions into stable reason_code values for Polygon source."""

    monkeypatch.setattr(sources_module, "HAS_MASSIVE_SDK", True)
    monkeypatch.setattr(sources_module.PolygonEventSource, "_build_client", lambda _self: object())

    def _raiser(*_: Any, **__: Any) -> list[MarketEvent]:
        exc_type = type(exc_name, (Exception,), {})
        raise exc_type("boom")  # type: ignore[misc]

    monkeypatch.setattr(sources_module.PolygonEventSource, "_fetch_dividends", _raiser)

    source = sources_module.PolygonEventSource(api_key="key")
    result = source.fetch_events(["AAPL"], start_ns=_utc_midnight_ns("2025-01-01"), end_ns=_utc_midnight_ns("2025-01-02"))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == expected_reason


def test_polygon_rest_event_source_fetch_events_parses_dividends_and_splits(monkeypatch) -> None:
    start_ns = _utc_midnight_ns("2025-01-01")
    end_ns = _utc_midnight_ns("2025-01-10")

    calls: list[tuple[str, dict[str, Any]]] = []

    class _Response:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    def _fake_get(url: str, *, params: dict[str, Any] | None = None, timeout: float | None = None) -> _Response:
        assert timeout is not None
        assert params is not None and params.get("apiKey") == "test-key"
        calls.append((url, dict(params)))

        if url.endswith("/stocks/v1/dividends"):
            assert "ex_dividend_date.gte" in params and "ex_dividend_date.lte" in params
            return _Response(
                {
                    "results": [
                        {
                            "ticker": "aapl",
                            "ex_dividend_date": "2025-01-02",
                            "cash_amount": 0.25,
                            "currency": "USD",
                            "pay_date": "2025-01-10",
                            "distribution_type": "CD",
                        },
                        # Duplicate (should be deduped)
                        {
                            "ticker": "AAPL",
                            "ex_dividend_date": "2025-01-02",
                            "cash_amount": 0.25,
                            "currency": "USD",
                            "pay_date": "2025-01-10",
                            "distribution_type": "CD",
                        },
                        # Out of range (should be filtered)
                        {"ticker": "AAPL", "ex_dividend_date": "2025-02-01", "cash_amount": 0.5},
                    ],
                    "status": "OK",
                }
            )

        if url.endswith("/stocks/v1/splits"):
            assert "execution_date.gte" in params and "execution_date.lte" in params
            return _Response(
                {
                    "results": [
                        {
                            "ticker": "AAPL",
                            "execution_date": "2025-01-02",
                            "split_from": 1,
                            "split_to": 4,
                            "adjustment_type": "forward_split",
                            "historical_adjustment_factor": 0.25,
                        },
                        {
                            "ticker": "AAPL",
                            "execution_date": "2025-01-03",
                            "split_from": 10,
                            "split_to": 1,
                            "adjustment_type": "reverse_split",
                            "historical_adjustment_factor": 10.0,
                        },
                    ],
                    "status": "OK",
                }
            )

        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(sources_module.requests, "get", _fake_get)

    source = sources_module.PolygonRestEventSource(api_key="test-key")
    result = source.fetch_events([" aapl "], start_ns, end_ns)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.source == "polygon_rest"
    assert result.data.symbols_covered == ["AAPL"]

    events = result.data.events
    assert {e.event_type for e in events} == {EventType.DIVIDEND, EventType.STOCK_SPLIT, EventType.REVERSE_SPLIT}
    assert [e.symbol for e in events] == ["AAPL", "AAPL", "AAPL"]

    dividend = next(e for e in events if e.event_type is EventType.DIVIDEND)
    assert dividend.risk_level is RiskLevel.LOW
    assert dividend.metadata is not None
    assert dividend.metadata["cash_amount"] == "0.25"
    assert dividend.metadata["distribution_type"] == "CD"

    reverse_split = next(e for e in events if e.event_type is EventType.REVERSE_SPLIT)
    assert reverse_split.risk_level is RiskLevel.HIGH
    assert reverse_split.metadata is not None
    assert reverse_split.metadata["split_from"] == "10"

    assert any(url.endswith("/stocks/v1/dividends") for url, _ in calls)
    assert any(url.endswith("/stocks/v1/splits") for url, _ in calls)


def test_polygon_rest_event_source_request_exception_continues_other_batches(monkeypatch) -> None:
    start_ns = _utc_midnight_ns("2025-01-01")
    end_ns = _utc_midnight_ns("2025-01-10")

    calls: list[tuple[str, dict[str, Any]]] = []

    class _Response:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    def _fake_get(url: str, *, params: dict[str, Any] | None = None, timeout: float | None = None) -> _Response:
        assert params is not None
        calls.append((url, dict(params)))

        if url.endswith("/stocks/v1/dividends"):
            ticker_any_of = str(params.get("ticker.any_of") or "")
            if "FAIL" in ticker_any_of:
                raise requests.RequestException("boom")
            return _Response({"results": [{"ticker": "OK", "ex_dividend_date": "2025-01-02", "cash_amount": 1.0}]})

        if url.endswith("/stocks/v1/splits"):
            return _Response({"results": []})

        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(sources_module.requests, "get", _fake_get)

    symbols = [f"FAIL{i}" for i in range(50)] + ["OK"]
    source = sources_module.PolygonRestEventSource(api_key="test-key")
    result = source.fetch_events(symbols, start_ns, end_ns)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert any(e.symbol == "OK" and e.event_type is EventType.DIVIDEND for e in result.data.events)
    assert len([c for c in calls if c[0].endswith("/stocks/v1/dividends")]) >= 2


def test_polygon_rest_event_source_missing_api_key_fails() -> None:
    source = sources_module.PolygonRestEventSource(api_key="")
    result = source.fetch_events(["AAPL"], start_ns=1, end_ns=2)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "MISSING_API_KEY"


def test_polygon_rest_event_source_invalid_range_fails() -> None:
    source = sources_module.PolygonRestEventSource(api_key="key")
    result = source.fetch_events(["AAPL"], start_ns=10, end_ns=1)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_RANGE"


def test_manual_file_source_reads_csv_and_applies_symbol_and_time_filters(tmp_path: Path) -> None:
    """Load events from CSV, normalize symbols, and filter by time range and symbols."""

    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "\n".join(
            [
                "symbol,event_type,event_date,risk_level,source",
                "aapl,EARNINGS,2025-01-02,HIGH,MANUAL",
                "AAPL,DIVIDEND,2025-01-08,,manual",
                "MSFT,EARNINGS,2025-01-02,HIGH,manual",
                "AAPL,EARNINGS,2025-03-01,HIGH,manual",
                ",EARNINGS,2025-01-02,HIGH,manual",
            ]
        ),
        encoding="utf-8",
    )

    source = sources_module.ManualFileSource(str(csv_path))
    start_ns = _utc_midnight_ns("2025-01-01")
    end_ns = _utc_midnight_ns("2025-01-10")
    result = source.fetch_events(["aapl"], start_ns, end_ns)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.source == "manual"
    assert result.data.symbols_covered == ["AAPL"]
    assert [e.event_type for e in result.data.events] == [EventType.EARNINGS, EventType.DIVIDEND]
    assert result.data.events[0].symbol == "AAPL"
    assert result.data.events[0].source == "manual"

    dividend = next(e for e in result.data.events if e.event_type is EventType.DIVIDEND)
    assert dividend.risk_level is RiskLevel.LOW


def test_manual_file_source_reads_json_and_normalizes_metadata(tmp_path: Path) -> None:
    """Load events from JSON list payload and keep metadata map entries."""

    json_path = tmp_path / "events.json"
    payload = [
        {
            "symbol": "aapl",
            "event_type": "EARNINGS",
            "event_date": "2025-01-02T13:30:00Z",
            "risk_level": "HIGH",
            "source": "MANUAL",
            "metadata": {"provider_event_id": "abc", "empty": " ", "skip": None},
        },
        {"symbol": "aapl", "event_type": "INVALID", "event_date": "2025-01-03", "risk_level": "INVALID"},
        "skip-me",
    ]
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    source = sources_module.ManualFileSource(str(json_path))
    start_ns = _utc_midnight_ns("2025-01-01")
    end_ns = _utc_midnight_ns("2025-01-10")
    result = source.fetch_events(["AAPL"], start_ns, end_ns)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert len(result.data.events) == 2

    first = result.data.events[0]
    assert first.symbol == "AAPL"
    assert first.source == "manual"
    assert first.metadata is not None and first.metadata == {"provider_event_id": "abc"}

    second = result.data.events[1]
    assert second.event_type is EventType.OTHER
    assert second.risk_level is RiskLevel.MEDIUM


def test_manual_file_source_missing_file_fails(tmp_path: Path) -> None:
    """Return FILE_NOT_FOUND when manual events file does not exist."""

    source = sources_module.ManualFileSource(str(tmp_path / "missing.csv"))
    result = source.fetch_events(["AAPL"], start_ns=1, end_ns=2)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "FILE_NOT_FOUND"


def test_manual_file_source_unsupported_format_fails(tmp_path: Path) -> None:
    """Reject unknown file formats with UNSUPPORTED_FORMAT."""

    path = tmp_path / "events.txt"
    path.write_text("x", encoding="utf-8")
    source = sources_module.ManualFileSource(str(path))
    result = source.fetch_events(["AAPL"], start_ns=1, end_ns=2)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "UNSUPPORTED_FORMAT"


def test_manual_file_source_invalid_json_returns_decode_error(tmp_path: Path) -> None:
    """Return DECODE_ERROR when JSON payload cannot be decoded."""

    path = tmp_path / "events.json"
    path.write_text("{not-json", encoding="utf-8")
    source = sources_module.ManualFileSource(str(path))
    result = source.fetch_events(["AAPL"], start_ns=1, end_ns=2)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "DECODE_ERROR"


def test_manual_file_source_csv_decode_error_returns_decode_error(monkeypatch, tmp_path: Path) -> None:
    """Return DECODE_ERROR when CSV parsing fails."""

    path = tmp_path / "events.csv"
    path.write_text("symbol,event_type,event_date\nAAPL,EARNINGS,2025-01-01\n", encoding="utf-8")

    class _BadReader:
        fieldnames = ["symbol", "event_type", "event_date"]

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def __iter__(self):
            raise sources_module.csv.Error("bad csv")  # type: ignore[misc]

    monkeypatch.setattr(sources_module.csv, "DictReader", _BadReader)
    source = sources_module.ManualFileSource(str(path))
    result = source.fetch_events(
        ["AAPL"], start_ns=_utc_midnight_ns("2025-01-01"), end_ns=_utc_midnight_ns("2025-01-02")
    )

    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "DECODE_ERROR"


def test_manual_file_source_format_errors_return_read_error(tmp_path: Path) -> None:
    """Return READ_ERROR when JSON shape is wrong or CSV headers are missing."""

    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    source = sources_module.ManualFileSource(str(bad_json_path))
    result = source.fetch_events(["AAPL"], start_ns=1, end_ns=2)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "READ_ERROR"

    bad_csv_path = tmp_path / "bad.csv"
    bad_csv_path.write_text("", encoding="utf-8")
    source2 = sources_module.ManualFileSource(str(bad_csv_path))
    result2 = source2.fetch_events(["AAPL"], start_ns=1, end_ns=2)
    assert result2.status is ResultStatus.FAILED
    assert result2.reason_code == "READ_ERROR"


def test_fetch_events_with_fallback_polygon_success_and_manual_overlay(monkeypatch) -> None:
    """Overlay manual events on Polygon results with manual taking precedence."""

    polygon_event = MarketEvent(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date=_utc_midnight_ns("2025-01-02"),
        risk_level=RiskLevel.HIGH,
        source="polygon",
    )
    manual_override = MarketEvent(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date=_utc_midnight_ns("2025-01-02"),
        risk_level=RiskLevel.CRITICAL,
        source="manual",
    )
    manual_extra = MarketEvent(
        symbol="MSFT",
        event_type=EventType.DIVIDEND,
        event_date=_utc_midnight_ns("2025-01-03"),
        risk_level=RiskLevel.LOW,
        source="manual",
    )

    polygon_snapshot = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        events=[polygon_event],
        source="polygon",
        symbols_covered=["AAPL", "MSFT"],
    )
    manual_snapshot = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        events=[manual_override, manual_extra],
        source="manual",
        symbols_covered=["AAPL", "MSFT"],
    )

    class _StubPolygon:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def fetch_events(self, _symbols: list[str], _start_ns: int, _end_ns: int) -> Result[EventSnapshot]:
            return Result.success(polygon_snapshot, reason_code="OK")

    class _StubManual:
        def __init__(self, file_path: str) -> None:
            self.file_path = file_path

        def fetch_events(self, _symbols: list[str], _start_ns: int, _end_ns: int) -> Result[EventSnapshot]:
            return Result.success(manual_snapshot, reason_code="OK")

    monkeypatch.setattr(sources_module, "PolygonEventSource", _StubPolygon)
    monkeypatch.setattr(sources_module, "ManualFileSource", _StubManual)
    monkeypatch.setenv("POLYGON_API_KEY", "key")

    config = EventGuardConfig(primary_source="polygon", manual_events_file="manual.csv")
    result = sources_module.fetch_events_with_fallback(config, ["AAPL", "MSFT"], start_ns=1, end_ns=2)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.source == "merged"
    assert result.data.symbols_covered == ["AAPL", "MSFT"]

    merged = {(e.symbol, e.event_type, e.event_date): e for e in result.data.events}
    assert merged[("AAPL", EventType.EARNINGS, _utc_midnight_ns("2025-01-02"))].risk_level is RiskLevel.CRITICAL
    assert ("MSFT", EventType.DIVIDEND, _utc_midnight_ns("2025-01-03")) in merged


def test_fetch_events_with_fallback_polygon_failure_falls_back_to_manual(monkeypatch) -> None:
    """Return DEGRADED manual snapshot when Polygon fails but manual succeeds."""

    manual_snapshot = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        events=[],
        source="manual",
        symbols_covered=["AAPL"],
    )

    class _StubPolygon:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def fetch_events(self, _symbols: list[str], _start_ns: int, _end_ns: int) -> Result[EventSnapshot]:
            return Result.failed(RuntimeError("down"), reason_code="HTTP_ERROR")

    class _StubManual:
        def __init__(self, file_path: str) -> None:
            self.file_path = file_path

        def fetch_events(self, _symbols: list[str], _start_ns: int, _end_ns: int) -> Result[EventSnapshot]:
            return Result.success(manual_snapshot, reason_code="OK")

    monkeypatch.setattr(sources_module, "PolygonEventSource", _StubPolygon)
    monkeypatch.setattr(sources_module, "ManualFileSource", _StubManual)
    monkeypatch.setenv("POLYGON_API_KEY", "key")

    config = EventGuardConfig(primary_source="polygon", manual_events_file="manual.csv")
    result = sources_module.fetch_events_with_fallback(config, ["AAPL"], start_ns=1, end_ns=2)

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "FALLBACK_MANUAL"
    assert result.data == manual_snapshot
    assert isinstance(result.error, RuntimeError)


def test_fetch_events_with_fallback_polygon_success_manual_failed_returns_degraded(monkeypatch) -> None:
    """Return DEGRADED Polygon snapshot when manual overlay fails."""

    polygon_snapshot = EventSnapshot(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        events=[],
        source="polygon",
        symbols_covered=["AAPL"],
    )

    class _StubPolygon:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def fetch_events(self, _symbols: list[str], _start_ns: int, _end_ns: int) -> Result[EventSnapshot]:
            return Result.success(polygon_snapshot, reason_code="OK")

    class _StubManual:
        def __init__(self, file_path: str) -> None:
            self.file_path = file_path

        def fetch_events(self, _symbols: list[str], _start_ns: int, _end_ns: int) -> Result[EventSnapshot]:
            return Result.failed(ValueError("bad manual"), reason_code="READ_ERROR")

    monkeypatch.setattr(sources_module, "PolygonEventSource", _StubPolygon)
    monkeypatch.setattr(sources_module, "ManualFileSource", _StubManual)
    monkeypatch.setenv("POLYGON_API_KEY", "key")

    config = EventGuardConfig(primary_source="polygon", manual_events_file="manual.csv")
    result = sources_module.fetch_events_with_fallback(config, ["AAPL"], start_ns=1, end_ns=2)

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "MANUAL_OVERRIDE_FAILED"
    assert result.data == polygon_snapshot


def test_fetch_events_with_fallback_both_fail_returns_degraded_empty(monkeypatch) -> None:
    """Return DEGRADED empty snapshot when both Polygon and manual fail."""

    class _StubPolygon:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def fetch_events(self, _symbols: list[str], _start_ns: int, _end_ns: int) -> Result[EventSnapshot]:
            return Result.failed(RuntimeError("down"), reason_code="HTTP_ERROR")

    class _StubManual:
        def __init__(self, file_path: str) -> None:
            self.file_path = file_path

        def fetch_events(self, _symbols: list[str], _start_ns: int, _end_ns: int) -> Result[EventSnapshot]:
            return Result.failed(FileNotFoundError("missing"), reason_code="FILE_NOT_FOUND")

    monkeypatch.setattr(sources_module, "PolygonEventSource", _StubPolygon)
    monkeypatch.setattr(sources_module, "ManualFileSource", _StubManual)
    monkeypatch.setenv("POLYGON_API_KEY", "key")

    config = EventGuardConfig(primary_source="polygon", manual_events_file="manual.csv")
    result = sources_module.fetch_events_with_fallback(config, [" aapl ", ""], start_ns=1, end_ns=2)

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "DEGRADED_EMPTY"
    assert result.data is not None
    assert result.data.source == "degraded"
    assert result.data.events == []
    assert result.data.symbols_covered == ["AAPL"]


def test_sources_private_helpers_chunked_and_parse_timestamp_ns() -> None:
    """Cover helper branches for chunking and timestamp parsing heuristics."""

    assert list(sources_module._chunked(["A", "B", "C"], size=2)) == [["A", "B"], ["C"]]
    with pytest.raises(ValueError):
        list(sources_module._chunked(["A"], size=0))

    assert sources_module._parse_timestamp_ns(1) == 1_000_000_000
    assert sources_module._parse_timestamp_ns(1_700_000_000_000) == 1_700_000_000_000_000_000
    # Numeric strings are treated as date-like by the current implementation.
    assert sources_module._parse_timestamp_ns("1700000000") is None

    class _StrLike:
        def __str__(self) -> str:
            return "1700000000"

    assert sources_module._parse_timestamp_ns(_StrLike()) == 1_700_000_000_000_000_000
    assert sources_module._parse_timestamp_ns("") is None
    assert sources_module._parse_timestamp_ns(0) is None
    assert sources_module._parse_timestamp_ns(-1) is None
