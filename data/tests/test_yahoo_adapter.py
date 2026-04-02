from __future__ import annotations

import builtins
import sys
from datetime import UTC, datetime
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from common.interface import ResultStatus
from data.yahoo_adapter import YahooDataAdapter


class _StubTicker:
    def __init__(self, df: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self._df = df
        self._error = error

    def history(self, **kwargs: Any) -> pd.DataFrame:
        if self._error is not None:
            raise self._error
        assert self._df is not None
        return self._df


def _install_stub_yfinance(monkeypatch: pytest.MonkeyPatch, ticker: _StubTicker) -> None:
    module = ModuleType("yfinance")

    def Ticker(symbol: str) -> _StubTicker:  # noqa: N802 - mimics yfinance API
        return ticker

    module.Ticker = Ticker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", module)


def test_yahoo_adapter_initialization() -> None:
    adapter = YahooDataAdapter()
    assert adapter.get_source_name() == "yahoo"


def test_fetch_price_series_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.DatetimeIndex([datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 2, tzinfo=UTC)])
    df = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [2.0, 3.0], "Low": [0.5, 1.5], "Close": [1.5, 2.5], "Volume": [10, 12]},
        index=idx,
    )
    _install_stub_yfinance(monkeypatch, _StubTicker(df=df))

    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("aapl", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.symbol == "AAPL"
    assert result.data.source == "yahoo"
    assert len(result.data.bars) == 2
    assert result.data.quality_flags["availability"] is True
    assert result.data.quality_flags["completeness"] is True
    assert result.data.quality_flags["freshness"] is True


def test_fetch_price_series_yfinance_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yfinance":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "MISSING_DEPENDENCY"


def test_fetch_price_series_yahoo_errors_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_yfinance(monkeypatch, _StubTicker(error=RuntimeError("boom")))
    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NETWORK_ERROR"


def test_fetch_price_series_quality_degraded_when_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.DatetimeIndex([datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 2, tzinfo=UTC)])
    df = pd.DataFrame(
        {"Open": [1.0, "bad"], "High": [2.0, 3.0], "Low": [0.5, 1.5], "Close": [1.5, 2.5], "Volume": [10, 12]},
        index=idx,
    )
    _install_stub_yfinance(monkeypatch, _StubTicker(df=df))

    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.quality_flags["completeness"] is False


def test_fetch_price_series_invalid_range_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.DatetimeIndex([datetime(2025, 1, 1, tzinfo=UTC)])
    df = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]}, index=idx)
    _install_stub_yfinance(monkeypatch, _StubTicker(df=df))

    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("AAPL", "2025-01-02", "2025-01-01")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_RANGE"


def test_fetch_price_series_empty_response_fails_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    _install_stub_yfinance(monkeypatch, _StubTicker(df=df))

    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-02")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NO_DATA"


def test_fetch_price_series_timestamp_fallback_from_date_index(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.Index([datetime(2025, 1, 1).date()], dtype=object)
    df = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=idx,
    )
    _install_stub_yfinance(monkeypatch, _StubTicker(df=df))

    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-01")
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert isinstance(result.data.bars[0].timestamp, int)


def test_fetch_price_series_timestamp_from_index_type_error_fails_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.Index(["bad-index"], dtype=object)
    df = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=idx,
    )
    _install_stub_yfinance(monkeypatch, _StubTicker(df=df))

    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-01")
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NO_DATA"


def test_fetch_price_series_timestamp_fallback_from_naive_datetime_index(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.Index([datetime(2025, 1, 1, 12, 0, 0)], dtype=object)
    df = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=idx,
    )
    _install_stub_yfinance(monkeypatch, _StubTicker(df=df))

    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("AAPL", "2025-01-01", "2025-01-01")
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert isinstance(result.data.bars[0].timestamp, int)


def test_fetch_price_series_quality_degraded_when_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.DatetimeIndex([datetime(2020, 1, 1, tzinfo=UTC)])
    df = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=idx,
    )
    _install_stub_yfinance(monkeypatch, _StubTicker(df=df))

    adapter = YahooDataAdapter()
    result = adapter.fetch_price_series("AAPL", "2020-01-01", "2020-01-10")
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.quality_flags["freshness"] is False
