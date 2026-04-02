from __future__ import annotations

import sys
from types import ModuleType

import pandas as pd
import pytest

from data.vix_provider import VIXProvider


class _FakeTicker:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def history(self, *args: object, **kwargs: object) -> pd.DataFrame:
        return self._df


def _install_fake_yfinance(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame) -> None:
    fake = ModuleType("yfinance")

    def _ticker(symbol: str) -> _FakeTicker:
        assert symbol == "^VIX"
        return _FakeTicker(df)

    fake.Ticker = _ticker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def test_fetch_vix_history_builds_vixdata_and_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {"Close": [14.0, 15.0, 26.0, 36.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], utc=True),
    )
    _install_fake_yfinance(monkeypatch, df)

    provider = VIXProvider()
    history = provider.fetch_vix_history(days=4)
    assert len(history) == 4

    assert history[0].value == 14.0
    assert history[0].level == "low"
    assert history[1].level == "normal"
    assert history[2].level == "elevated"
    assert history[3].level == "extreme"

    assert history[0].change_pct == 0.0
    assert history[1].change_pct == pytest.approx((15.0 - 14.0) / 14.0)


def test_fetch_vix_current_returns_latest_point(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {"Close": [20.0, 22.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
    )
    _install_fake_yfinance(monkeypatch, df)

    provider = VIXProvider()
    current = provider.fetch_vix_current()
    assert current is not None
    assert current.value == 22.0
    assert current.change_pct == pytest.approx((22.0 - 20.0) / 20.0)


def test_fetch_vix_history_gracefully_handles_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def _import(name: str, *args: object, **kwargs: object) -> object:
        if name == "yfinance":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _import)
    provider = VIXProvider()
    assert provider.fetch_vix_history(days=5) == []

