from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from data.interface import PriceBar
from data.short_data_provider import ShortDataProvider, ShortInterestData, ShortVolumeData
from scanner.indicator_scoring import compute_indicator_score
from scanner.interface import IndicatorScoringConfig


def _bar(ts: datetime, *, close: float) -> PriceBar:
    return PriceBar(timestamp=int(ts.timestamp() * 1_000_000_000), open=close, high=close, low=close, close=close, volume=100)


def test_fetch_short_interest_falls_back_to_finra(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ShortDataProvider(polygon_api_key="test-key")

    call_order: list[str] = []

    def fake_polygon(ticker: str) -> ShortInterestData | None:
        call_order.append("polygon")
        return None

    def fake_finra(ticker: str) -> ShortInterestData | None:
        call_order.append("finra")
        return ShortInterestData(
            ticker=ticker,
            short_interest=100,
            shares_outstanding=1000,
            short_percent=0.1,
            short_ratio=1.0,
            settlement_date="2026-01-01",
        )

    monkeypatch.setattr(provider, "_fetch_short_interest_polygon", fake_polygon)
    monkeypatch.setattr(provider, "_fetch_short_interest_finra", fake_finra)

    result = provider.fetch_short_interest("aapl")
    assert result is not None
    assert result.ticker == "AAPL"
    assert call_order == ["polygon", "finra"]


def test_fetch_short_volume_falls_back_to_finra(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ShortDataProvider(polygon_api_key="test-key")

    call_order: list[str] = []

    def fake_polygon(ticker: str, *, days: int = 30) -> list[ShortVolumeData] | None:
        call_order.append("polygon")
        return None

    def fake_finra(ticker: str, *, days: int = 30) -> list[ShortVolumeData] | None:
        call_order.append("finra")
        return [
            ShortVolumeData(ticker=ticker, date="2026-01-02", short_volume=40, total_volume=100, short_volume_ratio=0.4),
        ]

    monkeypatch.setattr(provider, "_fetch_short_volume_polygon", fake_polygon)
    monkeypatch.setattr(provider, "_fetch_short_volume_finra", fake_finra)

    result = provider.fetch_short_volume("aapl", days=5)
    assert result is not None
    assert len(result) == 1
    assert call_order == ["polygon", "finra"]


def test_short_scoring_bullish_squeeze_potential() -> None:
    class _StubShortProvider:
        def fetch_short_interest(self, ticker: str) -> ShortInterestData | None:
            return ShortInterestData(
                ticker=ticker,
                short_interest=300,
                shares_outstanding=1000,
                short_percent=0.30,
                short_ratio=2.0,
                settlement_date="2026-01-01",
            )

        def fetch_short_volume(self, ticker: str, days: int = 30) -> list[ShortVolumeData] | None:
            return [
                ShortVolumeData(ticker=ticker, date="2026-01-02", short_volume=10, total_volume=100, short_volume_ratio=0.10),
            ]

    now = datetime(2026, 1, 12, tzinfo=UTC)
    bars = [_bar(now - timedelta(days=10 - idx), close=100 + idx) for idx in range(11)]

    config = IndicatorScoringConfig(
        enabled=True,
        combination_mode="weighted_avg",
        rsi_enabled=False,
        macd_enabled=False,
        atr_enabled=False,
        bbands_enabled=False,
        obv_enabled=False,
        short_enabled=True,
        short_weight=1.0,
        short_squeeze_threshold=0.20,
        kdj_enabled=False,
    )

    score, meta = compute_indicator_score(bars, config, symbol="AAPL", short_data_provider=_StubShortProvider())

    assert meta["indicators"]["short"]["score"] == pytest.approx(score)
    assert meta["short_data"]["short_squeeze_potential"] is True
    assert meta["short_data"]["short_interest_pct"] == pytest.approx(0.30)


def test_short_scoring_high_short_volume_bearish() -> None:
    class _StubShortProvider:
        def fetch_short_interest(self, ticker: str) -> ShortInterestData | None:
            return None

        def fetch_short_volume(self, ticker: str, days: int = 30) -> list[ShortVolumeData] | None:
            return [
                ShortVolumeData(ticker=ticker, date="2026-01-02", short_volume=50, total_volume=100, short_volume_ratio=0.50),
            ]

    now = datetime(2026, 1, 12, tzinfo=UTC)
    bars = [_bar(now - timedelta(days=10 - idx), close=100.0) for idx in range(11)]

    config = IndicatorScoringConfig(
        enabled=True,
        combination_mode="weighted_avg",
        rsi_enabled=False,
        macd_enabled=False,
        atr_enabled=False,
        bbands_enabled=False,
        obv_enabled=False,
        short_enabled=True,
        short_weight=1.0,
        kdj_enabled=False,
    )

    score, meta = compute_indicator_score(bars, config, symbol="AAPL", short_data_provider=_StubShortProvider())
    assert score == pytest.approx(0.375)
    assert meta["short_data"]["short_volume_ratio"] == pytest.approx(0.50)
