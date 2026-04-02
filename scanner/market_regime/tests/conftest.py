from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from data.interface import PriceBar, PriceSeriesSnapshot


def _now_ns() -> int:
    return int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)


def make_price_bar(
    *,
    ts: int,
    close: float,
    high: float | None = None,
    low: float | None = None,
    open: float | None = None,
    volume: int = 1_000_000,
) -> PriceBar:
    c = float(close)
    return PriceBar(
        timestamp=ts,
        open=float(open if open is not None else c),
        high=float(high if high is not None else c * 1.01),
        low=float(low if low is not None else c * 0.99),
        close=c,
        volume=int(volume),
    )


def make_series_snapshot(symbol: str, closes: list[float], *, base_ts: int = 1_700_000_000_000_000_000) -> PriceSeriesSnapshot:
    bars = [make_price_bar(ts=base_ts + i, close=close) for i, close in enumerate(closes)]
    return PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=_now_ns(),
        symbol=symbol,
        timeframe="1d",
        bars=bars,
        source="test",
        quality_flags={},
    )


@dataclass(frozen=True, slots=True)
class DummyBar:
    close: float
    volume: float


@pytest.fixture()
def confirmation_bars() -> list[DummyBar]:
    return [
        DummyBar(close=99.0, volume=100.0),
        DummyBar(close=100.0, volume=100.0),
        DummyBar(close=101.0, volume=250.0),
        DummyBar(close=102.0, volume=260.0),
    ]

