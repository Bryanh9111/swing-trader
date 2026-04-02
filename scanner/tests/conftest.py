from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pytest

from scanner.interface import ScannerConfig


@dataclass(frozen=True, slots=True)
class DummyPriceBar:
    open: float
    high: float
    low: float
    close: float
    volume: float


def make_bar(
    *,
    close: float,
    volume: float,
    high: float | None = None,
    low: float | None = None,
    open: float | None = None,
) -> DummyPriceBar:
    bar_open = close if open is None else open
    bar_high = close if high is None else high
    bar_low = close if low is None else low
    return DummyPriceBar(open=float(bar_open), high=float(bar_high), low=float(bar_low), close=float(close), volume=float(volume))


def make_bars(
    closes: Iterable[float],
    *,
    volume: float | Iterable[float],
    high_pct: float = 0.01,
    low_pct: float = 0.01,
) -> list[DummyPriceBar]:
    closes_list = list(closes)
    if isinstance(volume, (int, float)):
        volumes = [float(volume) for _ in closes_list]
    else:
        volumes = [float(v) for v in volume]
    assert len(volumes) == len(closes_list)

    bars: list[DummyPriceBar] = []
    for close, vol in zip(closes_list, volumes, strict=True):
        bars.append(
            make_bar(
                close=float(close),
                volume=float(vol),
                high=float(close) * (1.0 + high_pct),
                low=float(close) * (1.0 - low_pct),
                open=float(close),
            )
        )
    return bars


def make_platform_series(
    *,
    total_bars: int = 120,
    base_price: float = 100.0,
    high_pct: float = 0.01,
    low_pct: float = 0.01,
    prev_volume: float = 2_000_000.0,
    recent_volume_high: float = 1_500_000.0,
    recent_volume_low: float = 1_000_000.0,
) -> list[DummyPriceBar]:
    """Create a deterministic 'platform' series with gentle oscillation and volume contraction."""
    if total_bars < 10:
        raise ValueError("total_bars must be >= 10")

    closes: list[float] = []
    for idx in range(total_bars):
        wobble = ((idx % 5) - 2) * 0.02  # +/- 0.04 range
        closes.append(base_price + wobble)

    volumes: list[float] = []
    tail = min(60, total_bars)
    split = total_bars - tail
    for idx in range(total_bars):
        if idx < split:
            volumes.append(prev_volume)
            continue

        tail_idx = idx - split
        if tail_idx < 30:
            volumes.append(recent_volume_high)
        elif tail_idx < tail - 5:
            volumes.append(recent_volume_low)
        else:
            # Last 5 bars: mild volume pickup near resistance to satisfy BreakthroughFilter
            # while keeping the overall platform window contracted vs the previous window.
            volumes.append(recent_volume_high)

    return make_bars(closes, volume=volumes, high_pct=high_pct, low_pct=low_pct)


def make_downtrend_series(
    *,
    total_bars: int = 120,
    start_price: float = 120.0,
    end_price: float = 80.0,
    volume: float = 1_500_000.0,
    high_pct: float = 0.01,
    low_pct: float = 0.01,
) -> list[DummyPriceBar]:
    if total_bars < 10:
        raise ValueError("total_bars must be >= 10")
    step = (end_price - start_price) / float(total_bars - 1)
    closes = [start_price + step * idx for idx in range(total_bars)]
    return make_bars(closes, volume=volume, high_pct=high_pct, low_pct=low_pct)


def make_breakout_series(*, base_series: list[DummyPriceBar], breakout_close: float = 125.0) -> list[DummyPriceBar]:
    if not base_series:
        raise ValueError("base_series must be non-empty")
    bars = list(base_series)
    last = bars[-1]
    bars[-1] = make_bar(
        close=breakout_close,
        volume=last.volume * 3.0,
        high=breakout_close * 1.02,
        low=breakout_close * 0.98,
        open=last.close,
    )
    return bars


@pytest.fixture()
def scanner_config() -> ScannerConfig:
    # Disable P0/P1 breakout confirmation by default for backward compatibility.
    # Tests that specifically test P0/P1 features should override these settings.
    return ScannerConfig(
        require_breakout_confirmation=False,
        require_breakout_volume_spike=False,
    )


@pytest.fixture()
def platform_bars() -> list[DummyPriceBar]:
    return make_platform_series()


@pytest.fixture()
def downtrend_bars() -> list[DummyPriceBar]:
    return make_downtrend_series()
