from __future__ import annotations

import math

import pytest

from data.interface import PriceBar
from indicators import compute_bbands_last, compute_bbands_series


def _make_bars(closes: list[float]) -> list[PriceBar]:
    return [
        PriceBar(
            timestamp=int(i * 1e9),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1000,
        )
        for i, c in enumerate(closes)
    ]


def test_bbands_last_insufficient_data_returns_none() -> None:
    assert compute_bbands_last([1.0, 2.0, 3.0], period=5, std_dev=2.0) is None


def test_bbands_series_insufficient_data_returns_all_none() -> None:
    assert compute_bbands_series([1.0, 2.0, 3.0], period=5, std_dev=2.0) == [None, None, None]


def test_bbands_invalid_period_raises() -> None:
    prices = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="period must be > 0"):
        compute_bbands_last(prices, period=0)
    with pytest.raises(ValueError, match="period must be > 0"):
        compute_bbands_series(prices, period=0)


def test_bbands_last_matches_manual_calculation() -> None:
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = compute_bbands_last(closes, period=5, std_dev=2.0)
    assert result is not None

    middle = sum(closes) / 5.0
    var = sum((x - middle) ** 2 for x in closes) / 5.0
    std = math.sqrt(var)
    upper = middle + 2.0 * std
    lower = middle - 2.0 * std
    bandwidth = (upper - lower) / middle
    percent_b = (closes[-1] - lower) / (upper - lower)

    assert result.middle == pytest.approx(middle)
    assert result.upper == pytest.approx(upper)
    assert result.lower == pytest.approx(lower)
    assert result.bandwidth == pytest.approx(bandwidth)
    assert result.percent_b == pytest.approx(percent_b)


def test_bbands_series_alignment_and_last_equals_last_api() -> None:
    closes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    series = compute_bbands_series(closes, period=5, std_dev=2.0)
    assert len(series) == len(closes)
    assert series[:4] == [None] * 4

    last_from_series = series[-1]
    last_direct = compute_bbands_last(closes, period=5, std_dev=2.0)
    assert last_from_series is not None
    assert last_direct is not None
    assert last_from_series == last_direct


def test_bbands_std_dev_zero_returns_neutral_percent_b() -> None:
    closes = [10.0] * 20
    result = compute_bbands_last(closes, period=20, std_dev=0.0)
    assert result is not None
    assert result.upper == 10.0
    assert result.middle == 10.0
    assert result.lower == 10.0
    assert result.bandwidth == 0.0
    assert result.percent_b == 0.5

    series = compute_bbands_series(closes, period=20, std_dev=0.0)
    assert series[:19] == [None] * 19
    assert series[-1] == result


def test_bbands_middle_zero_returns_none() -> None:
    closes = [0.0, 0.0, 0.0]
    assert compute_bbands_last(closes, period=3, std_dev=2.0) is None
    assert compute_bbands_series(closes, period=3, std_dev=2.0) == [None, None, None]


def test_bbands_accepts_pricebar_input() -> None:
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    bars = _make_bars(closes)
    from_bars = compute_bbands_last(bars, period=5, std_dev=2.0)
    from_floats = compute_bbands_last(closes, period=5, std_dev=2.0)
    assert from_bars == from_floats
