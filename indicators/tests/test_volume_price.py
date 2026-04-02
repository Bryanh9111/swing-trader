from __future__ import annotations

import math

import pytest

from data.interface import PriceBar
from indicators import (
    compute_obv_last,
    compute_obv_series,
    compute_volume_price_divergence,
    compute_vpt_last,
    compute_vpt_series,
)


def _make_bars(closes: list[float], volumes: list[int] | None = None) -> list[PriceBar]:
    if volumes is None:
        volumes = [1000] * len(closes)
    assert len(volumes) == len(closes)
    return [
        PriceBar(
            timestamp=int(i * 1e9),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=int(volumes[i]),
        )
        for i, c in enumerate(closes)
    ]


class TestOBV:
    def test_obv_empty(self) -> None:
        assert compute_obv_last([]) is None
        assert compute_obv_series([]) == []

    def test_obv_single_bar_is_none(self) -> None:
        bars = _make_bars([10.0], [123])
        assert compute_obv_last(bars) is None
        assert compute_obv_series(bars) == [None]

    def test_obv_series_matches_known_values(self) -> None:
        bars = _make_bars(
            [10.0, 11.0, 10.0, 10.0, 12.0],
            [100, 200, 300, 400, 500],
        )
        assert compute_obv_series(bars) == [None, 200.0, -100.0, -100.0, 400.0]
        assert compute_obv_last(bars) == 400.0


class TestVPT:
    def test_vpt_empty(self) -> None:
        assert compute_vpt_last([]) is None
        assert compute_vpt_series([]) == []

    def test_vpt_single_bar_is_none(self) -> None:
        bars = _make_bars([10.0], [123])
        assert compute_vpt_last(bars) is None
        assert compute_vpt_series(bars) == [None]

    def test_vpt_series_matches_known_values(self) -> None:
        bars = _make_bars([10.0, 11.0, 10.0], [100, 200, 300])
        series = compute_vpt_series(bars)
        assert series[0] is None
        assert series[1] == pytest.approx(20.0)
        assert series[2] == pytest.approx(20.0 + 300.0 * (-1.0 / 11.0))
        assert compute_vpt_last(bars) == pytest.approx(series[-1])

    def test_vpt_prev_close_zero_returns_none_last(self) -> None:
        bars = _make_bars([0.0, 10.0, 11.0], [100, 200, 300])
        series = compute_vpt_series(bars)
        assert series == [None, None, None]
        assert compute_vpt_last(bars) is None


class TestVolumePriceDivergence:
    def test_divergence_invalid_lookback(self) -> None:
        bars = _make_bars([10.0, 11.0], [100, 100])
        with pytest.raises(ValueError, match="lookback must be > 1"):
            compute_volume_price_divergence(bars, lookback=1)

    def test_divergence_insufficient_data(self) -> None:
        bars = _make_bars([10.0, 11.0, 12.0], [100, 100, 100])
        assert compute_volume_price_divergence(bars, lookback=20) is None

    def test_bearish_divergence_price_new_high_obv_not_new_high(self) -> None:
        bars = _make_bars([10.0, 12.0, 11.0, 13.0], [1000, 5000, 6000, 100])
        div = compute_volume_price_divergence(bars, lookback=4)
        assert div is not None
        assert div.divergence_type == "bearish"
        assert div.price_trend == "up"
        assert div.obv_trend == "down"
        assert 0.0 < div.strength <= 1.0
        assert math.isfinite(div.strength)

    def test_bullish_divergence_price_new_low_obv_not_new_low(self) -> None:
        bars = _make_bars([10.0, 8.0, 9.0, 7.0], [1000, 10000, 1000, 100])
        div = compute_volume_price_divergence(bars, lookback=4)
        assert div is not None
        assert div.divergence_type == "bullish"
        assert div.price_trend == "down"
        assert div.obv_trend == "up"
        assert 0.0 < div.strength <= 1.0
        assert math.isfinite(div.strength)

    def test_no_divergence_returns_none_type(self) -> None:
        bars = _make_bars([10.0, 11.0, 12.0, 13.0], [100, 100, 100, 100])
        div = compute_volume_price_divergence(bars, lookback=4)
        assert div is not None
        assert div.divergence_type == "none"
        assert div.strength == 0.0
