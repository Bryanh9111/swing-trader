from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.interface import ResultStatus
from scanner.filters import PullbackConfirmationFilter
from scanner.interface import PullbackConfirmationConfig, ScannerConfig


@dataclass(frozen=True, slots=True)
class DummyPriceBar:
    open: float
    high: float
    low: float
    close: float
    volume: float


def make_bar(*, open: float, high: float, low: float, close: float, volume: float = 1_000_000.0) -> DummyPriceBar:
    return DummyPriceBar(open=open, high=high, low=low, close=close, volume=volume)


def _config(*, enabled: bool = True, lookback_days: int = 5, pullback_tolerance_pct: float = 0.02) -> ScannerConfig:
    return ScannerConfig(
        pullback_confirmation=PullbackConfirmationConfig(
            enabled=enabled,
            lookback_days=lookback_days,
            pullback_tolerance_pct=pullback_tolerance_pct,
            require_volume_increase=False,
            volume_increase_ratio=1.5,
        )
    )


def test_passes_when_breakthrough_pullback_and_hold() -> None:
    resistance = 100.0
    pre = [make_bar(open=99.0, high=resistance, low=98.0, close=99.0) for _ in range(10)]

    window = [
        make_bar(open=99.0, high=105.0, low=103.0, close=104.0),  # breakthrough
        make_bar(open=104.0, high=101.0, low=99.0, close=100.5),  # pullback into 98~102 band
        make_bar(open=100.5, high=103.0, low=100.0, close=101.0),  # holding above resistance
        make_bar(open=101.0, high=102.0, low=100.5, close=101.5),
        make_bar(open=101.5, high=102.0, low=101.0, close=101.2),
    ]
    bars = pre + window

    result = PullbackConfirmationFilter(enabled=True).apply(bars, _config())
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.reason_code == "OK"


def test_fails_when_no_breakthrough() -> None:
    pre = [make_bar(open=99.0, high=100.0, low=98.0, close=99.0) for _ in range(10)]
    window = [make_bar(open=99.0, high=100.0, low=98.5, close=99.0) for _ in range(5)]
    bars = pre + window

    result = PullbackConfirmationFilter().apply(bars, _config())
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "NO_BREAKTHROUGH"


def test_fails_when_no_pullback() -> None:
    pre = [make_bar(open=99.0, high=100.0, low=98.0, close=99.0) for _ in range(10)]
    window = [
        make_bar(open=99.0, high=105.0, low=103.0, close=104.0),  # breakthrough
        make_bar(open=104.0, high=110.0, low=103.5, close=109.0),  # stays above band (low > 102)
        make_bar(open=109.0, high=111.0, low=108.0, close=110.0),
        make_bar(open=110.0, high=111.0, low=108.0, close=109.0),
        make_bar(open=109.0, high=110.0, low=108.0, close=109.5),
    ]
    bars = pre + window

    result = PullbackConfirmationFilter().apply(bars, _config())
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "NO_PULLBACK"


def test_fails_when_not_holding_above() -> None:
    pre = [make_bar(open=99.0, high=100.0, low=98.0, close=99.0) for _ in range(10)]
    window = [
        make_bar(open=99.0, high=105.0, low=103.0, close=104.0),  # breakthrough
        make_bar(open=104.0, high=101.0, low=99.0, close=99.5),  # pullback into band
        make_bar(open=99.5, high=100.0, low=98.5, close=99.8),  # fails to reclaim > resistance
        make_bar(open=99.8, high=100.0, low=99.0, close=99.9),
        make_bar(open=99.9, high=100.0, low=99.0, close=100.0),
    ]
    bars = pre + window

    result = PullbackConfirmationFilter().apply(bars, _config())
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "NOT_HOLDING_ABOVE"


def test_insufficient_bars() -> None:
    bars = [make_bar(open=10.0, high=10.0, low=9.0, close=10.0) for _ in range(5)]
    result = PullbackConfirmationFilter().apply(bars, _config(lookback_days=5))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BARS"


def test_disabled_filter_passes() -> None:
    bars = [make_bar(open=10.0, high=10.0, low=9.0, close=10.0) for _ in range(1)]
    result = PullbackConfirmationFilter(enabled=False).apply(bars, _config(enabled=False))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.reason_code == "FILTER_DISABLED"

