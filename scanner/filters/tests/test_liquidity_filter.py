from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.interface import ResultStatus
from scanner.filters import LiquidityFilter
from scanner.interface import LiquidityFilterConfig, ScannerConfig


@dataclass(frozen=True, slots=True)
class DummyPriceBar:
    open: float
    high: float
    low: float
    close: float
    volume: float


def make_bar(*, close: float, volume: float, high_pct: float = 0.01, low_pct: float = 0.01) -> DummyPriceBar:
    return DummyPriceBar(
        open=float(close),
        high=float(close) * (1.0 + float(high_pct)),
        low=float(close) * (1.0 - float(low_pct)),
        close=float(close),
        volume=float(volume),
    )


def make_bars(*, count: int, close: float, volumes: list[float], high_pct: float = 0.01, low_pct: float = 0.01) -> list[DummyPriceBar]:
    assert len(volumes) == count
    return [make_bar(close=close, volume=v, high_pct=high_pct, low_pct=low_pct) for v in volumes]


def _cfg(
    *,
    enabled: bool = True,
    min_liquidity_score: float = 0.3,
    min_dollar_volume: float = 1_000_000.0,
    max_spread_proxy: float = 0.05,
) -> ScannerConfig:
    return ScannerConfig(
        liquidity_filter=LiquidityFilterConfig(
            enabled=enabled,
            min_liquidity_score=min_liquidity_score,
            min_dollar_volume=min_dollar_volume,
            max_spread_proxy=max_spread_proxy,
        )
    )


@pytest.mark.parametrize(
    ("volumes", "expected_trend"),
    [
        ([100.0] * 40 + [120.0] * 20, "improving"),
        ([100.0] * 40 + [70.0] * 20, "deteriorating"),
        ([100.0] * 40 + [85.0] * 20, "stable"),
    ],
)
def test_volume_trend_classification(volumes: list[float], expected_trend: str) -> None:
    bars = make_bars(count=60, close=10.0, volumes=volumes, high_pct=0.01, low_pct=0.01)
    result = LiquidityFilter().apply(bars, _cfg())
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    metrics = result.data.features["liquidity_metrics"]
    assert metrics["volume_trend"] == expected_trend


def test_liquidity_score_formula_and_clamping() -> None:
    # avg dollar volume 20d = 10 * 10_000 = 100k; below min_dollar_volume -> low score
    bars = make_bars(count=60, close=10.0, volumes=[10_000.0] * 60, high_pct=0.0245, low_pct=0.0245)
    result = LiquidityFilter().apply(bars, _cfg(min_dollar_volume=1_000_000.0, max_spread_proxy=0.05))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    metrics = result.data.features["liquidity_metrics"]
    assert 0.0 <= metrics["liquidity_score"] <= 1.0
    assert result.data.passed is False
    assert result.reason_code == "LOW_LIQUIDITY"


def test_unusual_activity_detection() -> None:
    # Baseline: range_pct ~2%, volume=100k; last bar: range_pct ~6%, volume=250k => unusual=True
    baseline = [100_000.0] * 59
    bars = make_bars(count=60, close=10.0, volumes=baseline + [250_000.0], high_pct=0.01, low_pct=0.01)
    bars[-1] = make_bar(close=10.0, volume=250_000.0, high_pct=0.03, low_pct=0.03)

    result = LiquidityFilter().apply(bars, _cfg())
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    metrics = result.data.features["liquidity_metrics"]
    assert metrics["unusual_activity"] is True
