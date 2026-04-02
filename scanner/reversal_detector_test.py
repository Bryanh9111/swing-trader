from __future__ import annotations

import pytest

from common.interface import ResultStatus

from scanner.reversal_detector import (
    ReversalCandidate,
    ReversalConfig,
    ReversalDetector,
)


def _make_bar(
    *,
    close: float,
    low: float | None = None,
    volume: float = 1_000_000,
) -> dict[str, float]:
    """Helper to build a dict-based price bar for tests."""

    return {
        "open": close,
        "high": close,
        "low": low if low is not None else close,
        "close": close,
        "volume": volume,
    }


def test_reversal_disabled_returns_none() -> None:
    config = ReversalConfig(enabled=False)
    detector = ReversalDetector(config)
    bars = [_make_bar(close=100 + i, volume=1_000_000 + i * 10_000) for i in range(25)]

    result = detector.detect("TEST", bars, regime="bear")

    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "REVERSAL_DISABLED"


def test_reversal_non_bear_regime_returns_none() -> None:
    config = ReversalConfig(enabled=True)
    detector = ReversalDetector(config)
    bars = [_make_bar(close=150 - i, volume=2_000_000 + i * 50_000) for i in range(25)]

    result = detector.detect("TEST", bars, regime="bull")

    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "NON_BEAR_REGIME"


def test_rsi_calculation_accuracy() -> None:
    detector = ReversalDetector(ReversalConfig(enabled=True))
    closes = [10, 11, 12, 11, 10, 12]
    bars = [_make_bar(close=price) for price in closes]

    rsi_values = detector._calculate_rsi(bars, period=3)

    assert rsi_values == pytest.approx([66.6667, 44.4444, 72.2222], rel=1e-4)


def test_volume_ratio_calculation() -> None:
    detector = ReversalDetector(ReversalConfig(enabled=True))
    volumes = [100_000, 120_000, 110_000, 300_000]
    bars = [_make_bar(close=10 + idx, volume=vol) for idx, vol in enumerate(volumes)]

    ratio = detector._calculate_volume_ratio(bars, lookback=3)

    assert ratio == pytest.approx(300_000 / ((100_000 + 120_000 + 110_000) / 3))


def test_price_stabilization_check() -> None:
    detector = ReversalDetector(ReversalConfig(enabled=True))
    lows = [10.0, 9.5, 9.2, 9.3, 9.35, 9.4]
    bars = [_make_bar(close=low + 0.5, low=low) for low in lows]

    stabilized = detector._check_price_stabilization(bars, days=2)
    not_stabilized = detector._check_price_stabilization(
        bars[:-1] + [_make_bar(close=9.0, low=9.0)]
        + [_make_bar(close=8.9, low=8.9)],
        days=2,
    )

    assert stabilized is True
    assert not_stabilized is False


def test_signal_strength_calculation() -> None:
    detector = ReversalDetector(ReversalConfig(enabled=True))

    strength = detector._calculate_signal_strength(rsi=25.0, volume_ratio=3.0, price_stabilized=True)

    assert strength == pytest.approx(0.7)


def test_full_reversal_detection() -> None:
    config = ReversalConfig(
        enabled=True,
        rsi_period=3,
        rsi_oversold=45.0,
        volume_lookback=2,
        volume_ratio_threshold=1.5,
        stabilization_days=2,
    )
    detector = ReversalDetector(config)

    closes = [10.0, 9.5, 9.0, 8.5, 8.0, 8.05, 8.1, 8.2, 8.3]
    lows = [price - 0.1 for price in closes]
    volumes = [100_000, 110_000, 115_000, 120_000, 130_000, 140_000, 150_000, 160_000, 400_000]
    bars = [
        {
            "open": close,
            "high": close + 0.2,
            "low": low,
            "close": close,
            "volume": volume,
        }
        for close, low, volume in zip(closes, lows, volumes, strict=False)
    ]

    result = detector.detect("REV", bars, regime="bear")

    assert result.status is ResultStatus.SUCCESS
    assert isinstance(result.data, ReversalCandidate)
    assert result.data.symbol == "REV"
    assert result.data.rsi_turning_up is True
    assert result.data.volume_ratio >= config.volume_ratio_threshold
    assert 0.0 <= result.data.signal_strength <= 1.0
