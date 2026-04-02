from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.interface import ResultStatus
from scanner.filters import BreakthroughFilter
from scanner.interface import ScannerConfig


@dataclass(frozen=True, slots=True)
class DummyPriceBar:
    open: float
    high: float
    low: float
    close: float
    volume: float


def make_bar(*, open: float, high: float, low: float, close: float, volume: float) -> DummyPriceBar:
    return DummyPriceBar(open=float(open), high=float(high), low=float(low), close=float(close), volume=float(volume))


def make_flat_bars(
    count: int,
    *,
    close: float = 99.0,
    high: float = 100.0,
    low: float = 98.0,
    volume: float = 100.0,
) -> list[DummyPriceBar]:
    return [make_bar(open=close, high=high, low=low, close=close, volume=volume) for _ in range(count)]


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_reason"),
    [
        ("high", "bad", "INVALID_HIGH_ARRAY"),
        ("close", "bad", "INVALID_CLOSE_ARRAY"),
        ("volume", "bad", "INVALID_VOLUME_ARRAY"),
        ("high", float("nan"), "NONFINITE_HIGH"),
        ("close", float("nan"), "NONFINITE_CLOSE"),
        ("volume", float("nan"), "NONFINITE_VOLUME"),
    ],
)
def test_invalid_input_arrays_fail(field: str, bad_value: object, expected_reason: str) -> None:
    bars = make_flat_bars(15, close=99.0, high=100.0, low=98.0, volume=100.0)
    bad = bars[0]
    values = {name: getattr(bad, name) for name in ("open", "high", "low", "close", "volume")}
    values[field] = bad_value
    bars[0] = DummyPriceBar(**values)  # type: ignore[arg-type]

    result = BreakthroughFilter().apply(bars, ScannerConfig(require_breakout_confirmation=False))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == expected_reason


def test_rejects_nonpositive_price_and_negative_volume() -> None:
    bars = make_flat_bars(15, close=99.0, high=100.0, low=98.0, volume=100.0)
    bars[0] = make_bar(open=0.0, high=100.0, low=98.0, close=0.0, volume=100.0)
    result = BreakthroughFilter().apply(bars, ScannerConfig(require_breakout_confirmation=False))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NONPOSITIVE_PRICE"

    bars = make_flat_bars(15, close=99.0, high=100.0, low=98.0, volume=100.0)
    bars[0] = make_bar(open=99.0, high=100.0, low=98.0, close=99.0, volume=-1.0)
    result = BreakthroughFilter().apply(bars, ScannerConfig(require_breakout_confirmation=False))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NEGATIVE_VOLUME"


def test_rejects_invalid_thresholds() -> None:
    bars = make_flat_bars(15, close=99.0, high=100.0, low=98.0, volume=100.0)
    result = BreakthroughFilter().apply(bars, ScannerConfig(require_breakout_confirmation=False, resistance_proximity_pct=0.0))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_PROXIMITY_PCT"

    result = BreakthroughFilter().apply(bars, ScannerConfig(require_breakout_confirmation=False, volume_increase_ratio=0.0))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_VOLUME_INCREASE_RATIO"


def test_fails_when_prior_volume_avg_is_zero() -> None:
    volumes = [100.0 for _ in range(5)] + [0.0 for _ in range(10)] + [150.0 for _ in range(5)]
    bars = make_flat_bars(20, close=99.0, high=100.0, low=98.0, volume=100.0)
    for idx, vol in enumerate(volumes):
        bars[idx] = make_bar(open=99.0, high=100.0, low=98.0, close=99.0, volume=vol)

    result = BreakthroughFilter().apply(bars, ScannerConfig(require_breakout_confirmation=False))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_PRIOR_VOLUME"


def test_breakout_run_index_is_zero_when_always_above_resistance() -> None:
    bars = make_flat_bars(30, close=150.0, high=100.0, low=98.0, volume=100.0)
    config = ScannerConfig(
        volume_increase_ratio=1.0,
        require_breakout_confirmation=True,
        breakout_confirmation_days=2,
        require_breakout_volume_spike=False,
    )

    result = BreakthroughFilter().apply(bars, config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.features["actual_breakout"] is True
    assert result.data.metadata["breakout"]["breakout_day_index"] == 0
    assert result.data.metadata["breakout"]["breakout_run_length"] == len(bars)


def test_confirmation_days_exceed_history_disables_actual_breakout() -> None:
    bars = make_flat_bars(15, close=99.0, high=100.0, low=98.0, volume=100.0)
    config = ScannerConfig(
        volume_increase_ratio=1.0,
        require_breakout_confirmation=True,
        breakout_confirmation_days=len(bars),
        require_breakout_volume_spike=False,
    )

    result = BreakthroughFilter().apply(bars, config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.features["actual_breakout"] is False
    assert result.data.passed is False


def test_invalid_breakout_volume_ratio_fails_when_spike_enabled() -> None:
    bars = make_flat_bars(20, close=99.0, high=100.0, low=98.0, volume=100.0)
    config = ScannerConfig(
        volume_increase_ratio=1.0,
        require_breakout_confirmation=True,
        breakout_confirmation_days=2,
        require_breakout_volume_spike=True,
        breakout_volume_ratio=0.0,
    )
    result = BreakthroughFilter().apply(bars, config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_BREAKOUT_VOLUME_RATIO"


def test_insufficient_bars_fails() -> None:
    bars = make_flat_bars(14, close=99.0, high=100.0, low=98.0, volume=100.0)
    result = BreakthroughFilter().apply(bars, ScannerConfig(require_breakout_confirmation=False))
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BARS"


def test_invalid_breakout_confirmation_days_fails_when_enabled() -> None:
    bars = make_flat_bars(20, close=99.0, high=100.0, low=98.0, volume=100.0)
    result = BreakthroughFilter().apply(
        bars,
        ScannerConfig(
            require_breakout_confirmation=True,
            breakout_confirmation_days=0,
            require_breakout_volume_spike=False,
        ),
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_BREAKOUT_CONFIRMATION_DAYS"


def test_actual_breakout_passes_with_confirmation_days() -> None:
    bars = make_flat_bars(28, close=99.0, high=100.0, low=98.0, volume=100.0)
    bars += [
        make_bar(open=100.0, high=105.0, low=99.0, close=101.0, volume=250.0),  # breakout day
        make_bar(open=101.0, high=106.0, low=100.0, close=102.0, volume=150.0),  # confirmation day
    ]

    config = ScannerConfig(
        volume_increase_ratio=1.1,
        require_breakout_confirmation=True,
        breakout_confirmation_days=2,
        require_breakout_volume_spike=False,
    )

    result = BreakthroughFilter().apply(bars, config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.features["actual_breakout"] is True
    assert result.data.metadata["breakout"]["breakout_day_index"] == len(bars) - 2


def test_actual_breakout_fails_when_not_all_confirmation_closes_above() -> None:
    bars = make_flat_bars(28, close=99.0, high=100.0, low=98.0, volume=100.0)
    bars += [
        make_bar(open=99.0, high=105.0, low=98.0, close=99.5, volume=250.0),  # not above resistance
        make_bar(open=99.5, high=106.0, low=99.0, close=102.0, volume=150.0),
    ]

    config = ScannerConfig(
        volume_increase_ratio=1.1,
        require_breakout_confirmation=True,
        breakout_confirmation_days=2,
        require_breakout_volume_spike=False,
    )

    result = BreakthroughFilter().apply(bars, config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.data.features["actual_breakout"] is False


def test_breakout_volume_confirmed_passes_when_spike_ratio_met() -> None:
    bars = make_flat_bars(28, close=99.0, high=100.0, low=98.0, volume=100.0)
    bars += [
        make_bar(open=100.0, high=105.0, low=99.0, close=101.0, volume=250.0),  # breakout day
        make_bar(open=101.0, high=106.0, low=100.0, close=102.0, volume=150.0),
    ]

    config = ScannerConfig(
        volume_increase_ratio=1.1,
        require_breakout_confirmation=True,
        breakout_confirmation_days=2,
        require_breakout_volume_spike=True,
        breakout_volume_ratio=1.5,
    )

    result = BreakthroughFilter().apply(bars, config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.features["breakout_volume_confirmed"] is True
    assert result.data.metadata["breakout"]["breakout_volume_ratio"] == pytest.approx(2.5)


def test_breakout_volume_confirmed_fails_when_spike_ratio_not_met() -> None:
    bars = make_flat_bars(28, close=99.0, high=100.0, low=98.0, volume=100.0)
    bars += [
        make_bar(open=100.0, high=105.0, low=99.0, close=101.0, volume=120.0),  # breakout day (too small)
        make_bar(open=101.0, high=106.0, low=100.0, close=102.0, volume=150.0),
    ]

    config = ScannerConfig(
        volume_increase_ratio=1.05,
        require_breakout_confirmation=True,
        breakout_confirmation_days=2,
        require_breakout_volume_spike=True,
        breakout_volume_ratio=1.5,
    )

    result = BreakthroughFilter().apply(bars, config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.features["actual_breakout"] is True
    assert result.data.features["breakout_volume_confirmed"] is False
    assert result.data.passed is False


def test_backward_compatible_legacy_mode_uses_near_resistance_only() -> None:
    bars = make_flat_bars(25, close=99.0, high=100.0, low=98.0, volume=100.0)
    bars += [
        make_bar(open=99.0, high=100.0, low=98.0, close=99.6, volume=200.0),
        make_bar(open=99.6, high=100.0, low=98.5, close=99.5, volume=200.0),
        make_bar(open=99.5, high=100.0, low=98.5, close=99.6, volume=200.0),
        make_bar(open=99.6, high=100.0, low=98.5, close=99.7, volume=200.0),
        make_bar(open=99.7, high=100.0, low=98.5, close=99.8, volume=200.0),
    ]

    config = ScannerConfig(
        resistance_proximity_pct=0.02,
        volume_increase_ratio=1.5,
        require_breakout_confirmation=False,
        breakout_confirmation_days=2,
        require_breakout_volume_spike=True,
        breakout_volume_ratio=1.5,
    )

    result = BreakthroughFilter().apply(bars, config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.features["actual_breakout"] is False
    assert result.data.passed is True


def test_breakout_volume_spike_requires_enough_history() -> None:
    bars = make_flat_bars(18, close=99.0, high=100.0, low=98.0, volume=100.0)
    bars += [
        make_bar(open=100.0, high=105.0, low=99.0, close=101.0, volume=250.0),
        make_bar(open=101.0, high=106.0, low=100.0, close=102.0, volume=150.0),
    ]

    config = ScannerConfig(
        volume_increase_ratio=1.05,
        require_breakout_confirmation=True,
        breakout_confirmation_days=2,
        require_breakout_volume_spike=True,
        breakout_volume_ratio=1.5,
    )

    result = BreakthroughFilter().apply(bars, config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.features["actual_breakout"] is True
    assert result.data.features["breakout_volume_confirmed"] is False
    assert result.data.passed is False
