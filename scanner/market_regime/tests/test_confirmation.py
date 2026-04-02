from __future__ import annotations

import pytest

from common.interface import ResultStatus
from scanner.market_regime.confirmation import BreakthroughConfirmation, BreakthroughConfirmationConfig

from .conftest import DummyBar


def test_confirmation_fails_with_insufficient_bars() -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=3)
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation([DummyBar(close=101.0, volume=100.0)], resistance=100.0)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BARS"


def test_confirmation_rejects_nonfinite_close() -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=1)
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation([DummyBar(close=float("nan"), volume=100.0)], resistance=100.0)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "NONFINITE_CLOSE"


def test_confirmation_passes_with_two_day_close_and_volume(confirmation_bars: list[DummyBar]) -> None:
    config = BreakthroughConfirmationConfig(
        require_confirmation_days=2,
        confirmation_close_above_resistance=True,
        require_volume_confirmation=True,
    )
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation(
        confirmation_bars,
        resistance=100.5,
        volume_increase_ratio=2.0,
        volume_lookback=2,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is True


def test_confirmation_fails_when_close_not_above(confirmation_bars: list[DummyBar]) -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=2)
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation(confirmation_bars, resistance=200.0)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is False
    assert result.reason_code == "CONFIRMATION_CLOSE_NOT_ABOVE"


def test_confirmation_fails_when_volume_not_met(confirmation_bars: list[DummyBar]) -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=2, require_volume_confirmation=True)
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation(
        confirmation_bars,
        resistance=100.5,
        volume_increase_ratio=100.0,
        volume_lookback=2,
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.data is False
    assert result.reason_code == "CONFIRMATION_VOLUME_NOT_MET"


@pytest.mark.parametrize("resistance", [0.0, float("nan")])
def test_confirmation_rejects_invalid_resistance(confirmation_bars: list[DummyBar], resistance: float) -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=1)
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation(confirmation_bars, resistance=resistance)
    assert result.status is ResultStatus.FAILED


def test_confirmation_rejects_invalid_confirmation_days(confirmation_bars: list[DummyBar]) -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=0)
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation(confirmation_bars, resistance=100.0)
    assert result.status is ResultStatus.FAILED


def test_confirmation_rejects_invalid_volume_lookback(confirmation_bars: list[DummyBar]) -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=1, require_volume_confirmation=True)
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation(confirmation_bars, resistance=100.0, volume_lookback=0)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_VOLUME_LOOKBACK"


def test_confirmation_rejects_invalid_volume_ratio(confirmation_bars: list[DummyBar]) -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=1, require_volume_confirmation=True)
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation(confirmation_bars, resistance=100.0, volume_increase_ratio=-1.0, volume_lookback=1)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_VOLUME_INCREASE_RATIO"


def test_confirmation_fails_with_insufficient_bars_for_volume_baseline() -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=2, require_volume_confirmation=True)
    checker = BreakthroughConfirmation(config)
    result = checker.check_confirmation(
        [DummyBar(close=101.0, volume=100.0), DummyBar(close=102.0, volume=200.0)],
        resistance=100.0,
        volume_increase_ratio=1.0,
        volume_lookback=10,
    )
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BARS_FOR_VOLUME"


def test_confirmation_rejects_invalid_volume_values() -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=1, require_volume_confirmation=True)
    checker = BreakthroughConfirmation(config)
    bars = [
        DummyBar(close=99.0, volume=-1.0),
        DummyBar(close=101.0, volume=100.0),
    ]
    result = checker.check_confirmation(bars, resistance=100.0, volume_lookback=1)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_VOLUME"


def test_confirmation_rejects_zero_baseline_volume() -> None:
    config = BreakthroughConfirmationConfig(require_confirmation_days=1, require_volume_confirmation=True)
    checker = BreakthroughConfirmation(config)
    bars = [
        DummyBar(close=99.0, volume=0.0),
        DummyBar(close=101.0, volume=1.0),
    ]
    result = checker.check_confirmation(bars, resistance=100.0, volume_increase_ratio=1.0, volume_lookback=1)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "ZERO_BASELINE_VOLUME"
