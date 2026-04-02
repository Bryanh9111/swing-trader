from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from common.interface import Result, ResultStatus
from scanner.filters import FilterChain
from scanner.filters.atr_filter import ATRFilter
from scanner.filters.etf_filter import ETFFilter
from scanner.filters.platform_days_filter import PlatformDaysFilter, calculate_platform_days
from scanner.interface import ScannerConfig

from .conftest import make_bars


def test_etf_filter_blocks_known_etf_by_default(scanner_config: ScannerConfig) -> None:
    result = ETFFilter(symbol="SPY").apply([], scanner_config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "ETF_FILTERED"
    assert result.data.features["is_etf"] is True


def test_platform_days_filter_passes_when_enough_days(scanner_config: ScannerConfig) -> None:
    bars = make_bars([100.0] * 20, volume=1_000_000.0)
    result = PlatformDaysFilter(window=15).apply(bars, scanner_config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.features["platform_days"] == 15


def test_platform_days_filter_fails_when_insufficient(scanner_config: ScannerConfig) -> None:
    bars = make_bars([100.0] * 10, volume=1_000_000.0)
    result = PlatformDaysFilter(window=5).apply(bars, scanner_config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "PLATFORM_DAYS_FILTERED"


def test_platform_days_filter_window_slice_failure(scanner_config: ScannerConfig) -> None:
    bars = make_bars([100.0] * 5, volume=1_000_000.0)
    result = PlatformDaysFilter(window=20).apply(bars, scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_BARS"


def test_calculate_platform_days_breaks_on_first_outside() -> None:
    bars = make_bars([100.0, 100.0, 120.0], volume=1_000_000.0)
    assert calculate_platform_days(bars, box_low=99.0, box_high=101.0) == 0


def test_filter_chain_rejects_invalid_logic(scanner_config: ScannerConfig) -> None:
    chain = FilterChain(filters=[], logic="X")  # type: ignore[arg-type]
    result = chain.execute([], scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CHAIN_INVALID_LOGIC"


def test_filter_chain_empty_returns_no_filters_enabled(scanner_config: ScannerConfig) -> None:
    chain = FilterChain(filters=[], logic="AND")
    result = chain.execute([], scanner_config)
    assert result.status is ResultStatus.SUCCESS
    assert result.reason_code == "NO_FILTERS_ENABLED"
    assert result.data is not None
    assert result.data.reasons == ["NO_FILTERS_ENABLED"]
    assert result.data.combined_score == 1.0


def test_filter_chain_duplicate_names_in_execute(scanner_config: ScannerConfig) -> None:
    from scanner.filters.base import BaseFilter, FilterResult

    class _DupFilter(BaseFilter):
        name = "dup"

        def _apply_filter(self, bars: Sequence[object], config: ScannerConfig) -> Result[FilterResult]:
            return Result.success(FilterResult(passed=True, reason="OK", score=1.0))

    chain = FilterChain(filters=[], logic="AND")
    chain.filters = [_DupFilter(), _DupFilter()]
    result = chain.execute([], scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "CHAIN_DUPLICATE_FILTER_NAME"


@pytest.mark.parametrize("score", ["bad", float("nan")])
def test_filter_chain_clamps_non_numeric_scores(score: object, scanner_config: ScannerConfig) -> None:
    from scanner.filters.base import BaseFilter, FilterResult

    class _WeirdScore(BaseFilter):
        name = "weird"

        def _apply_filter(self, bars: Sequence[object], config: ScannerConfig) -> Result[FilterResult]:
            return Result.success(FilterResult(passed=True, reason="OK", score=score))  # type: ignore[arg-type]

    chain = FilterChain(filters=[_WeirdScore()], logic="AND")
    result = chain.execute([], scanner_config)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.combined_score == 0.0
    assert math.isfinite(result.data.combined_score)


def test_platform_days_filter_window_empty(monkeypatch: pytest.MonkeyPatch, scanner_config: ScannerConfig) -> None:
    from scanner import detector as detector_module

    monkeypatch.setattr(detector_module, "_window_slice", lambda bars, window: Result.success([]))
    bars = make_bars([100.0] * 5, volume=1_000_000.0)
    result = PlatformDaysFilter(window=5).apply(bars, scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "WINDOW_EMPTY"


def test_atr_filter_handles_empty_bars(scanner_config: ScannerConfig) -> None:
    result = ATRFilter().apply([], scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INSUFFICIENT_ATR_BARS"


def test_atr_filter_handles_invalid_last_close(scanner_config: ScannerConfig) -> None:
    bars = make_bars([100.0] * 20, volume=1_000_000.0)
    bars[-1] = bars[-1].__class__(open=100.0, high=101.0, low=99.0, close=float("nan"), volume=1_000_000.0)
    result = ATRFilter().apply(bars, scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_LAST_CLOSE"


def test_atr_filter_handles_invalid_atr_pct(monkeypatch: pytest.MonkeyPatch, scanner_config: ScannerConfig) -> None:
    from scanner import detector as detector_module

    monkeypatch.setattr(detector_module, "calculate_atr", lambda bars, period: Result.success(0.0))
    bars = make_bars([100.0] * 20, volume=1_000_000.0)
    result = ATRFilter().apply(bars, scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "INVALID_ATR_PCT"


def test_base_filter_rejects_empty_result(scanner_config: ScannerConfig) -> None:
    from scanner.filters.base import BaseFilter

    class _EmptyFilter(BaseFilter):
        name = "empty"

        def _apply_filter(self, bars: Sequence[object], config: ScannerConfig) -> Result[object]:
            return Result.success(None)  # type: ignore[arg-type]

    result = _EmptyFilter().apply([], scanner_config)
    assert result.status is ResultStatus.FAILED
    assert result.reason_code == "FILTER_EMPTY_RESULT"
