from __future__ import annotations

from collections.abc import Sequence

from common.interface import Result, ResultStatus
from scanner.filters import BaseFilter, FilterChain, FilterResult
from scanner.interface import ScannerConfig


class _PassFilter(BaseFilter):
    name = "pass"

    def __init__(self, *, score: float = 1.0, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._score = score

    def _apply_filter(self, bars: Sequence[object], config: ScannerConfig) -> Result[FilterResult]:
        return Result.success(FilterResult(passed=True, reason="PASS", score=self._score))


class _FailFilter(BaseFilter):
    name = "fail"

    def _apply_filter(self, bars: Sequence[object], config: ScannerConfig) -> Result[FilterResult]:
        return Result.success(FilterResult(passed=False, reason="FAIL", score=0.0))


class _BoomFilter(BaseFilter):
    name = "boom"

    def _apply_filter(self, bars: Sequence[object], config: ScannerConfig) -> Result[FilterResult]:
        raise RuntimeError("boom")


def test_filter_chain_and_passes(platform_bars: list[object], scanner_config: ScannerConfig) -> None:
    class _PassA(_PassFilter):
        name = "pass_a"

    class _PassB(_PassFilter):
        name = "pass_b"

    chain = FilterChain(filters=[_PassA(score=0.8), _PassB(score=0.6)], logic="AND")
    result = chain.execute(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.combined_score == (0.8 + 0.6) / 2.0


def test_filter_chain_and_fails(platform_bars: list[object], scanner_config: ScannerConfig) -> None:
    chain = FilterChain(filters=[_PassFilter(), _FailFilter()], logic="AND")
    result = chain.execute(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert "fail: FAIL" in result.data.reasons


def test_filter_chain_or_passes(platform_bars: list[object], scanner_config: ScannerConfig) -> None:
    chain = FilterChain(filters=[_FailFilter(), _PassFilter(score=0.7)], logic="OR")
    result = chain.execute(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.filter_results["fail"].passed is False
    assert result.data.filter_results["pass"].passed is True


def test_filter_chain_disabled_filters_reported(platform_bars: list[object], scanner_config: ScannerConfig) -> None:
    class _DisabledFilter(_PassFilter):
        name = "disabled"

    chain = FilterChain(filters=[_DisabledFilter(enabled=False)], logic="AND")
    result = chain.execute(platform_bars, scanner_config)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.data.filter_results["disabled"].reason == "FILTER_DISABLED"


def test_filter_chain_failed_filter_degrades(platform_bars: list[object], scanner_config: ScannerConfig) -> None:
    chain = FilterChain(filters=[_PassFilter(), _BoomFilter()], logic="AND")
    result = chain.execute(platform_bars, scanner_config)

    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert "pass" in result.data.filter_results
