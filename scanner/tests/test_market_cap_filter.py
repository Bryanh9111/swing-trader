from __future__ import annotations

import pytest

from common.interface import ResultStatus
from scanner.filters import MarketCapFilter
from scanner.interface import ScannerConfig

from .conftest import make_bars


def _context_with_market_cap(market_cap: float | None) -> dict[str, object]:
    meta: dict[str, object] = {}
    if market_cap is not None:
        meta["market_cap"] = market_cap
    return {"meta": meta}


def test_large_cap_filtering() -> None:
    config = ScannerConfig()
    bars = make_bars([100.0] * 60, volume=1_000_000.0, high_pct=0.01, low_pct=0.01)  # atr_pct ~ 2%
    filt = MarketCapFilter(window=30)

    result = filt.apply(bars, config, context=_context_with_market_cap(20_000_000_000.0))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "ATR_TOO_HIGH_FOR_MARKET_CAP"


def test_mid_cap_filtering() -> None:
    config = ScannerConfig()
    bars = make_bars([100.0] * 60, volume=50_000.0, high_pct=0.007, low_pct=0.007)  # dv ~ $5M
    filt = MarketCapFilter(window=30)

    result = filt.apply(bars, config, context=_context_with_market_cap(5_000_000_000.0))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "INSUFFICIENT_LIQUIDITY_FOR_MARKET_CAP"


def test_small_cap_filtering() -> None:
    config = ScannerConfig()
    bars = make_bars([100.0] * 60, volume=60_000.0, high_pct=0.02, low_pct=0.02)  # atr_pct ~ 4%, dv ~ $6M
    filt = MarketCapFilter(window=30)

    result = filt.apply(bars, config, context=_context_with_market_cap(500_000_000.0))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True


def test_micro_cap_excluded() -> None:
    config = ScannerConfig()
    bars = make_bars([100.0] * 60, volume=1_000_000.0, high_pct=0.007, low_pct=0.007)
    filt = MarketCapFilter(window=30)

    result = filt.apply(bars, config, context=_context_with_market_cap(100_000_000.0))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is False
    assert result.reason_code == "MICRO_CAP_EXCLUDED"


def test_no_market_cap_data_passes() -> None:
    config = ScannerConfig()
    bars = make_bars([100.0] * 60, volume=10_000.0, high_pct=0.03, low_pct=0.03)
    filt = MarketCapFilter(window=30)

    result = filt.apply(bars, config, context=_context_with_market_cap(None))
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.passed is True
    assert result.reason_code == "MARKET_CAP_UNKNOWN"

