from __future__ import annotations

import pytest

from common.interface import ResultStatus
from data.vix_provider import VIXData
from scanner.market_regime import detector as detector_module
from scanner.market_regime.detector import MarketRegimeDetector
from scanner.market_regime.interface import MarketRegime

from .conftest import make_series_snapshot


class _StubVIXProvider:
    def __init__(self, vix: VIXData | None) -> None:
        self._vix = vix

    def fetch_vix_current(self) -> VIXData | None:
        return self._vix


def test_vix_extreme_forces_bear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    spy_snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(120)])

    detector = MarketRegimeDetector(
        index_symbols=("SPY",),
        vix_enabled=True,
        vix_extreme_threshold=35.0,
        vix_elevated_threshold=25.0,
        vix_spike_threshold=0.20,
        vix_provider=_StubVIXProvider(
            VIXData(timestamp=1, value=40.0, change_pct=0.05, level="extreme"),
        ),
    )
    result = detector.detect({"SPY": spy_snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.BEAR
    assert result.data.vix_data is not None
    assert result.data.vix_data.value == 40.0


def test_vix_elevated_biases_to_choppy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    spy_snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(120)])

    detector = MarketRegimeDetector(
        index_symbols=("SPY",),
        vix_enabled=True,
        vix_extreme_threshold=35.0,
        vix_elevated_threshold=25.0,
        vix_provider=_StubVIXProvider(
            VIXData(timestamp=1, value=30.0, change_pct=0.01, level="elevated"),
        ),
    )
    result = detector.detect({"SPY": spy_snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.CHOPPY


def test_vix_spike_forces_choppy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    spy_snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(120)])

    detector = MarketRegimeDetector(
        index_symbols=("SPY",),
        vix_enabled=True,
        vix_spike_threshold=0.20,
        vix_spike_forces_choppy=True,
        vix_provider=_StubVIXProvider(
            VIXData(timestamp=1, value=22.0, change_pct=0.25, level="normal"),
        ),
    )
    result = detector.detect({"SPY": spy_snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.regime is MarketRegime.CHOPPY
    assert result.data.vix_spike_detected is True


def test_vix_unavailable_does_not_block_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_compute_adx_last", lambda *args, **kwargs: 30.0)
    spy_snapshot = make_series_snapshot("SPY", closes=[100.0 + i * 0.5 for i in range(120)])

    detector = MarketRegimeDetector(
        index_symbols=("SPY",),
        vix_enabled=True,
        vix_provider=_StubVIXProvider(None),
    )
    result = detector.detect({"SPY": spy_snapshot})
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.vix_data is None

