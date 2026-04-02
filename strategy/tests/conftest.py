from __future__ import annotations

import msgspec
import pytest

from data.interface import PriceBar, PriceSeriesSnapshot
from event_guard.interface import TradeConstraints
from scanner.interface import CandidateSet, PlatformCandidate, PlatformFeatures
from strategy.interface import StrategyEngineConfig


class PriceSeriesSnapshotWithATR(PriceSeriesSnapshot, frozen=True, kw_only=True):
    atr_pct: float | None = msgspec.field(default=None)


def _make_sample_price_bars(*, count: int = 30) -> list[PriceBar]:
    bars: list[PriceBar] = []
    for idx in range(count):
        close = 50.0 + ((idx % 5) - 2) * 0.5  # 49~51 wobble
        bars.append(
            PriceBar(
                timestamp=1_700_000_000_000_000_000 + idx * 60_000_000_000,
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1_000_000,
            )
        )
    return bars


@pytest.fixture()
def sample_price_bars() -> list[PriceBar]:
    return _make_sample_price_bars(count=30)


@pytest.fixture()
def sample_market_data(sample_price_bars: list[PriceBar]) -> PriceSeriesSnapshot:
    return PriceSeriesSnapshotWithATR(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1_700_000_000_000_000_000,
        symbol="AAPL",
        timeframe="1d",
        bars=sample_price_bars,
        source="test",
        quality_flags={},
        atr_pct=0.02,
    )


@pytest.fixture()
def sample_candidate() -> PlatformCandidate:
    return PlatformCandidate(
        symbol="AAPL",
        detected_at=1_700_000_000_000_000_000,
        window=30,
        score=0.8,
        features=PlatformFeatures(
            box_range=0.04,
            box_low=49.0,
            box_high=51.0,
            ma_diff=0.01,
            volatility=0.02,
            atr_pct=0.02,
            volume_change_ratio=0.7,
            volume_stability=0.2,
            avg_dollar_volume=50_000_000.0,
            box_quality=0.9,
            support_level=49.0,
            resistance_level=51.0,
        ),
        invalidation_level=48.5,
        target_level=52.0,
        reasons=["price_platform"],
    )


@pytest.fixture()
def sample_candidate_set(sample_candidate: PlatformCandidate) -> CandidateSet:
    return CandidateSet(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1_700_000_000_000_000_000,
        candidates=[sample_candidate],
        total_scanned=1,
        total_detected=1,
        config_snapshot={},
        data_source="test",
        universe_source="test",
    )


@pytest.fixture()
def sample_constraints() -> TradeConstraints:
    return TradeConstraints(
        symbol="AAPL",
        can_open_new=True,
        max_position_size=1000.0,
    )


@pytest.fixture()
def default_config() -> StrategyEngineConfig:
    return StrategyEngineConfig()

