from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime

import pytest

from data.interface import PriceBar, PriceSeriesSnapshot, QualityAssessment, normalize_date


def test_price_bar_creation() -> None:
    bar = PriceBar(timestamp=123, open=1.0, high=2.0, low=0.5, close=1.5, volume=10)
    assert bar.timestamp == 123
    assert bar.open == 1.0


def test_price_series_snapshot_creation() -> None:
    bar = PriceBar(timestamp=123, open=1.0, high=2.0, low=0.5, close=1.5, volume=10)
    snapshot = PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=456,
        symbol="AAPL",
        timeframe="1D",
        bars=[bar],
        source="polygon",
        quality_flags={"availability": True},
    )
    assert snapshot.symbol == "AAPL"
    assert snapshot.bars[0] == bar


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2025, 1, 1), date(2025, 1, 1)),
        (datetime(2025, 1, 1, 5, 0, 0), date(2025, 1, 1)),
        (datetime(2025, 1, 1, 5, 0, 0, tzinfo=UTC), date(2025, 1, 1)),
        ("2025-01-01", date(2025, 1, 1)),
    ],
)
def test_normalize_date_supported_inputs(value: object, expected: date) -> None:
    assert normalize_date(value) == expected  # type: ignore[arg-type]


def test_normalize_date_invalid_type_raises() -> None:
    with pytest.raises(TypeError):
        normalize_date(123)  # type: ignore[arg-type]


def test_quality_assessment_to_flags_merges_details() -> None:
    assessment = QualityAssessment(
        availability=True,
        completeness=False,
        freshness=True,
        stability=True,
        details={"provider": "unit-test", "missing_rows": 3},
    )
    flags = assessment.to_flags()
    assert flags["availability"] is True
    assert flags["completeness"] is False
    assert flags["provider"] == "unit-test"
    assert flags["missing_rows"] == 3


def test_quality_assessment_is_frozen() -> None:
    assessment = QualityAssessment(
        availability=True,
        completeness=True,
        freshness=True,
        stability=True,
        details={},
    )
    with pytest.raises(FrozenInstanceError):
        assessment.availability = False  # type: ignore[misc]

