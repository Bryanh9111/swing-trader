from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import msgspec
import pytest
import logging

from common.exceptions import DataSourceUnavailableError, ValidationError
from common.interface import DomainEvent, EventBus, Result, ResultStatus
from plugins.interface import PluginCategory
from universe.builder import UniverseBuilder
from universe.interface import EquityInfo, UniverseFilterCriteria


class CapturingEventBus(EventBus):
    def __init__(self) -> None:
        self.published: list[tuple[str, DomainEvent]] = []

    def publish(self, topic: str, event: DomainEvent) -> None:
        self.published.append((topic, event))

    def subscribe(self, pattern: str, handler: Callable[[DomainEvent], None]) -> None:
        return None


class RaisingEventBus(EventBus):
    def __init__(self, factory: Callable[[], Exception]) -> None:
        self._factory = factory
        self.calls: list[tuple[str, DomainEvent]] = []

    def publish(self, topic: str, event: DomainEvent) -> None:
        self.calls.append((topic, event))
        raise self._factory()

    def subscribe(self, pattern: str, handler: Callable[[DomainEvent], None]) -> None:
        return None


class ListLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))


class BindableListLogger:
    def __init__(self) -> None:
        self.bind_calls: list[dict[str, Any]] = []
        self.entries: list[tuple[str, str, dict[str, Any]]] = []

    def bind(self, **new_values: Any) -> "BindableListLogger":
        self.bind_calls.append(dict(new_values))
        return self

    def info(self, event: str, **kwargs: Any) -> None:
        self.entries.append(("info", event, dict(kwargs)))

    def warning(self, event: str, **kwargs: Any) -> None:
        self.entries.append(("warning", event, dict(kwargs)))


def _make_equity(
    *,
    symbol: str,
    exchange: str,
    price: float = 10.0,
    avg_dollar_volume_20d: float = 25_000_000.0,
    market_cap: float = 2_000_000_000.0,
    is_otc: bool = False,
    is_halted: bool = False,
    sector: str | None = None,
) -> EquityInfo:
    return EquityInfo(
        symbol=symbol,
        exchange=exchange,
        price=price,
        avg_dollar_volume_20d=avg_dollar_volume_20d,
        market_cap=market_cap,
        is_otc=is_otc,
        is_halted=is_halted,
        sector=sector,
    )


@pytest.fixture
def valid_criteria() -> UniverseFilterCriteria:
    return UniverseFilterCriteria(
        exchanges=["NYSE", "NASDAQ"],
        min_price=0.0,
        max_price=2000.0,
        min_avg_dollar_volume_20d=10_000_000.0,
        min_market_cap=None,
        exclude_otc=True,
        exclude_halted=True,
        max_results=None,
    )


@pytest.fixture
def sample_equities() -> list[EquityInfo]:
    return [
        _make_equity(symbol="AAPL", exchange="nasdaq", price=195.0, sector="Technology"),
        _make_equity(symbol="IBM", exchange="NYSE", price=160.0, sector="Technology"),
        _make_equity(symbol="LOWP", exchange="NYSE", price=4.0),
        _make_equity(symbol="HIGH", exchange="NASDAQ", price=250.0),
        _make_equity(symbol="ILLQ", exchange="NASDAQ", avg_dollar_volume_20d=1_000.0),
        _make_equity(symbol="SMALL", exchange="NASDAQ", market_cap=500_000_000.0),
        _make_equity(symbol="OTCQ", exchange="NASDAQ", is_otc=True),
        _make_equity(symbol="HALT", exchange="NYSE", is_halted=True),
        _make_equity(symbol="AMEX1", exchange="AMEX"),
    ]


@pytest.fixture
def event_bus() -> CapturingEventBus:
    return CapturingEventBus()


@pytest.fixture
def bindable_logger() -> BindableListLogger:
    return BindableListLogger()


def test_metadata_verification() -> None:
    metadata = UniverseBuilder.metadata
    assert metadata.name == "universe_builder"
    assert metadata.version == "1.0.0"
    assert metadata.category is PluginCategory.DATA_SOURCE
    assert metadata.enabled is True
    assert "universe" in (metadata.description or "").lower()


def test_constructor_with_universe_filter_criteria(valid_criteria: UniverseFilterCriteria) -> None:
    builder = UniverseBuilder(valid_criteria)

    assert builder._criteria == valid_criteria
    assert builder._run_id == "unknown"
    assert builder._module_name == builder.metadata.name
    assert builder._system_version == "dev"


@dataclass
class FakeRunContext:
    logger: Any
    event_bus: Any
    run_id: str
    system_version: str


def test_init_with_context_mapping_sets_run_metadata(
    valid_criteria: UniverseFilterCriteria,
    event_bus: CapturingEventBus,
    bindable_logger: BindableListLogger,
) -> None:
    builder = UniverseBuilder(valid_criteria)

    result = builder.init(
        {
            "logger": bindable_logger,
            "event_bus": event_bus,
            "run_id": "run-123",
            "system_version": "sys-1.2.3",
            "module_name": "custom_universe_builder",
        }
    )

    assert result.status is ResultStatus.SUCCESS
    assert builder._event_bus is event_bus
    assert builder._run_id == "run-123"
    assert builder._system_version == "sys-1.2.3"
    assert builder._module_name == "custom_universe_builder"
    assert {"module": "custom_universe_builder", "run_id": "run-123"} in bindable_logger.bind_calls


def test_init_with_context_run_context_fallbacks(
    valid_criteria: UniverseFilterCriteria,
    event_bus: CapturingEventBus,
) -> None:
    logger = ListLogger()
    run_context = FakeRunContext(
        logger=logger,
        event_bus=event_bus,
        run_id="run-from-context",
        system_version="system-from-context",
    )
    builder = UniverseBuilder(valid_criteria)

    result = builder.init({"run_context": run_context})

    assert result.status is ResultStatus.SUCCESS
    assert builder._event_bus is event_bus
    assert builder._run_id == "run-from-context"
    assert builder._system_version == "system-from-context"


def test_init_returns_failed_when_criteria_invalid() -> None:
    invalid_criteria = UniverseFilterCriteria(
        exchanges=[],
        min_price=5.0,
        max_price=None,
        min_avg_dollar_volume_20d=10_000_000.0,
        min_market_cap=None,
        exclude_otc=True,
        exclude_halted=True,
        max_results=None,
    )
    builder = UniverseBuilder(invalid_criteria)

    result = builder.init({})

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ValidationError)
    assert result.reason_code == builder._CONFIG_INVALID_REASON


def test_validate_config_normalises_exchanges_and_types() -> None:
    criteria = UniverseFilterCriteria(
        exchanges=[" nyse ", "NASDAQ", "nasdaq", ""],
        min_price=5.0,
        max_price=None,
        min_avg_dollar_volume_20d=0,
        min_market_cap=None,
        exclude_otc=True,
        exclude_halted=False,
        max_results=100,
    )
    builder = UniverseBuilder(criteria)

    result = builder.validate_config(criteria)

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.exchanges == ["NYSE", "NASDAQ"]
    assert result.data.min_price == 0.0
    assert result.data.max_price == 2000.0
    assert result.data.min_market_cap is None
    assert result.data.exclude_otc is True
    assert result.data.exclude_halted is False
    assert result.data.max_results == 100
    assert isinstance(result.data.min_price, float)


@pytest.mark.parametrize(
    ("criteria_overrides", "expected_substring"),
    [
        ({"exchanges": []}, "exchanges"),
        ({"min_price": -1.0}, "min_price"),
        ({"max_price": 0.0}, "max_price"),
        ({"max_price": 1.0, "min_price": 5.0}, "max_price"),
        ({"min_avg_dollar_volume_20d": -0.01}, "min_avg_dollar_volume_20d"),
        ({"min_market_cap": -1.0}, "min_market_cap"),
        ({"max_results": 0}, "max_results"),
    ],
)
def test_validate_config_invalid_inputs_fail(
    criteria_overrides: Mapping[str, Any],
    expected_substring: str,
) -> None:
    base = UniverseFilterCriteria(
        exchanges=["NYSE"],
        min_price=5.0,
        max_price=None,
        min_avg_dollar_volume_20d=10_000_000.0,
        min_market_cap=None,
        exclude_otc=True,
        exclude_halted=True,
        max_results=None,
    )
    criteria = UniverseFilterCriteria(
        exchanges=criteria_overrides.get("exchanges", base.exchanges),
        min_price=criteria_overrides.get("min_price", base.min_price),
        max_price=criteria_overrides.get("max_price", base.max_price),
        min_avg_dollar_volume_20d=criteria_overrides.get(
            "min_avg_dollar_volume_20d", base.min_avg_dollar_volume_20d
        ),
        min_market_cap=criteria_overrides.get("min_market_cap", base.min_market_cap),
        exclude_otc=criteria_overrides.get("exclude_otc", base.exclude_otc),
        exclude_halted=criteria_overrides.get("exclude_halted", base.exclude_halted),
        max_results=criteria_overrides.get("max_results", base.max_results),
    )
    builder = UniverseBuilder(criteria)

    result = builder.validate_config(criteria)

    assert result.status is ResultStatus.FAILED
    assert isinstance(result.error, ValidationError)
    assert result.reason_code == builder._CONFIG_INVALID_REASON
    assert expected_substring in str(result.error).lower()


def test_format_log_includes_payload() -> None:
    formatted = UniverseBuilder._format_log("event", {"x": 1, "y": "a"})
    assert formatted.startswith("event |")
    assert "x=1" in formatted
    assert "y='a'" in formatted


def test_apply_filters_exchange_filtering(
    valid_criteria: UniverseFilterCriteria,
    sample_equities: Sequence[EquityInfo],
) -> None:
    builder = UniverseBuilder(valid_criteria)

    filtered = builder._apply_filters(sample_equities, valid_criteria)

    symbols = {equity.symbol for equity in filtered}
    assert "AMEX1" in symbols
    assert "AAPL" in symbols
    assert "IBM" in symbols


def test_apply_filters_price_floor_excludes_zero_and_negative(
    valid_criteria: UniverseFilterCriteria,
) -> None:
    builder = UniverseBuilder(valid_criteria)

    equities = [
        _make_equity(symbol="ZERO", exchange="NYSE", price=0.0),
        _make_equity(symbol="NEG", exchange="NYSE", price=-1.0),
        _make_equity(symbol="POS", exchange="NYSE", price=0.01),
    ]
    filtered = builder._apply_filters(equities, valid_criteria)

    symbols = {equity.symbol for equity in filtered}
    assert "ZERO" not in symbols
    assert "NEG" not in symbols
    assert "POS" in symbols


def test_apply_filters_price_ceiling_excludes_above_2000(
    valid_criteria: UniverseFilterCriteria,
) -> None:
    builder = UniverseBuilder(valid_criteria)

    equities = [
        _make_equity(symbol="OK", exchange="NYSE", price=2000.0),
        _make_equity(symbol="TOO_HIGH", exchange="NYSE", price=2000.01),
    ]
    filtered = builder._apply_filters(equities, valid_criteria)

    symbols = {equity.symbol for equity in filtered}
    assert "OK" in symbols
    assert "TOO_HIGH" not in symbols


def test_apply_filters_liquidity_filtering(
    valid_criteria: UniverseFilterCriteria,
    sample_equities: Sequence[EquityInfo],
) -> None:
    builder = UniverseBuilder(valid_criteria)

    filtered = builder._apply_filters(sample_equities, valid_criteria)

    symbols = {equity.symbol for equity in filtered}
    assert "ILLQ" not in symbols


def test_apply_filters_does_not_require_market_cap(
    valid_criteria: UniverseFilterCriteria,
    sample_equities: Sequence[EquityInfo],
) -> None:
    builder = UniverseBuilder(valid_criteria)

    equities = list(sample_equities) + [
        _make_equity(symbol="NOCAP", exchange="NYSE", market_cap=None),
    ]
    filtered = builder._apply_filters(sample_equities, valid_criteria)

    symbols = {equity.symbol for equity in filtered}
    assert "SMALL" in symbols

    filtered_with_none = builder._apply_filters(equities, valid_criteria)
    symbols_with_none = {equity.symbol for equity in filtered_with_none}
    assert "NOCAP" in symbols_with_none


def test_apply_filters_exclude_otc_filtering(
    valid_criteria: UniverseFilterCriteria,
    sample_equities: Sequence[EquityInfo],
) -> None:
    builder = UniverseBuilder(valid_criteria)

    filtered = builder._apply_filters(sample_equities, valid_criteria)

    symbols = {equity.symbol for equity in filtered}
    assert "OTCQ" not in symbols


def test_apply_filters_exclude_halted_filtering(
    valid_criteria: UniverseFilterCriteria,
    sample_equities: Sequence[EquityInfo],
) -> None:
    builder = UniverseBuilder(valid_criteria)

    filtered = builder._apply_filters(sample_equities, valid_criteria)

    symbols = {equity.symbol for equity in filtered}
    assert "HALT" not in symbols


def test_apply_filters_include_otc_when_disabled(
    valid_criteria: UniverseFilterCriteria,
    sample_equities: Sequence[EquityInfo],
) -> None:
    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["exclude_otc"] = False
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    filtered = builder._apply_filters(sample_equities, criteria)

    symbols = {equity.symbol for equity in filtered}
    assert "OTCQ" in symbols


def test_apply_filters_include_halted_when_disabled(
    valid_criteria: UniverseFilterCriteria,
    sample_equities: Sequence[EquityInfo],
) -> None:
    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["exclude_halted"] = False
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    filtered = builder._apply_filters(sample_equities, criteria)

    symbols = {equity.symbol for equity in filtered}
    assert "HALT" in symbols


def test_apply_filters_max_results_limit(
    valid_criteria: UniverseFilterCriteria,
) -> None:
    equities = [
        _make_equity(symbol="BIG", exchange="NYSE", price=10.0, avg_dollar_volume_20d=50_000_000.0, market_cap=50_000_000_000.0),
        _make_equity(symbol="SMALL", exchange="NYSE", price=10.0, avg_dollar_volume_20d=20_000_000.0, market_cap=2_000_000_000.0),
        _make_equity(symbol="UNK", exchange="NYSE", price=10.0, avg_dollar_volume_20d=100_000_000.0, market_cap=None),
    ]
    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["max_results"] = 2
    criteria_payload["min_avg_dollar_volume_20d"] = 0.0
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    filtered = builder._apply_filters(equities, criteria)

    assert [equity.symbol for equity in filtered] == ["SMALL", "UNK"]


def test_apply_filters_sorting_downweights_large_caps_when_capped(
    valid_criteria: UniverseFilterCriteria,
) -> None:
    equities = [
        _make_equity(symbol="BIG", exchange="NYSE", price=10.0, avg_dollar_volume_20d=500_000_000.0, market_cap=100_000_000_000.0),
        _make_equity(symbol="SMALL", exchange="NYSE", price=10.0, avg_dollar_volume_20d=50_000_000.0, market_cap=2_000_000_000.0),
    ]
    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["max_results"] = 1
    criteria_payload["min_avg_dollar_volume_20d"] = 0.0
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    filtered = builder._apply_filters(equities, criteria)

    assert [equity.symbol for equity in filtered] == ["SMALL"]


def test_apply_filters_with_details_enrich_enabled(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["min_avg_dollar_volume_20d"] = 0.0
    criteria_payload["enable_details_enrich"] = True
    criteria_payload["details_enrich_top_k"] = 3
    criteria_payload["max_results"] = None
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    equities = [
        _make_equity(symbol="S1", exchange="NYSE", avg_dollar_volume_20d=100.0, market_cap=None),
        _make_equity(symbol="S2", exchange="NYSE", avg_dollar_volume_20d=90.0, market_cap=None),
        _make_equity(symbol="S3", exchange="NYSE", avg_dollar_volume_20d=80.0, market_cap=None),
        _make_equity(symbol="S4", exchange="NYSE", avg_dollar_volume_20d=70.0, market_cap=None),
        _make_equity(symbol="S5", exchange="NYSE", avg_dollar_volume_20d=60.0, market_cap=None),
    ]

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    called: dict[str, Any] = {}

    def _fake_details(symbols: list[str]) -> dict[str, dict[str, Any]]:
        called["symbols"] = list(symbols)
        return {
            "S1": {"market_cap": 5_000_000_000.0, "sector": "Tech"},
            "S2": {"market_cap": 20_000_000_000.0, "sector": "Finance"},
            "S3": {"market_cap": 6_000_000_000.0, "sector": None},
        }

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_ticker_details", _fake_details)

    filtered = builder._apply_filters(equities, criteria)

    assert called["symbols"] == ["S1", "S2", "S3"]
    assert [equity.symbol for equity in filtered] == ["S1", "S3", "S4", "S5", "S2"]
    by_symbol = {equity.symbol: equity for equity in filtered}
    assert by_symbol["S1"].market_cap == 5_000_000_000.0
    assert by_symbol["S1"].sector == "Tech"


def test_apply_filters_with_details_enrich_disabled(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["min_avg_dollar_volume_20d"] = 0.0
    criteria_payload["enable_details_enrich"] = False
    criteria_payload["details_enrich_top_k"] = 3
    criteria_payload["max_results"] = None
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    equities = [
        _make_equity(symbol="S1", exchange="NYSE", avg_dollar_volume_20d=100.0, market_cap=None),
        _make_equity(symbol="S2", exchange="NYSE", avg_dollar_volume_20d=90.0, market_cap=None),
        _make_equity(symbol="S3", exchange="NYSE", avg_dollar_volume_20d=80.0, market_cap=None),
    ]

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    def _should_not_be_called(symbols: list[str]) -> dict[str, dict[str, Any]]:
        raise AssertionError("details enrichment should be disabled")

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_ticker_details", _should_not_be_called)

    filtered = builder._apply_filters(equities, criteria)
    assert [equity.symbol for equity in filtered] == ["S1", "S2", "S3"]


@pytest.mark.parametrize(
    ("details_payload", "expected_status", "expect_warning", "expect_revert"),
    [
        (
            {
                "S1": {"market_cap": 5_000_000_000.0},
                "S2": {"market_cap": 6_000_000_000.0},
            },
            ResultStatus.SUCCESS,
            False,
            False,
        ),
        (
            {
                "S1": {"market_cap": 5_000_000_000.0},
            },
            ResultStatus.DEGRADED,
            True,
            True,
        ),
    ],
)
def test_apply_filters_with_partial_enrichment_success(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    details_payload: dict[str, dict[str, Any]],
    expected_status: ResultStatus,
    expect_warning: bool,
    expect_revert: bool,
) -> None:
    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["min_avg_dollar_volume_20d"] = 0.0
    criteria_payload["enable_details_enrich"] = True
    criteria_payload["details_enrich_top_k"] = 4
    criteria_payload["max_results"] = None
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    equities = [
        _make_equity(symbol="S1", exchange="NYSE", avg_dollar_volume_20d=100.0, market_cap=None),
        _make_equity(symbol="S2", exchange="NYSE", avg_dollar_volume_20d=90.0, market_cap=None),
        _make_equity(symbol="S3", exchange="NYSE", avg_dollar_volume_20d=80.0, market_cap=None),
        _make_equity(symbol="S4", exchange="NYSE", avg_dollar_volume_20d=70.0, market_cap=None),
        _make_equity(symbol="S5", exchange="NYSE", avg_dollar_volume_20d=60.0, market_cap=None),
    ]

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    caplog.set_level(logging.INFO, logger="universe.builder")

    captured: dict[str, Any] = {}

    def _fake_details(symbols: list[str]) -> dict[str, dict[str, Any]]:
        captured["symbols"] = list(symbols)
        return dict(details_payload)

    def fetch_polygon() -> Result[list[EquityInfo]]:
        return Result.success(list(equities))

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)
    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_ticker_details", _fake_details)

    result = builder.execute()

    assert captured["symbols"] == ["S1", "S2", "S3", "S4"]
    assert result.status is expected_status
    assert result.data is not None

    by_symbol = {equity.symbol: equity for equity in result.data.equities}
    if expect_revert:
        assert by_symbol["S1"].market_cap is None
        assert by_symbol["S2"].market_cap is None
    else:
        assert by_symbol["S1"].market_cap == 5_000_000_000.0
        assert by_symbol["S2"].market_cap == details_payload.get("S2", {}).get("market_cap")

    warning_hit = any("universe.details_enrich_low_coverage" in rec.message for rec in caplog.records)
    assert warning_hit is expect_warning

    if expected_status is ResultStatus.DEGRADED:
        assert result.reason_code == builder._DETAILS_ENRICH_DEGRADED_REASON


def test_apply_filters_with_enrichment_failure(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["min_avg_dollar_volume_20d"] = 0.0
    criteria_payload["enable_details_enrich"] = True
    criteria_payload["details_enrich_top_k"] = 3
    criteria_payload["max_results"] = None
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    equities = [
        _make_equity(symbol="S1", exchange="NYSE", avg_dollar_volume_20d=100.0, market_cap=None),
        _make_equity(symbol="S2", exchange="NYSE", avg_dollar_volume_20d=90.0, market_cap=None),
        _make_equity(symbol="S3", exchange="NYSE", avg_dollar_volume_20d=80.0, market_cap=None),
    ]

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    caplog.set_level(logging.WARNING, logger="universe.builder")

    def _fake_details(symbols: list[str]) -> dict[str, dict[str, Any]]:
        raise RuntimeError("details down")

    def fetch_polygon() -> Result[list[EquityInfo]]:
        return Result.success(list(equities))

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)
    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_ticker_details", _fake_details)

    result = builder.execute()

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == builder._DETAILS_ENRICH_DEGRADED_REASON
    assert result.data is not None
    assert all(equity.market_cap is None for equity in result.data.equities)
    assert any("universe.details_enrich_failed" in rec.message for rec in caplog.records)


def test_apply_filters_top_k_calculation(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    captured: dict[str, Any] = {}

    def _fake_details(symbols: list[str]) -> dict[str, dict[str, Any]]:
        captured["symbols"] = list(symbols)
        return {symbol: {"market_cap": 1.0} for symbol in symbols}

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_ticker_details", _fake_details)

    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["min_avg_dollar_volume_20d"] = 0.0
    criteria_payload["enable_details_enrich"] = True
    criteria_payload["details_enrich_top_k"] = 500
    criteria_payload["max_results"] = None
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    equities = [
        _make_equity(symbol=f"SYM{i:04d}", exchange="NYSE", avg_dollar_volume_20d=float(10_000 - i), market_cap=None)
        for i in range(3000)
    ]
    builder._apply_filters(equities, criteria)
    assert len(captured["symbols"]) == 500
    assert captured["symbols"][0] == "SYM0000"

    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["min_avg_dollar_volume_20d"] = 0.0
    criteria_payload["enable_details_enrich"] = True
    criteria_payload["details_enrich_top_k"] = 500
    criteria_payload["details_enrich_multiplier"] = 1.5
    criteria_payload["max_results"] = 100
    criteria2 = UniverseFilterCriteria(**criteria_payload)
    builder2 = UniverseBuilder(criteria2)

    equities2 = [
        _make_equity(symbol=f"X{i:04d}", exchange="NYSE", avg_dollar_volume_20d=float(10_000 - i), market_cap=None)
        for i in range(200)
    ]
    builder2._apply_filters(equities2, criteria2)
    assert len(captured["symbols"]) == 150


def test_apply_filters_no_polygon_api_key(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import patch

    criteria_payload = msgspec.to_builtins(valid_criteria)
    criteria_payload["min_avg_dollar_volume_20d"] = 0.0
    criteria_payload["enable_details_enrich"] = True
    criteria_payload["details_enrich_top_k"] = 3
    criteria_payload["max_results"] = None
    criteria = UniverseFilterCriteria(**criteria_payload)
    builder = UniverseBuilder(criteria)

    equities = [
        _make_equity(symbol="S1", exchange="NYSE", avg_dollar_volume_20d=100.0, market_cap=None),
        _make_equity(symbol="S2", exchange="NYSE", avg_dollar_volume_20d=90.0, market_cap=None),
        _make_equity(symbol="S3", exchange="NYSE", avg_dollar_volume_20d=80.0, market_cap=None),
    ]

    def _should_not_be_called(symbols: list[str]) -> dict[str, dict[str, Any]]:
        raise AssertionError("details enrichment should be skipped without API key")

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_ticker_details", _should_not_be_called)

    def _env_get(key: str, default: Any = None) -> Any:
        if key == "POLYGON_API_KEY":
            return None
        return default

    with patch("os.environ.get", side_effect=_env_get):
        filtered = builder._apply_filters(equities, criteria)

    assert [equity.symbol for equity in filtered] == ["S1", "S2", "S3"]


def test_execute_success_fetches_filters_and_returns_snapshot(
    valid_criteria: UniverseFilterCriteria,
    sample_equities: Sequence[EquityInfo],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = UniverseBuilder(valid_criteria)

    def fetch_polygon() -> Result[list[EquityInfo]]:
        return Result.success(list(sample_equities))

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)

    result = builder.execute()

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.source == "polygon"
    assert result.data.total_candidates == len(sample_equities)
    assert result.data.total_filtered == len(result.data.equities)
    assert result.data.total_filtered == 6


def test_execute_emits_universe_built_event_known_run_id(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
    event_bus: CapturingEventBus,
) -> None:
    builder = UniverseBuilder(valid_criteria)
    builder.init({"event_bus": event_bus, "run_id": "run-123"})

    def fetch_polygon() -> Result[list[EquityInfo]]:
        return Result.success([_make_equity(symbol="AAPL", exchange="NASDAQ", price=100.0)])

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)

    result = builder.execute()

    assert result.status is ResultStatus.SUCCESS
    assert len(event_bus.published) == 1
    topic, event = event_bus.published[0]
    assert topic == "runs.run-123.universe_builder.universe.built"
    assert event.event_type == "universe.built"
    assert event.run_id == "run-123"
    assert event.data is not None
    assert event.data["total_candidates"] == 1
    assert event.data["total_filtered"] == 1


def test_execute_emits_universe_built_event_unknown_run_id_defaults_to_events_topic(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
    event_bus: CapturingEventBus,
) -> None:
    builder = UniverseBuilder(valid_criteria)
    builder.init({"event_bus": event_bus})

    def fetch_polygon() -> Result[list[EquityInfo]]:
        return Result.success([])

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)

    result = builder.execute()

    assert result.status is ResultStatus.SUCCESS
    assert len(event_bus.published) == 1
    topic, event = event_bus.published[0]
    assert topic == "events.universe.built"
    assert event.run_id == "unknown"


def test_execute_event_publish_failure_is_suppressed(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = UniverseBuilder(valid_criteria)
    builder.init({"event_bus": RaisingEventBus(lambda: RuntimeError("bus down")), "run_id": "run-123"})

    def fetch_polygon() -> Result[list[EquityInfo]]:
        return Result.success([])

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)

    result = builder.execute()

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None


def test_execute_includes_system_version_from_context(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = UniverseBuilder(valid_criteria)
    builder.init({"system_version": "system-xyz"})

    def fetch_polygon() -> Result[list[EquityInfo]]:
        return Result.success([])

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)

    result = builder.execute()

    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.system_version == "system-xyz"


def test_execute_degraded_when_polygon_fails_but_fmp_succeeds(
    valid_criteria: UniverseFilterCriteria,
    sample_equities: Sequence[EquityInfo],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = UniverseBuilder(valid_criteria)
    calls: list[str] = []

    def fetch_polygon() -> Result[list[EquityInfo]]:
        calls.append("polygon")
        err = DataSourceUnavailableError(
            "polygon unavailable",
            module="tests.universe",
            reason_code="TEST_POLYGON_FAILED",
        )
        return Result.failed(err, err.reason_code)

    def fetch_fmp() -> Result[list[EquityInfo]]:
        calls.append("fmp")
        return Result.success(list(sample_equities))

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)
    monkeypatch.setattr("universe.builder.data_sources.fetch_fmp_universe", fetch_fmp)

    result = builder.execute()

    assert calls == ["polygon", "fmp"]
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.source == "fmp"
    assert result.reason_code == builder._SOURCE_FALLBACK_REASON


def test_execute_degraded_when_polygon_raises_permission_error_but_fmp_succeeds(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = UniverseBuilder(valid_criteria)
    calls: list[str] = []

    def fetch_polygon() -> Result[list[EquityInfo]]:
        calls.append("polygon")
        raise PermissionError("denied")

    def fetch_fmp() -> Result[list[EquityInfo]]:
        calls.append("fmp")
        return Result.success([])

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)
    monkeypatch.setattr("universe.builder.data_sources.fetch_fmp_universe", fetch_fmp)

    result = builder.execute()

    assert calls == ["polygon", "fmp"]
    assert result.status is ResultStatus.DEGRADED
    assert isinstance(result.error, DataSourceUnavailableError)
    assert result.reason_code == builder._SOURCE_FALLBACK_REASON


def test_execute_degraded_when_polygon_fails_and_fmp_fails_but_cache_succeeds(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = UniverseBuilder(valid_criteria)
    calls: list[str] = []

    def fetch_polygon() -> Result[list[EquityInfo]]:
        calls.append("polygon")
        err = DataSourceUnavailableError(
            "polygon down",
            module="tests.universe",
            reason_code="TEST_POLYGON_DOWN",
        )
        return Result.failed(err, err.reason_code)

    def fetch_fmp() -> Result[list[EquityInfo]]:
        calls.append("fmp")
        err = RuntimeError("fmp down")
        return Result.failed(err, "TEST_FMP_DOWN")

    def fetch_cache() -> Result[list[EquityInfo]]:
        calls.append("cache")
        return Result.success([])

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)
    monkeypatch.setattr("universe.builder.data_sources.fetch_fmp_universe", fetch_fmp)
    monkeypatch.setattr("universe.builder.data_sources.fetch_cached_universe", fetch_cache)

    result = builder.execute()

    assert calls == ["polygon", "fmp", "cache"]
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert result.data.source == "cache"
    assert result.reason_code == builder._SOURCE_FALLBACK_REASON


def test_execute_degraded_when_all_sources_fail_returns_empty_snapshot(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = UniverseBuilder(valid_criteria)
    calls: list[str] = []

    def fetch_polygon() -> Result[list[EquityInfo]]:
        calls.append("polygon")
        raise RuntimeError("polygon crash")

    def fetch_fmp() -> Result[list[EquityInfo]]:
        calls.append("fmp")
        err = RuntimeError("fmp crash")
        return Result.failed(err, "TEST_FMP_FAILED")

    def fetch_cache() -> Result[list[EquityInfo]]:
        calls.append("cache")
        raise RuntimeError("cache crash")

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)
    monkeypatch.setattr("universe.builder.data_sources.fetch_fmp_universe", fetch_fmp)
    monkeypatch.setattr("universe.builder.data_sources.fetch_cached_universe", fetch_cache)

    result = builder.execute()

    assert calls == ["polygon", "fmp", "cache"]
    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == builder._ALL_SOURCES_FAILED_REASON
    assert result.data is not None
    assert result.data.source == "cache_fallback"
    assert result.data.total_candidates == 0
    assert result.data.total_filtered == 0
    assert result.data.equities == []


def test_execute_degraded_when_source_returns_degraded_data(
    valid_criteria: UniverseFilterCriteria,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = UniverseBuilder(valid_criteria)

    def fetch_polygon() -> Result[list[EquityInfo]]:
        equities = [_make_equity(symbol="AAPL", exchange="NASDAQ", price=100.0)]
        return Result.degraded(equities, RuntimeError("partial"), "TEST_DEGRADED_SOURCE")

    monkeypatch.setattr("universe.builder.data_sources.fetch_polygon_universe", fetch_polygon)

    result = builder.execute()

    assert result.status is ResultStatus.DEGRADED
    assert result.reason_code == "TEST_DEGRADED_SOURCE"
    assert result.data is not None
    assert result.data.source == "polygon"
    assert [equity.symbol for equity in result.data.equities] == ["AAPL"]


def test_cleanup_returns_success(valid_criteria: UniverseFilterCriteria) -> None:
    builder = UniverseBuilder(valid_criteria)

    result = builder.cleanup()

    assert result.status is ResultStatus.SUCCESS
