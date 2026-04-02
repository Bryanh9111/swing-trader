from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping

import msgspec
import pytest

from common.exceptions import OperationalError, PartialDataError, RecoverableError
from common.events import InMemoryEventBus
from common.interface import Result, ResultStatus
from common.logging import StructlogLoggerFactory
from data.interface import PriceBar, PriceSeriesSnapshot
from data.orchestrator import DataOrchestrator
from journal import ArtifactManager, JournalWriter
from orchestrator import DataPlugin, EODScanOrchestrator, ScannerPlugin, UniversePlugin, register_real_plugins
from plugins.registry import PluginRegistry
from scanner.interface import CandidateSet, PlatformCandidate, PlatformFeatures, ScannerConfig
from universe.builder import UniverseBuilder
from universe.interface import EquityInfo, UniverseFilterCriteria, UniverseSnapshot


@pytest.fixture()
def universe_snapshot() -> UniverseSnapshot:
    criteria = UniverseFilterCriteria(
        exchanges=["NASDAQ"],
        min_price=5.0,
        max_price=None,
        min_avg_dollar_volume_20d=0.0,
        min_market_cap=None,
        exclude_otc=True,
        exclude_halted=True,
        max_results=None,
    )
    equities = [
        EquityInfo(
            symbol="AAPL",
            exchange="NASDAQ",
            price=195.0,
            avg_dollar_volume_20d=12_500_000_000.0,
            market_cap=3_000_000_000_000.0,
            is_otc=False,
            is_halted=False,
            sector="Technology",
        ),
        EquityInfo(
            symbol="MSFT",
            exchange="NASDAQ",
            price=410.0,
            avg_dollar_volume_20d=9_500_000_000.0,
            market_cap=3_100_000_000_000.0,
            is_otc=False,
            is_halted=False,
            sector="Technology",
        ),
    ]
    return UniverseSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1_700_000_000_000_000_000,
        source="cache",
        equities=equities,
        filter_criteria=msgspec.to_builtins(criteria),
        total_candidates=len(equities),
        total_filtered=len(equities),
    )


@pytest.fixture()
def universe_payload(universe_snapshot: UniverseSnapshot) -> Mapping[str, Any]:
    return msgspec.to_builtins(universe_snapshot)


@pytest.fixture()
def price_bars() -> list[PriceBar]:
    base_ts = 1_700_000_000_000_000_000
    return [
        PriceBar(timestamp=base_ts, open=100.0, high=103.0, low=99.0, close=102.0, volume=1_000_000),
        PriceBar(timestamp=base_ts + 86_400_000_000_000, open=102.0, high=104.0, low=101.0, close=103.0, volume=900_000),
        PriceBar(timestamp=base_ts + 2 * 86_400_000_000_000, open=103.0, high=105.0, low=102.0, close=104.0, volume=850_000),
    ]


def _price_series(symbol: str, bars: list[PriceBar], *, source: str = "yahoo") -> PriceSeriesSnapshot:
    return PriceSeriesSnapshot(
        schema_version="1.0.0",
        system_version="test",
        asof_timestamp=1_700_000_000_000_000_000,
        symbol=symbol,
        timeframe="1d",
        bars=bars,
        source=source,
        quality_flags={"availability": True, "completeness": True, "freshness": True, "stability": True},
    )


def _platform_candidate(symbol: str, *, window: int, detected_at: int) -> PlatformCandidate:
    return PlatformCandidate(
        symbol=symbol,
        detected_at=detected_at,
        window=window,
        score=0.88,
        features=PlatformFeatures(
            box_range=0.05,
            box_low=100.0,
            box_high=105.0,
            ma_diff=0.01,
            volatility=0.02,
            volume_change_ratio=0.8,
            volume_stability=0.25,
            avg_dollar_volume=50_000_000.0,
            box_quality=0.9,
        ),
        invalidation_level=99.0,
        target_level=120.0,
        reasons=["test"],
        meta={"window": window},
    )


def test_universe_plugin_success_degraded_failed(monkeypatch: pytest.MonkeyPatch, universe_snapshot: UniverseSnapshot) -> None:
    plugin = UniversePlugin()
    assert plugin.validate_config({}).status is ResultStatus.SUCCESS
    assert plugin.init({}).status is ResultStatus.SUCCESS

    expected_payload = msgspec.to_builtins(universe_snapshot)

    def fake_execute_success(self: UniverseBuilder, payload: None = None) -> Result[UniverseSnapshot]:
        return Result.success(universe_snapshot)

    def fake_execute_degraded(self: UniverseBuilder, payload: None = None) -> Result[UniverseSnapshot]:
        error = RecoverableError("Universe incomplete.", module="tests.universe", reason_code="TEST_UNIVERSE_DEGRADED")
        return Result.degraded(universe_snapshot, error, error.reason_code)

    def fake_execute_failed(self: UniverseBuilder, payload: None = None) -> Result[UniverseSnapshot]:
        error = OperationalError("Universe failed.", module="tests.universe", reason_code="TEST_UNIVERSE_FAILED")
        return Result.failed(error, error.reason_code)

    monkeypatch.setattr(UniverseBuilder, "execute", fake_execute_success)
    result = plugin.execute({})
    assert result.status is ResultStatus.SUCCESS
    assert result.data == expected_payload

    monkeypatch.setattr(UniverseBuilder, "execute", fake_execute_degraded)
    result = plugin.execute({})
    assert result.status is ResultStatus.DEGRADED
    assert result.data == expected_payload
    assert isinstance(result.error, RecoverableError)
    assert result.reason_code == "TEST_UNIVERSE_DEGRADED"

    monkeypatch.setattr(UniverseBuilder, "execute", fake_execute_failed)
    result = plugin.execute({})
    assert result.status is ResultStatus.FAILED
    assert result.data is None
    assert isinstance(result.error, OperationalError)
    assert result.reason_code == "TEST_UNIVERSE_FAILED"


def test_data_plugin_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, universe_payload: Mapping[str, Any], price_bars: list[PriceBar]) -> None:
    plugin = DataPlugin(config={"cache_dir": str(tmp_path / "cache"), "lookback_days": 10, "enable_incremental_cache": False})
    assert plugin.validate_config(plugin.config).status is ResultStatus.SUCCESS
    assert plugin.init({}).status is ResultStatus.SUCCESS

    def fake_fetch(self: DataOrchestrator, symbol: str, start: date, end: date) -> Result[PriceSeriesSnapshot]:
        return Result.success(_price_series(symbol, price_bars))

    monkeypatch.setattr(DataOrchestrator, "fetch_with_cache", fake_fetch)

    result = plugin.execute(universe_payload)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert set(result.data["series_by_symbol"].keys()) == {"AAPL", "MSFT"}  # type: ignore[union-attr]


def test_data_plugin_with_incremental_cache_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    universe_payload: Mapping[str, Any],
    price_bars: list[PriceBar],
) -> None:
    plugin = DataPlugin(config={"cache_dir": str(tmp_path / "cache"), "lookback_days": 10, "enable_incremental_cache": True})
    assert plugin.validate_config(plugin.config).status is ResultStatus.SUCCESS
    assert plugin.init({}).status is ResultStatus.SUCCESS

    calls: list[str] = []

    def fake_fetch_incremental(self: DataOrchestrator, symbol: str, start: date, end: date) -> Result[PriceSeriesSnapshot]:
        calls.append(symbol)
        return Result.success(_price_series(symbol, price_bars))

    def should_not_call(self: DataOrchestrator, symbol: str, start: date, end: date) -> Result[PriceSeriesSnapshot]:
        raise AssertionError("fetch_with_cache should not be called when incremental cache is enabled")

    monkeypatch.setattr(DataOrchestrator, "fetch_with_incremental_cache", fake_fetch_incremental)
    monkeypatch.setattr(DataOrchestrator, "fetch_with_cache", should_not_call)

    result = plugin.execute(universe_payload)
    assert result.status is ResultStatus.SUCCESS
    assert set(calls) == {"AAPL", "MSFT"}


def test_data_plugin_with_incremental_cache_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    universe_payload: Mapping[str, Any],
    price_bars: list[PriceBar],
) -> None:
    plugin = DataPlugin(config={"cache_dir": str(tmp_path / "cache"), "lookback_days": 10, "enable_incremental_cache": False})
    assert plugin.validate_config(plugin.config).status is ResultStatus.SUCCESS
    assert plugin.init({}).status is ResultStatus.SUCCESS

    calls: list[str] = []

    def fake_fetch(self: DataOrchestrator, symbol: str, start: date, end: date) -> Result[PriceSeriesSnapshot]:
        calls.append(symbol)
        return Result.success(_price_series(symbol, price_bars))

    def should_not_call(self: DataOrchestrator, symbol: str, start: date, end: date) -> Result[PriceSeriesSnapshot]:
        raise AssertionError("fetch_with_incremental_cache should not be called when disabled")

    monkeypatch.setattr(DataOrchestrator, "fetch_with_cache", fake_fetch)
    monkeypatch.setattr(DataOrchestrator, "fetch_with_incremental_cache", should_not_call)

    result = plugin.execute(universe_payload)
    assert result.status is ResultStatus.SUCCESS
    assert set(calls) == {"AAPL", "MSFT"}


def test_data_plugin_degraded_partial_symbol_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    universe_payload: Mapping[str, Any],
    price_bars: list[PriceBar],
) -> None:
    plugin = DataPlugin(config={"cache_dir": str(tmp_path / "cache"), "lookback_days": 10, "enable_incremental_cache": False})
    assert plugin.validate_config(plugin.config).status is ResultStatus.SUCCESS
    assert plugin.init({}).status is ResultStatus.SUCCESS

    def fake_fetch(self: DataOrchestrator, symbol: str, start: date, end: date) -> Result[PriceSeriesSnapshot]:
        if symbol == "MSFT":
            error = OperationalError("MSFT fetch failed.", module="tests.data", reason_code="TEST_MSFT_FAILED")
            return Result.failed(error, error.reason_code)
        return Result.success(_price_series(symbol, price_bars))

    monkeypatch.setattr(DataOrchestrator, "fetch_with_cache", fake_fetch)

    result = plugin.execute(universe_payload)
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert set(result.data["series_by_symbol"].keys()) == {"AAPL"}  # type: ignore[union-attr]
    assert isinstance(result.error, PartialDataError)
    assert result.reason_code == "DATA_PARTIAL"


def test_scanner_plugin_multi_window_aggregation(monkeypatch: pytest.MonkeyPatch, universe_snapshot: UniverseSnapshot, price_bars: list[PriceBar]) -> None:
    plugin = ScannerPlugin(config={"windows": [10, 20], "window_weights": {10: 0.5, 20: 0.5}})
    assert plugin.validate_config(plugin.config).status is ResultStatus.SUCCESS
    assert plugin.init({}).status is ResultStatus.SUCCESS

    payload = {
        "universe": msgspec.to_builtins(universe_snapshot),
        "series_by_symbol": {
            "AAPL": msgspec.to_builtins(_price_series("AAPL", price_bars)),
            "MSFT": msgspec.to_builtins(_price_series("MSFT", price_bars)),
        },
    }

    calls: list[tuple[str, int]] = []

    def fake_detect(symbol: str, bars: list[PriceBar], window: int, config: ScannerConfig, detected_at: int) -> Result[PlatformCandidate | None]:
        calls.append((symbol, window))
        if (symbol, window) in {("AAPL", 10), ("MSFT", 20)}:
            return Result.success(_platform_candidate(symbol, window=window, detected_at=detected_at))
        return Result.success(None)

    monkeypatch.setattr("orchestrator.plugins.detect_platform_candidate", fake_detect)

    result = plugin.execute(payload)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert len(calls) == 4

    candidate_set = msgspec.convert(dict(result.data), type=CandidateSet)  # type: ignore[arg-type]
    assert candidate_set.total_scanned == 2
    assert candidate_set.total_detected == 2
    assert {c.symbol for c in candidate_set.candidates} == {"AAPL", "MSFT"}


def test_scanner_plugin_degraded_when_some_calls_fail(
    monkeypatch: pytest.MonkeyPatch,
    universe_snapshot: UniverseSnapshot,
    price_bars: list[PriceBar],
) -> None:
    plugin = ScannerPlugin(config={"windows": [10, 20], "window_weights": {10: 0.5, 20: 0.5}})
    assert plugin.validate_config(plugin.config).status is ResultStatus.SUCCESS
    assert plugin.init({}).status is ResultStatus.SUCCESS

    payload = {
        "universe": msgspec.to_builtins(universe_snapshot),
        "series_by_symbol": {
            "AAPL": msgspec.to_builtins(_price_series("AAPL", price_bars)),
            "MSFT": msgspec.to_builtins(_price_series("MSFT", price_bars)),
        },
    }

    def fake_detect(symbol: str, bars: list[PriceBar], window: int, config: ScannerConfig, detected_at: int) -> Result[PlatformCandidate | None]:
        if (symbol, window) == ("AAPL", 10):
            error = OperationalError("detect failed", module="tests.scanner", reason_code="TEST_DETECT_FAILED")
            return Result.failed(error, error.reason_code)
        return Result.success(None)

    monkeypatch.setattr("orchestrator.plugins.detect_platform_candidate", fake_detect)

    result = plugin.execute(payload)
    assert result.status is ResultStatus.DEGRADED
    assert result.data is not None
    assert isinstance(result.error, PartialDataError)
    assert result.reason_code == "SCANNER_DEGRADED"


def test_scanner_plugin_failed_when_all_symbols_fail(
    monkeypatch: pytest.MonkeyPatch,
    universe_snapshot: UniverseSnapshot,
    price_bars: list[PriceBar],
) -> None:
    plugin = ScannerPlugin(config={"windows": [10, 20], "window_weights": {10: 0.5, 20: 0.5}})
    assert plugin.validate_config(plugin.config).status is ResultStatus.SUCCESS
    assert plugin.init({}).status is ResultStatus.SUCCESS

    payload = {
        "universe": msgspec.to_builtins(universe_snapshot),
        "series_by_symbol": {
            "AAPL": msgspec.to_builtins(_price_series("AAPL", price_bars)),
            "MSFT": msgspec.to_builtins(_price_series("MSFT", price_bars)),
        },
    }

    def fake_detect(symbol: str, bars: list[PriceBar], window: int, config: ScannerConfig, detected_at: int) -> Result[PlatformCandidate | None]:
        error = OperationalError("detect failed", module="tests.scanner", reason_code="TEST_ALL_FAILED")
        return Result.failed(error, error.reason_code)

    monkeypatch.setattr("orchestrator.plugins.detect_platform_candidate", fake_detect)

    result = plugin.execute(payload)
    assert result.status is ResultStatus.FAILED
    assert result.data is None
    assert isinstance(result.error, PartialDataError)
    assert result.reason_code == "SCANNER_ALL_FAILED"


@pytest.fixture()
def orchestrator_env_real(tmp_path: Path) -> EODScanOrchestrator:
    artifact_manager = ArtifactManager(base_path=tmp_path / "artifacts")
    journal_writer = JournalWriter(artifact_manager)
    event_bus = InMemoryEventBus()
    logger_factory = StructlogLoggerFactory()
    registry = PluginRegistry()
    register_real_plugins(registry)
    return EODScanOrchestrator(
        journal_writer=journal_writer,
        event_bus=event_bus,
        logger_factory=logger_factory,
        plugin_registry=registry,
    )


@pytest.mark.parametrize(
    ("msft_data_failed", "expected_status"),
    [
        (False, ResultStatus.SUCCESS),
        (True, ResultStatus.DEGRADED),
    ],
)
def test_end_to_end_eod_scan_real_plugins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    orchestrator_env_real: EODScanOrchestrator,
    universe_snapshot: UniverseSnapshot,
    price_bars: list[PriceBar],
    msft_data_failed: bool,
    expected_status: ResultStatus,
) -> None:
    def fake_universe_execute(self: UniverseBuilder, payload: None = None) -> Result[UniverseSnapshot]:
        return Result.success(universe_snapshot)

    def fake_fetch(self: DataOrchestrator, symbol: str, start: date, end: date) -> Result[PriceSeriesSnapshot]:
        if msft_data_failed and symbol == "MSFT":
            error = OperationalError("MSFT fetch failed.", module="tests.data", reason_code="TEST_MSFT_FAILED")
            return Result.failed(error, error.reason_code)
        return Result.success(_price_series(symbol, price_bars))

    def fake_detect(symbol: str, bars: list[PriceBar], window: int, config: ScannerConfig, detected_at: int) -> Result[PlatformCandidate | None]:
        if symbol == "AAPL" and window == 20:
            return Result.success(_platform_candidate(symbol, window=window, detected_at=detected_at))
        return Result.success(None)

    monkeypatch.setattr(UniverseBuilder, "execute", fake_universe_execute)
    monkeypatch.setattr(DataOrchestrator, "fetch_with_cache", fake_fetch)
    monkeypatch.setattr("orchestrator.plugins.detect_platform_candidate", fake_detect)

    config: dict[str, Any] = {
        "mode": "DRY_RUN",
        "plugins": {
            "data_sources": [
                {"name": "universe_real", "config": {}},
                {"name": "data_real", "config": {"cache_dir": str(tmp_path / "cache"), "lookback_days": 10, "enable_incremental_cache": False}},
            ],
            "scanners": [
                {"name": "scanner_real", "config": {"windows": [20, 30], "window_weights": {20: 0.5, 30: 0.5}}},
            ],
        },
    }

    run_result = orchestrator_env_real.execute_run(config)
    assert run_result.status is expected_status
    assert run_result.data is not None
    assert run_result.data.candidates_count == 1
