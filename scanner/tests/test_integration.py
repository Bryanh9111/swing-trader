from __future__ import annotations

import msgspec
import msgspec.json
import pytest
from typing import Any

from common.interface import Result, ResultStatus
from scanner import CandidateSet, ScannerConfig, detect_platform_candidate

from .conftest import make_breakout_series, make_downtrend_series, make_platform_series

if not hasattr(msgspec.json, "DecodeError"):
    setattr(msgspec.json, "DecodeError", msgspec.DecodeError)
if not hasattr(msgspec.json, "EncodeError"):
    setattr(msgspec.json, "EncodeError", msgspec.EncodeError)


def _normalize_config_snapshot(config_snapshot: dict[str, Any]) -> None:
    window_weights = config_snapshot.get("window_weights")
    if isinstance(window_weights, dict):
        config_snapshot["window_weights"] = {str(k): v for k, v in window_weights.items()}

    for key in ("indicator_scoring", "adx_entry_filter", "ma200_trend_filter"):
        value = config_snapshot.get(key)
        if isinstance(value, msgspec.Struct):
            config_snapshot[key] = msgspec.to_builtins(value)

    indicator_scoring = config_snapshot.get("indicator_scoring")
    if isinstance(indicator_scoring, dict):
        for key in ("rsi_optimal_range", "atr_optimal_range"):
            value = indicator_scoring.get(key)
            if isinstance(value, tuple):
                indicator_scoring[key] = list(value)


def _run_multi_window_scan(
    *,
    symbol: str,
    bars: list[object],
    config: ScannerConfig,
    detected_at: int,
    schema_version: str = "1.0.0",
    system_version: str = "deadbeef",
    data_source: str = "test",
    universe_source: str = "test",
) -> Result[CandidateSet]:
    candidates = []
    failures: list[Result[object]] = []
    for window in config.windows:
        result = detect_platform_candidate(symbol, bars, window, config, detected_at=detected_at)
        if result.status is ResultStatus.SUCCESS and result.data is not None:
            candidates.append(result.data)
        elif result.status is ResultStatus.FAILED:
            failures.append(result)

    snapshot = CandidateSet(
        schema_version=schema_version,
        system_version=system_version,
        asof_timestamp=detected_at,
        candidates=candidates,
        total_scanned=1,
        total_detected=len(candidates),
        config_snapshot=msgspec.to_builtins(config),
        data_source=data_source,
        universe_source=universe_source,
    )

    if failures and candidates:
        return Result.degraded(
            snapshot,
            error=failures[0].error or RuntimeError("window scan failed"),
            reason_code="WINDOW_SCAN_DEGRADED",
        )
    if failures and not candidates:
        return Result.failed(failures[0].error or RuntimeError("window scan failed"), "WINDOW_SCAN_FAILED")
    return Result.success(snapshot)


@pytest.mark.integration
def test_end_to_end_multi_window_detection_produces_candidates() -> None:
    config = ScannerConfig(require_breakout_confirmation=False, require_breakout_volume_spike=False)
    bars = make_platform_series(total_bars=120)

    candidates = []
    for window in [20, 30, 60]:
        result = detect_platform_candidate(
            symbol="AAPL",
            bars=bars,
            window=window,
            config=config,
            detected_at=1,
        )
        assert result.status is ResultStatus.SUCCESS
        assert result.data is not None
        candidates.append(result.data)

    assert {c.window for c in candidates} == {20, 30, 60}
    assert all(0.0 <= c.score <= 1.0 for c in candidates)


@pytest.mark.integration
def test_real_world_like_sequences_filter_candidates_vs_non_candidates() -> None:
    config = ScannerConfig(require_breakout_confirmation=False, require_breakout_volume_spike=False)
    platform = make_platform_series(total_bars=120)
    breakout = make_breakout_series(base_series=platform, breakout_close=130.0)
    downtrend = make_downtrend_series(total_bars=120, start_price=120.0, end_price=80.0)

    ok = detect_platform_candidate("OK", platform, 30, config, detected_at=1)
    assert ok.status is ResultStatus.SUCCESS
    assert ok.data is not None

    breakout_result = detect_platform_candidate("BRK", breakout, 30, config, detected_at=1)
    assert breakout_result.status is ResultStatus.SUCCESS
    assert breakout_result.data is None
    assert breakout_result.reason_code == "PRICE_RULES_NOT_MET"

    downtrend_result = detect_platform_candidate("DWN", downtrend, 30, config, detected_at=1)
    assert downtrend_result.status is ResultStatus.SUCCESS
    assert downtrend_result.data is None
    assert downtrend_result.reason_code == "PRICE_RULES_NOT_MET"


@pytest.mark.integration
def test_candidate_set_snapshot_roundtrip_from_detection_results() -> None:
    config = ScannerConfig(require_breakout_confirmation=False, require_breakout_volume_spike=False)
    bars = make_platform_series(total_bars=120)
    results = [
        detect_platform_candidate("AAPL", bars, w, config, detected_at=1) for w in [20, 30, 60]
    ]

    emitted = [r.data for r in results if r.status is ResultStatus.SUCCESS and r.data is not None]
    assert len(emitted) >= 1
    for candidate in emitted:
        config_snapshot = candidate.meta.get("config_snapshot")
        if isinstance(config_snapshot, dict):
            _normalize_config_snapshot(config_snapshot)

    config_snapshot = dict(msgspec.to_builtins(config))
    config_snapshot["window_weights"] = {str(k): v for k, v in config.window_weights.items()}
    _normalize_config_snapshot(config_snapshot)
    snapshot = CandidateSet(
        schema_version="1.0.0",
        system_version="deadbeef",
        asof_timestamp=1,
        candidates=emitted,
        total_scanned=1,
        total_detected=len(emitted),
        config_snapshot=config_snapshot,
        data_source="test",
        universe_source="test",
    )
    payload = msgspec.json.encode(snapshot)
    decoded = msgspec.json.decode(payload, type=CandidateSet)
    # Compare via JSON roundtrip to normalize any Struct/dict differences
    assert msgspec.json.encode(decoded) == msgspec.json.encode(snapshot)


@pytest.mark.integration
def test_insufficient_bars_returns_success_with_none() -> None:
    """Insufficient bars should return SUCCESS with None for backtest tolerance."""
    config = ScannerConfig(require_breakout_confirmation=False, require_breakout_volume_spike=False)
    result = detect_platform_candidate("BAD", [], 30, config, detected_at=1)
    # Insufficient bars now returns SUCCESS(None) instead of FAILED
    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "INSUFFICIENT_BARS"


@pytest.mark.integration
def test_multi_window_scan_with_partial_data_returns_success_with_available_candidates() -> None:
    """When some windows have insufficient data, we still succeed with available candidates."""
    config = ScannerConfig(windows=[20, 30, 60], require_breakout_confirmation=False, require_breakout_volume_spike=False)
    # 60D window requires 120 bars; provide enough for 20D only.
    bars = make_platform_series(total_bars=60)
    result = _run_multi_window_scan(symbol="AAPL", bars=bars, config=config, detected_at=1)
    # INSUFFICIENT_BARS now returns SUCCESS(None), so no failures are recorded
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    # At least one window should produce a candidate
    assert result.data.total_detected >= 1


@pytest.mark.integration
def test_multi_window_scan_with_empty_bars_returns_success_with_no_candidates() -> None:
    """Empty bars means all windows return INSUFFICIENT_BARS (now SUCCESS with None)."""
    config = ScannerConfig(windows=[20, 30, 60], require_breakout_confirmation=False, require_breakout_volume_spike=False)
    result = _run_multi_window_scan(symbol="AAPL", bars=[], config=config, detected_at=1)
    # INSUFFICIENT_BARS now returns SUCCESS(None), so the scan succeeds with no candidates
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None
    assert result.data.total_detected == 0
