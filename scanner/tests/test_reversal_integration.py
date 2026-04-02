from __future__ import annotations

from scanner.detector import detect_platform_candidate
from scanner.interface import ScannerConfig
from scanner.reversal_detector import ReversalConfig

from .conftest import DummyPriceBar, make_bars, make_platform_series


def _make_reversal_bars(total_bars: int = 45) -> list[DummyPriceBar]:
    """Construct a downtrend with a late oversold bounce and volume spike."""

    if total_bars < 10:
        raise ValueError("total_bars must be >= 10")

    prefix = [100.0 - idx * 0.5 for idx in range(total_bars - 9)]
    tail_template = [10.0, 9.5, 9.0, 8.5, 8.0, 8.05, 8.1, 8.2, 8.3]
    offset = (prefix[-1] - tail_template[0]) if prefix else 0.0
    tail = [value + offset for value in tail_template]
    closes = prefix + tail

    base_volumes = [1_200_000.0 for _ in range(total_bars - 9)]
    tail_volumes = [
        1_000_000.0,
        900_000.0,
        850_000.0,
        800_000.0,
        750_000.0,
        700_000.0,
        650_000.0,
        600_000.0,
        2_200_000.0,
    ]
    volumes = base_volumes + tail_volumes
    return make_bars(closes, volume=volumes, high_pct=0.005, low_pct=0.005)


def _make_reversal_config() -> ReversalConfig:
    return ReversalConfig(
        enabled=True,
        rsi_period=3,
        rsi_oversold=45.0,
        volume_lookback=2,
        volume_ratio_threshold=1.5,
        stabilization_days=2,
    )


def _make_scanner_config(**overrides: object) -> ScannerConfig:
    config_kwargs: dict[str, object] = {
        "windows": [30],
        "window_weights": {30: 1.0},
        "use_filter_chain_detector": False,
        "use_reversal_detector": True,
        "reversal": _make_reversal_config(),
    }
    config_kwargs.update(overrides)
    return ScannerConfig(**config_kwargs)


def test_reversal_detection_in_bear_regime() -> None:
    config = _make_scanner_config()
    bars = _make_reversal_bars()

    result = detect_platform_candidate("REV", bars, 30, config, detected_at=1, regime="bear")

    assert result.data is not None, result.reason_code
    assert result.data.signal_type == "reversal"
    assert result.data.reversal_signal_strength is not None and result.data.reversal_signal_strength > 0
    assert result.data.reversal_rsi is not None
    assert result.data.reversal_volume_ratio is not None and result.data.reversal_volume_ratio >= 1.5


def test_reversal_skipped_in_bull_regime() -> None:
    config = _make_scanner_config()
    bars = _make_reversal_bars()

    result = detect_platform_candidate("REV", bars, 30, config, detected_at=1, regime="bull")

    assert result.data is None
    assert result.reason_code != "REVERSAL_SIGNAL"


def test_reversal_disabled_by_config() -> None:
    config = _make_scanner_config(use_reversal_detector=False)
    bars = _make_reversal_bars()

    result = detect_platform_candidate("REV", bars, 30, config, detected_at=1, regime="bear")

    assert result.data is None


def test_platform_detection_takes_priority() -> None:
    config = _make_scanner_config()
    bars = make_platform_series()

    result = detect_platform_candidate("PLAT", bars, 30, config, detected_at=1, regime="bear")

    assert result.data is not None
    assert result.data.signal_type == "platform_breakout"
    assert result.data.reversal_signal_strength is None
