from __future__ import annotations

from common.interface import ResultStatus
from scanner import ScannerConfig, detect_platform_candidate
from scanner.interface import PullbackConfirmationConfig

from .conftest import make_bar, make_platform_series


def _bars_with_last_bar_breakthrough_no_pullback() -> list[object]:
    bars = make_platform_series()
    last = bars[-1]
    bars[-1] = make_bar(
        close=last.close + 1.5,
        open=last.close,
        high=last.close + 1.5,
        low=last.close + 0.5,
        volume=last.volume,
    )
    return list(bars)


def test_score_threshold_skip_high_score() -> None:
    bars = _bars_with_last_bar_breakthrough_no_pullback()
    config = ScannerConfig(
        use_pullback_confirmation_filter=True,
        pullback_confirmation=PullbackConfirmationConfig(
            enabled=True,
            score_threshold_for_skip=0.90,
        ),
        require_breakout_confirmation=False,
        require_breakout_volume_spike=False,
    )

    result = detect_platform_candidate("SKIP", bars, 30, config, detected_at=1)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is not None

    filter_results = result.data.meta["filter_chain"]["filter_results"]
    assert filter_results["pullback_confirmation"]["passed"] is False


def test_score_threshold_does_not_skip_low_score() -> None:
    bars = _bars_with_last_bar_breakthrough_no_pullback()
    config = ScannerConfig(
        use_pullback_confirmation_filter=True,
        pullback_confirmation=PullbackConfirmationConfig(
            enabled=True,
            score_threshold_for_skip=1.01,
        ),
        require_breakout_confirmation=False,
        require_breakout_volume_spike=False,
    )

    result = detect_platform_candidate("NOSKIP", bars, 30, config, detected_at=1)
    assert result.status is ResultStatus.SUCCESS
    assert result.data is None
    assert result.reason_code == "PULLBACK_CONFIRMATION_FILTERED"
