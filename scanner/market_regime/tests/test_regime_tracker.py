from __future__ import annotations

import pytest

from scanner.market_regime.interface import MarketRegime, RegimeConfirmationConfig
from scanner.market_regime.regime_tracker import RegimeTransitionTracker


def test_tracker_initializes_confirmed_regime() -> None:
    tracker = RegimeTransitionTracker(config=RegimeConfirmationConfig(confirmation_days=3, reset_on_opposite=True))

    assert tracker.current_confirmed_regime is None
    assert tracker.pending_regime is None
    assert tracker.pending_days == 0

    assert tracker.update(MarketRegime.BULL) is MarketRegime.BULL
    assert tracker.current_confirmed_regime is MarketRegime.BULL
    assert tracker.pending_regime is None
    assert tracker.pending_days == 0


def test_tracker_ignores_unknown_detections() -> None:
    tracker = RegimeTransitionTracker(config=RegimeConfirmationConfig(confirmation_days=3, reset_on_opposite=True))
    assert tracker.update(MarketRegime.UNKNOWN) is MarketRegime.UNKNOWN

    tracker.update(MarketRegime.CHOPPY)
    assert tracker.update(MarketRegime.UNKNOWN) is MarketRegime.CHOPPY
    assert tracker.current_confirmed_regime is MarketRegime.CHOPPY


def test_tracker_requires_consecutive_days_to_switch() -> None:
    tracker = RegimeTransitionTracker(config=RegimeConfirmationConfig(confirmation_days=3, reset_on_opposite=True))
    tracker.update(MarketRegime.BULL)

    # Start BEAR transition.
    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BULL
    assert tracker.pending_regime is MarketRegime.BEAR
    assert tracker.pending_days == 1

    # Still pending.
    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BULL
    assert tracker.pending_days == 2

    # Confirm on day 3.
    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BEAR
    assert tracker.current_confirmed_regime is MarketRegime.BEAR
    assert tracker.pending_regime is None
    assert tracker.pending_days == 0


def test_tracker_resets_pending_when_returning_to_confirmed_default() -> None:
    tracker = RegimeTransitionTracker(config=RegimeConfirmationConfig(confirmation_days=3, reset_on_opposite=True))
    tracker.update(MarketRegime.BULL)

    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BULL
    assert tracker.pending_regime is MarketRegime.BEAR
    assert tracker.pending_days == 1

    # Return to confirmed resets in strict mode.
    assert tracker.update(MarketRegime.BULL) is MarketRegime.BULL
    assert tracker.pending_regime is None
    assert tracker.pending_days == 0

    # Transition restarts from 1.
    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BULL
    assert tracker.pending_regime is MarketRegime.BEAR
    assert tracker.pending_days == 1


def test_tracker_relaxed_mode_allows_choppy_noise_without_reset() -> None:
    tracker = RegimeTransitionTracker(config=RegimeConfirmationConfig(confirmation_days=3, reset_on_opposite=False))
    tracker.update(MarketRegime.BULL)

    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BULL
    assert tracker.pending_regime is MarketRegime.BEAR
    assert tracker.pending_days == 1

    # CHOPPY days do not reset pending confirmation when reset_on_opposite=False.
    assert tracker.update(MarketRegime.CHOPPY) is MarketRegime.BULL
    assert tracker.pending_regime is MarketRegime.BEAR
    assert tracker.pending_days == 1

    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BULL
    assert tracker.pending_days == 2

    assert tracker.update(MarketRegime.CHOPPY) is MarketRegime.BULL
    assert tracker.pending_days == 2

    # Third BEAR detection confirms.
    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BEAR
    assert tracker.current_confirmed_regime is MarketRegime.BEAR
    assert tracker.pending_regime is None
    assert tracker.pending_days == 0


def test_tracker_relaxed_mode_resets_on_opposite_direction() -> None:
    tracker = RegimeTransitionTracker(config=RegimeConfirmationConfig(confirmation_days=3, reset_on_opposite=False))
    tracker.update(MarketRegime.BULL)

    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BULL
    assert tracker.pending_regime is MarketRegime.BEAR
    assert tracker.pending_days == 1

    # Opposite (BULL) resets BEAR pending in relaxed mode too.
    assert tracker.update(MarketRegime.BULL) is MarketRegime.BULL
    assert tracker.pending_regime is None
    assert tracker.pending_days == 0


@pytest.mark.parametrize("confirmation_days", [0, -1])
def test_tracker_invalid_confirmation_days_falls_back_to_one(confirmation_days: int) -> None:
    tracker = RegimeTransitionTracker(config=RegimeConfirmationConfig(confirmation_days=confirmation_days, reset_on_opposite=True))
    assert tracker.update(MarketRegime.BULL) is MarketRegime.BULL
    assert tracker.update(MarketRegime.BEAR) is MarketRegime.BEAR


def test_tracker_reset_clears_state() -> None:
    tracker = RegimeTransitionTracker(config=RegimeConfirmationConfig(confirmation_days=3, reset_on_opposite=True))
    tracker.update(MarketRegime.BULL)
    tracker.update(MarketRegime.BEAR)
    assert tracker.pending_regime is not None

    tracker.reset()
    assert tracker.current_confirmed_regime is None
    assert tracker.pending_regime is None
    assert tracker.pending_days == 0

