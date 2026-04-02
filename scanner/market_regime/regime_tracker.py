"""Regime transition tracking with multi-day confirmation.

This module smooths noisy daily regime classification by requiring a new regime
to be detected for N days before switching the confirmed regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from .interface import MarketRegime, RegimeConfirmationConfig

__all__ = ["RegimeTransitionTracker"]


def _is_opposite(left: MarketRegime, right: MarketRegime) -> bool:
    """Return True when `left` is the directional opposite of `right`."""

    return (left is MarketRegime.BULL and right is MarketRegime.BEAR) or (left is MarketRegime.BEAR and right is MarketRegime.BULL)


@dataclass(slots=True)
class RegimeTransitionTracker:
    """Track confirmed regime with an N-day confirmation requirement.

    Core behavior (default): the detector must emit the same new regime for
    `confirmation_days` consecutive updates before the confirmed regime switches.

    If `reset_on_opposite` is False, non-opposite "noise" regimes (e.g. CHOPPY)
    do not reset the confirmation counter once a pending directional transition
    is in progress; only the opposite directional regime resets the counter.
    """

    config: RegimeConfirmationConfig = RegimeConfirmationConfig()

    current_confirmed_regime: MarketRegime | None = None
    pending_regime: MarketRegime | None = None
    pending_days: int = 0

    def reset(self) -> None:
        self.current_confirmed_regime = None
        self.pending_regime = None
        self.pending_days = 0

    def update(self, detected_regime: MarketRegime) -> MarketRegime:
        """Update the tracker with a new detection and return confirmed regime."""

        logger = structlog.get_logger(__name__).bind(module="market_regime.regime_tracker")

        confirmation_days = int(self.config.confirmation_days)
        if confirmation_days <= 0:
            confirmation_days = 1

        if detected_regime is MarketRegime.UNKNOWN:
            return self.current_confirmed_regime or MarketRegime.UNKNOWN

        if self.current_confirmed_regime is None:
            self.current_confirmed_regime = detected_regime
            logger.info("regime.initial_confirmed", regime=detected_regime.value)
            return detected_regime

        if detected_regime is self.current_confirmed_regime:
            should_reset = self.pending_regime is not None and (
                self.config.reset_on_opposite or _is_opposite(detected_regime, self.pending_regime)
            )
            if should_reset:
                logger.debug(
                    "regime.pending_reset",
                    confirmed=self.current_confirmed_regime.value,
                    pending=self.pending_regime.value,
                    pending_days=self.pending_days,
                    reason="returned_to_confirmed",
                )
                self.pending_regime = None
                self.pending_days = 0
            return self.current_confirmed_regime

        if confirmation_days == 1:
            previous = self.current_confirmed_regime
            self.current_confirmed_regime = detected_regime
            self.pending_regime = None
            self.pending_days = 0
            logger.info("regime.confirmed_switch", previous=previous.value, current=detected_regime.value, confirmation_days=1)
            return detected_regime

        if self.pending_regime is None:
            self.pending_regime = detected_regime
            self.pending_days = 1
            logger.debug(
                "regime.pending_start",
                confirmed=self.current_confirmed_regime.value,
                pending=detected_regime.value,
                pending_days=self.pending_days,
                confirmation_days=confirmation_days,
            )
            return self.current_confirmed_regime

        if detected_regime is self.pending_regime:
            self.pending_days += 1
        else:
            if self.config.reset_on_opposite:
                self.pending_regime = detected_regime
                self.pending_days = 1
            else:
                if _is_opposite(detected_regime, self.pending_regime):
                    self.pending_regime = detected_regime
                    self.pending_days = 1
                # Non-opposite noise: keep existing pending regime and counter.

        if self.pending_regime is None:
            return self.current_confirmed_regime

        if self.pending_days >= confirmation_days:
            previous = self.current_confirmed_regime
            self.current_confirmed_regime = self.pending_regime
            confirmed = self.current_confirmed_regime
            logger.info(
                "regime.confirmed_switch",
                previous=previous.value,
                current=confirmed.value,
                confirmation_days=confirmation_days,
            )
            self.pending_regime = None
            self.pending_days = 0
            return confirmed

        return self.current_confirmed_regime
