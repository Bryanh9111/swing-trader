"""Market regime detection and configuration utilities.

This package provides:
- Market regime classification (bull/bear/choppy/unknown) based on market index
  time series (e.g., SPY/QQQ).
- Regime-specific configuration loading/merging from `config/regimes/`.
- Multi-day breakthrough confirmation helpers.
"""

from __future__ import annotations

from .confirmation import BreakthroughConfirmation, BreakthroughConfirmationConfig
from .config_loader import RegimeConfigLoader
from .detector import MarketRegimeDetector
from .interface import MarketRegime, RegimeConfirmationConfig, RegimeConfig, RegimeDetectionResult
from .regime_tracker import RegimeTransitionTracker

__all__ = [
    "BreakthroughConfirmation",
    "BreakthroughConfirmationConfig",
    "MarketRegime",
    "RegimeConfirmationConfig",
    "MarketRegimeDetector",
    "RegimeConfig",
    "RegimeConfigLoader",
    "RegimeDetectionResult",
    "RegimeTransitionTracker",
]
