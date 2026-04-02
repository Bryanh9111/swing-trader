"""Pluggable trend-continuation pattern detectors.

The framework ships with a single demo pattern (MA Crossover).
Add your own detectors by implementing the ``TrendPatternDetector`` protocol
defined in ``scanner.patterns.interface``.

Usage:
    from scanner.patterns import MACrossoverDetector

    detector = MACrossoverDetector()
    result = detector.detect(symbol, bars, date)
    if result.detected:
        print(f"Pattern: {result.pattern_type}, Score: {result.score}")
"""

from .interface import (
    TrendPatternConfig,
    TrendPatternResult,
    TrendPatternDetector,
)

__all__ = [
    # Interface
    "TrendPatternConfig",
    "TrendPatternResult",
    "TrendPatternDetector",
    # Concrete detectors
    "MACrossoverDetector",
    "MACrossoverConfig",
]


def __getattr__(name: str):
    if name == "MACrossoverDetector":
        from .ma_crossover import MACrossoverDetector
        return MACrossoverDetector
    if name == "MACrossoverConfig":
        from .ma_crossover import MACrossoverConfig
        return MACrossoverConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
