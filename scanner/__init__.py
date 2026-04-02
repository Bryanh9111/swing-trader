"""Phase 2.3 Scanner package (platform/consolidation detection).

The scanner is responsible for detecting "platform" / consolidation patterns in
validated OHLCV price series snapshots and emitting structured candidates for
downstream ranking, signal generation, and journaling.

This package follows the Phase 1 module structure conventions used across AST:
- Public, stable imports are re-exported from the package root.
- Core algorithms live in ``scanner.detector``.
- Typed schemas and plugin contracts live in ``scanner.interface``.

Public API
----------
Types (from ``scanner.interface``):
    - :class:`~scanner.interface.ScannerConfig`
    - :class:`~scanner.interface.PlatformFeatures`
    - :class:`~scanner.interface.PlatformCandidate`
    - :class:`~scanner.interface.CandidateSet`
    - :class:`~scanner.interface.ScannerPlugin`

Functions (from ``scanner.detector``):
    - :func:`~scanner.detector.detect_platform_candidate` (main entry point)
    - :func:`~scanner.detector.detect_price_platform`
    - :func:`~scanner.detector.detect_volume_platform`
    - :func:`~scanner.detector.calculate_atr`
    - :func:`~scanner.detector.detect_support_resistance`

Usage examples
--------------

Detect candidates from a single symbol's bar series:

.. code-block:: python

    from scanner import ScannerConfig, detect_platform_candidate
    from common.interface import ResultStatus

    config = ScannerConfig()
    result = detect_platform_candidate(bars, symbol="AAPL", config=config, detected_at=0)
    if result.status is ResultStatus.SUCCESS:
        candidate = result.data
        print(candidate.symbol, candidate.score, candidate.features.box_range)

Use the lower-level detectors directly:

.. code-block:: python

    from scanner import detect_price_platform, detect_volume_platform

    price_result = detect_price_platform(bars, window=30, config=ScannerConfig())
    volume_result = detect_volume_platform(bars, window=30, config=ScannerConfig())

Compose and run a FilterChain directly (advanced usage):

.. code-block:: python

    from scanner import (
        FilterChain,
        PricePlatformFilter,
        VolumePlatformFilter,
        ATRFilter,
        BoxQualityFilter,
        ScannerConfig,
    )

    chain = FilterChain(logic="AND")
    chain.add_filter(PricePlatformFilter(window=30))
    chain.add_filter(VolumePlatformFilter(window=30))
    chain.add_filter(ATRFilter())
    chain.add_filter(BoxQualityFilter(window=30))
    chain_result = chain.execute(bars, ScannerConfig())
"""

# Interface exports
from .interface import (
    CandidateSet,
    PlatformCandidate,
    PlatformFeatures,
    ScannerConfig,
    ScannerPlugin,
)

# Detector exports
from .detector import (
    calculate_atr,
    detect_platform_candidate,
    detect_price_platform,
    detect_support_resistance,
    detect_volume_platform,
)

# Reversal detector exports
from .reversal_detector import ReversalCandidate, ReversalConfig, ReversalDetector

# Trend pattern router exports (Growth/AI stock patterns)
from .trend_pattern_router import (
    TrendPatternRouter,
    TrendPatternRouterConfig,
    detect_trend_pattern_candidate,
)

# Trend pattern detector exports
from .patterns import (
    TrendPatternConfig,
    TrendPatternResult,
    TrendPatternDetector,
    MACrossoverDetector,
    MACrossoverConfig,
)

# Gate exports
from .gates import (
    GateConfig,
    GateResult,
    Gate,
    SectorRegimeGate,
    SectorRegimeConfig,
    RelativeStrengthGate,
    RelativeStrengthConfig,
)

# Filter exports (advanced usage)
from .filters import (
    ATRFilter,
    BoxQualityFilter,
    BreakthroughFilter,
    FilterChain,
    LowPositionFilter,
    MarketCapFilter,
    PricePlatformFilter,
    RapidDeclineFilter,
    VolumePlatformFilter,
)

__all__ = [
    # Interface exports
    "ScannerConfig",
    "PlatformFeatures",
    "PlatformCandidate",
    "CandidateSet",
    "ScannerPlugin",
    # Detector exports
    "detect_platform_candidate",
    "detect_price_platform",
    "detect_volume_platform",
    "calculate_atr",
    "detect_support_resistance",
    "ReversalConfig",
    "ReversalCandidate",
    "ReversalDetector",
    # Trend pattern router exports
    "TrendPatternRouter",
    "TrendPatternRouterConfig",
    "detect_trend_pattern_candidate",
    # Trend pattern detector exports
    "TrendPatternConfig",
    "TrendPatternResult",
    "TrendPatternDetector",
    "MACrossoverDetector",
    "MACrossoverConfig",
    # Gate exports
    "GateConfig",
    "GateResult",
    "Gate",
    "SectorRegimeGate",
    "SectorRegimeConfig",
    "RelativeStrengthGate",
    "RelativeStrengthConfig",
    # Filter exports
    "FilterChain",
    "PricePlatformFilter",
    "VolumePlatformFilter",
    "ATRFilter",
    "BoxQualityFilter",
    "LowPositionFilter",
    "RapidDeclineFilter",
    "BreakthroughFilter",
    "MarketCapFilter",
]

__version__ = "0.1.0"
