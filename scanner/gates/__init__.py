"""Gate conditions for trend pattern detectors.

Gates control when trend continuation patterns are allowed to fire,
preventing signals during unfavorable market conditions.

Gate types:
- Sector Regime: XLK/SOXX must be bullish (EMA stack + positive slope)
- Relative Strength: Stock must outperform QQQ/SPY (RS line rising)

Usage:
    from scanner.gates import SectorRegimeGate, RelativeStrengthGate

    sector_gate = SectorRegimeGate()
    rs_gate = RelativeStrengthGate()

    sector_result = sector_gate.evaluate(symbol, bars, date, benchmark_bars=xlk_bars)
    rs_result = rs_gate.evaluate(symbol, bars, date, benchmark_bars=qqq_bars)

    if sector_result.passed and rs_result.passed:
        # Safe to check trend patterns
        pattern_result = detector.detect(..., sector_bullish=True, rs_strong=True)
"""

from .interface import (
    GateConfig,
    GateResult,
    Gate,
)

__all__ = [
    # Interface
    "GateConfig",
    "GateResult",
    "Gate",
    # Concrete gates (lazy imports)
    "SectorRegimeGate",
    "SectorRegimeConfig",
    "RelativeStrengthGate",
    "RelativeStrengthConfig",
]


# Lazy imports for concrete implementations
def __getattr__(name: str):
    if name == "SectorRegimeGate":
        from .sector_regime import SectorRegimeGate
        return SectorRegimeGate
    if name == "SectorRegimeConfig":
        from .sector_regime import SectorRegimeConfig
        return SectorRegimeConfig
    if name == "RelativeStrengthGate":
        from .relative_strength import RelativeStrengthGate
        return RelativeStrengthGate
    if name == "RelativeStrengthConfig":
        from .relative_strength import RelativeStrengthConfig
        return RelativeStrengthConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
