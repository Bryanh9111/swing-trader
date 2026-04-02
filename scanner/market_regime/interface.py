"""Public interfaces for the market regime module."""

from __future__ import annotations

from enum import Enum
from typing import Any, TYPE_CHECKING

import msgspec

__all__ = [
    "MarketRegime",
    "RegimeConfirmationConfig",
    "RegimeDetectionResult",
    "RegimeConfig",
]

if TYPE_CHECKING:
    from data.vix_provider import VIXData
else:  # pragma: no cover
    VIXData = Any  # type: ignore[misc,assignment]


class MarketRegime(str, Enum):
    """Coarse market regime classifications."""

    BULL = "bull"
    BEAR = "bear"
    CHOPPY = "choppy"
    UNKNOWN = "unknown"


class RegimeConfirmationConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for multi-day regime transition confirmation."""

    confirmation_days: int = 3
    reset_on_opposite: bool = True


class RegimeDetectionResult(msgspec.Struct, frozen=True, kw_only=True):
    """Outcome of a single regime detection evaluation."""

    regime: MarketRegime
    market_symbol: str
    adx: float | None
    ma50_slope: float | None
    volatility: float | None
    is_trending: bool
    is_stable: bool
    # Real VIX readings (optional)
    vix_data: VIXData | None = None
    # VIX proxy signals (optional, auxiliary)
    vix_proxy_symbol: str | None = None
    vix_proxy_change: float | None = None  # Daily % change of VIX proxy ETF
    vix_spike_detected: bool = False  # True if VIX proxy spiked above threshold
    details: dict[str, Any] = msgspec.field(default_factory=dict)


class RegimeConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Regime-specific parameter overrides loaded from YAML.

    The loader merges a `base.yaml` mapping with regime-specific overrides.
    This class keeps the merged result split by logical sections to make it
    easy to apply the values into existing config trees.
    """

    regime_name: str
    description: str
    enabled: bool = True
    scanner: dict[str, Any] = msgspec.field(default_factory=dict)
    market_regime: dict[str, Any] = msgspec.field(default_factory=dict)
    strategy: dict[str, Any] = msgspec.field(default_factory=dict)
    breakthrough: dict[str, Any] = msgspec.field(default_factory=dict)
    risk_gate: dict[str, Any] = msgspec.field(default_factory=dict)
    # V21: BULL confirmation and risk overlay
    bull_confirmation: dict[str, Any] = msgspec.field(default_factory=dict)
    risk_overlay: dict[str, Any] = msgspec.field(default_factory=dict)
    # V21: Regime-aware capital allocation overrides
    capital_allocation: dict[str, Any] = msgspec.field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Return a plain mapping representation for downstream merging."""

        return {
            "regime_name": self.regime_name,
            "description": self.description,
            "enabled": self.enabled,
            "scanner": dict(self.scanner),
            "market_regime": dict(self.market_regime),
            "strategy": dict(self.strategy),
            "breakthrough": dict(self.breakthrough),
            "risk_gate": dict(self.risk_gate),
            "bull_confirmation": dict(self.bull_confirmation),
            "risk_overlay": dict(self.risk_overlay),
            "capital_allocation": dict(self.capital_allocation),
        }
