"""Risk Gate package (Phase 3.3).

The Risk Gate is AST's pre-trade enforcement boundary:
strategy intents -> risk checks -> ALLOW/BLOCK/DOWNGRADE decisions.
"""

from risk_gate.checks import (  # noqa: F401
    ConcentrationCheck,
    DailyLossCheck,
    DrawdownCheck,
    LeverageCheck,
    MaxPositionCheck,
    OrderCountCheck,
    PriceBandCheck,
    RateLimitCheck,
    ReduceOnlyCheck,
    VolatilityHaltCheck,
)
from risk_gate.engine import evaluate_intents  # noqa: F401
from risk_gate.interface import (  # noqa: F401
    CheckResult,
    CheckStatus,
    DecisionType,
    OperationalRiskConfig,
    PortfolioRiskConfig,
    RiskCheckContext,
    RiskCheckProtocol,
    RiskDecision,
    RiskDecisionSet,
    RiskGateConfig,
    SafeModeState,
    SymbolRiskConfig,
)

__all__ = [
    "evaluate_intents",
    "CheckResult",
    "CheckStatus",
    "DecisionType",
    "OperationalRiskConfig",
    "PortfolioRiskConfig",
    "RiskCheckContext",
    "RiskCheckProtocol",
    "RiskDecision",
    "RiskDecisionSet",
    "RiskGateConfig",
    "SafeModeState",
    "SymbolRiskConfig",
    "ConcentrationCheck",
    "DailyLossCheck",
    "DrawdownCheck",
    "LeverageCheck",
    "MaxPositionCheck",
    "OrderCountCheck",
    "PriceBandCheck",
    "RateLimitCheck",
    "ReduceOnlyCheck",
    "VolatilityHaltCheck",
]

