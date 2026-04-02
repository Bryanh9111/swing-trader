"""Strategy Engine module (Phase 3.2) for AST trade intent generation.

This module converts Scanner candidates into deterministic trading intents
with position sizing and price strategies, integrated with Event Guard constraints.

Public API:
    Interface schemas:
        - IntentType, TradeIntent, IntentGroup, OrderIntentSet
        - StrategyEngineConfig, IntentSnapshot
        - PositionSizerProtocol, PricePolicyProtocol

    Core functions:
        - generate_intents() - Main entry point

    Position sizers:
        - FixedPercentSizer, FixedRiskSizer, VolatilityScaledSizer
        - create_position_sizer()

    Price policies:
        - ATRBracketPolicy, TrailingStopPolicy, LadderEntryPolicy
        - create_price_policy()

Usage example:
    ```python
    from strategy import (
        generate_intents,
        create_position_sizer,
        create_price_policy,
        StrategyEngineConfig,
    )

    config = StrategyEngineConfig(
        position_sizer="fixed_risk",
        price_policy="atr_bracket",
    )
    sizer = create_position_sizer(config)
    policy = create_price_policy(config)

    result = generate_intents(
        candidates=candidate_set,
        constraints=constraints_dict,
        market_data=market_data_dict,
        account_equity=100000.0,
        config=config,
        position_sizer=sizer,
        price_policy=policy,
    )
    ```
"""

from strategy.engine import generate_intents
from strategy.interface import (
    IntentSnapshot,
    IntentType,
    OrderIntentSet,
    PositionSizerProtocol,
    PricePolicyProtocol,
    StrategyEngineConfig,
    TradeIntent,
    IntentGroup,
)
from strategy.pricing import (
    ATRBracketPolicy,
    LadderEntryPolicy,
    TrailingStopPolicy,
    create_price_policy,
)
from strategy.sizing import (
    FixedPercentSizer,
    FixedRiskSizer,
    VolatilityScaledSizer,
    create_position_sizer,
)

__all__ = [
    # Core function
    "generate_intents",
    # Schemas
    "IntentType",
    "TradeIntent",
    "IntentGroup",
    "OrderIntentSet",
    "StrategyEngineConfig",
    "IntentSnapshot",
    # Protocols
    "PositionSizerProtocol",
    "PricePolicyProtocol",
    # Position sizers
    "FixedPercentSizer",
    "FixedRiskSizer",
    "VolatilityScaledSizer",
    "create_position_sizer",
    # Price policies
    "ATRBracketPolicy",
    "TrailingStopPolicy",
    "LadderEntryPolicy",
    "create_price_policy",
]
