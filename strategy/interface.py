"""Strategy Engine interface contracts for Phase 3.2 trade intent generation.

This module defines the Phase 3.2 Strategy Engine boundary for Automated Swing
Trader (AST). The Strategy Engine consumes upstream candidates (e.g.,
``scanner.interface.CandidateSet``) and market data snapshots (e.g.,
``data.interface.PriceSeriesSnapshot``), integrates Event Guard trading
constraints (``event_guard.interface.TradeConstraints``), and emits deterministic
"trade intents" suitable for downstream order execution.

Key design patterns captured here:

- Bracket / OrderList pattern (atomic transaction unit): an ``IntentGroup``
  models an entry intent plus linked stop-loss/take-profit intents, similar to
  NautilusTrader ``OrderList``.
- Deterministic idempotency: every intent/group uses a deterministic identifier
  so replays produce stable outputs (see ``TradeIntent.intent_id``).
- Trade constraints integration: constraints are evaluated before emitting
  intents; applied constraints are recorded in ``OrderIntentSet``.

All schemas follow the repo-wide conventions:

- ``msgspec.Struct`` value objects with ``frozen=True`` and ``kw_only=True``.
- Nanosecond Unix epoch timestamps represented as ``int`` (UTC).
- Journal-friendly snapshots embed ``journal.interface.SnapshotBase`` metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, TYPE_CHECKING, TypeAlias, runtime_checkable

import msgspec

from common.interface import Result
from journal.interface import SnapshotBase

if TYPE_CHECKING:
    from data.interface import PriceSeriesSnapshot
    from event_guard.interface import TradeConstraints
else:  # pragma: no cover - type-only imports may not exist in early phases.
    PriceSeriesSnapshot = Any  # type: ignore[assignment]
    TradeConstraints = Any  # type: ignore[assignment]

__all__ = [
    "TimestampNs",
    "IntentType",
    "TransactionCostConfig",
    "TradeIntent",
    "IntentGroup",
    "OrderIntentSet",
    "IndicatorsConfig",
    "StrategyEngineConfig",
    "StrategyOutput",
    "IntentSnapshot",
    "PositionSizerProtocol",
    "PricePolicyProtocol",
]

TimestampNs: TypeAlias = int


class TransactionCostConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Transaction cost configuration for realistic pricing.

    Applied to both entry and exit prices to simulate real trading costs:
    - Entry: price * (1 + spread_pct + slippage_pct) + commission_per_trade / quantity
    - Exit: price * (1 - spread_pct - slippage_pct) - commission_per_trade / quantity

    Attributes:
        spread_pct: Bid-ask spread cost as percentage (default: 0.001 = 0.1%)
        slippage_pct: Expected slippage as percentage (default: 0.0005 = 0.05%)
        commission_per_trade: Flat fee per trade in dollars (default: $1.00)
    """

    spread_pct: float = msgspec.field(default=0.001)  # 0.1% spread
    slippage_pct: float = msgspec.field(default=0.0005)  # 0.05% slippage
    commission_per_trade: float = msgspec.field(default=1.0)  # $1 commission


class IntentType(str, Enum):
    """Enumerate the supported trade intent types for the Strategy Engine.

    These intent types provide a stable vocabulary for downstream components
    (execution adapters, journaling, analytics) regardless of the concrete
    broker/order API.
    """

    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    REDUCE_POSITION = "REDUCE_POSITION"
    CANCEL_PENDING = "CANCEL_PENDING"


class TradeIntent(msgspec.Struct, frozen=True, kw_only=True):
    """A single trade intent emitted by the Strategy Engine.

    ``TradeIntent`` models one atomic action (e.g., open long, place stop loss)
    and is designed to be combined into higher-level transaction units via
    :class:`~strategy.interface.IntentGroup`.

    Deterministic intent ids (Pattern 2):
        ``intent_id`` is expected to be generated deterministically for
        idempotency and replay stability, using a template like:

        ``O-<datetime>-<trader_tag>-<strategy_tag>-<count>``

        with inputs derived from a stable hash of
        ``(strategy_id, symbol, bar_ts, intent_type, ladder_level)``.

    Attributes:
        intent_id: Deterministic identifier for the intent.
        symbol: Instrument identifier (e.g., ``"AAPL"``).
        intent_type: Intent classification (open/close/SL/TP/etc.).
        quantity: Position size in shares. Sign convention is not used here;
            direction is implied by ``intent_type`` and downstream execution.
        created_at_ns: Nanosecond Unix epoch timestamp when the intent was
            created (UTC).

        entry_price: Optional entry price; ``None`` means market.
        stop_loss_price: Optional stop-loss price.
        take_profit_price: Optional take-profit price.

        parent_intent_id: Optional parent intent id for bracket linkage (e.g.,
            SL/TP referencing the entry).
        linked_intent_ids: Optional linked intent ids for bracket linkage (e.g.,
            entry referencing SL/TP).

        reduce_only: When True, the intent must not increase exposure (cannot
            open a new position); it may only reduce an existing one.
        contingency_type: Optional contingency type for transaction semantics
            (e.g., ``"OTO"``, ``"OUO"``). Used to express entry vs bracket legs.
        ladder_level: Optional ladder level index for staged entries.

        reason_codes: Machine-readable reasons for why the intent exists
            (e.g., ``"SCAN_SIGNAL"``, ``"RISK_REDUCTION"``).
        metadata: Optional string-to-string metadata for debugging, provenance,
            or execution hints.
    """

    intent_id: str
    symbol: str
    intent_type: IntentType
    quantity: float
    created_at_ns: TimestampNs

    entry_price: float | None = msgspec.field(default=None)
    stop_loss_price: float | None = msgspec.field(default=None)
    take_profit_price: float | None = msgspec.field(default=None)

    parent_intent_id: str | None = msgspec.field(default=None)
    linked_intent_ids: list[str] = msgspec.field(default_factory=list)

    reduce_only: bool = msgspec.field(default=False)
    contingency_type: str | None = msgspec.field(default=None)
    ladder_level: int | None = msgspec.field(default=None)

    reason_codes: list[str] = msgspec.field(default_factory=list)
    metadata: dict[str, str] | None = msgspec.field(default=None)


class IntentGroup(msgspec.Struct, frozen=True, kw_only=True):
    """A transaction unit containing one entry and its contingent intents.

    ``IntentGroup`` mirrors the "Bracket/OrderList pattern" (Pattern 1): it
    groups one entry intent together with its dependent stop-loss and
    take-profit intents, which must be treated as an atomic unit by downstream
    execution layers.

    Common bracket semantics:
        - Entry intent may set ``contingency_type="OTO"`` and include
          ``linked_intent_ids`` pointing at the generated SL/TP intents.
        - SL/TP intents typically set ``reduce_only=True`` and reference the
          entry intent via ``parent_intent_id``.

    Attributes:
        group_id: Deterministic identifier for the group (idempotent output).
        symbol: Instrument identifier.
        intents: Ordered list of intents (usually entry then SL/TP legs).
        contingency_type: Group-level contingency type (defaults to
            ``"OUO"``: one-updates-other).
        created_at_ns: Nanosecond Unix epoch timestamp when the group was
            created (UTC).
    """

    group_id: str
    symbol: str
    intents: list[TradeIntent]
    created_at_ns: TimestampNs
    contingency_type: str = msgspec.field(default="OUO")


class OrderIntentSet(SnapshotBase, frozen=True, kw_only=True):
    """Batch output of generated trade intents for journaling and replay.

    ``OrderIntentSet`` is the Strategy Engine output snapshot for a single
    evaluation pass (e.g., "run strategy for these candidates at T").
    It embeds the standard journaling metadata envelope from
    :class:`~journal.interface.SnapshotBase`.

    Attributes:
        schema_version: See ``journal.interface.SnapshotBase``.
        system_version: See ``journal.interface.SnapshotBase``.
        asof_timestamp: See ``journal.interface.SnapshotBase``.

        intent_groups: Generated intent groups (each is an atomic transaction
            unit).
        constraints_applied: Record of applied trade constraints by symbol.
            Keyed by ``symbol`` with a value payload that mirrors or summarizes
            ``event_guard.interface.TradeConstraints`` for audit/debugging.
        source_candidates: Source candidate symbol list (for provenance and
            reconciliation with upstream scanners).
    """

    intent_groups: list[IntentGroup] = msgspec.field(default_factory=list)
    constraints_applied: dict[str, dict[str, Any]] = msgspec.field(default_factory=dict)
    source_candidates: list[str] = msgspec.field(default_factory=list)
    degradation_events: list[dict[str, Any]] = msgspec.field(default_factory=list)


class IndicatorsConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for indicator-based entry confirmation checks.

    This configuration is owned by the Strategy Engine and controls whether
    RSI/MACD checks are applied as *additional* filters during entry intent
    generation. The top-level ``enabled`` flag is a master switch: when False,
    indicator checks are skipped entirely for backward compatibility.

    Attributes:
        enabled: Master switch for indicator confirmation checks.

        rsi_enabled: Enable RSI confirmation when indicator checks are enabled.
        rsi_period: RSI lookback period (number of bars).
        rsi_overbought: RSI threshold considered overbought.
        rsi_oversold: RSI threshold considered oversold.

        macd_enabled: Enable MACD confirmation when indicator checks are enabled.
        macd_fast: MACD fast EMA period.
        macd_slow: MACD slow EMA period.
        macd_signal: MACD signal EMA period.
        macd_require_bullish: When True, require MACD histogram > 0.
    """

    enabled: bool = msgspec.field(default=False)
    rsi_enabled: bool = msgspec.field(default=True)
    rsi_period: int = msgspec.field(default=14)
    rsi_overbought: float = msgspec.field(default=70.0)
    rsi_oversold: float = msgspec.field(default=30.0)
    macd_enabled: bool = msgspec.field(default=True)
    macd_fast: int = msgspec.field(default=12)
    macd_slow: int = msgspec.field(default=26)
    macd_signal: int = msgspec.field(default=9)
    macd_require_bullish: bool = msgspec.field(default=True)


class StrategyEngineConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration controlling Strategy Engine sizing and pricing policies.

    The Strategy Engine is composed from two main concerns:

    - Position sizing (``position_sizer``): determine share quantity based on
      equity, risk, and constraints.
    - Price policy (``price_policy``): determine entry/stop/target levels.

    Attributes:
        position_sizer: Position sizing algorithm name. Expected values:
            ``"fixed_percent"``, ``"fixed_risk"``, ``"volatility_scaled"``,
            ``"quality_scaled"`` (new).

        # Quality-Scaled Sizing config (new)
        quality_scaling_enabled: Master switch for quality-based scaling when supported
            by the selected position sizing algorithm (e.g., adaptive sizing).
        base_position_pct: Base allocation as fraction of equity (e.g., 0.05 = 5%).
        min_position_pct: Minimum position size as fraction of equity to avoid
            excessive transaction costs.
        min_score_threshold: Minimum signal quality score to consider trading
            (signals below this are skipped).
        quality_multiplier_excellent: Allocation multiplier for excellent signals
            (score >= quality_threshold_excellent).
        quality_multiplier_good: Allocation multiplier for good signals
            (quality_threshold_good <= score < quality_threshold_excellent).
        quality_multiplier_acceptable: Allocation multiplier for acceptable signals
            (quality_threshold_acceptable <= score < quality_threshold_good).
        quality_threshold_excellent: Score threshold for excellent tier.
        quality_threshold_good: Score threshold for good tier.
        quality_threshold_acceptable: Score threshold for acceptable tier.
        max_commission_ratio: Maximum allowed ratio of commission to position value
            (e.g., 0.02 = 2%). Positions with higher ratios are skipped.
        position_size_pct: Fraction of account equity allocated to a position
            for ``fixed_percent`` sizing.
        risk_per_trade_pct: Fraction of equity to risk per trade for
            ``fixed_risk`` sizing.
        max_position_pct: Hard cap fraction of account equity per position.

        price_policy: Price/stop/target policy name. Expected values:
            ``"atr_bracket"``, ``"trailing_stop"``, ``"ladder_entry"``.
        atr_multiplier: ATR multiple used by ATR-based bracket policies.
        trailing_offset_pct: Trailing stop offset expressed as a fraction of
            price (e.g., 0.05 == 5%).
        tp_sl_ratio: Take profit to stop loss distance ratio (e.g., 1.5 means
            TP distance is 1.5x SL distance). Used by pricing policies to
            calculate asymmetric risk-reward brackets.
        dynamic_tp_sl_enabled: Enable dynamic tp_sl_ratio selection based on volatility.
        tp_sl_ratio_low_vol: Take profit to stop loss ratio when volatility is low.
        tp_sl_ratio_high_vol: Take profit to stop loss ratio when volatility is high.
        low_volatility_threshold: ATR% threshold below which volatility is considered low.
        high_volatility_threshold: ATR% threshold above which volatility is considered high.

        entry_strategy: Entry construction strategy. Expected values:
            ``"single"`` or ``"ladder"``.
        ladder_levels: Number of ladder levels for staged entry.
        ladder_spacing_pct: Percentage spacing between ladder entry levels.

        use_conservative_defaults: When data is missing or uncertain, instruct
            the engine to degrade into conservative behaviours (e.g., smaller
            size, wider stops, or no-trade).

        indicators: Indicator confirmation configuration for entry filtering.
        transaction_costs: Transaction cost parameters (spread, slippage, commission)
    """

    position_sizer: str = msgspec.field(default="fixed_percent")
    position_size_pct: float = msgspec.field(default=0.02)
    risk_per_trade_pct: float = msgspec.field(default=0.01)
    max_position_pct: float = msgspec.field(default=0.1)

    # Quality-Scaled Position Sizing config (new)
    quality_scaling_enabled: bool = msgspec.field(default=True)
    base_position_pct: float = msgspec.field(default=0.05)  # Base position pct (5%)
    min_position_pct: float = msgspec.field(default=0.025)  # Min position pct (2.5%)
    min_score_threshold: float = msgspec.field(default=0.85)  # Min signal quality threshold

    # Signal quality tier multipliers (score -> multiplier)
    quality_multiplier_excellent: float = msgspec.field(default=2.0)  # score >= 0.95
    quality_multiplier_good: float = msgspec.field(default=1.5)       # 0.90 <= score < 0.95
    quality_multiplier_acceptable: float = msgspec.field(default=1.0)  # 0.85 <= score < 0.90

    # Signal quality tier thresholds
    quality_threshold_excellent: float = msgspec.field(default=0.95)
    quality_threshold_good: float = msgspec.field(default=0.90)
    quality_threshold_acceptable: float = msgspec.field(default=0.85)

    # Transaction cost optimization
    max_commission_ratio: float = msgspec.field(default=0.02)  # Max commission-to-position ratio (2%)

    price_policy: str = msgspec.field(default="atr_bracket")
    atr_multiplier: float = msgspec.field(default=2.0)
    trailing_offset_pct: float = msgspec.field(default=0.05)
    tp_sl_ratio: float = msgspec.field(default=1.5)

    # Dynamic tp_sl_ratio config
    dynamic_tp_sl_enabled: bool = msgspec.field(default=True)
    tp_sl_ratio_low_vol: float = msgspec.field(default=1.5)    # Low volatility
    tp_sl_ratio_high_vol: float = msgspec.field(default=2.5)   # High volatility
    low_volatility_threshold: float = msgspec.field(default=0.02)  # ATR < 2%
    high_volatility_threshold: float = msgspec.field(default=0.03)  # ATR > 3%

    # Regime TP/SL constraint params (new)
    stop_loss_pct: float = msgspec.field(default=0.06)  # Max stop-loss pct cap (6%)
    take_profit_pct: float = msgspec.field(default=0.20)  # Take-profit pct (20%)
    use_fixed_tp: bool = msgspec.field(default=False)  # True=use take_profit_pct directly, False=use ratio

    entry_strategy: str = msgspec.field(default="single")
    ladder_levels: int = msgspec.field(default=3)
    ladder_spacing_pct: float = msgspec.field(default=0.02)

    use_conservative_defaults: bool = msgspec.field(default=True)
    indicators: IndicatorsConfig = msgspec.field(default_factory=IndicatorsConfig)
    transaction_costs: TransactionCostConfig = msgspec.field(
        default_factory=TransactionCostConfig
    )


class IntentSnapshot(SnapshotBase, frozen=True, kw_only=True):
    """High-level journal snapshot for Strategy Engine outputs and degradations.

    ``IntentSnapshot`` is a container snapshot used by the orchestrator for
    journaling and replay. It captures one or more intent-set evaluations along
    with any recorded degradation events (fallbacks, missing data, or policy
    blocks).

    Attributes:
        schema_version: See ``journal.interface.SnapshotBase``.
        system_version: See ``journal.interface.SnapshotBase``.
        asof_timestamp: See ``journal.interface.SnapshotBase``.

        intent_sets: One or more emitted intent sets (batched outputs).
        degradation_events: Structured records describing degraded decisions.
            This is intentionally ``dict[str, Any]`` to allow incremental schema
            evolution while keeping journaling stable.
    """

    SCHEMA_VERSION = "1.2.0"

    intent_sets: list[OrderIntentSet] = msgspec.field(default_factory=list)
    degradation_events: list[dict[str, Any]] = msgspec.field(default_factory=list)


class StrategyOutput(SnapshotBase, frozen=True, kw_only=True):
    """Strategy Engine plugin output snapshot for journaling and replay.

    This snapshot wraps the complete Strategy Engine output including generated
    trade intents and pass-through data from upstream modules. It embeds the
    standard snapshot metadata envelope from ``journal.interface.SnapshotBase``.

    Attributes:
        schema_version: See ``journal.interface.SnapshotBase``.
        system_version: See ``journal.interface.SnapshotBase``.
        asof_timestamp: See ``journal.interface.SnapshotBase``.
        intents: Serialized OrderIntentSet containing generated trade intents.
        candidates: Pass-through CandidateSet from upstream scanner (serialized).
        constraints: Pass-through TradeConstraints from Event Guard (serialized).
    """

    SCHEMA_VERSION = "1.2.0"

    intents: dict[str, Any] = msgspec.field(default_factory=dict)
    candidates: dict[str, Any] = msgspec.field(default_factory=dict)
    constraints: dict[str, Any] = msgspec.field(default_factory=dict)


@runtime_checkable
class PositionSizerProtocol(Protocol):
    """Protocol for calculating share quantity given risk/equity/constraints."""

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        account_equity: float,
        config: StrategyEngineConfig,
        constraints: TradeConstraints | None = None,
        market_data: PriceSeriesSnapshot | None = None,
    ) -> Result[float]:
        """Compute position size, returning the number of shares.

        Args:
            symbol: Instrument identifier.
            entry_price: Intended entry price.
            stop_loss_price: Intended stop-loss price.
            account_equity: Current account equity in quote currency.
            config: Strategy Engine configuration.
            constraints: Trade constraints (from Event Guard), if available.
            market_data: Optional market data snapshot (e.g., for ATR-based sizing).

        Returns:
            Result[float]: Successful result holds the computed share quantity.
        """


@runtime_checkable
class PricePolicyProtocol(Protocol):
    """Protocol for computing entry/stop/target prices from market data."""

    def calculate_entry_price(
        self,
        symbol: str,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Compute the entry price for the requested side.

        Args:
            symbol: Instrument identifier.
            side: ``"LONG"`` or ``"SHORT"``.
            config: Strategy Engine configuration.
            market_data: Market data snapshot used for the computation.
        """

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Compute the stop-loss price for a planned entry."""

    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        config: StrategyEngineConfig,
        market_data: PriceSeriesSnapshot,
    ) -> Result[float]:
        """Compute the take-profit price for a planned entry."""
