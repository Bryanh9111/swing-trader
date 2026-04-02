"""Risk Gate interface contracts for Phase 3.3 pre-trade enforcement.

This module defines the public schemas and protocols for AST's Risk Gate. The
Risk Gate is the final approval authority for all trade intents emitted by the
Strategy Engine, enforcing:

- Portfolio-level caps (leverage, drawdown, daily loss, concentration).
- Symbol-level caps (max position, price band, volatility halt, event windows).
- Operational controls (order limits, rate limits, pacing).
- Minimal Safe Mode (degrade or halt trading on critical failures).

All schemas follow repo-wide conventions:

- Immutable value objects built with ``msgspec.Struct(frozen=True, kw_only=True)``.
- Stable, machine-readable reason codes plus ``details`` payloads for audit.
- Journal-friendly snapshot schemas inherit ``journal.interface.SnapshotBase``.
- Interfaces return ``common.interface.Result[T]`` to capture success/degraded/fail outcomes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, TYPE_CHECKING, TypeAlias, runtime_checkable

import msgspec

from common.interface import Result
from journal.interface import SnapshotBase

if TYPE_CHECKING:
    from strategy.interface import TradeIntent
    from strategy.interface import OrderIntentSet

    from data.interface import PriceSeriesSnapshot
else:  # pragma: no cover - type-only imports may not exist in early phases.
    TradeIntent = Any  # type: ignore[assignment]
    OrderIntentSet = Any  # type: ignore[assignment]
    PriceSeriesSnapshot = Any  # type: ignore[assignment]

Position: TypeAlias = Any
TimestampNs: TypeAlias = int

__all__ = [
    "TimestampNs",
    "DecisionType",
    "SafeModeState",
    "CheckStatus",
    "CheckResult",
    "RiskDecision",
    "RiskDecisionSet",
    "RiskGateOutput",
    "PortfolioRiskConfig",
    "SymbolRiskConfig",
    "OperationalRiskConfig",
    "RiskGateConfig",
    "RiskCheckContext",
    "RiskCheckProtocol",
]


class DecisionType(str, Enum):
    """High-level Risk Gate decision for a single intent.

    Values:
        ALLOW: The intent passes all checks and can proceed to execution.
        BLOCK: The intent fails at least one check and must not be executed.
        DOWNGRADE: The intent is rejected and the system should enter or
            remain in Safe Mode (typically allowing only reduce/cancel/protect
            operations).
    """

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    DOWNGRADE = "DOWNGRADE"


class SafeModeState(str, Enum):
    """Trading system Safe Mode state (circuit breaker).

    Values:
        ACTIVE: Normal operations; intents may be evaluated normally.
        SAFE_REDUCING: Degraded mode; only allow reduce/cancel/protect actions.
        HALTED: Hard stop; reject all intents (including reductions).
    """

    ACTIVE = "ACTIVE"
    SAFE_REDUCING = "SAFE_REDUCING"
    HALTED = "HALTED"


class CheckStatus(str, Enum):
    """Discrete pass/fail outcome for an individual risk check."""

    PASS = "PASS"
    FAIL = "FAIL"


class CheckResult(msgspec.Struct, frozen=True, kw_only=True):
    """Outcome of running a single risk check.

    Risk checks should use stable, machine-readable reason codes so the Risk
    Gate can aggregate outcomes deterministically.

    Attributes:
        status: PASS when the check allows the intent; FAIL when the check
            blocks or triggers downgrade behaviour.
        reason_codes: One or more reason codes explaining the check outcome.
            For PASS, this may be empty. For FAIL, it should be non-empty.
        details: Free-form diagnostic payload (e.g., computed values and
            thresholds). This is intentionally flexible to allow incremental
            schema evolution without breaking journal consumers.
    """

    status: CheckStatus
    reason_codes: list[str] = msgspec.field(default_factory=list)
    details: dict[str, Any] = msgspec.field(default_factory=dict)


class RiskDecision(msgspec.Struct, frozen=True, kw_only=True):
    """Risk Gate decision for a single ``TradeIntent``.

    Attributes:
        intent_id: Identifier of the intent under evaluation.
        decision_type: ALLOW/BLOCK/DOWNGRADE decision label.
        reason_codes: Aggregated reason codes describing the decision.
        details: Aggregated details payload for auditing/debugging.
        checked_at_ns: Nanosecond Unix epoch timestamp when the decision was produced.
    """

    intent_id: str
    decision_type: DecisionType
    reason_codes: list[str] = msgspec.field(default_factory=list)
    details: dict[str, Any] = msgspec.field(default_factory=dict)
    checked_at_ns: TimestampNs


class RiskDecisionSet(SnapshotBase, frozen=True, kw_only=True):
    """Journal snapshot containing Risk Gate decisions for a batch of intents.

    Attributes:
        schema_version: See ``journal.interface.SnapshotBase``.
        system_version: See ``journal.interface.SnapshotBase``.
        asof_timestamp: See ``journal.interface.SnapshotBase``.

        decisions: One decision per evaluated intent.
        safe_mode_active: True when Safe Mode is currently active.
        safe_mode_reason: Optional reason string explaining why Safe Mode is active.
        constraints_snapshot: Serializable snapshot of the current risk
            constraints/thresholds for auditing.
    """

    decisions: list[RiskDecision] = msgspec.field(default_factory=list)
    safe_mode_active: bool = msgspec.field(default=False)
    safe_mode_reason: str | None = msgspec.field(default=None)
    constraints_snapshot: dict[str, Any] = msgspec.field(default_factory=dict)


class RiskGateOutput(SnapshotBase, frozen=True, kw_only=True):
    """High-level wrapper snapshot for Risk Gate evaluation outputs."""

    SCHEMA_VERSION = "1.3.0"

    decisions: RiskDecisionSet
    intents: OrderIntentSet
    constraints: dict[str, Any] = msgspec.field(default_factory=dict)


class PortfolioRiskConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Portfolio-level risk thresholds.

    Attributes:
        max_leverage: Maximum allowed leverage defined as
            ``gross_exposure / account_equity``.
        max_drawdown_pct: Maximum allowed drawdown fraction relative to peak
            equity (e.g., 0.2 == 20%).
        max_daily_loss_pct: Maximum allowed daily loss fraction relative to
            day-start equity (e.g., 0.02 == 2%).
        max_concentration_pct: Maximum allowed single-symbol concentration
            fraction relative to equity (e.g., 0.25 == 25%).
    """

    max_leverage: float = msgspec.field(default=1.5)
    max_drawdown_pct: float = msgspec.field(default=0.2)
    max_daily_loss_pct: float = msgspec.field(default=0.02)
    max_concentration_pct: float = msgspec.field(default=0.25)
    max_capital_usage: float | None = msgspec.field(default=None)
    capital_usage_type: str = msgspec.field(default="absolute")


class SymbolRiskConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Symbol-level risk thresholds.

    Attributes:
        max_position_size: Maximum allowed absolute position size for a single
            symbol. Units (shares vs notional) are defined by the execution
            layer and should be applied consistently by all checks.
        max_price_band_bps: Maximum allowed deviation in basis points between
            an intended order price and a reference price (mid/last). When
            ``use_dynamic_price_band`` is enabled, this serves as the minimum
            floor for the dynamic threshold.
        max_volatility_z_score: Maximum allowed absolute z-score of recent
            returns before triggering a volatility halt.
        event_window_hours: Time window around scheduled events during which
            new entries should be blocked.
        use_dynamic_price_band: Enable ATR-based dynamic price band threshold.
            When True, the effective threshold is computed as:
            ``max(max_price_band_bps, atr_pct * atr_multiplier * 10000)``
            where ``atr_pct = ATR / reference_price``.
        atr_multiplier: Multiplier for ATR-based dynamic threshold. Typical
            values: 2.0 (tight), 3.0 (balanced), 4.0 (loose). Only used when
            ``use_dynamic_price_band`` is True.
        atr_period: Lookback period for ATR calculation (default: 14). Only
            used when ``use_dynamic_price_band`` is True.
    """

    max_position_size: float = msgspec.field(default=10_000.0)
    max_price_band_bps: float = msgspec.field(default=250.0)
    max_volatility_z_score: float = msgspec.field(default=3.0)
    event_window_hours: float = msgspec.field(default=24.0)
    use_dynamic_price_band: bool = msgspec.field(default=False)
    atr_multiplier: float = msgspec.field(default=3.0)
    atr_period: int = msgspec.field(default=14)


class OperationalRiskConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Operational safety thresholds for a single run.

    Attributes:
        max_orders_per_run: Maximum number of orders/intents allowed in a run.
        max_new_positions_per_day: Maximum number of new positions allowed per day.
            When exceeded, select top N new-position intents by score (default: 15).
        rate_limit_per_second: Maximum submit rate for intents/orders.
        max_order_count: Hard cap on total order count (including cancel/replace)
            in a run/session, used as a circuit breaker.
    """

    max_orders_per_run: int = msgspec.field(default=50)
    max_new_positions_per_day: int = msgspec.field(default=15)
    rate_limit_per_second: float = msgspec.field(default=5.0)
    max_order_count: int = msgspec.field(default=200)


class RiskGateConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Aggregate configuration for the Risk Gate.

    The Risk Gate groups thresholds by the primary required context:

    - Portfolio: account equity, PnL, and cross-symbol exposure.
    - Symbol: per-symbol position and market data.
    - Operational: system pacing and run-level guardrails.
    """

    portfolio: PortfolioRiskConfig = msgspec.field(default_factory=PortfolioRiskConfig)
    symbol: SymbolRiskConfig = msgspec.field(default_factory=SymbolRiskConfig)
    operational: OperationalRiskConfig = msgspec.field(default_factory=OperationalRiskConfig)


class RiskCheckContext(msgspec.Struct, frozen=True, kw_only=True):
    """Context required to evaluate risk checks for one or more intents.

    The Risk Gate should be invoked with a coherent, point-in-time context so
    all checks operate on consistent inputs.

    Attributes:
        portfolio_snapshot: Optional raw portfolio snapshot payload (broker- or
            orchestrator-specific). This is intentionally untyped during early
            phases.
        account_equity: Current account equity in quote currency.
        positions: Current positions keyed by symbol.
        market_data: Market data snapshots keyed by symbol.
        intents: Optional batch intents for cross-intent checks (e.g., bracket legs).
        safe_mode_state: Current Safe Mode state of the system.
    """

    portfolio_snapshot: Any | None = msgspec.field(default=None)
    account_equity: float
    positions: dict[str, Position] = msgspec.field(default_factory=dict)
    market_data: dict[str, PriceSeriesSnapshot] = msgspec.field(default_factory=dict)
    intents: list[TradeIntent] = msgspec.field(default_factory=list)
    safe_mode_state: SafeModeState = msgspec.field(default=SafeModeState.ACTIVE)


@runtime_checkable
class RiskCheckProtocol(Protocol):
    """Protocol for pluggable risk checks executed by the Risk Gate.

    Implementations should be deterministic and side-effect free. On missing
    critical inputs, checks should prefer fail-closed outcomes (i.e., return
    FAIL reason codes, or return a degraded/failed ``Result`` that the Risk Gate
    can treat as a BLOCK/DOWNGRADE according to policy).
    """

    def check(self, intent: TradeIntent, context: RiskCheckContext, config: RiskGateConfig) -> Result[CheckResult]:
        """Evaluate a single risk check for a given intent.

        Args:
            intent: Strategy Engine trade intent to evaluate.
            context: Point-in-time risk context (equity, positions, market data,
                and Safe Mode state).
            config: Risk Gate configuration thresholds.

        Returns:
            Result[CheckResult]: PASS/FAIL outcome with reason codes and details.
        """
