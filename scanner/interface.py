"""Scanner interfaces for detecting consolidation (platform) patterns.

This module defines the Phase 2.3 scanner boundary for Automated Swing Trader
(AST). The scanner consumes a universe snapshot plus validated price series
snapshots, detects platform/consolidation patterns across multiple lookback
windows, and returns a versioned ``CandidateSet`` snapshot.

The schemas below intentionally follow the Phase 1 architecture patterns:

- Immutable, keyword-only value objects built on ``msgspec.Struct`` for
  deterministic serialization.
- Versioned snapshot payloads derived from ``journal.interface.SnapshotBase``.
- A runtime-checkable ``ScannerPlugin`` protocol built on the shared
  ``plugins.interface.PluginBase`` contract.
"""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

import msgspec

from journal.interface import SnapshotBase
from plugins.interface import PluginBase
from scanner.reversal_detector import ReversalConfig

if TYPE_CHECKING:
    from data.interface import PriceBar, PriceSeriesSnapshot
    from universe.interface import EquityInfo, UniverseSnapshot
else:  # pragma: no cover - type-only imports may not exist in early phases.
    PriceBar = Any  # type: ignore[assignment]
    PriceSeriesSnapshot = Any  # type: ignore[assignment]
    EquityInfo = Any  # type: ignore[assignment]
    UniverseSnapshot = Any  # type: ignore[assignment]

__all__ = [
    "ScannerConfig",
    "IndicatorScoringConfig",
    "ADXEntryFilterConfig",
    "MA200TrendFilterConfig",
    "LiquidityFilterConfig",
    "PullbackConfirmationConfig",
    "PlatformFeatures",
    "PlatformCandidate",
    "CandidateSet",
    "ScannerPlugin",
]


class ADXEntryFilterConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for ADX entry filtering (trend confirmation)."""

    enabled: bool = False
    min_adx: float = 20.0
    period: int = 14


class MA200TrendFilterConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for MA200 trend filtering (long-term trend confirmation)."""

    enabled: bool = True
    period: int = 200
    fallback_periods: list[int] = msgspec.field(default_factory=lambda: [200, 100])
    require_above: bool = True
    tolerance_pct: float = 0.02


class PullbackConfirmationConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for pullback confirmation filtering (breakout + pullback + hold)."""

    enabled: bool = False
    lookback_days: int = 5
    pullback_tolerance_pct: float = 0.02
    require_volume_increase: bool = False
    volume_increase_ratio: float = 1.5
    # When set, candidates with score >= threshold can skip pullback confirmation.
    # None means do not skip based on score.
    score_threshold_for_skip: float | None = None


class LiquidityFilterConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for liquidity monitoring (soft filter)."""

    enabled: bool = True
    min_liquidity_score: float = 0.3
    min_dollar_volume: float = 1_000_000.0
    max_spread_proxy: float = 0.05


class ScannerConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration controlling platform-period detection and scoring.

    The scanner evaluates platform candidates across multiple rolling windows
    (e.g., 20/30/60 trading days). Each window produces a per-window score that
    is aggregated using ``window_weights``.

    Attributes:
        windows: Lookback windows (in trading days) used for detection and
            multi-window scoring.
        box_threshold: Maximum allowed platform height expressed as a fraction
            of price (e.g., 0.10 == 10% high/low range).
        ma_diff_threshold: Maximum allowed moving-average dispersion within the
            window, measured as ``std(MAs) / mean(MAs)``.
        volatility_threshold: Maximum allowed return volatility within the
            window, measured as standard deviation of returns.
        volume_change_threshold: Maximum allowed ratio of recent to prior
            average volume (<= threshold indicates contraction).
        volume_stability_threshold: Maximum allowed coefficient of variation
            for volume (``std(volume) / mean(volume)``).
        volume_increase_threshold: Minimum required ratio for breakout volume
            (>= threshold indicates expansion).
        min_box_quality: Minimum box quality score (0~1) required for a
            candidate to be emitted.
        min_atr_pct: Minimum ATR percentage (ATR / close) for liquidity/energy
            screening, typically used for US equities.
        max_atr_pct: Maximum ATR percentage (ATR / close) to avoid overly
            volatile symbols.
        min_dollar_volume: Minimum average dollar volume (e.g., 20D) to avoid
            illiquid symbols.
        window_weights: Per-window weights used to compute a weighted score.
            Keys should be present for each member of ``windows``.

        decline_threshold: Minimum decline percentage from historical high for
            :class:`~scanner.filters.low_position_filter.LowPositionFilter`.
        min_periods: Minimum number of bars since the historical high for
            :class:`~scanner.filters.low_position_filter.LowPositionFilter`.
        rapid_decline_threshold: Minimum peak-to-trough decline within the
            recent window for :class:`~scanner.filters.rapid_decline_filter.RapidDeclineFilter`.
        rapid_decline_days: Lookback window size (in bars) used by
            :class:`~scanner.filters.rapid_decline_filter.RapidDeclineFilter`.
        resistance_proximity_pct: Maximum percentage distance between the
            latest close and recent resistance for
            :class:`~scanner.filters.breakthrough_filter.BreakthroughFilter`.
        volume_increase_ratio: Minimum ratio of recent average volume to prior
            average volume for :class:`~scanner.filters.breakthrough_filter.BreakthroughFilter`.

        use_low_position_filter: Enable :class:`~scanner.filters.low_position_filter.LowPositionFilter`.
        use_rapid_decline_filter: Enable :class:`~scanner.filters.rapid_decline_filter.RapidDeclineFilter`.
        use_breakthrough_filter: Enable :class:`~scanner.filters.breakthrough_filter.BreakthroughFilter`.
        use_pullback_confirmation_filter: Enable :class:`~scanner.filters.pullback_confirmation_filter.PullbackConfirmationFilter`.

        use_filter_chain_detector: Use :class:`~scanner.filters.chain.FilterChain` in
            :func:`~scanner.detector.detect_platform_candidate`. Set to False to use the
            legacy detector implementation.

        indicator_scoring: Optional indicator-based score adjustment configuration. When
            enabled, indicator scoring influences ranking only (never filtering).
    """

    windows: list[int] = msgspec.field(default_factory=lambda: [20, 30, 60])

    box_threshold: float = 0.06  # US market: 6% (stricter than A-share 10%)
    # Reason: US quality platforms typically have <5-7% range.
    # A-share platforms can be looser (8-12%) due to different market dynamics ("zhuangjia" control).
    ma_diff_threshold: float = 0.02
    volatility_threshold: float = 0.035  # US market: 3.5% (higher than A-share 2.5%)
    # Reason: US (especially tech) daily volatility of 3-5% is common; using 2.5% is overly strict.
    # A-share volatility distribution is different (daily limit rules + different participant structure).

    volume_change_threshold: float = 0.9
    volume_stability_threshold: float = 0.5
    volume_increase_threshold: float = 1.5

    min_box_quality: float = 0.6

    min_atr_pct: float = 0.01
    max_atr_pct: float = 0.05
    min_dollar_volume: float = 5_000_000  # US market: $5M (more conservative than $1M)
    # Reason: $1M average dollar volume is often micro-cap territory with liquidity + spread traps.
    # US market micro-caps have higher gap risk (pre/after-hours) and are more vulnerable to manipulation.

    decline_threshold: float = 0.10  # US market: 10% (safer than A-share 20%)
    # Reason: 20% drawdown in US stocks often signals fundamental deterioration (not "accumulation").
    # A-share "low position" logic relies on different market behavior and can be a value-trap in US.
    min_periods: int = 20

    rapid_decline_threshold: float = 0.08  # US market: 8% (avoid post-earnings falling knives)
    # Reason: Rapid declines in US often come from earnings misses, regulatory issues, guidance cuts, etc.
    # Treating 15%+ drops as "washout" increases tail-risk during earnings season.
    rapid_decline_days: int = 30

    resistance_proximity_pct: float = 0.02
    volume_increase_ratio: float = 1.5  # US market: 1.5x (stronger breakout confirmation)
    # Reason: Institutional participation typically shows up as clearer volume expansion in US equities.
    # A-share breakouts can work with weaker expansion due to different liquidity + participant mix.

    # Breakthrough confirmation (close above resistance for N days)
    require_breakout_confirmation: bool = True
    breakout_confirmation_days: int = 2

    # Breakout day volume spike confirmation (volume expansion on breakout day)
    require_breakout_volume_spike: bool = True
    breakout_volume_ratio: float = 1.5

    use_low_position_filter: bool = False
    # US market: DISABLED for safety
    # Risk: In US equities, a 20%+ drawdown often indicates fundamental deterioration,
    #       not "accumulation" by institutions (unlike A-share "zhuangjia" behavior).
    # Example: Meta -76% in 2022 (multiple "stabilizations" were false signals).
    # TODO: Re-enable after adding fundamental filters (revenue growth, earnings trend).

    use_rapid_decline_filter: bool = False
    # US market: DISABLED for safety
    # Risk: Rapid declines are often tied to earnings misses, regulatory issues, CEO departure, etc.
    # Example: SVB Bank -60% in 3 days → "stabilization" → $0.
    # TODO: Re-enable after integrating Event Guard (earnings calendar) + sentiment analysis.

    use_breakthrough_filter: bool = True  # US market: ENABLED (generic resistance breakout logic)
    pullback_confirmation: PullbackConfirmationConfig | None = None
    use_pullback_confirmation_filter: bool = False
    use_filter_chain_detector: bool = True

    adx_entry_filter: ADXEntryFilterConfig = msgspec.field(default_factory=lambda: ADXEntryFilterConfig())
    ma200_trend_filter: MA200TrendFilterConfig = msgspec.field(default_factory=lambda: MA200TrendFilterConfig())
    liquidity_filter: LiquidityFilterConfig = msgspec.field(default_factory=LiquidityFilterConfig)

    use_event_guard_filter: bool = True
    earnings_window_days: int = 10

    use_market_cap_filter: bool = True

    # Phase 6: Fundamental enrichment (information only; no hard filtering)
    enrich_with_fundamentals: bool = False

    # Seasonal adaptation (auto-relax parameters during low-volume months)
    enable_seasonal_adaptation: bool = False
    seasonal_low_volume_months: list[int] = msgspec.field(default_factory=lambda: [8, 12])
    seasonal_adjustments: dict[str, float] = msgspec.field(
        default_factory=lambda: {"volume_increase_ratio": 1.2, "min_dollar_volume": 500000.0}
    )

    window_weights: dict[int, float] = msgspec.field(
        default_factory=lambda: {20: 0.3, 30: 0.4, 60: 0.3},
    )

    indicator_scoring: IndicatorScoringConfig = msgspec.field(default_factory=lambda: IndicatorScoringConfig())

    reversal: ReversalConfig = msgspec.field(default_factory=ReversalConfig)
    use_reversal_detector: bool = False

    # Trend pattern detection (Growth/AI stocks)
    # When enabled, scanner also checks for EMA Pullback, Gap Pullback, and Flag Breakout patterns
    # These patterns are gated by sector regime (XLK bullish) and relative strength (stock > QQQ)
    use_trend_pattern_detector: bool = False
    trend_pattern_config: dict[str, Any] | None = None  # Optional detailed config


class IndicatorScoringConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Configuration for optional indicator-based score adjustment.

    When enabled, indicator scores are combined with the base scanner score
    to produce a final adjusted score. This is OPTIONAL and does NOT filter
    candidates; it only influences ranking.

    Attributes:
        enabled: Master switch for indicator scoring.
        combination_mode: How to combine indicator scores with base score.
            - "multiply": final = base * indicator_score (default)
            - "weighted_avg": final = base * base_weight + indicator * (1-base_weight)
            - "min": final = min(base, indicator_score)
        base_weight: Weight for base scanner score in weighted_avg mode (0-1).

        rsi_enabled: Enable RSI scoring.
        rsi_weight: Weight for RSI in indicator score calculation.
        rsi_optimal_range: (min, max) RSI values considered optimal (score=1.0).
            Values outside this range get lower scores.

        macd_enabled: Enable MACD scoring.
        macd_weight: Weight for MACD in indicator score calculation.
        macd_histogram_threshold: Minimum histogram value for score=1.0.
            Negative values get lower scores.

        atr_enabled: Enable ATR scoring.
        atr_weight: Weight for ATR in indicator score calculation.
        atr_optimal_range: (min, max) ATR% values considered optimal.
            Too low (illiquid) or too high (volatile) get lower scores.

        bbands_enabled: Enable Bollinger Bands scoring.
        bbands_weight: Weight for Bollinger Bands in indicator score calculation.
        bbands_period: Lookback window size for Bollinger Bands.
        bbands_std_dev: Standard deviation multiplier for Bollinger Bands.

        obv_enabled: Enable OBV/divergence scoring.
        obv_weight: Weight for OBV/divergence in indicator score calculation.
        divergence_lookback: Lookback window for divergence detection.

        short_enabled: Enable short interest/volume scoring (requires external data).
        short_weight: Weight for short score in indicator score calculation.
        short_squeeze_threshold: short% threshold (0-1) to flag squeeze potential.

        kdj_enabled: Enable KDJ scoring (momentum/stochastic indicator).
        kdj_weight: Weight for KDJ in indicator score calculation.
        kdj_n_period: Lookback period for highest high / lowest low (default 9).
        kdj_k_period: Smoothing period for K line (default 3).
        kdj_d_period: Smoothing period for D line (default 3).
        kdj_oversold: J value threshold for oversold condition (bullish signal).
        kdj_overbought: J value threshold for overbought condition (bearish signal).
    """

    enabled: bool = msgspec.field(default=False)
    combination_mode: str = msgspec.field(default="multiply")
    base_weight: float = msgspec.field(default=0.7)

    rsi_enabled: bool = msgspec.field(default=True)
    rsi_weight: float = msgspec.field(default=0.3)
    rsi_optimal_range: tuple[float, float] = msgspec.field(default=(30.0, 70.0))

    macd_enabled: bool = msgspec.field(default=True)
    macd_weight: float = msgspec.field(default=0.4)
    macd_histogram_threshold: float = msgspec.field(default=0.0)

    atr_enabled: bool = msgspec.field(default=True)
    atr_weight: float = msgspec.field(default=0.3)
    atr_optimal_range: tuple[float, float] = msgspec.field(default=(0.01, 0.03))

    bbands_enabled: bool = msgspec.field(default=True)
    bbands_weight: float = msgspec.field(default=0.3)
    bbands_period: int = msgspec.field(default=20)
    bbands_std_dev: float = msgspec.field(default=2.0)

    obv_enabled: bool = msgspec.field(default=True)
    obv_weight: float = msgspec.field(default=0.3)
    divergence_lookback: int = msgspec.field(default=20)

    short_enabled: bool = msgspec.field(default=False)
    short_weight: float = msgspec.field(default=0.2)
    short_squeeze_threshold: float = msgspec.field(default=0.20)

    kdj_enabled: bool = msgspec.field(default=True)
    kdj_weight: float = msgspec.field(default=0.3)
    kdj_n_period: int = msgspec.field(default=9)
    kdj_k_period: int = msgspec.field(default=3)
    kdj_d_period: int = msgspec.field(default=3)
    kdj_oversold: float = msgspec.field(default=20.0)
    kdj_overbought: float = msgspec.field(default=80.0)


class PlatformFeatures(msgspec.Struct, frozen=True, kw_only=True):
    """Computed features describing a detected platform/consolidation period.

    Features are grouped into price, volume, and structure components. Some
    fields (e.g., ATR, breakout volume, support/resistance) may be unavailable
    depending on the data source or on whether a breakout context exists.

    Attributes:
        box_range: Platform height expressed as ``(box_high - box_low) / box_low``.
        box_low: Lower bound of the detected platform (support region).
        box_high: Upper bound of the detected platform (resistance region).
        ma_diff: Moving-average convergence metric (lower is tighter).
        volatility: Return volatility (standard deviation of returns).
        atr_pct: Average true range as a fraction of close (ATR / close), if
            available.
        volume_change_ratio: Ratio of recent average volume to the previous
            average volume (lower implies contraction).
        volume_stability: Coefficient of variation for volume (lower implies
            steadier activity).
        volume_increase_ratio: Breakout volume ratio relative to baseline, if
            evaluated in a breakout context.
        avg_dollar_volume: Average traded dollar volume over the window.
        box_quality: Box quality score (0~1) combining support/resistance,
            containment, trend, and height factors.
        support_level: Primary detected support level, if available.
        resistance_level: Primary detected resistance level, if available.
    """

    box_range: float
    box_low: float
    box_high: float
    ma_diff: float
    volatility: float
    atr_pct: float | None = None

    volume_change_ratio: float
    volume_stability: float
    volume_increase_ratio: float | None = None
    avg_dollar_volume: float

    box_quality: float | None = None
    support_level: float | None = None
    resistance_level: float | None = None


class PlatformCandidate(msgspec.Struct, frozen=True, kw_only=True):
    """A single platform candidate detected for a symbol and lookback window.

    Attributes:
        symbol: The instrument identifier (e.g., ticker).
        detected_at: Nanosecond Unix epoch timestamp when detection occurred.
        window: Lookback window (in trading days) used for detection.
        score: Composite score (0~1) used for ranking and filtering.
        features: Structured feature payload describing the platform.
        invalidation_level: Price level that invalidates the setup (stop-loss).
        target_level: Price level representing the expected target (take-profit).
        reasons: Human-readable reasons or rule keys supporting inclusion.
        meta: Additional metadata such as parameter snapshots, data lineage,
            model versions, or debugging context.
    """

    symbol: str
    detected_at: int
    window: int
    score: float

    features: PlatformFeatures

    invalidation_level: float
    target_level: float

    reasons: list[str]

    meta: dict[str, Any] = msgspec.field(default_factory=dict)

    signal_type: str = "platform_breakout"
    reversal_rsi: float | None = None
    reversal_volume_ratio: float | None = None
    reversal_signal_strength: float | None = None


class CandidateSet(SnapshotBase, frozen=True, kw_only=True):
    """Versioned snapshot capturing detected platform candidates for a run.

    The snapshot follows the Phase 1 journaling pattern by inheriting the
    metadata envelope from ``SnapshotBase``.

    Attributes:
        candidates: Ordered list of detected candidates across all windows and
            symbols.
        total_scanned: Total number of symbols evaluated.
        total_detected: Total number of candidates emitted.
        config_snapshot: Serialized configuration used for detection, typically
            derived from ``ScannerConfig``.
        data_source: Price data origin (e.g., ``"polygon"``, ``"yahoo"``, ``"cache"``).
        universe_source: Universe origin (e.g., ``"polygon"``, ``"fmp"``, ``"cache"``).
    """

    candidates: list[PlatformCandidate]

    total_scanned: int
    total_detected: int

    config_snapshot: dict[str, Any]
    data_source: str
    universe_source: str


@runtime_checkable
class ScannerPlugin(
    PluginBase[
        ScannerConfig,
        tuple[UniverseSnapshot, dict[str, PriceSeriesSnapshot]],
        CandidateSet,
    ],
    Protocol,
):
    """Protocol contract for scanner plugins.

    Input:
        ``(UniverseSnapshot, {symbol: PriceSeriesSnapshot})``

    Config:
        ``ScannerConfig``

    Output:
        ``CandidateSet``
    """
