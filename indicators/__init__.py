"""Technical Indicators module for Scanner and Strategy.

Provides RSI, MACD, EMA, SMA calculations compatible with PriceBar sequences.
"""

from indicators.config import EMAConfig, IndicatorsConfig, MACDConfig, RSIConfig, SMAConfig
from indicators.interface import (
    BollingerBandsResult,
    KDJResult,
    MACDResult,
    IndicatorProtocol,
    InputPrices,
    VolumePriceDivergence,
    compute_atr_last,
    compute_bbands_last,
    compute_bbands_series,
    compute_ema_last,
    compute_ema_series,
    compute_kdj_last,
    compute_kdj_series,
    compute_macd_last,
    compute_macd_series,
    compute_obv_last,
    compute_obv_series,
    compute_rsi_last,
    compute_rsi_series,
    compute_sma_last,
    compute_sma_series,
    compute_volume_price_divergence,
    compute_vpt_last,
    compute_vpt_series,
)

__all__ = [
    # Results
    "MACDResult",
    "BollingerBandsResult",
    "KDJResult",
    "InputPrices",
    "IndicatorProtocol",
    "VolumePriceDivergence",
    # RSI
    "compute_rsi_last",
    "compute_rsi_series",
    # MACD
    "compute_macd_last",
    "compute_macd_series",
    # EMA
    "compute_ema_last",
    "compute_ema_series",
    # SMA
    "compute_sma_last",
    "compute_sma_series",
    # ATR
    "compute_atr_last",
    # Bollinger Bands
    "compute_bbands_last",
    "compute_bbands_series",
    # KDJ
    "compute_kdj_last",
    "compute_kdj_series",
    # Volume-Price
    "compute_obv_last",
    "compute_obv_series",
    "compute_vpt_last",
    "compute_vpt_series",
    "compute_volume_price_divergence",
    # Config
    "RSIConfig",
    "MACDConfig",
    "EMAConfig",
    "SMAConfig",
    "IndicatorsConfig",
]
