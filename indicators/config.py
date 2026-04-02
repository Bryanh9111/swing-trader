"""Configuration schema for the technical indicators module."""

from __future__ import annotations

import msgspec


class RSIConfig(msgspec.Struct, frozen=True, kw_only=True):
    """RSI indicator configuration."""

    period: int = 14


class MACDConfig(msgspec.Struct, frozen=True, kw_only=True):
    """MACD indicator configuration."""

    fast: int = 12
    slow: int = 26
    signal: int = 9


class EMAConfig(msgspec.Struct, frozen=True, kw_only=True):
    """EMA indicator configuration."""

    periods: list[int] = msgspec.field(default_factory=lambda: [12, 26])


class SMAConfig(msgspec.Struct, frozen=True, kw_only=True):
    """SMA indicator configuration."""

    periods: list[int] = msgspec.field(default_factory=lambda: [20, 50, 200])


class IndicatorsConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Top-level configuration for technical indicators."""

    rsi: RSIConfig = RSIConfig()
    macd: MACDConfig = MACDConfig()
    ema: EMAConfig = EMAConfig()
    sma: SMAConfig = SMAConfig()


__all__ = [
    "RSIConfig",
    "MACDConfig",
    "EMAConfig",
    "SMAConfig",
    "IndicatorsConfig",
]
