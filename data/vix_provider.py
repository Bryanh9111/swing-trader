"""VIX data provider (Yahoo Finance via yfinance)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import msgspec
import structlog

__all__ = ["VIXData", "VIXProvider"]


class VIXData(msgspec.Struct, frozen=True, kw_only=True):
    """Point-in-time VIX reading with simple derived fields."""

    timestamp: int
    value: float
    change_pct: float
    level: str  # "low" | "normal" | "elevated" | "extreme"


def _to_timestamp_ns(index_value: object) -> int:
    if isinstance(index_value, datetime):
        dt = index_value if index_value.tzinfo is not None else index_value.replace(tzinfo=UTC)
        return int(dt.astimezone(UTC).timestamp() * 1_000_000_000)
    if hasattr(index_value, "to_pydatetime"):
        try:
            dt2 = index_value.to_pydatetime()
            if isinstance(dt2, datetime):
                dt2 = dt2 if dt2.tzinfo is not None else dt2.replace(tzinfo=UTC)
                return int(dt2.astimezone(UTC).timestamp() * 1_000_000_000)
        except Exception:  # noqa: BLE001
            pass
    return time.time_ns()


def _classify_vix_level(value: float) -> str:
    if not math.isfinite(value):
        return "normal"
    if value < 15.0:
        return "low"
    if value <= 25.0:
        return "normal"
    if value <= 35.0:
        return "elevated"
    return "extreme"


@dataclass(frozen=True, slots=True)
class VIXProvider:
    """Fetch VIX data from Yahoo Finance (symbol: ^VIX)."""

    symbol: str = "^VIX"
    requests_timeout_seconds: float = 20.0
    _logger: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_logger", structlog.get_logger(__name__).bind(source="vix", symbol=self.symbol))

    def fetch_vix_current(self) -> VIXData | None:
        history = self.fetch_vix_history(days=2)
        return history[-1] if history else None

    def fetch_vix_history(self, days: int = 30) -> list[VIXData]:
        if days <= 0:
            return []

        try:
            import yfinance as yf
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("vix.missing_dependency", error=str(exc)[:200])
            return []

        # Fetch slightly more than requested to ensure we can compute day-over-day change.
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=int(days) + 14)

        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(
                interval="1d",
                start=start.date().isoformat(),
                end=(end.date() + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=False,
                timeout=float(self.requests_timeout_seconds),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("vix.request_failed", error=str(exc)[:200])
            return []

        if df is None or len(df) < 1:
            return []

        closes: list[float] = []
        timestamps: list[int] = []
        for row in df.itertuples():
            try:
                close = float(getattr(row, "Close"))
            except Exception:  # noqa: BLE001
                continue
            if not math.isfinite(close):
                continue
            closes.append(close)
            timestamps.append(_to_timestamp_ns(getattr(row, "Index")))

        if len(closes) < 1:
            return []

        start_idx = max(0, len(closes) - int(days))
        payload: list[VIXData] = []

        for idx in range(start_idx, len(closes)):
            value = float(closes[idx])
            if idx == 0:
                change_pct = 0.0
            else:
                prev = float(closes[idx - 1])
                change_pct = 0.0 if prev == 0 else (value - prev) / prev

            payload.append(
                VIXData(
                    timestamp=int(timestamps[idx]),
                    value=value,
                    change_pct=float(change_pct),
                    level=_classify_vix_level(value),
                )
            )

        return payload
