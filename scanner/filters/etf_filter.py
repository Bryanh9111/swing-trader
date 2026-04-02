"""ETF symbol filter.

This filter rejects symbols that match common ETF ticker patterns so the
platform scanner focuses on single-name equities by default.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, TYPE_CHECKING

from common.interface import Result
from common.utils import is_etf
from scanner.interface import ScannerConfig

from .base import BaseFilter, FilterResult

if TYPE_CHECKING:
    from data.interface import PriceBar
else:  # pragma: no cover - type-only import may not exist in early phases.
    PriceBar = Any  # type: ignore[assignment]


class ETFFilter(BaseFilter):
    name: ClassVar[str] = "etf_filter"

    def __init__(self, *, symbol: str, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._symbol = str(symbol)

    def _apply_filter(self, bars: Sequence[PriceBar], config: ScannerConfig) -> Result[FilterResult]:
        exclude_etfs = bool(getattr(config, "exclude_etfs", True))
        etf = is_etf(self._symbol)
        passed = (not exclude_etfs) or (not etf)
        reason_code = "OK" if passed else "ETF_FILTERED"

        return Result.success(
            FilterResult(
                passed=passed,
                reason="OK" if passed else reason_code,
                score=1.0 if passed else 0.0,
                features={"symbol": self._symbol, "is_etf": etf},
                metadata={"exclude_etfs": exclude_etfs},
            ),
            reason_code=reason_code,
        )
