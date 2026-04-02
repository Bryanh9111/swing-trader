"""Fundamental metrics adapter (primary: YFinance).

Polygon fundamentals endpoints are unavailable (403) on the current plan, so we
use ``yfinance`` as the primary fallback source for basic screening metrics.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import msgspec

from common.interface import Result

__all__ = ["fetch_fundamentals"]


class _FundamentalCacheEntry(msgspec.Struct, frozen=True, kw_only=True):
    saved_at_ns: int
    metrics: dict[str, Any]


_CACHE_DIR = Path(".cache/fundamentals")
_CACHE_TTL_SECONDS = 24 * 60 * 60


def fetch_fundamentals(symbols: list[str], timeout: float = 5.0) -> Result[dict[str, dict[str, Any]]]:
    """Fetch fundamental metrics for multiple symbols.

    Notes:
        - Best-effort per-symbol: failures don't block other symbols.
        - Cache TTL is 24 hours (fundamentals update slowly).
        - Partial success returns ``Result.success`` with reason_code
          ``FUNDAMENTALS_DEGRADED``.
    """

    normalized = _normalize_symbols(symbols)
    if not normalized:
        return Result.failed(ValueError("symbols must be a non-empty list of tickers."), "INVALID_SYMBOLS")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    decoder = msgspec.json.Decoder(type=_FundamentalCacheEntry)
    encoder = msgspec.json.Encoder()

    cached: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for symbol in normalized:
        hit = _cache_get(symbol, decoder)
        if hit is None:
            missing.append(symbol)
            continue
        cached[symbol] = hit

    if not missing:
        return Result.success(cached, reason_code="OK")

    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        if cached:
            return Result.success(cached, reason_code="FUNDAMENTALS_DEGRADED")
        return Result.failed(exc, reason_code="MISSING_DEPENDENCY")

    max_workers = min(8, max(1, len(missing)))
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    def _task(sym: str) -> dict[str, Any]:
        ticker = yf.Ticker(sym)
        info = ticker.info  # may issue network requests lazily
        if not isinstance(info, dict) or not info:
            raise RuntimeError("Empty fundamentals payload.")
        return _extract_metrics(info)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fundamentals") as pool:
        futures = {pool.submit(_task, sym): sym for sym in missing}
        for fut, sym in list(futures.items()):
            try:
                metrics = fut.result(timeout=float(timeout))
            except FutureTimeoutError:
                failures[sym] = "TIMEOUT"
                continue
            except Exception as exc:  # noqa: BLE001
                failures[sym] = type(exc).__name__
                continue

            successes[sym] = metrics
            _cache_set(sym, metrics, encoder)

    merged = {**cached, **successes}
    if not merged:
        return Result.failed(RuntimeError("All fundamentals fetches failed."), "FUNDAMENTALS_ALL_FAILED")

    if failures:
        return Result.success(merged, reason_code="FUNDAMENTALS_DEGRADED")

    return Result.success(merged, reason_code="OK")


def _normalize_symbols(symbols: list[str]) -> list[str]:
    if not isinstance(symbols, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        if not isinstance(raw, str):
            continue
        sym = raw.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        normalized.append(sym)
    return normalized


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_").upper()
    return _CACHE_DIR / f"{safe}.json"


def _cache_get(symbol: str, decoder: msgspec.json.Decoder) -> dict[str, Any] | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None

    try:
        entry = decoder.decode(path.read_bytes())
    except Exception:  # noqa: BLE001
        return None

    age_seconds = (time.time_ns() - int(entry.saved_at_ns)) / 1_000_000_000
    if _CACHE_TTL_SECONDS > 0 and age_seconds > _CACHE_TTL_SECONDS:
        try:
            path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return None

    if not isinstance(entry.metrics, dict):
        return None
    return dict(entry.metrics)


def _cache_set(symbol: str, metrics: dict[str, Any], encoder: msgspec.json.Encoder) -> None:
    path = _cache_path(symbol)
    entry = _FundamentalCacheEntry(saved_at_ns=time.time_ns(), metrics=dict(metrics))
    try:
        payload = encoder.encode(entry)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
    except Exception:  # noqa: BLE001
        return


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
    except Exception:  # noqa: BLE001
        return None
    if f != f:  # NaN
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return f


def _extract_metrics(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "trailing_pe": _safe_float(info.get("trailingPE")),
        "price_to_book": _safe_float(info.get("priceToBook")),
        "return_on_equity": _safe_float(info.get("returnOnEquity")),
        "profit_margin": _safe_float(info.get("profitMargins")),
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "current_ratio": _safe_float(info.get("currentRatio")),
        "earnings_growth": _safe_float(info.get("earningsGrowth")),
        "revenue_growth": _safe_float(info.get("revenueGrowth")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "peg_ratio": _safe_float(info.get("pegRatio")),
    }
