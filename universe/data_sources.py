"""Universe reference data sources.

Polygon/FMP integrations are planned as primary/secondary sources.
For Phase 2.1 we implement a practical tertiary fallback using ``yfinance``
with a local on-disk cache and TTL enforcement.
"""

from __future__ import annotations

import os
import inspect
import time
from datetime import datetime, timedelta
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import msgspec
import requests
import structlog

from common.interface import Result, ResultStatus

from .interface import EquityInfo

__all__ = [
    "fetch_polygon_universe",
    "fetch_polygon_ticker_details",
    "fetch_fmp_universe",
    "fetch_cached_universe",
]

_CACHE_DIR = Path(".cache/universe")
_CACHE_PATH = _CACHE_DIR / "yfinance_universe.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60

_POLYGON_TICKERS_URL = "https://api.polygon.io/v3/reference/tickers"
_POLYGON_TICKER_DETAILS_URL = "https://api.polygon.io/v3/reference/tickers/{ticker}"
_POLYGON_REQUEST_TIMEOUT_SECONDS = 15.0


def fetch_polygon_ticker_details(
    symbols: list[str],
    *,
    max_workers: int = 10,
    rate_limit_per_second: int = 10,
    timeout_seconds: float = 30.0,
) -> dict[str, dict[str, Any]]:
    """Fetch ticker details for ``symbols`` using Polygon's ticker details endpoint.

    This is a best-effort enrichment call used to increase market cap coverage
    without impacting the core universe build path.
    """

    import concurrent.futures

    from data.rate_limiter import TokenBucketRateLimiter

    logger = structlog.get_logger(__name__).bind(source="polygon", endpoint="/v3/reference/tickers/{ticker}")

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        logger.warning("polygon.details.missing_api_key")
        return {}

    requested_symbols = [symbol.strip().upper() for symbol in symbols if isinstance(symbol, str) and symbol.strip()]
    if not requested_symbols:
        return {}

    limiter = TokenBucketRateLimiter(
        calls_per_minute=rate_limit_per_second * 60,
        bucket_size=rate_limit_per_second,
    )

    def _fetch_one(symbol: str) -> tuple[str, dict[str, Any] | None]:
        acquired = False
        try:
            acquired = limiter.acquire(timeout_seconds=timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("polygon.details.rate_limiter_failed", symbol=symbol, error=str(exc)[:200])
            return symbol, None

        if not acquired:
            logger.warning("polygon.details.rate_limit_timeout", symbol=symbol)
            return symbol, None

        url = _POLYGON_TICKER_DETAILS_URL.format(ticker=symbol)
        try:
            response = requests.get(url, params={"apiKey": api_key}, timeout=timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("polygon.details.fetch_failed", symbol=symbol, error=str(exc)[:200])
            return symbol, None

        if response.status_code != 200:
            logger.warning("polygon.details.http_error", symbol=symbol, status=response.status_code)
            return symbol, None

        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("polygon.details.json_decode_failed", symbol=symbol, error=str(exc)[:200])
            return symbol, None

        results_field = payload.get("results")
        if not isinstance(results_field, dict):
            logger.warning("polygon.details.unexpected_payload", symbol=symbol)
            return symbol, None

        market_cap = _as_float(results_field.get("market_cap"))
        sector = _as_nonempty_str(results_field.get("sic_description")) or _as_nonempty_str(results_field.get("sector"))

        if market_cap is None and sector is None:
            return symbol, None

        return symbol, {"market_cap": market_cap, "sector": sector}

    results: dict[str, dict[str, Any]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_one, symbol) for symbol in requested_symbols]
        for future in concurrent.futures.as_completed(futures):
            try:
                symbol, details = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("polygon.details.future_failed", error=str(exc)[:200])
                continue
            if details is not None:
                results[symbol] = details

    logger.info(
        "polygon.details.batch_complete",
        requested=len(requested_symbols),
        succeeded=len(results),
        coverage=(len(results) / len(requested_symbols)) if requested_symbols else 0.0,
    )

    return results

try:
    from massive import RESTClient  # type: ignore[import-not-found]

    HAS_MASSIVE_SDK = True
except ImportError:  # pragma: no cover
    HAS_MASSIVE_SDK = False
    RESTClient = None  # type: ignore[assignment]


class _UniverseCacheEntry(msgspec.Struct, frozen=True, kw_only=True):
    saved_at_ns: int
    equities: list[EquityInfo]


@dataclass(frozen=True, slots=True)
class _Candidate:
    symbol: str
    exchange: str | None
    price: float | None
    market_cap: float | None
    avg_dollar_volume_20d: float | None
    sector: str | None


def fetch_polygon_universe() -> Result[list[EquityInfo]]:
    """Fetch the US equity universe reference data from Polygon."""

    if HAS_MASSIVE_SDK:
        sdk_result = _fetch_polygon_universe_with_sdk()
        if sdk_result.status == ResultStatus.SUCCESS:
            return sdk_result

        logger = structlog.get_logger(__name__).bind(source="polygon", impl="massive_sdk")
        logger.warning(
            "polygon.universe.massive_sdk_failed_using_fallback",
            reason_code=sdk_result.reason_code,
            error=str(sdk_result.error)[:200] if sdk_result.error else None,
        )
        return _fetch_polygon_universe_fallback()

    logger = structlog.get_logger(__name__).bind(source="polygon", impl="fallback", endpoint="/v3/reference/tickers")
    logger.warning("polygon.universe.massive_sdk_missing_using_fallback")
    return _fetch_polygon_universe_fallback()


def _fetch_polygon_universe_with_sdk() -> Result[list[EquityInfo]]:
    logger = structlog.get_logger(__name__).bind(
        source="polygon",
        impl="massive_sdk",
        endpoints=[
            "/v3/reference/tickers",
            "/v2/aggs/grouped/locale/us/market/stocks/{date}",
        ],
    )
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        logger.warning("polygon.universe.missing_api_key")
        return Result.failed(RuntimeError("POLYGON_API_KEY is not set."), "MISSING_API_KEY")

    if RESTClient is None:  # pragma: no cover
        logger.warning("polygon.universe.massive_sdk_unavailable")
        return Result.failed(RuntimeError("massive SDK is not available."), "NOT_AVAILABLE")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    with suppress(Exception):
        if "timeout" in inspect.signature(RESTClient).parameters:
            client_kwargs["timeout"] = _POLYGON_REQUEST_TIMEOUT_SECONDS
    client = RESTClient(**client_kwargs)

    tickers_by_symbol: dict[str, dict[str, Any]] = {}
    aggs_by_symbol: dict[str, Any] = {}

    calls_config: dict[str, int] = {"tickers": 0, "grouped_aggs": 0}

    MIN_PRICE = 0.0
    MAX_PRICE = 2000.0
    # Use a low pre-filter threshold here; the real filtering happens in
    # builder.py using the configured min_avg_dollar_volume_20d.
    # Note: the "avg" dollar volume computed below is actually single-day
    # (close * volume), not a true 20-day average, so keep this loose.
    MIN_AVG_DOLLAR_VOLUME = 50_000.0

    try:
        # BUG FIX: active=False in Polygon SDK means "only delisted",
        # NOT "include all".  This caused the universe to contain zero
        # active tickers (AAPL, MSFT, etc. were missing) and only ~49
        # symbols survived the grouped-aggs intersection.
        #
        # For live/paper trading we need active stocks.  Omitting the
        # parameter would return both active and delisted, but that
        # doubles the API pagination (~40K vs ~20K tickers).  Since
        # grouped_daily_aggs already filters to symbols that traded
        # recently, active=True is correct and sufficient here.
        tickers_iter = client.list_tickers(market="stocks", active=True, limit=1000)
        for item in tickers_iter:
            symbol = _as_symbol(_get_field(item, "ticker", "symbol"))
            if not symbol:
                continue
            exchange = _as_nonempty_str(_get_field(item, "primary_exchange", "exchange"))
            if not exchange:
                continue
            name = _as_nonempty_str(_get_field(item, "name"))
            market_cap = _as_float(_get_field(item, "market_cap", "marketCap"))
            asset_type = _as_nonempty_str(_get_field(item, "type"))
            tickers_by_symbol[symbol] = {"exchange": exchange, "name": name, "market_cap": market_cap, "asset_type": asset_type}
        calls_config["tickers"] = max(1, (len(tickers_by_symbol) + 999) // 1000) if tickers_by_symbol else 1

        def _normalize_grouped_results(response: Any) -> list[Any]:
            results = _get_field(response, "results")
            if isinstance(results, list):
                return results
            if isinstance(response, list):
                return response
            if isinstance(response, dict) and isinstance(response.get("results"), list):
                return response["results"]
            if isinstance(response, Iterable) and not isinstance(response, (str, bytes, dict)):
                with suppress(Exception):
                    return list(response)
            return []

        grouped_aggs_date: str | None = None
        grouped_results: list[Any] = []

        today = datetime.now().date()
        candidate = today - timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)

        for attempt in range(8):
            date_str = candidate.strftime("%Y-%m-%d")
            grouped_aggs_date = date_str
            grouped_response = client.get_grouped_daily_aggs(date_str)
            calls_config["grouped_aggs"] += 1
            grouped_results = _normalize_grouped_results(grouped_response)
            if grouped_results:
                break
            candidate -= timedelta(days=1)
            while candidate.weekday() >= 5:
                candidate -= timedelta(days=1)
            if attempt == 0:
                logger.info("polygon.universe.grouped_aggs_empty_trying_previous_day", date=date_str)

        if not grouped_results:
            logger.warning("polygon.universe.no_grouped_aggs", date=grouped_aggs_date)
            return Result.failed(
                RuntimeError(f"No grouped daily aggregates returned for {grouped_aggs_date}."),
                "NO_DATA",
            )

        for agg in grouped_results:
            symbol = _as_symbol(_get_field(agg, "ticker", "symbol"))
            if not symbol or symbol not in tickers_by_symbol:
                continue
            aggs_by_symbol[symbol] = agg

    except Exception as exc:  # noqa: BLE001
        exc_name = type(exc).__name__
        logger.warning("polygon.universe.sdk_error", error_type=exc_name, error=str(exc)[:200])
        if exc_name == "AuthError":
            return Result.failed(exc, "MISSING_API_KEY")
        if exc_name == "BadResponse":
            return Result.failed(exc, "HTTP_ERROR")
        return Result.failed(exc, "HTTP_ERROR")

    if not tickers_by_symbol:
        logger.warning("polygon.universe.no_data")
        return Result.failed(RuntimeError("No tickers returned from Polygon."), "NO_DATA")

    preliminary_equities: list[EquityInfo] = []
    filtered_counts = {"price": 0, "otc": 0, "adv": 0}

    for symbol, agg in aggs_by_symbol.items():
        base = tickers_by_symbol.get(symbol)
        if not base:
            continue
        exchange = base["exchange"]
        name = base.get("name")
        market_cap = _as_float(base.get("market_cap"))

        close_price = _as_float(_get_field(agg, "close"))
        volume = _as_float(_get_field(agg, "volume"))
        is_otc = _as_bool(_get_field(agg, "otc")) or False

        if close_price is None or not (MIN_PRICE < close_price <= MAX_PRICE):
            filtered_counts["price"] += 1
            continue
        if is_otc:
            filtered_counts["otc"] += 1
            continue

        avg_dollar_volume = close_price * volume if volume is not None else None
        if avg_dollar_volume is None or avg_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
            filtered_counts["adv"] += 1
            continue

        preliminary_equities.append(
            EquityInfo(
                symbol=symbol,
                name=name,
                exchange=exchange,
                price=close_price,
                market_cap=market_cap,
                avg_dollar_volume_20d=avg_dollar_volume,
                is_otc=is_otc,
                is_halted=False,
                sector=None,
                asset_type=base.get("asset_type"),
            )
        )

    if not preliminary_equities:
        logger.warning(
            "polygon.universe.preliminary_filter_empty",
            total_tickers=len(tickers_by_symbol),
            includes_delisted=True,
            grouped_symbols=len(aggs_by_symbol),
            filtered_counts=filtered_counts,
        )
        return Result.failed(RuntimeError("No tickers remained after preliminary Polygon filtering."), "NO_DATA")

    logger.info(
        "polygon.universe.preliminary_filtered",
        total_tickers=len(tickers_by_symbol),
        includes_delisted=True,
        grouped_symbols=len(aggs_by_symbol),
        preliminary_symbols=len(preliminary_equities),
        grouped_aggs_date=grouped_aggs_date,
        filtered_counts=filtered_counts,
        api_calls_estimated=calls_config,
    )

    equities = preliminary_equities
    market_cap_covered = sum(1 for equity in equities if equity.market_cap is not None)
    logger.info(
        "polygon.universe.sdk_fetched",
        total_tickers=len(tickers_by_symbol),
        includes_delisted=True,
        grouped_symbols=len(aggs_by_symbol),
        total_equities=len(equities),
        grouped_aggs_date=grouped_aggs_date,
        preliminary_filtered=len(preliminary_equities),
        api_calls_estimated=calls_config,
        market_cap_source="list_tickers_only",
        details_enrich_enabled=False,
        coverage={"market_cap": market_cap_covered / len(equities)},
    )

    return Result.success(equities, reason_code="OK")


def _fetch_polygon_universe_fallback() -> Result[list[EquityInfo]]:
    logger = structlog.get_logger(__name__).bind(source="polygon", impl="fallback", endpoint="/v3/reference/tickers")
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        logger.warning("polygon.universe.missing_api_key")
        return Result.failed(RuntimeError("POLYGON_API_KEY is not set."), "MISSING_API_KEY")

    decoder = msgspec.json.Decoder(type=dict[str, Any])

    equities: list[EquityInfo] = []
    url: str | None = _POLYGON_TICKERS_URL
    params: dict[str, Any] | None = {
        "active": "true",
        "market": "stocks",
        "limit": "1000",
        "apiKey": api_key,
    }

    page = 0
    seen_urls: set[str] = set()

    while url:
        page += 1
        if url in seen_urls:
            logger.warning("polygon.universe.pagination_loop", page=page, url=url)
            return Result.failed(RuntimeError("Polygon pagination loop detected."), "DECODE_ERROR")
        seen_urls.add(url)

        request_started_ns = time.time_ns()
        try:
            response = requests.get(url, params=params, timeout=_POLYGON_REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            logger.warning("polygon.universe.request_failed", page=page, url=url, error=str(exc))
            return Result.failed(exc, "HTTP_ERROR")

        latency_ms = (time.time_ns() - request_started_ns) / 1_000_000
        if response.status_code != 200:
            logger.warning(
                "polygon.universe.http_error",
                page=page,
                url=url,
                status_code=response.status_code,
                latency_ms=latency_ms,
                body=response.text[:200],
            )
            return Result.failed(RuntimeError(f"Polygon HTTP {response.status_code}"), "HTTP_ERROR")

        try:
            payload = decoder.decode(response.content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("polygon.universe.decode_error", page=page, error=repr(exc))
            return Result.failed(exc, "DECODE_ERROR")

        results = payload.get("results") or []
        if not isinstance(results, list):
            logger.warning("polygon.universe.invalid_results_type", page=page, results_type=type(results).__name__)
            return Result.failed(RuntimeError("Polygon response 'results' is not a list."), "DECODE_ERROR")

        extracted = 0
        skipped = 0
        for item in results:
            if not isinstance(item, dict):
                skipped += 1
                continue
            ticker = item.get("ticker")
            if not isinstance(ticker, str) or not ticker.strip():
                skipped += 1
                continue

            name = item.get("name")
            name_str = name.strip() if isinstance(name, str) and name.strip() else None

            primary_exchange = item.get("primary_exchange")
            if not isinstance(primary_exchange, str) or not primary_exchange.strip():
                skipped += 1
                continue

            asset_type_raw = item.get("type")
            asset_type_str = asset_type_raw.strip() if isinstance(asset_type_raw, str) and asset_type_raw.strip() else None

            equities.append(
                EquityInfo(
                    symbol=ticker.strip().upper(),
                    name=name_str,
                    exchange=primary_exchange.strip(),
                    price=None,
                    market_cap=None,
                    avg_dollar_volume_20d=None,
                    is_otc=False,
                    is_halted=False,
                    sector=None,
                    asset_type=asset_type_str,
                )
            )
            extracted += 1

        next_url = payload.get("next_url")
        if isinstance(next_url, str) and next_url.strip():
            url = _ensure_api_key(next_url.strip(), api_key=api_key)
            params = None
        else:
            url = None
            params = None

        logger.info(
            "polygon.universe.page_fetched",
            page=page,
            extracted=extracted,
            skipped=skipped,
            total=len(equities),
            has_next=bool(url),
            latency_ms=latency_ms,
        )

    if not equities:
        logger.warning("polygon.universe.no_data")
        return Result.failed(RuntimeError("No tickers returned from Polygon."), "NO_DATA")

    return Result.success(equities, reason_code="OK")


def _get_field(obj: Any, *names: str) -> Any:
    for name in names:
        if obj is None:
            continue
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        with suppress(Exception):
            return getattr(obj, name)
    return None


def _as_nonempty_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _as_symbol(value: Any) -> str | None:
    symbol = _as_nonempty_str(value)
    return symbol.upper() if symbol else None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        with suppress(Exception):
            return float(stripped)
    return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return None
        if stripped in {"true", "1", "yes", "y", "t"}:
            return True
        if stripped in {"false", "0", "no", "n", "f"}:
            return False
    return None


def fetch_fmp_universe() -> Result[list[EquityInfo]]:
    """Fetch the US equity universe reference data from FMP (stub)."""

    # Return failed to trigger fallback to next source (yfinance/cache)
    return Result.failed(RuntimeError("FMP universe source not yet implemented."), "NOT_IMPLEMENTED")


def _ensure_api_key(url: str, *, api_key: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query.get("apiKey"):
        return url
    query["apiKey"] = api_key
    rebuilt = parsed._replace(query=urlencode(query))
    return urlunparse(rebuilt)


def fetch_cached_universe() -> Result[list[EquityInfo]]:
    """Fetch a cached universe snapshot, refreshing via ``yfinance`` when stale.

    Behaviour:
    - If a fresh cache exists (<= 24h), returns ``Result.success`` with cached data.
    - If cache is missing/expired, attempts a live refresh via ``yfinance``.
    - Network failure:
        - with stale cache -> ``Result.degraded`` using stale data
        - without cache -> ``Result.failed``
    - Partial symbol failures -> ``Result.degraded`` with the partial universe.
    """

    cache_read = _read_cache(_CACHE_PATH, ttl_seconds=_CACHE_TTL_SECONDS)
    if cache_read.reason_code == "CACHE_HIT" and cache_read.equities is not None:
        return Result.success(cache_read.equities, reason_code="CACHE_HIT")

    stale_equities = cache_read.equities

    live = _fetch_yfinance_universe()
    if live.status == "success":
        _write_cache_best_effort(_CACHE_PATH, live.equities)
        return Result.success(live.equities, reason_code="OK")

    if live.status == "degraded":
        _write_cache_best_effort(_CACHE_PATH, live.equities)
        return Result.degraded(live.equities, live.error or RuntimeError("Unknown yfinance degraded error."), "QUALITY_DEGRADED")

    # live.status == "failed"
    if stale_equities is not None:
        return Result.degraded(stale_equities, live.error or RuntimeError("Unknown yfinance failure."), "NETWORK_ERROR_USING_STALE_CACHE")
    return Result.failed(live.error or RuntimeError("Unknown yfinance failure."), "NETWORK_ERROR")


class _FetchOutcome(msgspec.Struct, frozen=True, kw_only=True):
    status: str  # "success" | "degraded" | "failed"
    equities: list[EquityInfo]
    error: BaseException | None = None


def _fetch_yfinance_universe() -> _FetchOutcome:
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        return _FetchOutcome(status="failed", equities=[], error=exc)

    tickers = _seed_universe_symbols()
    if not tickers:
        return _FetchOutcome(status="failed", equities=[], error=RuntimeError("No seed tickers available."))

    BATCH_SIZE = 100
    ticker_list = sorted(tickers)
    all_bars: dict[tuple[str, ...], object] = {}
    batch_errors: list[BaseException] = []

    for i in range(0, len(ticker_list), BATCH_SIZE):
        batch = ticker_list[i : i + BATCH_SIZE]
        try:
            batch_bars = yf.download(
                tickers=" ".join(batch),
                period="1mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
            if batch_bars is None:
                batch_errors.append(RuntimeError(f"Batch download returned None ({batch[0]}..{batch[-1]})."))
                continue
            all_bars[tuple(batch)] = batch_bars
        except Exception as exc:  # noqa: BLE001
            batch_errors.append(exc)
            continue

    if not all_bars:
        return _FetchOutcome(status="failed", equities=[], error=RuntimeError("All batches failed."))

    candidates: list[_Candidate] = []
    per_symbol_errors: list[BaseException] = []
    for batch_tickers, bars in all_bars.items():
        for symbol in batch_tickers:
            try:
                avg_dv = _avg_dollar_volume_20d_from_download(bars, symbol)
                last_close = _last_close_from_download(bars, symbol)
            except Exception as exc:  # noqa: BLE001
                per_symbol_errors.append(exc)
                continue

            if last_close is None or not (5.0 <= last_close <= 500.0):
                continue

            exchange: str | None = None
            market_cap: float | None = None
            sector: str | None = None
            price: float = last_close
            try:
                ticker = yf.Ticker(symbol)
                fast = getattr(ticker, "fast_info", None)
                if isinstance(fast, dict):
                    price = _coerce_float(fast.get("last_price")) or price
                    market_cap = _coerce_float(fast.get("market_cap"))
                    exchange = _normalize_exchange(fast.get("exchange"), fast.get("quote_type"))

                if market_cap is None or exchange is None or sector is None:
                    info = getattr(ticker, "info", None)
                    if isinstance(info, dict):
                        market_cap = market_cap or _coerce_float(info.get("marketCap"))
                        exchange = exchange or _normalize_exchange(info.get("exchange"), info.get("fullExchangeName"))
                        sector = sector or _coerce_str(info.get("sector"))
                        price = _coerce_float(info.get("regularMarketPrice")) or price
                        if avg_dv is None:
                            avg_vol = _coerce_float(info.get("averageVolume")) or _coerce_float(
                                info.get("averageDailyVolume10Day")
                            )
                            if avg_vol is not None:
                                avg_dv = float(avg_vol) * float(price)
            except Exception as exc:  # noqa: BLE001
                per_symbol_errors.append(exc)
                continue

            if market_cap is None or market_cap <= 1_000_000_000.0:
                continue
            if exchange is None or exchange == "OTC":
                continue

            candidates.append(
                _Candidate(
                    symbol=symbol,
                    exchange=exchange,
                    price=price,
                    market_cap=market_cap,
                    avg_dollar_volume_20d=avg_dv,
                    sector=sector,
                )
            )

    equities = _select_and_normalize_universe(candidates)
    if not equities:
        return _FetchOutcome(status="failed", equities=[], error=RuntimeError("No equities matched the filters."))

    if batch_errors or per_symbol_errors or len(equities) < 50:
        return _FetchOutcome(
            status="degraded",
            equities=equities,
            error=RuntimeError("Universe fetch partially degraded."),
        )

    return _FetchOutcome(status="success", equities=equities)


def _fetch_nasdaq_api_seed() -> set[str]:
    """Fetch all US stock symbols from the NASDAQ official API (~7000+ tickers).

    API: https://api.nasdaq.com/api/screener/stocks
    Params: tableonly=true&limit=10000&download=true

    Returns:
        Set of stock symbols; empty set on failure.
    """
    try:
        import requests
    except Exception:  # noqa: BLE001
        return set()

    try:
        url = "https://api.nasdaq.com/api/screener/stocks"
        params = {
            "tableonly": "true",
            "limit": 10000,
            "offset": 0,
            "download": "true",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        response = requests.get(url, params=params, headers=headers, timeout=15)

        if response.status_code != 200:
            return set()

        data = response.json()
        if "data" not in data or "rows" not in data["data"]:
            return set()

        rows = data["data"]["rows"]
        symbols: set[str] = set()

        for row in rows:
            symbol = row.get("symbol")
            if isinstance(symbol, str) and symbol.strip():
                symbols.add(symbol.strip().upper())

        return symbols

    except Exception:  # noqa: BLE001
        return set()


def _seed_universe_symbols() -> set[str]:
    """Get seed stock symbols, preferring dynamic fetch with static fallback.

    Returns:
        ~7000+ symbols (dynamic) or ~150 (static fallback).
    """
    dynamic_seed = _fetch_nasdaq_api_seed()
    if dynamic_seed and len(dynamic_seed) >= 100:
        return dynamic_seed

    symbols: set[str] = set()
    symbols.update(_builtin_nasdaq_100_seed())
    symbols.update(_builtin_large_cap_seed())
    return symbols


def _builtin_tech_seed() -> set[str]:
    return {
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "GOOG",
        "META",
        "AVGO",
        "COST",
        "NFLX",
        "AMD",
        "INTC",
        "ADBE",
        "CRM",
        "ORCL",
        "CSCO",
        "QCOM",
        "TSLA",
        "ASML",
        "AMAT",
        "INTU",
        "TXN",
        "BKNG",
        "NOW",
        "ISRG",
        "MU",
        "PANW",
        "CRWD",
        "SNPS",
        "CDNS",
        "KLAC",
        "LRCX",
        "PYPL",
        "SQ",
        "UBER",
        "ABNB",
        "SHOP",
        "ARM",
        "SMCI",
        "MRVL",
    }


def _builtin_nasdaq_100_seed() -> set[str]:
    # Static seed to avoid relying on scraping/web access at runtime.
    # This does not need to be perfectly up-to-date; it is a starting set used
    # to query yfinance, and the final output is filtered by liquidity/cap.
    return {
        *_builtin_tech_seed(),
        "ADP",
        "ANSS",
        "AZN",
        "BIIB",
        "BKR",
        "CCEP",
        "CHTR",
        "CMCSA",
        "COP",
        "CSX",
        "CTAS",
        "DASH",
        "DDOG",
        "DXCM",
        "EA",
        "EXC",
        "FANG",
        "FTNT",
        "GEHC",
        "GILD",
        "HON",
        "IDXX",
        "ILMN",
        "JD",
        "KDP",
        "KHC",
        "LULU",
        "MAR",
        "MCHP",
        "MDLZ",
        "MELI",
        "MNST",
        "MRNA",
        "NXPI",
        "ODFL",
        "ON",
        "PDD",
        "PEP",
        "REGN",
        "ROP",
        "ROST",
        "SBUX",
        "TEAM",
        "TMUS",
        "VRTX",
        "WBD",
        "WDAY",
        "XEL",
        "ZS",
    }


def _builtin_large_cap_seed() -> set[str]:
    # NYSE/other large liquid names to help hit the 50-100 target if the NASDAQ
    # set experiences quote outages or data gaps.
    return {
        "BRK-B",
        "JPM",
        "V",
        "MA",
        "LLY",
        "UNH",
        "PG",
        "XOM",
        "CVX",
        "HD",
        "ABBV",
        "KO",
        "MRK",
        "PEP",
        "C",
        "BAC",
        "WMT",
        "COST",
    }


def _normalize_symbol(value: str) -> str | None:
    sym = value.strip().upper()
    if not sym:
        return None
    sym = sym.replace(".", "-")
    sym = sym.split(" ")[0]
    sym = sym.split("\n")[0]
    return sym if sym.isascii() else None


def _avg_dollar_volume_20d_from_download(download_df: object, symbol: str) -> float | None:
    import pandas as pd

    df = download_df
    if df is None or not hasattr(df, "columns"):
        return None

    if isinstance(df.columns, pd.MultiIndex):
        if (symbol, "Close") not in df.columns or (symbol, "Volume") not in df.columns:
            return None
        close = df[(symbol, "Close")].dropna()
        vol = df[(symbol, "Volume")].dropna()
    else:
        if "Close" not in df.columns or "Volume" not in df.columns:
            return None
        close = df["Close"].dropna()
        vol = df["Volume"].dropna()

    if close.empty or vol.empty:
        return None

    joined = pd.concat([close, vol], axis=1).dropna()
    joined.columns = ["close", "volume"]
    if len(joined) == 0:
        return None
    joined = joined.tail(20)
    dv = (joined["close"].astype(float) * joined["volume"].astype(float)).mean()
    return float(dv) if dv == dv else None


def _last_close_from_download(download_df: object, symbol: str) -> float | None:
    import pandas as pd

    df = download_df
    if df is None or not hasattr(df, "columns"):
        return None

    if isinstance(df.columns, pd.MultiIndex):
        if (symbol, "Close") not in df.columns:
            return None
        close = df[(symbol, "Close")].dropna()
    else:
        if "Close" not in df.columns:
            return None
        close = df["Close"].dropna()

    if close.empty:
        return None
    return float(close.iloc[-1])


def _normalize_exchange(exchange: object, fallback: object | None = None) -> str | None:
    raw = _coerce_str(exchange) or _coerce_str(fallback)
    if not raw:
        return None
    upper = raw.strip().upper()
    if upper in {"NMS", "NGM", "NCM", "NASDAQ"}:
        return "NASDAQ"
    if upper in {"NYQ", "NYSE"}:
        return "NYSE"
    if upper in {"ASE", "AMEX"}:
        return "AMEX"
    if "NASDAQ" in upper:
        return "NASDAQ"
    if "NEW YORK STOCK EXCHANGE" in upper or upper == "NY STOCK EXCHANGE":
        return "NYSE"
    if upper in {"PNK", "OTC", "OTCMKTS"} or "OTC" in upper:
        return "OTC"
    return None


def _coerce_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _select_and_normalize_universe(candidates: Iterable[_Candidate]) -> list[EquityInfo]:
    cleaned: list[_Candidate] = []
    for candidate in candidates:
        if not candidate.symbol or candidate.price is None or candidate.market_cap is None or candidate.exchange is None:
            continue
        if not (5.0 <= candidate.price <= 500.0):
            continue
        if candidate.market_cap <= 1_000_000_000.0:
            continue
        if candidate.exchange == "OTC":
            continue
        cleaned.append(candidate)

    def rank_key(candidate: _Candidate) -> tuple[int, int, float, float]:
        nasdaq_rank = 0 if candidate.exchange == "NASDAQ" else 1
        tech_rank = 0 if (candidate.sector or "").strip().lower() in {"technology", "tech"} else 1
        market_cap = float(candidate.market_cap or 0.0)
        dv = float(candidate.avg_dollar_volume_20d or 0.0)
        return (nasdaq_rank, tech_rank, -market_cap, -dv)

    cleaned.sort(key=rank_key)
    selected = cleaned[:250]

    return [
        EquityInfo(
            symbol=candidate.symbol,
            exchange=candidate.exchange,
            price=float(candidate.price),
            avg_dollar_volume_20d=float(candidate.avg_dollar_volume_20d or 0.0),
            market_cap=float(candidate.market_cap),
            is_otc=False,
            is_halted=False,
            sector=candidate.sector,
        )
        for candidate in selected
    ]


@dataclass(frozen=True, slots=True)
class _CacheReadResult:
    equities: list[EquityInfo] | None
    reason_code: str


def _read_cache(path: Path, *, ttl_seconds: int) -> _CacheReadResult:
    if not path.exists():
        return _CacheReadResult(equities=None, reason_code="CACHE_MISS")

    decoder = msgspec.json.Decoder(type=_UniverseCacheEntry)
    try:
        entry = decoder.decode(path.read_bytes())
    except Exception:  # noqa: BLE001
        return _CacheReadResult(equities=None, reason_code="CACHE_READ_FAILED")

    if ttl_seconds > 0:
        age_seconds = (time.time_ns() - entry.saved_at_ns) / 1_000_000_000
        if age_seconds > ttl_seconds:
            return _CacheReadResult(equities=entry.equities, reason_code="CACHE_EXPIRED")

    return _CacheReadResult(equities=entry.equities, reason_code="CACHE_HIT")


def _write_cache_best_effort(path: Path, equities: list[EquityInfo]) -> None:
    with suppress(Exception):
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = _UniverseCacheEntry(saved_at_ns=time.time_ns(), equities=equities)
        payload = msgspec.json.Encoder().encode(entry)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
