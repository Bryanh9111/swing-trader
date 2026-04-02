"""Data layer for fetching and validating market price data.

Phase 2.2 introduces a dedicated data layer that can fetch, normalize, cache,
and validate OHLCV time series snapshots from pluggable data sources.
"""

from .cache import FileCache
from .interface import DataSourceAdapter, PriceBar, PriceSeriesSnapshot
from .orchestrator import DataOrchestrator
from .polygon_adapter import PolygonDataAdapter
from .rate_limiter import TokenBucketRateLimiter
from .s3_adapter import S3DataAdapter
from .yahoo_adapter import YahooDataAdapter

__all__ = [
    "DataOrchestrator",
    "DataSourceAdapter",
    "FileCache",
    "PolygonDataAdapter",
    "PriceBar",
    "PriceSeriesSnapshot",
    "S3DataAdapter",
    "TokenBucketRateLimiter",
    "YahooDataAdapter",
]

