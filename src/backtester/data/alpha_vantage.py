"""Alpha Vantage daily OHLCV fetcher with on-disk cache.

API key is read from the ``ALPHA_VANTAGE_API_KEY`` environment variable.
Cached responses live under ``data_cache/alpha_vantage/`` keyed by symbol +
fetch date, so repeated runs don't burn the free-tier quota.

API key is read from ``ALPHA_VANTAGE_API_KEY``.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data_cache" / "alpha_vantage"


def _cache_path(symbol: str, day: date) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_{day.isoformat()}.csv"


def fetch_daily(
    symbol: str,
    *,
    api_key: str | None = None,
    outputsize: str = "full",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV from Alpha Vantage. Returns DataFrame indexed by date.

    Reads ``ALPHA_VANTAGE_API_KEY`` from env when ``api_key`` is None.
    """
    today = date.today()
    if cache:
        path = _cache_path(symbol, today)
        if path.exists():
            return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")

    key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY not set. Export it or pass api_key explicitly."
        )

    from alpha_vantage.timeseries import TimeSeries  # imported lazily

    ts = TimeSeries(key=key, output_format="pandas")
    df, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    )
    df.index.name = "datetime"
    df.sort_index(inplace=True)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_cache_path(symbol, today))
    return df
