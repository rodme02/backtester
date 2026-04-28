"""Yahoo Finance daily OHLCV with on-disk cache.

Free, no API key. Cache is keyed by ``symbol + (start, end) + fetch_date``
so repeated calls within a day re-use the file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data_cache" / "yfinance"


def _cache_path(symbol: str, start: str, end: str, today: date) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_{start}_{end}_{today.isoformat()}.csv"


def fetch_daily(
    symbol: str,
    *,
    start: str = "2010-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance.

    Returns a DataFrame indexed by ``datetime`` with lower-cased
    ``open/high/low/close/volume`` columns to match the rest of the
    framework.
    """
    end = end or date.today().isoformat()
    today = date.today()
    if cache:
        path = _cache_path(symbol, start, end, today)
        if path.exists():
            return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")

    import yfinance as yf  # imported lazily so the rest of the lib stays light

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol} {start}..{end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index.name = "datetime"
    keep = [c for c in ("open", "high", "low", "close", "adj_close", "volume") if c in df.columns]
    df = df[keep].sort_index()

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_cache_path(symbol, start, end, today))
    return df
