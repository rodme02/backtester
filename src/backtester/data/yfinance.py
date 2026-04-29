"""Yahoo Finance daily OHLCV with on-disk cache + transient-error retry.

Free, no API key. Cache is keyed by ``symbol + (start, end) + fetch_date``
so repeated calls within a day re-use the file. yfinance's HTTP layer
times out frequently; we retry with exponential backoff before giving
up.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd

from ._fixture import fixture_mode_active, load_fixture

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
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance.

    Returns a DataFrame indexed by ``datetime`` with lower-cased
    ``open/high/low/close/volume`` columns to match the rest of the
    framework.
    """
    if fixture_mode_active():
        df = load_fixture(f"yfinance_{symbol.upper()}.csv")
        if df is not None:
            return df
    end = end or date.today().isoformat()
    today = date.today()
    if cache:
        path = _cache_path(symbol, start, end, today)
        if path.exists():
            return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")

    import yfinance as yf  # imported lazily so the rest of the lib stays light

    last_exc: Exception | None = None
    df: pd.DataFrame | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                symbol, start=start, end=end, progress=False, auto_adjust=False
            )
            if df is not None and not df.empty:
                break
        except Exception as exc:  # noqa: BLE001 — yfinance throws bare Exception
            last_exc = exc
        time.sleep(backoff ** attempt)
    if df is None or df.empty:
        raise RuntimeError(
            f"yfinance returned no rows for {symbol} {start}..{end}"
            + (f" (last error: {last_exc})" if last_exc else "")
        )
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
