"""FRED macroeconomic series with on-disk cache.

Requires ``FRED_API_KEY`` in environment (free; sign up at
https://fred.stlouisfed.org/). Useful series for the equity case study:

- ``VIXCLS``  — VIX, equity-market implied vol.
- ``T10Y2Y``  — 10y minus 2y Treasury yield (term-structure slope).
- ``BAA10Y``  — Moody's BAA corporate yield minus 10y Treasury (credit
  spread).
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from ._cache import cache_path
from ._fixture import fixture_mode_active, load_fixture


def _cache_path(series_id: str, today: date):
    return cache_path("fred", series_id, today=today)


def fetch_series(
    series_id: str,
    *,
    api_key: str | None = None,
    cache: bool = True,
) -> pd.Series:
    """Fetch a FRED series by ID, return as a date-indexed pandas Series."""
    if fixture_mode_active():
        df = load_fixture(f"fred_{series_id}.csv", source="fred")
        if df is not None:
            return df.iloc[:, 0]
    today = date.today()
    if cache:
        path = _cache_path(series_id, today)
        if path.exists():
            df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
            return df[series_id]

    load_dotenv(override=False)  # re-read .env in case it was added after import.
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/ and export it."
        )

    from fredapi import Fred  # imported lazily

    series = Fred(api_key=key).get_series(series_id)
    series.index.name = "datetime"
    series.name = series_id

    if cache:
        path = _cache_path(series_id, today)
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_frame().to_csv(path)
    return series
