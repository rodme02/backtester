"""Binance public-API loaders with on-disk cache.

No API key required for the endpoints used here (klines, funding-rate
history, open-interest history). Endpoints:

- Spot klines:           ``GET /api/v3/klines``
- USDT-M futures klines: ``GET /fapi/v1/klines``
- Funding-rate history:  ``GET /fapi/v1/fundingRate``
- Open-interest history: ``GET /futures/data/openInterestHist``

All loaders return DataFrames indexed by ``datetime`` (UTC).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from ._fixture import fixture_mode_active, load_fixture

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data_cache" / "binance"

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"


def _cache_path(name: str, today: date) -> Path:
    return CACHE_DIR / f"{name}_{today.isoformat()}.csv"


def _read_cache(name: str) -> pd.DataFrame | None:
    path = _cache_path(name, date.today())
    if path.exists():
        return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    return None


def _write_cache(name: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(name, date.today()))


def _to_ts(d: str | datetime | None) -> int | None:
    if d is None:
        return None
    if isinstance(d, str):
        d = datetime.fromisoformat(d)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    *,
    interval: str = "1d",
    market: Literal["spot", "futures"] = "spot",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV klines. Returns columns: open, high, low, close, volume."""
    if fixture_mode_active():
        df = load_fixture(f"binance_klines_{market}_{symbol.upper()}_{interval}.csv")
        if df is not None:
            if getattr(df.index, "tz", None) is None:
                df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
            return df
    name = f"klines_{market}_{symbol.upper()}_{interval}_{start}_{end}"
    if cache and (cached := _read_cache(name)) is not None:
        return cached

    base = SPOT_BASE + "/api/v3/klines" if market == "spot" else FUTURES_BASE + "/fapi/v1/klines"
    params: dict[str, object] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": 1000,
    }
    start_ts = _to_ts(start)
    end_ts = _to_ts(end)
    if start_ts is not None:
        params["startTime"] = start_ts
    if end_ts is not None:
        params["endTime"] = end_ts

    resp = requests.get(base, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise RuntimeError(f"Binance returned no klines for {symbol} {interval}")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tqav", "_ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]].astype(float)

    if cache:
        _write_cache(name, df)
    return df


def fetch_funding_rate(
    symbol: str,
    *,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch perpetual funding-rate history (8h granularity)."""
    if fixture_mode_active():
        df = load_fixture(f"binance_funding_{symbol.upper()}.csv")
        if df is not None:
            if getattr(df.index, "tz", None) is None:
                df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
            return df
    name = f"funding_{symbol.upper()}_{start}_{end}"
    if cache and (cached := _read_cache(name)) is not None:
        return cached

    params: dict[str, object] = {"symbol": symbol.upper(), "limit": 1000}
    start_ts = _to_ts(start)
    end_ts = _to_ts(end)
    if start_ts is not None:
        params["startTime"] = start_ts
    if end_ts is not None:
        params["endTime"] = end_ts

    resp = requests.get(FUTURES_BASE + "/fapi/v1/fundingRate", params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame(columns=["funding_rate"], index=pd.DatetimeIndex([], name="datetime"))

    df = pd.DataFrame(raw)
    df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df.set_index("datetime")[["fundingRate"]].astype(float)
    df.columns = ["funding_rate"]

    if cache:
        _write_cache(name, df)
    return df


def fetch_premium_index_klines(
    symbol: str,
    *,
    interval: str = "1d",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch perp/spot premium-index klines (basis history).

    The premium index is a smoothed perp/spot price-gap measure;
    ``close`` is the daily basis level.
    """
    if fixture_mode_active():
        df = load_fixture(f"binance_premium_{symbol.upper()}_{interval}.csv")
        if df is not None:
            if getattr(df.index, "tz", None) is None:
                df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
            return df
    name = f"premium_{symbol.upper()}_{interval}_{start}_{end}"
    if cache and (cached := _read_cache(name)) is not None:
        return cached

    params: dict[str, object] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": 1500,
    }
    start_ts = _to_ts(start)
    end_ts = _to_ts(end)
    if start_ts is not None:
        params["startTime"] = start_ts
    if end_ts is not None:
        params["endTime"] = end_ts

    resp = requests.get(
        FUTURES_BASE + "/fapi/v1/premiumIndexKlines", params=params, timeout=15
    )
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame(
            columns=["premium"], index=pd.DatetimeIndex([], name="datetime")
        )

    cols = [
        "open_time", "open", "high", "low", "close", "_v",
        "close_time", "_qav", "_n", "_tb", "_tq", "_ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("datetime")[["close"]].astype(float)
    df.columns = ["premium"]

    if cache:
        _write_cache(name, df)
    return df


def fetch_long_short_ratio(
    symbol: str,
    *,
    period: str = "1d",
    kind: Literal["top_account", "top_position", "global_account"] = "top_account",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch top-trader / global long-short ratio.

    Note Binance's REST history is rolling 30 days only; collect
    prospectively for longer windows.
    """
    if fixture_mode_active():
        df = load_fixture(f"binance_lsr_{kind}_{symbol.upper()}_{period}.csv")
        if df is not None:
            return df
    name = f"lsr_{kind}_{symbol.upper()}_{period}_{start}_{end}"
    if cache and (cached := _read_cache(name)) is not None:
        return cached

    endpoint = {
        "top_account": "/futures/data/topLongShortAccountRatio",
        "top_position": "/futures/data/topLongShortPositionRatio",
        "global_account": "/futures/data/globalLongShortAccountRatio",
    }[kind]

    params: dict[str, object] = {
        "symbol": symbol.upper(),
        "period": period,
        "limit": 500,
    }
    start_ts = _to_ts(start)
    end_ts = _to_ts(end)
    if start_ts is not None:
        params["startTime"] = start_ts
    if end_ts is not None:
        params["endTime"] = end_ts

    resp = requests.get(FUTURES_BASE + endpoint, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame(
            columns=["long_ratio", "short_ratio", "long_short_ratio"],
            index=pd.DatetimeIndex([], name="datetime"),
        )

    df = pd.DataFrame(raw)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")[
        ["longAccount", "shortAccount", "longShortRatio"]
    ].astype(float)
    df.columns = ["long_ratio", "short_ratio", "long_short_ratio"]

    if cache:
        _write_cache(name, df)
    return df
