"""Build CI fixtures for every (data-loader, key) combination the notebooks call.

Run this once locally (with internet) to populate ``samples/fixtures/``;
commit the resulting CSVs. The CI ``notebooks`` job sets
``BACKTESTER_FIXTURE_MODE=1`` and reads from those fixtures instead of
hitting live APIs.

Usage::

    python scripts/build_fixtures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Make sure fixture mode is OFF for this run — we want live data.
os.environ.pop("BACKTESTER_FIXTURE_MODE", None)

import pandas as pd  # noqa: E402

from backtester.data.binance import (  # noqa: E402
    fetch_funding_rate,
    fetch_klines,
    fetch_premium_index_klines,
)
from backtester.data.fred import fetch_series as fred_fetch  # noqa: E402
from backtester.data.universe import load_universe  # noqa: E402
from backtester.data.yfinance import fetch_daily as yf_fetch  # noqa: E402

OUT = ROOT / "samples" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def write(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    if df is None or df.empty:
        print(f"  SKIP {name} (empty)")
        return
    if df.index.name is None:
        df.index.name = "datetime"
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.to_csv(path)
    print(f"  wrote {path.name} ({len(df)} rows)")


def fetch_paginated(loader, symbol: str, *, start: str, end: str, **kw) -> pd.DataFrame:
    """Page through a Binance loader to cover [start, end] in 1000-row chunks."""
    chunks = []
    cursor = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    seen_max = pd.Timestamp("1970-01-01", tz="UTC")
    while cursor < end_ts:
        chunk = loader(
            symbol, start=cursor.isoformat(), end=end_ts.isoformat(),
            cache=False, **kw,
        )
        if chunk.empty:
            break
        # Make chunk index tz-aware UTC for comparison.
        if getattr(chunk.index, "tz", None) is None:
            chunk.index = pd.DatetimeIndex(chunk.index).tz_localize("UTC")
        chunk_max = chunk.index.max()
        if chunk_max <= seen_max:
            break
        seen_max = chunk_max
        chunks.append(chunk)
        cursor = chunk_max + pd.Timedelta(seconds=1)
        if chunk.shape[0] < 1000:
            break
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks).sort_index()
    return out[~out.index.duplicated(keep="first")]


# ---------- yfinance equities ----------
universe = load_universe()
tickers = sorted(universe["ticker"].tolist()) + ["SPY"]
print(f"[yfinance] {len(tickers)} tickers")
for t in tickers:
    try:
        df = yf_fetch(t, start="2010-01-01", end="2024-12-31")
        write(df, f"yfinance_{t.upper()}.csv")
    except Exception as exc:
        print(f"  FAIL {t}: {exc!s:80.80}")

# ---------- FRED macro ----------
fred_ids = ["VIXCLS", "T10Y2Y", "BAA10Y"]
print(f"\n[fred] {len(fred_ids)} series")
for sid in fred_ids:
    try:
        s = fred_fetch(sid)
        write(s.to_frame(), f"fred_{sid}.csv")
    except Exception as exc:
        print(f"  FAIL {sid}: {exc!s:80.80}")

# ---------- Binance crypto ----------
crypto_symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT",
]
print(f"\n[binance] {len(crypto_symbols)} symbols")
for sym in crypto_symbols:
    try:
        klines = fetch_paginated(
            lambda s, **kw: fetch_klines(s, interval="1d", market="futures", **kw),
            sym, start="2021-01-01T00:00:00", end="2024-12-31T23:59:59",
        )
        write(klines, f"binance_klines_futures_{sym}_1d.csv")
    except Exception as exc:
        print(f"  klines {sym} FAIL: {exc!s:80.80}")

    try:
        funding = fetch_paginated(
            fetch_funding_rate, sym, start="2021-01-01", end="2024-12-31",
        )
        write(funding, f"binance_funding_{sym}.csv")
    except Exception as exc:
        print(f"  funding {sym} FAIL: {exc!s:80.80}")

    try:
        premium = fetch_paginated(
            lambda s, **kw: fetch_premium_index_klines(s, interval="1d", **kw),
            sym, start="2021-01-01", end="2024-12-31",
        )
        write(premium, f"binance_premium_{sym}_1d.csv")
    except Exception as exc:
        print(f"  premium {sym} FAIL: {exc!s:80.80}")

print(f"\nFixtures written to {OUT}")
