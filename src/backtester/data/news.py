"""Per-ticker news headlines from yfinance with on-disk cache.

`yfinance.Ticker(symbol).news` returns the most recent ~10 headlines
(typically last 24h on the public feed). For a daily-rebalance survey
this is a strict data-availability constraint — multi-year sentiment
backtests need a paid news archive. The cache here is keyed by
``(ticker, fetch_date)`` so reruns the same day are offline.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from ._fixture import fixture_mode_active, load_fixture

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data_cache" / "news"


def _cache_path(symbol: str, today: date) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_{today.isoformat()}.json"


def _normalise(item: dict) -> dict:
    """Flatten yfinance's nested news item to {title, published_at, ...}."""
    content = item.get("content", item)
    title = content.get("title", "") or item.get("title", "")
    pub = (
        content.get("pubDate")
        or content.get("providerPublishTime")
        or item.get("providerPublishTime")
    )
    if isinstance(pub, (int, float)):
        published_at = datetime.fromtimestamp(pub, tz=timezone.utc).isoformat()
    elif isinstance(pub, str):
        published_at = pub
    else:
        published_at = None
    summary = content.get("summary") or content.get("description") or ""
    publisher = content.get("provider", {}).get("displayName") if isinstance(
        content.get("provider"), dict
    ) else item.get("publisher", "")
    return {
        "title": title.strip(),
        "summary": summary.strip(),
        "publisher": publisher,
        "published_at": published_at,
    }


def fetch_recent_news(
    symbol: str,
    *,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch the most-recent yfinance news for ``symbol``.

    Returns a DataFrame with columns ``title``, ``summary``, ``publisher``,
    ``published_at``, indexed by ``datetime`` (parsed publication time).
    Empty DataFrame on failure.
    """
    if fixture_mode_active():
        df = load_fixture(f"news_{symbol.upper()}.csv")
        if df is not None:
            return df

    today = date.today()
    if cache:
        path = _cache_path(symbol, today)
        if path.exists():
            items = json.loads(path.read_text())
            return _items_to_df(items)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "fetch_recent_news requires yfinance; install with: pip install yfinance"
        ) from exc

    raw = yf.Ticker(symbol).news or []
    items = [_normalise(it) for it in raw]
    items = [it for it in items if it["title"]]

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(symbol, today).write_text(json.dumps(items, indent=2))
    return _items_to_df(items)


def _items_to_df(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(
            columns=["title", "summary", "publisher", "published_at"],
            index=pd.DatetimeIndex([], name="datetime"),
        )
    df = pd.DataFrame(items)
    df["datetime"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    return df
