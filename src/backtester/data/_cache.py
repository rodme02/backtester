"""Shared on-disk cache layout for live data loaders.

Each loader (yfinance, fred, binance, news) keys its CSV / JSON cache
files by ``(source, name, fetch_date)`` under ``data_cache/``. This
module owns that layout so the four loaders don't each re-derive
``REPO_ROOT`` and rebuild path strings.

LLM sentiment uses a different key (per-headline hash, not date) and
keeps its own helper.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = REPO_ROOT / "data_cache"


def cache_path(
    source: str,
    name: str,
    *,
    today: date | None = None,
    suffix: str = ".csv",
) -> Path:
    """Return ``data_cache/<source>/<name>_<YYYY-MM-DD><suffix>``.

    ``today`` defaults to ``date.today()``. Caller is responsible for
    ``parent.mkdir(parents=True, exist_ok=True)`` before writing.
    """
    today = today or date.today()
    return CACHE_ROOT / source / f"{name}_{today.isoformat()}{suffix}"
