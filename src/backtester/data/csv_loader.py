"""CSV loaders for OHLCV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = REPO_ROOT / "samples" / "ohlcv"


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV with `datetime` as index.

    Expected columns: datetime, open, high, low, close, volume.
    """
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df.sort_index(inplace=True)
    expected = {"open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing columns: {sorted(missing)}")
    return df


def load_samples(symbol: str) -> pd.DataFrame:
    """Load a bundled sample dataset by symbol (e.g. 'AAPL')."""
    path = SAMPLES_DIR / f"{symbol.upper()}.csv"
    if not path.exists():
        available = sorted(p.stem for p in SAMPLES_DIR.glob("*.csv"))
        raise FileNotFoundError(f"No sample for {symbol!r}. Available: {available}")
    return load_csv(path)
