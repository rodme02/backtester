"""Point-in-time-aware ticker universe loader.

Survivorship bias is the silent killer of equity backtests: training on
*today's* index members lets you pretend you would have held the
winners. This loader returns only the tickers whose
``first_eligible`` date is ``<=`` the requested date.

The bundled snapshot ``samples/universe_us_liquid.csv`` is a hand-curated
list of large-cap US tickers with conservative first-eligible dates
(IPO/listing dates where known, ``2000-01-01`` for older blue-chips).
It is **not** an exact reproduction of historical S&P 100 membership —
treat it as a "liquid mega-cap" universe. Replace with a paid,
properly point-in-time membership feed for production-quality
research.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIVERSE = REPO_ROOT / "samples" / "universe_us_liquid.csv"


def load_universe(path: str | Path = DEFAULT_UNIVERSE) -> pd.DataFrame:
    """Load a (ticker, first_eligible) snapshot."""
    df = pd.read_csv(path, parse_dates=["first_eligible"])
    if "last_eligible" in df.columns:
        df["last_eligible"] = pd.to_datetime(df["last_eligible"])
    return df


def eligible_tickers(
    universe: pd.DataFrame, as_of: pd.Timestamp | str
) -> list[str]:
    """Return tickers eligible at ``as_of`` (inclusive)."""
    as_of_ts = pd.Timestamp(as_of)
    mask = universe["first_eligible"] <= as_of_ts
    if "last_eligible" in universe.columns:
        not_dropped = universe["last_eligible"].isna() | (
            universe["last_eligible"] >= as_of_ts
        )
        mask &= not_dropped
    return universe.loc[mask, "ticker"].tolist()
