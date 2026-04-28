"""Universe-wide cross-sectional features.

Inputs are wide DataFrames indexed by date, columns by ticker. For
each row (date) the function ranks tickers and returns a same-shape
DataFrame of ranks scaled to ``[0, 1]``. The ranking at date ``t``
uses only data observable at ``t``, so the operation is per-row and
inherently leakage-free.
"""

from __future__ import annotations

import pandas as pd


def momentum_rank(
    prices: pd.DataFrame,
    *,
    lookback: int = 126,
    skip: int = 21,
) -> pd.DataFrame:
    """Cross-sectional rank of total return over [t-lookback, t-skip].

    The ``skip`` window (default ~1 month) excludes the most-recent
    period, which captures short-term mean-reversion rather than
    momentum (Jegadeesh & Titman 1993).
    """
    forward = prices.shift(skip)
    past = prices.shift(skip + lookback)
    momentum = forward / past - 1.0
    return momentum.rank(axis=1, pct=True)


def vol_rank(
    returns: pd.DataFrame,
    *,
    window: int = 60,
) -> pd.DataFrame:
    """Cross-sectional rank of trailing realised vol."""
    vol = returns.rolling(window=window, min_periods=window).std(ddof=1)
    return vol.rank(axis=1, pct=True)
