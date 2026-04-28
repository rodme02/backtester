"""Regime tagging for per-regime performance breakdowns.

Two simple, transparent regime taxonomies — both computable from the
benchmark's price series alone, no labels from the future:

- ``trend_regimes``  — bull when price > 200d SMA, bear otherwise.
- ``vol_regimes``    — high/low based on the trailing realised-vol
  quantile of the benchmark's daily returns.

``per_regime_metrics`` slices a strategy-return series by regime tag
and computes a metric on each slice.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def trend_regimes(prices: pd.Series, *, window: int = 200) -> pd.Series:
    """Bull/bear tag from a moving-average crossover on the benchmark.

    NaN until the SMA window is filled.
    """
    sma = prices.rolling(window=window, min_periods=window).mean()
    out = pd.Series(np.where(prices > sma, "bull", "bear"), index=prices.index)
    out[sma.isna()] = pd.NA
    return out


def vol_regimes(
    returns: pd.Series, *, window: int = 60, quantile: float = 0.5
) -> pd.Series:
    """High/low realised-vol tag from a rolling stdev quantile.

    Uses an *expanding* quantile so the regime label at time ``t``
    depends only on information up to ``t``.
    """
    rolling_vol = returns.rolling(window=window, min_periods=window).std(ddof=1)
    expanding_q = rolling_vol.expanding(min_periods=window).quantile(quantile)
    out = pd.Series(
        np.where(rolling_vol > expanding_q, "high_vol", "low_vol"),
        index=returns.index,
    )
    out[rolling_vol.isna() | expanding_q.isna()] = pd.NA
    return out


def per_regime_metrics(
    returns: pd.Series,
    regimes: pd.Series,
    metric: Callable[[np.ndarray], float],
) -> dict[str, float]:
    """Apply ``metric`` to slices of ``returns`` grouped by regime tag."""
    aligned = pd.concat([returns.rename("r"), regimes.rename("regime")], axis=1).dropna()
    out: dict[str, float] = {}
    for tag, group in aligned.groupby("regime"):
        if group.empty:
            continue
        out[str(tag)] = float(metric(group["r"].to_numpy()))
    return out
