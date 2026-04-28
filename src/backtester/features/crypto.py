"""Crypto-specific features built from Binance market data.

Funding rate, basis (perp - spot), and open-interest delta carry signal
that doesn't exist in equities. Funding-rate prints arrive every 8h on
Binance; ``align_funding`` resamples to a daily calendar with one bar
of lag (so the feature at date ``t`` uses funding paid at the most
recent close <= t-1).

Causality invariant: feature at ``t`` depends only on inputs <= ``t``.
Tested in ``tests/test_features_leakage.py``.
"""

from __future__ import annotations

import pandas as pd


def align_funding(funding: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Resample 8h funding to a daily calendar with a one-day lag.

    ``funding`` is a Series indexed by funding-print timestamps. We
    sum within each calendar day (each day has 3 prints), shift by 1
    day, then reindex to ``calendar`` with forward-fill.
    """
    if funding.empty:
        return pd.Series(index=calendar, dtype=float)
    # Coerce to DatetimeIndex (CSV round-trip can leave a plain Index).
    idx = pd.DatetimeIndex(pd.to_datetime(funding.index, utc=True, errors="coerce"))
    fr = pd.Series(funding.values, index=idx).dropna()
    # Aggregate to daily: total funding paid per day.
    daily = fr.groupby(fr.index.tz_convert(None).normalize()).sum()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily.shift(1).reindex(calendar, method="ffill")


def funding_features(
    funding: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    short_change: int = 1,
    long_change: int = 7,
) -> pd.DataFrame:
    """Funding-rate level + short/long-window changes."""
    aligned = align_funding(funding, calendar)
    return pd.DataFrame(
        {
            "funding_level": aligned,
            f"funding_chg_{short_change}d": aligned - aligned.shift(short_change),
            f"funding_chg_{long_change}d": aligned - aligned.shift(long_change),
        },
        index=calendar,
    )


def basis(spot_close: pd.Series, perp_close: pd.Series) -> pd.Series:
    """Perp-vs-spot basis as a fraction. NaN where either side is missing."""
    aligned = pd.concat([spot_close.rename("spot"), perp_close.rename("perp")], axis=1)
    return (aligned["perp"] / aligned["spot"]) - 1.0


def open_interest_delta(oi: pd.Series, *, window: int = 1) -> pd.Series:
    """Log-change in open interest over ``window`` periods."""
    import numpy as np

    return np.log(oi).diff(window)
