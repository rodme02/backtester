"""Per-(date, ticker) sentiment factor from per-headline scores.

Pipeline:

1. ``score_headline`` (in ``data/llm.py``) gives one score per
   (ticker, headline).
2. ``daily_sentiment`` aggregates to a per-day-per-ticker number
   (mean of scores published that day; falls back to the prior day's
   score if no news arrived).
3. ``lagged_sentiment_factor`` takes a trailing window mean and
   shifts by one trading day to prevent same-day leakage.

Causality invariant: factor at ``(t, ticker)`` uses only scores from
headlines published strictly *before* ``t``.
"""

from __future__ import annotations

import pandas as pd


def daily_sentiment(
    scores: pd.Series,
    *,
    timestamps: pd.Series,
    tickers: pd.Series,
) -> pd.DataFrame:
    """Aggregate per-headline scores to per-day-per-ticker mean.

    Parameters
    ----------
    scores
        Float scores in [-1, 1], one per headline.
    timestamps
        UTC publication timestamps, one per headline.
    tickers
        Ticker symbol, one per headline.

    Returns
    -------
    DataFrame indexed by ``datetime`` (UTC midnight, i.e. publication
    date), columns are tickers, values are the mean of the day's
    headline scores.
    """
    df = pd.DataFrame(
        {"score": scores.values, "ticker": tickers.values, "datetime": timestamps.values}
    )
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return pd.DataFrame()
    df["date"] = df["datetime"].dt.tz_convert(None).dt.normalize()
    daily = (
        df.groupby(["date", "ticker"])["score"].mean()
        .unstack("ticker")
        .sort_index()
    )
    daily.index.name = "datetime"
    return daily


def lagged_sentiment_factor(
    daily: pd.DataFrame,
    *,
    window: int = 7,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Trailing-window mean of daily sentiment, lagged.

    The lag prevents same-day leakage: the factor at date ``t`` uses
    only scores observed at ``t - lag_days`` and earlier.
    """
    if daily.empty:
        return daily
    rolled = daily.rolling(window=window, min_periods=1).mean()
    return rolled.shift(lag_days)
