"""Sentiment-factor leakage and aggregation tests."""

import numpy as np
import pandas as pd
import pytest

from backtester.features.sentiment import daily_sentiment, lagged_sentiment_factor


def test_daily_sentiment_means_per_day_ticker():
    ts = pd.to_datetime(
        ["2024-01-01 09:00", "2024-01-01 18:00", "2024-01-02 09:00"], utc=True
    )
    daily = daily_sentiment(
        scores=pd.Series([0.4, -0.2, 0.5]),
        timestamps=pd.Series(ts),
        tickers=pd.Series(["AAPL", "AAPL", "AAPL"]),
    )
    # Day 1 mean = 0.1, day 2 mean = 0.5
    assert daily.shape == (2, 1)
    assert daily.iloc[0, 0] == pytest.approx(0.1)
    assert daily.iloc[1, 0] == pytest.approx(0.5)


def test_lagged_factor_shifts_by_lag_days():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    daily = pd.DataFrame({"AAPL": np.arange(10.0)}, index=idx)
    factor = lagged_sentiment_factor(daily, window=1, lag_days=1)
    # First row should be NaN (lagged 1d, nothing before).
    assert pd.isna(factor.iloc[0, 0])
    # Subsequent rows should equal the *previous* day's value.
    np.testing.assert_array_equal(
        factor["AAPL"].iloc[1:].to_numpy(),
        daily["AAPL"].iloc[:-1].to_numpy(),
    )


def test_lagged_factor_window_smooths():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    daily = pd.DataFrame({"AAPL": np.arange(10.0)}, index=idx)
    factor = lagged_sentiment_factor(daily, window=3, lag_days=1)
    # At t=4 (idx[4]), the lagged-window-3 mean uses days idx[1], idx[2], idx[3] = 1,2,3 -> 2.0
    assert factor.iloc[4, 0] == pytest.approx(2.0)


def test_lagged_factor_no_future_leakage():
    """Property: factor[t] depends only on scores at indices < t."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    raw = pd.DataFrame(
        {"AAPL": rng.uniform(-1, 1, size=30), "MSFT": rng.uniform(-1, 1, size=30)},
        index=idx,
    )
    full = lagged_sentiment_factor(raw, window=5, lag_days=1)

    # Truncate raw at t=15; recompute factor; values at indices [0..14] must match.
    truncated = raw.iloc[:16]
    partial = lagged_sentiment_factor(truncated, window=5, lag_days=1)
    # Compare overlap.
    overlap = partial.index.intersection(full.index)
    pd.testing.assert_frame_equal(full.loc[overlap], partial.loc[overlap])


def test_empty_input_returns_empty():
    out = daily_sentiment(
        scores=pd.Series([], dtype=float),
        timestamps=pd.Series([], dtype="datetime64[ns, UTC]"),
        tickers=pd.Series([], dtype=str),
    )
    assert out.empty
