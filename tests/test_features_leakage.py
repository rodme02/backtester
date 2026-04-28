"""Property tests: every feature must be causal.

For any feature ``f`` and prefix length ``t``, ``f(series[:t+k])``
evaluated at index ``t`` must equal ``f(series)`` at index ``t``,
for any extra-data window ``k > 0``. If they differ, the feature is
peeking into the future.

We test on multiple ``t`` and ``k`` to make accidental causality bugs
hard to slip past.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.features import (
    atr,
    basis,
    funding_features,
    log_returns,
    macd,
    macro_features,
    momentum_rank,
    open_interest_delta,
    rolling_volatility,
    rsi,
    vol_rank,
)


@pytest.fixture(scope="module")
def ohlc():
    rng = np.random.default_rng(42)
    n = 600
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.012, size=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx)
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, size=n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.005, size=n)))
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


@pytest.fixture(scope="module")
def panel(ohlc):
    rng = np.random.default_rng(7)
    cols = list("ABCDE")
    idx = ohlc.index
    rets = rng.normal(0.0003, 0.01, size=(len(idx), len(cols)))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols
    )
    returns = prices.pct_change()
    return prices, returns


def _assert_causal(feature_full: pd.Series, builder, *args, **kwargs):
    """For random t and k, the prefix-evaluated feature must equal the full one at t."""
    n = len(feature_full)
    rng = np.random.default_rng(0)
    for _ in range(5):
        t = int(rng.integers(n // 2, n - 1))
        k = int(rng.integers(1, min(50, n - t)))
        truncated_inputs = [a.iloc[: t + 1] for a in args]
        partial = builder(*truncated_inputs, **kwargs)
        full_val = feature_full.iloc[t]
        partial_val = partial.iloc[t]
        if pd.isna(full_val) and pd.isna(partial_val):
            continue
        assert full_val == pytest.approx(partial_val, rel=1e-9, abs=1e-12), (
            f"Causality violated at t={t}, k={k}"
        )


def test_log_returns_causal(ohlc):
    full = log_returns(ohlc["close"])
    _assert_causal(full, log_returns, ohlc["close"])


def test_rolling_volatility_causal(ohlc):
    rets = log_returns(ohlc["close"])
    full = rolling_volatility(rets)
    _assert_causal(full, rolling_volatility, rets)


def test_rsi_causal(ohlc):
    full = rsi(ohlc["close"])
    _assert_causal(full, rsi, ohlc["close"])


def test_macd_causal(ohlc):
    full_df = macd(ohlc["close"])
    _assert_causal(full_df["macd_line"], lambda p: macd(p)["macd_line"], ohlc["close"])
    _assert_causal(
        full_df["macd_signal"], lambda p: macd(p)["macd_signal"], ohlc["close"]
    )


def test_atr_causal(ohlc):
    full = atr(ohlc["high"], ohlc["low"], ohlc["close"])
    rng = np.random.default_rng(1)
    n = len(full)
    for _ in range(5):
        t = int(rng.integers(n // 2, n - 1))
        partial = atr(
            ohlc["high"].iloc[: t + 1],
            ohlc["low"].iloc[: t + 1],
            ohlc["close"].iloc[: t + 1],
        )
        if pd.isna(full.iloc[t]):
            continue
        assert full.iloc[t] == pytest.approx(partial.iloc[t], rel=1e-9)


def test_momentum_rank_causal(panel):
    prices, _ = panel
    full = momentum_rank(prices)
    rng = np.random.default_rng(2)
    n = len(full)
    for _ in range(5):
        t = int(rng.integers(n // 2, n - 1))
        partial = momentum_rank(prices.iloc[: t + 1])
        # Compare per-row (NaN-safe).
        f_row = full.iloc[t].dropna()
        p_row = partial.iloc[t].dropna()
        pd.testing.assert_series_equal(f_row, p_row, check_names=False)


def test_vol_rank_causal(panel):
    _, returns = panel
    full = vol_rank(returns)
    rng = np.random.default_rng(3)
    n = len(full)
    for _ in range(5):
        t = int(rng.integers(n // 2, n - 1))
        partial = vol_rank(returns.iloc[: t + 1])
        f_row = full.iloc[t].dropna()
        p_row = partial.iloc[t].dropna()
        pd.testing.assert_series_equal(f_row, p_row, check_names=False)


def test_funding_features_lagged():
    cal = pd.date_range("2023-01-01", periods=60, freq="D")
    # 3 funding prints per day, simulated as evenly spaced.
    funding_idx = pd.date_range("2023-01-01", periods=60 * 3, freq="8h", tz="UTC")
    funding = pd.Series(np.linspace(-0.01, 0.01, num=180), index=funding_idx, name="fr")
    feats = funding_features(funding, cal)
    # First-day level should be NaN (lag of 1 day).
    assert pd.isna(feats["funding_level"].iloc[0])
    # First-day-with-data level should equal sum of the previous day's prints.
    expected = funding.iloc[:3].sum()  # day 0 -> appears as level on day 1
    assert feats["funding_level"].iloc[1] == pytest.approx(expected)


def test_basis_signal_aligned():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    spot = pd.Series(100.0 * (1 + np.arange(10) * 0.01), index=idx)
    perp = spot * 1.001
    b = basis(spot, perp)
    assert b.iloc[0] == pytest.approx(0.001)


def test_open_interest_delta_basic():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    oi = pd.Series(np.exp(np.arange(10) * 0.05), index=idx)
    d = open_interest_delta(oi, window=1)
    # log-diff of exp(0.05 * t) is 0.05 each step.
    assert d.iloc[1:].round(6).eq(0.05).all()


def test_macro_features_lagged_and_aligned():
    cal = pd.date_range("2020-01-01", periods=100, freq="B")
    raw = pd.Series(np.linspace(10.0, 20.0, num=100), index=cal, name="VIXCLS")
    feats = macro_features({"VIXCLS": raw}, cal, change_window=5)
    # The level on day t must equal the raw value on day t-1
    # (one-period lag) — verify via spot-check on a random day.
    t = 50
    assert feats["VIXCLS_level"].iloc[t] == pytest.approx(raw.iloc[t - 1])
    # Change column has 5-day diff applied on top of the lag, so the
    # first 6 rows should be NaN.
    assert feats["VIXCLS_change"].iloc[:6].isna().all()
