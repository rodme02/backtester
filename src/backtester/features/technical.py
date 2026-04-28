"""Single-asset technical features.

All builders take a price (or OHLC) Series/DataFrame and return a
Series of the same index. Values for indices where the feature is
not yet defined (warm-up window) are NaN — never zero, never the
last valid value carried forward, both of which would be leakage by
imputation.

Causality invariant: feature value at index ``t`` is a function of
inputs at indices ``<= t`` only. ``tests/test_features_leakage.py``
enforces this via property tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Periodic log returns. First value is NaN."""
    return np.log(prices).diff()


def rolling_volatility(
    returns: pd.Series, *, window: int = 20, annualise: int | None = 252
) -> pd.Series:
    """Trailing realised volatility (sample std of log returns)."""
    vol = returns.rolling(window=window, min_periods=window).std(ddof=1)
    if annualise:
        vol = vol * np.sqrt(annualise)
    return vol


def rsi(prices: pd.Series, *, period: int = 14) -> pd.Series:
    """Wilder's RSI on log returns."""
    delta = prices.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    # Wilder's smoothing = EMA with alpha = 1/period.
    avg_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    prices: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Standard MACD: returns columns ``macd_line`` and ``macd_signal``."""
    ema_fast = prices.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd_line": line, "macd_signal": sig})


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, *, period: int = 14
) -> pd.Series:
    """Average True Range (Wilder)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
