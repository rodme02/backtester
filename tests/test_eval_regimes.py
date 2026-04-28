import numpy as np
import pandas as pd

from backtester.eval.regimes import per_regime_metrics, trend_regimes, vol_regimes


def _price_path(n: int = 600, drift: float = 0.0005, sigma: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, sigma, size=n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx)
    return prices, pd.Series(rets, index=idx)


def test_trend_regimes_basic_shape():
    prices, _ = _price_path()
    tags = trend_regimes(prices, window=50)
    # First 49 should be NaN (sma not yet filled).
    assert tags.iloc[:49].isna().all()
    assert set(tags.dropna().unique()).issubset({"bull", "bear"})


def test_trend_regimes_uptrend_is_bullish():
    # Strong drift -> price persistently above its moving average.
    prices, _ = _price_path(n=500, drift=0.002, sigma=0.005, seed=1)
    tags = trend_regimes(prices, window=50).dropna()
    assert (tags == "bull").mean() > 0.7


def test_vol_regimes_balance_around_quantile():
    _, rets = _price_path(n=500, drift=0.0, sigma=0.01, seed=2)
    tags = vol_regimes(rets, window=30, quantile=0.5).dropna()
    # Roughly half-half by construction — wide tolerance.
    high_share = (tags == "high_vol").mean()
    assert 0.3 < high_share < 0.7


def test_per_regime_metrics_segments_correctly():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    rets = pd.Series([0.01, -0.01, 0.02, 0.0, 0.0, -0.02, 0.0, 0.01, -0.01, 0.0], index=idx)
    tags = pd.Series(
        ["bull", "bull", "bull", "bull", "bull", "bear", "bear", "bear", "bear", "bear"],
        index=idx,
    )
    means = per_regime_metrics(rets, tags, metric=lambda x: float(x.mean()))
    assert means["bull"] == sum([0.01, -0.01, 0.02, 0.0, 0.0]) / 5
    assert means["bear"] == sum([-0.02, 0.0, 0.01, -0.01, 0.0]) / 5
