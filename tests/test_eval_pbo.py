"""PBO property tests."""

import numpy as np
import pandas as pd
import pytest

from backtester.eval.pbo import probability_of_backtest_overfitting


def test_pbo_low_when_one_clear_winner():
    """A genuinely-best strategy should have low PBO: it's also best OOS."""
    rng = np.random.default_rng(0)
    T, S = 600, 6
    base = rng.normal(0.0, 0.01, size=(T, S))
    # Strategy 0 has a real positive drift; others are noise.
    base[:, 0] += 0.003
    out = probability_of_backtest_overfitting(base, n_splits=4)
    assert out["pbo"] < 0.3
    assert out["performance_degradation"] < 0.5  # IS-best stays good OOS


def test_pbo_high_under_pure_noise():
    """Winner's-curse / regression-to-the-mean: under pure i.i.d.
    noise, the IS-best is the lucky-in-IS strategy, which regresses
    OOS — PBO is well above 0.5 (and grows with S)."""
    rng = np.random.default_rng(1)
    T, S = 800, 8
    arr = rng.normal(0.0, 0.01, size=(T, S))
    out = probability_of_backtest_overfitting(arr, n_splits=4)
    assert out["pbo"] > 0.5


def test_pbo_returns_dict_with_expected_keys():
    rng = np.random.default_rng(2)
    arr = rng.normal(0.0, 0.01, size=(400, 4))
    out = probability_of_backtest_overfitting(arr, n_splits=3)
    assert {"pbo", "median_logit", "performance_degradation", "n_partitions"}.issubset(
        out.keys()
    )
    assert out["n_partitions"] > 0


def test_pbo_dataframe_input_works():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0.0, 0.01, size=(400, 4)), columns=list("ABCD"))
    out = probability_of_backtest_overfitting(df, n_splits=3)
    assert 0.0 <= out["pbo"] <= 1.0


def test_pbo_invalid_args():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros(100), n_splits=3)  # 1-D
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((400, 1)), n_splits=3)  # S=1
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((400, 4)), n_splits=1)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((10, 4)), n_splits=4)  # T too small
