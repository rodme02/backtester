"""MDA importance tests."""

import numpy as np
import pandas as pd
import pytest

from backtester.eval.feature_importance import mda_manual, mda_sklearn


def _separable_dataset(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    x_signal = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    logits = 0.8 * x_signal + 0.05 * rng.normal(size=n)
    y = (logits > 0).astype(int)
    X = pd.DataFrame({"signal": x_signal, "noise": x_noise})
    return X, pd.Series(y)


def test_mda_sklearn_ranks_signal_above_noise():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = _separable_dataset()
    model = HistGradientBoostingClassifier(max_iter=80, random_state=0).fit(X, y)
    fi = mda_sklearn(model, X, y, n_repeats=5, random_state=0)
    assert list(fi.columns) == ["importance_mean", "importance_std"]
    assert fi.loc["signal", "importance_mean"] > fi.loc["noise", "importance_mean"]


def test_mda_manual_matches_sklearn_directionally():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = _separable_dataset()
    model = HistGradientBoostingClassifier(max_iter=80, random_state=0).fit(X, y)
    fi = mda_manual(lambda Xp: model.predict(Xp.values), X, y, n_repeats=5, random_state=0)
    assert fi.loc["signal", "importance_mean"] > fi.loc["noise", "importance_mean"]
