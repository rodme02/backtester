import numpy as np
import pandas as pd
import pytest

from backtester.models import GBMClassifier


def _separable_dataset(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.5, size=n)
    logits = 0.8 * x1 - 0.6 * x2 + noise
    y = (logits > 0).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y)


def test_gbm_fits_and_outputs_probabilities():
    X, y = _separable_dataset()
    model = GBMClassifier(max_iter=80, learning_rate=0.1)
    model.fit(X.iloc[:400], y.iloc[:400])
    proba = model.predict_proba(X.iloc[400:])
    assert proba.shape == (200,)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_gbm_beats_random_on_separable_data():
    X, y = _separable_dataset()
    model = GBMClassifier(max_iter=80, learning_rate=0.1)
    model.fit(X.iloc[:400], y.iloc[:400])
    proba = model.predict_proba(X.iloc[400:])
    accuracy = ((proba > 0.5) == y.iloc[400:].to_numpy()).mean()
    assert accuracy > 0.6


def test_predict_before_fit_raises():
    model = GBMClassifier()
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame({"x1": [0.0]}))


def test_feature_importance_returns_aligned_series():
    X, y = _separable_dataset()
    model = GBMClassifier(max_iter=40, learning_rate=0.1).fit(X, y)
    fi = model.feature_importance
    assert list(fi.index) == ["x1", "x2"]
