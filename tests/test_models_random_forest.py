import numpy as np
import pandas as pd
import pytest

from backtester.models import RandomForest


def _separable_dataset(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.8 * x1 - 0.6 * x2 + rng.normal(scale=0.4, size=n)
    y = (logits > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2}), pd.Series(y)


def test_rf_fits_and_outputs_probabilities():
    X, y = _separable_dataset()
    model = RandomForest(n_estimators=80).fit(X.iloc[:400], y.iloc[:400])
    proba = model.predict_proba(X.iloc[400:])
    assert proba.shape == (200,)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_rf_beats_random_on_separable_data():
    X, y = _separable_dataset()
    model = RandomForest(n_estimators=80).fit(X.iloc[:400], y.iloc[:400])
    proba = model.predict_proba(X.iloc[400:])
    accuracy = ((proba > 0.5) == y.iloc[400:].to_numpy()).mean()
    assert accuracy > 0.6


def test_rf_supports_sample_weight():
    X, y = _separable_dataset()
    weights = np.linspace(0.1, 1.9, num=len(X))
    model = RandomForest(n_estimators=40).fit(X, y, sample_weight=weights)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)


def test_rf_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        RandomForest().predict_proba(pd.DataFrame({"x1": [0.0]}))


def test_rf_feature_importance_aligned():
    X, y = _separable_dataset()
    model = RandomForest(n_estimators=40).fit(X, y)
    fi = model.feature_importance
    assert set(fi.index) == {"x1", "x2"}
    assert (fi >= 0).all()
