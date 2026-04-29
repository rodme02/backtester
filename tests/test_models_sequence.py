import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from backtester.models import (  # noqa: E402
    LSTMClassifier,
    TCNClassifier,
    TransformerClassifier,
    stack_sequences,
)


def _toy_panel(n_dates: int = 80, tickers=("A", "B", "C")):
    rng = np.random.default_rng(0)
    rows = []
    for t in tickers:
        for d in pd.date_range("2024-01-01", periods=n_dates, freq="B"):
            rows.append((d, t, rng.normal(), rng.normal()))
    df = pd.DataFrame(rows, columns=["datetime", "ticker", "f1", "f2"]).set_index(
        ["datetime", "ticker"]
    )
    return df


def test_stack_sequences_shape():
    panel = _toy_panel(n_dates=20)
    flat, idx = stack_sequences(panel, lookback=5, feature_cols=["f1", "f2"])
    assert flat.shape == ((20 - 5 + 1) * 3, 5 * 2)
    assert idx.names == ["datetime", "ticker"]


def test_stack_sequences_no_leakage():
    """Window ending at date t must use only inputs <= t."""
    panel = _toy_panel(n_dates=15)
    full, _ = stack_sequences(panel, lookback=4, feature_cols=["f1", "f2"])
    truncated_panel = panel.loc[panel.index.get_level_values(0) <= "2024-01-12"]
    truncated, _ = stack_sequences(truncated_panel, lookback=4, feature_cols=["f1", "f2"])
    # Compare overlapping rows
    common = full.index.intersection(truncated.index)
    pd.testing.assert_frame_equal(full.loc[common], truncated.loc[common])


@pytest.mark.parametrize(
    "Model", [LSTMClassifier, TCNClassifier, TransformerClassifier]
)
def test_sequence_model_fits_and_predicts(Model):
    rng = np.random.default_rng(0)
    n, lookback, n_features = 200, 8, 3
    X = pd.DataFrame(rng.normal(size=(n, lookback * n_features)))
    y = pd.Series(rng.integers(0, 2, size=n))
    model = Model(lookback=lookback, n_features=n_features, hidden=16, epochs=2, batch_size=64)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (n,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_before_fit_raises():
    model = LSTMClassifier(lookback=4, n_features=2)
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame(np.zeros((1, 8))))
