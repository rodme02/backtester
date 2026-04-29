"""L2-regularised logistic regression baseline.

The linear sanity check: if a low-capacity model achieves the same OOS
performance as a tuned GBM, the GBM is just learning noise. Wraps
``sklearn.linear_model.LogisticRegression`` with ``saga`` solver and
on-the-fly standardisation, exposing the project's uniform
``fit / predict_proba`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LogisticBaseline:
    C: float = 1.0
    """Inverse L2 regularisation strength (sklearn convention)."""
    max_iter: int = 1000
    random_state: int = 17

    _model: Any = field(default=None, init=False, repr=False)
    _scaler: Any = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> LogisticBaseline:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._feature_names = list(X.columns)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.values)
        self._model = LogisticRegression(
            C=self.C,
            solver="saga",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        y_arr = np.asarray(y).astype(int)
        self._model.fit(Xs, y_arr, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict_proba().")
        X_aligned = X[self._feature_names]
        Xs = self._scaler.transform(X_aligned.values)
        return self._model.predict_proba(Xs)[:, 1]

    @property
    def coefficients(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("fit() must be called before coefficients.")
        return pd.Series(self._model.coef_[0], index=self._feature_names).sort_values(
            key=lambda s: s.abs(), ascending=False
        )
