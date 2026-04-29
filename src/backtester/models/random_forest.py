"""Random forest classifier (alternative tree ensemble).

Pairs with ``GBMClassifier`` for the tabular bake-off. The two are in
the same family but differ in how trees are combined (bagging +
random feature subsets vs sequential boosting), so a divergent verdict
between them tells us something about the *kind* of failure (high
variance vs high bias).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RandomForest:
    n_estimators: int = 400
    max_depth: int | None = 12
    min_samples_leaf: int = 30
    max_features: str | float | None = "sqrt"
    random_state: int = 17

    _model: Any = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> RandomForest:
        from sklearn.ensemble import RandomForestClassifier

        self._feature_names = list(X.columns)
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        y_arr = np.asarray(y).astype(int)
        self._model.fit(X.values, y_arr, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict_proba().")
        X_aligned = X[self._feature_names]
        return self._model.predict_proba(X_aligned.values)[:, 1]

    @property
    def feature_importance(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("fit() must be called before feature_importance.")
        return pd.Series(
            self._model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)
