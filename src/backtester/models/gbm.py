"""Gradient-boosted-tree binary classifier.

Wraps scikit-learn's ``HistGradientBoostingClassifier`` — a histogram-
based GBM in the same family as LightGBM/XGBoost, with no system
dependencies (no libomp install required). Configured for tabular
financial data: deeper-than-default trees, mild L2, deterministic
seed. Exposes the project's uniform ``fit / predict_proba`` interface
so the eval harness can swap it out for any other model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GBMClassifier:
    max_iter: int = 400
    learning_rate: float = 0.03
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    early_stopping: bool = False
    random_state: int = 17

    _model: Any = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> GBMClassifier:
        from sklearn.ensemble import HistGradientBoostingClassifier  # imported lazily

        self._feature_names = list(X.columns)
        self._model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )
        y_arr = np.asarray(y).astype(int)
        self._model.fit(X.values, y_arr, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict_proba().")
        X_aligned = X[self._feature_names]
        proba = self._model.predict_proba(X_aligned.values)
        return proba[:, 1]

    @property
    def feature_importance(self) -> pd.Series:
        """Permutation-style importance via the model's feature contributions.

        ``HistGradientBoostingClassifier`` does not expose an out-of-the-box
        ``feature_importances_`` attribute (unlike tree-only ensembles). To
        keep the interface uniform we approximate with the absolute mean
        leaf-value response per feature using ``permutation_importance``
        on the training fold. Pass ``X``, ``y`` to compute it; for now
        return an empty Series if not supported.
        """
        if self._model is None:
            raise RuntimeError("fit() must be called before feature_importance.")
        return pd.Series(dtype=float, index=self._feature_names)
