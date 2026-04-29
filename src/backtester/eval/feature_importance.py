"""Mean Decrease Accuracy (MDA) feature importance (López de Prado AFML §8.4).

Wraps scikit-learn's ``permutation_importance`` for any classifier
that exposes ``predict_proba``. For sequence models that don't fit
the sklearn estimator API, ``mda_manual`` runs the same logic with a
user-supplied predict callable.

Why MDA over GBM split-counts:
- Computed at evaluation time, not training; reflects OOS behaviour.
- Classifier-agnostic.
- Immune to high-cardinality bias (split-counts favour features with
  more split candidates, which inflates importance for noisy
  high-cardinality columns).

Caveat: MDA underestimates correlated features (permuting A still
leaves correlated B carrying the signal). Cluster features by
correlation if you need a "block MDA."
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score


def mda_sklearn(
    estimator,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int = 10,
    random_state: int = 0,
    scoring: str = "accuracy",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Permutation-based MDA via sklearn.

    Returns a DataFrame indexed by feature name with columns
    ``importance_mean``, ``importance_std``.
    """
    y_arr = np.asarray(y).astype(int)
    result = permutation_importance(
        estimator,
        X.values,
        y_arr,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    return pd.DataFrame(
        {
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        },
        index=X.columns,
    ).sort_values("importance_mean", ascending=False)


def mda_manual(
    predict: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """MDA for any model exposing a ``predict(X) -> y_pred`` callable.

    Useful for sequence models / torch wrappers that don't satisfy the
    sklearn estimator contract.
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y).astype(int)
    base = accuracy_score(y_arr, predict(X))

    importances: dict[str, list[float]] = {col: [] for col in X.columns}
    for _ in range(n_repeats):
        for col in X.columns:
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            score = accuracy_score(y_arr, predict(X_perm))
            importances[col].append(base - score)

    rows = {
        col: {
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
        }
        for col, drops in importances.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").sort_values(
        "importance_mean", ascending=False
    )
