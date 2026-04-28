"""Thin model wrappers exposing a uniform interface to the eval harness.

Every model exposes ``fit(X, y)`` and ``predict_proba(X) -> np.ndarray
of shape (n,)`` returning the predicted probability of the positive
class. The harness drives them through the walk-forward CV uniformly,
so swapping models is a one-line change in a notebook.
"""

from .gbm import GBMClassifier

__all__ = ["GBMClassifier"]
