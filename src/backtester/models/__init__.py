"""Thin model wrappers exposing a uniform interface to the eval harness.

Every model exposes ``fit(X, y)`` and ``predict_proba(X) -> np.ndarray
of shape (n,)`` returning the predicted probability of the positive
class. The harness drives them through the walk-forward CV uniformly,
so swapping models is a one-line change in a notebook.
"""

from .gbm import GBMClassifier


def _try_import_sequence():
    try:
        from .sequence import LSTMClassifier, TCNClassifier, stack_sequences

        return LSTMClassifier, TCNClassifier, stack_sequences
    except ImportError:  # torch not installed
        return None, None, None


LSTMClassifier, TCNClassifier, stack_sequences = _try_import_sequence()

__all__ = ["GBMClassifier", "LSTMClassifier", "TCNClassifier", "stack_sequences"]
