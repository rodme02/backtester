"""Thin model wrappers exposing a uniform interface to the eval harness.

Every model exposes ``fit(X, y, sample_weight=None)`` and
``predict_proba(X) -> np.ndarray`` of shape ``(n,)`` (probability of
the positive class). The harness drives them through walk-forward /
CPCV uniformly, so swapping models is a one-line change in a notebook.

Tabular models are eager imports (sklearn is a hard dependency).
Sequence models live behind the ``[ml]`` extra and raise an
informative ``ImportError`` when called without torch installed.
"""

from .gbm import GBMClassifier
from .linear import LogisticBaseline
from .random_forest import RandomForest


def _import_sequence():
    try:
        from .sequence import LSTMClassifier, TCNClassifier, stack_sequences
    except ImportError:  # torch not installed
        msg = "sequence models require torch; install with: pip install -e '.[ml]'"

        class _Missing:
            def __init__(self, *args, **kwargs):
                raise ImportError(msg)

            def __call__(self, *args, **kwargs):  # called when stack_sequences is invoked
                raise ImportError(msg)

        sentinel = _Missing  # type: ignore[assignment]
        return sentinel, sentinel, sentinel
    return LSTMClassifier, TCNClassifier, stack_sequences


LSTMClassifier, TCNClassifier, stack_sequences = _import_sequence()

__all__ = [
    "GBMClassifier",
    "LSTMClassifier",
    "LogisticBaseline",
    "RandomForest",
    "TCNClassifier",
    "stack_sequences",
]
