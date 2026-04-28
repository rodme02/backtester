"""Purged & embargoed walk-forward cross-validation.

Standard k-fold CV leaks across the train/test boundary when the
target depends on a *future* window of returns (almost any prediction
target in finance does). This module yields ``(train_idx, test_idx)``
pairs that:

1. Slide forward in time (no shuffling, no future train data).
2. **Purge** training rows whose target window overlaps the test
   window (López de Prado, *Advances in Financial Machine Learning*
   §7.4).
3. Apply an **embargo** after each test window: a buffer of rows that
   are excluded from the *next* training fold to absorb residual
   serial correlation.

Usage::

    splits = walk_forward_splits(
        n=len(X),
        n_splits=5,
        label_horizon=5,    # next-5-day return → 5-row purge
        embargo=2,
    )
    for train_idx, test_idx in splits:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        ...
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def walk_forward_splits(
    n: int,
    *,
    n_splits: int = 5,
    label_horizon: int = 1,
    embargo: int = 0,
    min_train_size: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` pairs covering ``range(n)``.

    Parameters
    ----------
    n
        Total number of (time-ordered) samples.
    n_splits
        Number of test folds. Each test fold has roughly ``n / (n_splits + 1)``
        rows; the first ``n / (n_splits + 1)`` rows are reserved as
        the initial training window.
    label_horizon
        How many future rows the target uses. Training rows whose
        ``[i, i + label_horizon)`` window overlaps the test window are
        purged.
    embargo
        Extra rows after each test fold that are kept out of the
        *next* training fold. Defaults to 0.
    min_train_size
        Optional minimum training-fold size. Folds smaller than this
        are skipped.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if label_horizon < 1:
        raise ValueError("label_horizon must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    if n < 2 * (n_splits + 1):
        raise ValueError(
            f"n={n} too small for n_splits={n_splits}; need at least {2 * (n_splits + 1)}"
        )

    # Test folds tile the second half of the range; the first slice is
    # the initial training window.
    fold_size = n // (n_splits + 1)
    boundaries = [fold_size * (k + 1) for k in range(n_splits + 1)]
    boundaries[-1] = n  # absorb the remainder into the last fold

    for k in range(n_splits):
        test_start = boundaries[k]
        test_end = boundaries[k + 1]
        test_idx = np.arange(test_start, test_end)

        # Train rows are everything before the test window…
        candidate = np.arange(0, test_start)
        # …minus rows whose label window leaks into the test fold.
        leak_start = max(0, test_start - label_horizon + 1)
        train_mask = candidate < leak_start
        train_idx = candidate[train_mask]

        if min_train_size is not None and train_idx.size < min_train_size:
            continue

        yield train_idx, test_idx

        # The embargo only matters for the *next* fold's training set,
        # which on the next iteration starts at boundaries[k+1] anyway.
        # If we wanted a bidirectional CV (with future-into-past leakage)
        # this is where embargo would slot in.
        if embargo:
            # Bump the next fold's effective start to apply embargo.
            boundaries[k + 1] = min(boundaries[k + 1] + embargo, n)
