"""Combinatorial Purged Cross-Validation (López de Prado, AFML §12).

Standard walk-forward gives **one** out-of-sample path. CPCV partitions
the timeline into ``n_groups`` contiguous groups and, for every choice
of ``k_test`` groups as the test set, trains on the remaining
``n_groups - k_test`` (with purge + embargo around each test segment),
yielding ``C(n_groups, k_test)`` train/test splits.

Crucially, the test predictions can be **stitched** across splits to
form ``n_paths = C(n_groups - 1, k_test - 1)`` distinct end-to-end OOS
paths — a *distribution* of realised performance, which is what
enables PBO and proper CPCV-Sharpe error bars.

Usage::

    bounds = group_bounds(n=len(X), n_groups=10)
    splits = list(cpcv_splits(n=len(X), n_groups=10, k_test=2,
                              label_horizon=5, embargo=2))
    preds_by_combo: dict[tuple[int, ...], pd.Series] = {}
    for train_idx, test_idx, combo in splits:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds_by_combo[combo] = pd.Series(
            model.predict_proba(X.iloc[test_idx]),
            index=test_idx,
        )

    paths = reconstruct_paths(preds_by_combo, bounds=bounds, k_test=2)
    # paths is a list[pd.Series]; each Series spans the full date range
    # of test groups (union over all groups).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def group_bounds(n: int, n_groups: int) -> list[tuple[int, int]]:
    """Return ``n_groups`` ``(start, end)`` pairs covering ``[0, n)``.

    All groups have ``n // n_groups`` rows except the last, which
    absorbs the remainder.
    """
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    if n < n_groups:
        raise ValueError(f"n={n} smaller than n_groups={n_groups}")
    base = n // n_groups
    bounds = [(i * base, (i + 1) * base) for i in range(n_groups)]
    bounds[-1] = (bounds[-1][0], n)
    return bounds


def cpcv_splits(
    n: int,
    *,
    n_groups: int = 10,
    k_test: int = 2,
    label_horizon: int = 1,
    embargo: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
    """Yield ``(train_idx, test_idx, group_ids)`` for every C(n_groups, k_test) split."""
    if n_groups < 3:
        raise ValueError("n_groups must be >= 3")
    if not 1 <= k_test < n_groups:
        raise ValueError("k_test must be in [1, n_groups)")
    if label_horizon < 1:
        raise ValueError("label_horizon must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    bounds = group_bounds(n, n_groups)

    for combo in combinations(range(n_groups), k_test):
        test_idx_parts = [np.arange(*bounds[g]) for g in combo]
        test_idx = np.concatenate(test_idx_parts)

        train_mask = np.ones(n, dtype=bool)
        for g in combo:
            start, end = bounds[g]
            lo = max(0, start - (label_horizon - 1))
            hi = min(n, end + embargo)
            train_mask[lo:hi] = False
        train_idx = np.where(train_mask)[0]
        yield train_idx, test_idx, combo


def n_paths(n_groups: int, k_test: int) -> int:
    """Number of distinct full-length OOS paths reconstructible from CPCV."""
    if n_groups < 3 or not 1 <= k_test < n_groups:
        raise ValueError("invalid n_groups / k_test")
    return comb(n_groups - 1, k_test - 1)


def reconstruct_paths(
    predictions_by_combo: Mapping[tuple[int, ...], pd.Series],
    *,
    bounds: list[tuple[int, int]],
    k_test: int,
) -> list[pd.Series]:
    """Stitch CPCV per-combo predictions into ``n_paths`` full-length series.

    Algorithm: every group ``g`` is covered by exactly
    ``C(n_groups - 1, k_test - 1)`` combos. For each path index
    ``p ∈ [0, n_paths)``, pick the *p*-th combo (in canonical order)
    that contains group ``g``; that combo's predictions sliced to
    group ``g`` form path ``p``'s contribution from group ``g``.

    Concatenating across all groups yields path ``p``.

    Parameters
    ----------
    predictions_by_combo
        Mapping ``combo_tuple -> Series`` indexed by sample positions.
        Each Series must contain predictions for *every* row in every
        group of the combo. (i.e., the test_idx returned by
        ``cpcv_splits`` for that combo.)
    bounds
        Group boundaries from :func:`group_bounds`.
    k_test
        Same as in :func:`cpcv_splits`.
    """
    n_groups = len(bounds)
    expected = n_paths(n_groups, k_test)

    combos_for_group: dict[int, list[tuple[int, ...]]] = {g: [] for g in range(n_groups)}
    for combo in sorted(predictions_by_combo.keys()):
        for g in combo:
            combos_for_group[g].append(combo)

    for g, combos in combos_for_group.items():
        if len(combos) != expected:
            raise ValueError(
                f"group {g} appears in {len(combos)} combos but expected {expected}; "
                "predictions_by_combo is incomplete"
            )

    paths: list[pd.Series] = []
    for p in range(expected):
        pieces: list[pd.Series] = []
        for g in range(n_groups):
            combo = combos_for_group[g][p]
            series = predictions_by_combo[combo]
            start, end = bounds[g]
            # Restrict this combo's predictions to rows in group g.
            mask = (series.index >= start) & (series.index < end)
            pieces.append(series.loc[mask])
        path = pd.concat(pieces).sort_index()
        paths.append(path)

    return paths
