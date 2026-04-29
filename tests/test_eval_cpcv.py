"""CPCV property tests: structure, purge/embargo correctness, path completeness."""

from math import comb

import numpy as np
import pandas as pd
import pytest

from backtester.eval.cpcv import (
    cpcv_splits,
    group_bounds,
    n_paths,
    reconstruct_paths,
)


def test_group_bounds_cover_full_range():
    bounds = group_bounds(n=100, n_groups=10)
    assert len(bounds) == 10
    assert bounds[0] == (0, 10)
    assert bounds[-1] == (90, 100)
    # All rows covered exactly once.
    covered = np.concatenate([np.arange(*b) for b in bounds])
    np.testing.assert_array_equal(covered, np.arange(100))


def test_group_bounds_remainder_absorbed_in_last_group():
    bounds = group_bounds(n=103, n_groups=10)
    assert bounds[-1] == (90, 103)


def test_cpcv_yields_full_combination_count():
    splits = list(cpcv_splits(n=100, n_groups=10, k_test=2))
    assert len(splits) == comb(10, 2) == 45


def test_cpcv_test_idx_size_matches_groups():
    bounds = group_bounds(n=100, n_groups=10)
    for _, test_idx, combo in cpcv_splits(n=100, n_groups=10, k_test=2):
        expected_size = sum(b[1] - b[0] for g, b in enumerate(bounds) if g in combo)
        assert test_idx.size == expected_size


def test_cpcv_train_disjoint_from_test():
    for train_idx, test_idx, _ in cpcv_splits(n=200, n_groups=10, k_test=2):
        assert np.intersect1d(train_idx, test_idx).size == 0


def test_cpcv_purge_excludes_label_horizon_rows():
    """No training row's [t, t+L) window may overlap any test group."""
    L = 5
    bounds = group_bounds(n=200, n_groups=10)
    for train_idx, _, combo in cpcv_splits(
        n=200, n_groups=10, k_test=2, label_horizon=L
    ):
        for g in combo:
            start = bounds[g][0]
            # Training rows in [start - L + 1, start) would have label
            # windows that touch the test group; they must be purged.
            forbidden = np.arange(max(0, start - L + 1), start)
            overlap = np.intersect1d(train_idx, forbidden)
            assert overlap.size == 0, f"purge violated near group {g}: {overlap}"


def test_cpcv_embargo_excludes_post_test_rows():
    e = 3
    bounds = group_bounds(n=200, n_groups=10)
    for train_idx, _, combo in cpcv_splits(
        n=200, n_groups=10, k_test=2, embargo=e
    ):
        for g in combo:
            end = bounds[g][1]
            forbidden = np.arange(end, min(200, end + e))
            overlap = np.intersect1d(train_idx, forbidden)
            assert overlap.size == 0, f"embargo violated after group {g}: {overlap}"


def test_n_paths_formula():
    assert n_paths(n_groups=10, k_test=2) == comb(9, 1) == 9
    assert n_paths(n_groups=6, k_test=2) == comb(5, 1) == 5
    assert n_paths(n_groups=8, k_test=3) == comb(7, 2) == 21


def test_reconstruct_paths_covers_full_range_without_overlap():
    n = 100
    n_groups = 5
    k_test = 2
    bounds = group_bounds(n, n_groups)
    splits = list(cpcv_splits(n=n, n_groups=n_groups, k_test=k_test))

    # Synthesise predictions: combo (a, b) -> Series of value `combo_id`
    # at each test position. Different combos produce different values
    # so we can verify the path-construction logic picks them correctly.
    preds_by_combo = {}
    for _, test_idx, combo in splits:
        # Use a hash of combo as the value (just unique per combo).
        v = hash(combo) % 1000
        preds_by_combo[combo] = pd.Series(v, index=test_idx, dtype=float)

    paths = reconstruct_paths(preds_by_combo, bounds=bounds, k_test=k_test)
    expected_paths = n_paths(n_groups, k_test)
    assert len(paths) == expected_paths

    # Each path must cover every row exactly once.
    for path in paths:
        assert len(path) == n, f"path length {len(path)} != n={n}"
        np.testing.assert_array_equal(path.index.to_numpy(), np.arange(n))


def test_reconstruct_paths_uses_each_group_once_per_path():
    """Each path's slice for group g must come from exactly one combo
    that contains g; across paths, each combo is used at most once per
    group.
    """
    n = 60
    n_groups = 6
    k_test = 2
    bounds = group_bounds(n, n_groups)
    splits = list(cpcv_splits(n=n, n_groups=n_groups, k_test=k_test))

    # Tag each combo with a unique value (sortable) so we can inspect.
    preds_by_combo = {}
    for _, test_idx, combo in splits:
        preds_by_combo[combo] = pd.Series(
            float(combo[0] * 10 + combo[1]), index=test_idx
        )

    paths = reconstruct_paths(preds_by_combo, bounds=bounds, k_test=k_test)
    # Per group, the set of combos used across all paths should equal
    # the set of all combos containing that group (each combo used once).
    for g, (start, end) in enumerate(bounds):
        combos_used = set()
        for path in paths:
            v = path.loc[start:end - 1].iloc[0]
            # Recover combo from the encoding.
            combos_used.add(int(v))
        all_combos_with_g = {
            c[0] * 10 + c[1] for c in preds_by_combo if g in c
        }
        assert combos_used == all_combos_with_g


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        list(cpcv_splits(n=100, n_groups=2))
    with pytest.raises(ValueError):
        list(cpcv_splits(n=100, n_groups=10, k_test=0))
    with pytest.raises(ValueError):
        list(cpcv_splits(n=100, n_groups=10, k_test=10))
    with pytest.raises(ValueError):
        list(cpcv_splits(n=100, n_groups=10, label_horizon=0))
    with pytest.raises(ValueError):
        list(cpcv_splits(n=100, n_groups=10, embargo=-1))
