import numpy as np
import pytest

from backtester.eval.walkforward import walk_forward_splits


def test_splits_cover_disjoint_test_regions():
    splits = list(walk_forward_splits(n=100, n_splits=4, label_horizon=1))
    assert len(splits) == 4
    test_indices = np.concatenate([test_idx for _, test_idx in splits])
    # All test indices unique.
    assert test_indices.size == np.unique(test_indices).size
    # Test indices cover the second half of the range (first fold is training).
    assert test_indices.min() >= 100 // 5
    assert test_indices.max() == 99


def test_train_always_precedes_test():
    for train_idx, test_idx in walk_forward_splits(n=200, n_splits=5):
        assert train_idx.max() < test_idx.min()


def test_purge_removes_label_horizon_rows():
    horizon = 5
    splits = list(
        walk_forward_splits(n=120, n_splits=3, label_horizon=horizon, embargo=0)
    )
    for train_idx, test_idx in splits:
        # No training row's [i, i + horizon) window overlaps the test fold.
        leak_threshold = test_idx.min() - horizon + 1
        assert (train_idx < leak_threshold).all()


def test_embargo_shrinks_test_coverage():
    no_embargo = list(walk_forward_splits(n=100, n_splits=4, embargo=0))
    with_embargo = list(walk_forward_splits(n=100, n_splits=4, embargo=5))
    assert len(no_embargo) == len(with_embargo)
    no_e_total = sum(len(t) for _, t in no_embargo)
    with_e_total = sum(len(t) for _, t in with_embargo)
    # Embargo shifts later fold boundaries forward, so total tested
    # observations cannot exceed the no-embargo case.
    assert with_e_total <= no_e_total


def test_min_train_size_skips_short_folds():
    splits = list(walk_forward_splits(n=60, n_splits=5, min_train_size=20))
    # First few folds should be skipped because train set < 20.
    assert all(len(train) >= 20 for train, _ in splits)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        list(walk_forward_splits(n=100, n_splits=0))
    with pytest.raises(ValueError):
        list(walk_forward_splits(n=100, label_horizon=0))
    with pytest.raises(ValueError):
        list(walk_forward_splits(n=5, n_splits=5))
