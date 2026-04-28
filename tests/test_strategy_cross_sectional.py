import pandas as pd
import pytest

from backtester.eval.costs import BPS, EQUITIES_LIQUID
from backtester.strategy import (
    apply_book_costs,
    daily_returns_from_book,
    long_short_quantile_weights,
)


def _scores_panel():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), t) for t in list("ABCDE")
        ] + [
            (pd.Timestamp("2024-01-02"), t) for t in list("ABCDE")
        ],
        names=["datetime", "ticker"],
    )
    scores = pd.Series(
        [0.9, 0.7, 0.5, 0.3, 0.1, 0.2, 0.8, 0.4, 0.6, 0.5],
        index=idx,
    )
    return scores


def test_long_short_weights_dollar_neutral():
    scores = _scores_panel()
    w = long_short_quantile_weights(scores, quantile=0.2)
    daily_sum = w.groupby(level=0).sum()
    assert (daily_sum.abs() < 1e-9).all()


def test_long_short_weights_equal_weighted_within_side():
    scores = _scores_panel()
    w = long_short_quantile_weights(scores, quantile=0.2)
    longs = w[w > 0]
    shorts = w[w < 0]
    assert longs.nunique() == 1
    assert shorts.nunique() == 1


def test_long_short_quantile_validation():
    with pytest.raises(ValueError):
        long_short_quantile_weights(pd.Series([1.0]), quantile=0.5)
    with pytest.raises(ValueError):
        long_short_quantile_weights(pd.Series([1.0]), quantile=0.0)
    with pytest.raises(ValueError):
        long_short_quantile_weights(pd.Series([1.0]), quantile=0.1)  # not multiindex


def test_daily_returns_from_book_basic():
    scores = _scores_panel()
    w = long_short_quantile_weights(scores, quantile=0.2)
    fwd = scores * 0.0 + 0.01  # +1% across the board
    daily = daily_returns_from_book(w, fwd)
    # Dollar-neutral book at +1% across all names should yield ~0 return.
    assert daily.abs().max() < 1e-9


def test_apply_book_costs_charges_initial_and_turnover():
    scores = _scores_panel()
    w = long_short_quantile_weights(scores, quantile=0.2)
    gross = pd.Series(0.0, index=w.index.get_level_values(0).unique())
    net = apply_book_costs(w, gross, EQUITIES_LIQUID)
    # First day: turnover = sum(|w|) = 2.0 (1 long + 1 short normalised).
    expected_first = -2.0 * EQUITIES_LIQUID.per_turnover_bps * BPS
    assert net.iloc[0] == pytest.approx(expected_first)
    # Day 2 turnover should be > 0 because picks change with scores.
    assert net.iloc[1] < 0
