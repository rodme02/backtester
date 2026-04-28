"""Cross-sectional long/short portfolio construction.

Inputs are tidy ``(datetime, ticker) -> score`` series produced by an
ML model (out-of-sample probabilities, alpha scores, etc.).

Pipeline:

1. ``long_short_quantile_weights(scores, q)`` → per-period dollar-
   neutral weights, equal-weighted within each side.
2. ``daily_returns_from_book(weights, forward_returns)`` → portfolio
   daily returns (gross, before costs).
3. ``apply_book_costs(weights, gross_returns, model)`` → returns net
   of book-level turnover costs.
"""

from __future__ import annotations

import pandas as pd

from ..eval.costs import BPS, CostModel


def long_short_quantile_weights(
    scores: pd.Series,
    *,
    quantile: float = 0.2,
) -> pd.Series:
    """Top-``q`` long / bottom-``q`` short, equal-weighted, dollar-neutral.

    ``scores`` must be a Series indexed by ``(datetime, ticker)``.
    Returns weights with the same index, summing to 0 each day.
    """
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be in (0, 0.5)")
    if not isinstance(scores.index, pd.MultiIndex):
        raise ValueError("scores must have a (datetime, ticker) MultiIndex")

    def _one_day(group: pd.Series) -> pd.Series:
        upper = group.quantile(1 - quantile)
        lower = group.quantile(quantile)
        weights = pd.Series(0.0, index=group.index)
        long_mask = group >= upper
        short_mask = group <= lower
        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())
        if n_long:
            weights[long_mask] = 1.0 / n_long
        if n_short:
            weights[short_mask] = -1.0 / n_short
        return weights

    return scores.groupby(level=0, group_keys=False).apply(_one_day)


def daily_returns_from_book(
    weights: pd.Series,
    forward_returns: pd.Series,
) -> pd.Series:
    """Daily portfolio returns from per-name weights and 1-period-ahead returns.

    ``weights`` and ``forward_returns`` share a ``(datetime, ticker)``
    index. Returns sum across tickers per date.
    """
    return (weights * forward_returns).groupby(level=0).sum().sort_index()


def apply_book_costs(
    weights: pd.Series,
    gross_returns: pd.Series,
    model: CostModel,
) -> pd.Series:
    """Subtract book-level turnover costs.

    Turnover at date ``t`` is the L1 distance between the weight vector
    at ``t`` and at ``t - 1`` (the initial entry counts as full
    turnover). Cost = turnover × ``model.per_turnover_bps`` × bps.
    """
    wide = weights.unstack("ticker").fillna(0.0).sort_index()
    book_turnover = wide.diff().abs().sum(axis=1)
    book_turnover.iloc[0] = wide.iloc[0].abs().sum()
    book_turnover = book_turnover.reindex(gross_returns.index).fillna(0.0)
    cost = book_turnover * model.per_turnover_bps * BPS
    return gross_returns - cost
