"""Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, Zhu 2017).

Given a ``(T × S)`` matrix of strategy returns (``T`` time bars,
``S`` strategy/configuration variants), CSCV (Combinatorially-
Symmetric Cross-Validation) splits the time axis into ``2·n_splits``
contiguous sub-periods. For every ``C(2n, n)`` partition into IS/OOS
halves:

1. Compute each strategy's IS Sharpe; pick the best.
2. Compute each strategy's OOS rank.
3. Take the OOS rank of the IS-best strategy (normalised to ``(0, 1)``).
4. Convert to logit; sign tells us whether the IS-best beat the
   OOS median (positive logit) or lost (negative logit).

PBO = fraction of partitions whose logit ≤ 0 (IS-best below OOS
median). High PBO → backtest overfit.

Reference: Bailey-Borwein-López de Prado-Zhu, "The Probability of
Backtest Overfitting" (Journal of Computational Finance, 2017).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _sharpe(arr: np.ndarray) -> np.ndarray:
    """Per-column Sharpe (mean / std). Returns 0 where std == 0."""
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1)
    out = np.zeros_like(mu)
    nonzero = sd > 0
    out[nonzero] = mu[nonzero] / sd[nonzero]
    return out


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame | np.ndarray,
    *,
    n_splits: int = 8,
) -> dict[str, float]:
    """PBO and supporting statistics.

    Parameters
    ----------
    returns_matrix
        ``(T, S)`` matrix of per-period strategy returns. Columns =
        strategies / hyperparameter combinations. Rows = time bars.
    n_splits
        Half-count of CSCV sub-periods. The time axis is split into
        ``2 * n_splits`` chunks; the partition uses ``n_splits`` for
        IS and ``n_splits`` for OOS. Default 8 → C(16, 8) = 12,870
        partitions, manageable for any S.

    Returns
    -------
    Dict with:

    - ``pbo``: probability of backtest overfitting (in ``[0, 1]``).
    - ``median_logit``: median over partitions of
      ``log(rank / (1 - rank))`` for the IS-best.
    - ``performance_degradation``: median (IS_best_sharpe -
      OOS_sharpe_of_IS_best) across partitions.
    - ``n_partitions``: actual number of partitions evaluated.
    """
    arr = np.asarray(returns_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T x S)")
    T, S = arr.shape
    if S < 2:
        raise ValueError("PBO requires >= 2 strategies")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    total_chunks = 2 * n_splits
    if total_chunks * 2 > T:
        raise ValueError(f"T={T} too small for {total_chunks} chunks")

    chunks = np.array_split(arr, total_chunks, axis=0)

    logits: list[float] = []
    degradations: list[float] = []
    for is_idx in combinations(range(total_chunks), n_splits):
        oos_idx = tuple(i for i in range(total_chunks) if i not in is_idx)
        is_arr = np.vstack([chunks[i] for i in is_idx])
        oos_arr = np.vstack([chunks[i] for i in oos_idx])

        sr_is = _sharpe(is_arr)
        sr_oos = _sharpe(oos_arr)

        n_star = int(np.argmax(sr_is))
        # Rank of IS-best in OOS (1 = worst, S = best); convert to
        # uniform-on-(0,1) by adding 1 / (S+1).
        oos_rank = (sr_oos.argsort().argsort()[n_star] + 1) / (S + 1)
        logits.append(float(np.log(oos_rank / (1 - oos_rank))))
        degradations.append(float(sr_is[n_star] - sr_oos[n_star]))

    logits_arr = np.asarray(logits)
    return {
        "pbo": float((logits_arr <= 0).mean()),
        "median_logit": float(np.median(logits_arr)),
        "performance_degradation": float(np.median(degradations)),
        "n_partitions": len(logits),
    }
