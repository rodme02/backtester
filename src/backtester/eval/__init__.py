"""Honest-evaluation harness: walk-forward CV, CPCV+PBO, statistics,
costs, regimes, feature importance.
"""

from .costs import CRYPTO_PERP, EQUITIES_LIQUID, CostModel, apply_costs
from .cpcv import cpcv_splits, group_bounds, n_paths, reconstruct_paths
from .regimes import per_regime_metrics, trend_regimes, vol_regimes
from .statistics import (
    annualised_sharpe,
    bootstrap_ci,
    deflated_sharpe_ratio,
    holm_correct,
    probabilistic_sharpe_ratio,
)
from .walkforward import walk_forward_splits

__all__ = [
    "CRYPTO_PERP",
    "CostModel",
    "EQUITIES_LIQUID",
    "annualised_sharpe",
    "apply_costs",
    "bootstrap_ci",
    "cpcv_splits",
    "deflated_sharpe_ratio",
    "group_bounds",
    "holm_correct",
    "n_paths",
    "per_regime_metrics",
    "probabilistic_sharpe_ratio",
    "reconstruct_paths",
    "trend_regimes",
    "vol_regimes",
    "walk_forward_splits",
]
