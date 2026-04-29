"""Honest-evaluation harness: walk-forward CV, CPCV+PBO, statistics,
costs, regimes, feature importance.
"""

from .costs import (
    CRYPTO_PERP,
    CRYPTO_PERP_WITH_FUNDING,
    EQUITIES_LIQUID,
    EQUITIES_LIQUID_WITH_BORROW,
    CostModel,
    apply_costs,
)
from .cpcv import cpcv_splits, group_bounds, n_paths, reconstruct_paths
from .feature_importance import mda_manual, mda_sklearn
from .pbo import probability_of_backtest_overfitting
from .regimes import per_regime_metrics, trend_regimes, vol_regimes
from .statistics import (
    annualised_sharpe,
    bootstrap_ci,
    deflated_sharpe_ratio,
    dsr_sensitivity,
    holm_correct,
    probabilistic_sharpe_ratio,
)
from .walkforward import walk_forward_splits

__all__ = [
    "CRYPTO_PERP",
    "CRYPTO_PERP_WITH_FUNDING",
    "CostModel",
    "EQUITIES_LIQUID",
    "EQUITIES_LIQUID_WITH_BORROW",
    "annualised_sharpe",
    "apply_costs",
    "bootstrap_ci",
    "cpcv_splits",
    "deflated_sharpe_ratio",
    "dsr_sensitivity",
    "group_bounds",
    "holm_correct",
    "mda_manual",
    "mda_sklearn",
    "n_paths",
    "probability_of_backtest_overfitting",
    "per_regime_metrics",
    "probabilistic_sharpe_ratio",
    "reconstruct_paths",
    "trend_regimes",
    "vol_regimes",
    "walk_forward_splits",
]
