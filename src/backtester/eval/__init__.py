"""Honest-evaluation harness: walk-forward CV, cost models, statistics, regimes."""

from .costs import CRYPTO_PERP, EQUITIES_LIQUID, CostModel, apply_costs
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
    "deflated_sharpe_ratio",
    "holm_correct",
    "probabilistic_sharpe_ratio",
    "walk_forward_splits",
]
