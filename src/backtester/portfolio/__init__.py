"""Portfolio construction helpers used by the case-study notebooks.

The case studies operate on per-(date, ticker) prediction series, not
on bar-by-bar event streams: model probabilities → quantile weights →
daily returns → cost adjustment → eval-harness inputs.
"""

from .cross_sectional import (
    apply_book_costs,
    daily_returns_from_book,
    long_short_quantile_weights,
)

__all__ = [
    "apply_book_costs",
    "daily_returns_from_book",
    "long_short_quantile_weights",
]
