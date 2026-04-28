"""Portfolio construction helpers used by the case-study notebooks.

These are kept separate from the ``strategies/`` (backtrader) package
on purpose: the ML-research case studies operate on per-(date, ticker)
prediction series, not on bar-by-bar event streams.
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
