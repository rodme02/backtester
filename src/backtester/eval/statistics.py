"""Statistical tests for trading strategies — the rigour layer.

Implements:

- ``probabilistic_sharpe_ratio`` (PSR, Bailey & López de Prado 2012):
  P(true SR > benchmark | observed SR, n, skew, kurtosis).
- ``deflated_sharpe_ratio`` (DSR, Bailey & López de Prado 2014):
  PSR with the benchmark inflated to account for multiple-trial bias.
- ``bootstrap_ci``: percentile bootstrap CI for any statistic of a
  return series.
- ``holm_correct``: Holm–Bonferroni step-down family-wise correction.

References:
- Bailey, López de Prado, "The Sharpe Ratio Efficient Frontier" (2012).
- Bailey, López de Prado, "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality" (2014).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

PERIODS_PER_YEAR_DEFAULT = 252  # trading days


def annualised_sharpe(
    returns: np.ndarray | Sequence[float],
    *,
    periods_per_year: int = PERIODS_PER_YEAR_DEFAULT,
    risk_free_per_period: float = 0.0,
) -> float:
    """Annualised Sharpe ratio of a periodic excess-return series.

    Returns 0.0 if the standard deviation is zero (degenerate series).
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    excess = r - risk_free_per_period
    sd = excess.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    returns: np.ndarray | Sequence[float],
    *,
    benchmark_sr: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR_DEFAULT,
) -> float:
    """Probability that the *true* Sharpe ratio exceeds ``benchmark_sr``.

    ``benchmark_sr`` is given on the same annualisation basis as the
    returned ``annualised_sharpe`` value (annualised by default).
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 4:
        return float("nan")

    sr_annual = annualised_sharpe(r, periods_per_year=periods_per_year)
    # Convert to per-period SR for the standard-error formula.
    sqrt_p = math.sqrt(periods_per_year)
    sr_p = sr_annual / sqrt_p
    bench_p = benchmark_sr / sqrt_p

    skew = float(stats.skew(r, bias=False))
    # Excess kurtosis (Fisher).
    kurt = float(stats.kurtosis(r, bias=False, fisher=True))

    # Variance of the SR estimator (per-period scale).
    var = (1.0 - skew * sr_p + 0.25 * (kurt) * sr_p**2) / (n - 1)
    if var <= 0 or not np.isfinite(var):
        return float("nan")
    z = (sr_p - bench_p) / math.sqrt(var)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray | Sequence[float],
    *,
    n_trials: int,
    trials_sr_var: float | None = None,
    periods_per_year: int = PERIODS_PER_YEAR_DEFAULT,
) -> float:
    """Deflated Sharpe Ratio.

    The benchmark SR is inflated to the expected maximum SR under the
    null across ``n_trials`` strategy candidates with cross-trial
    Sharpe-ratio variance ``trials_sr_var`` (annualised).

    ``trials_sr_var`` is a methodological choice, not an empirical
    measurement. Reasonable values:

    - ``1.0`` (default) — *worst case*: trial Sharpes are completely
      independent. Very pessimistic for closely-related models on the
      same data.
    - ``0.25–0.5`` — realistic for trials that share data and most
      hyperparameters but differ in 1–2 axes.
    - ``0.05–0.10`` — strong overlap (e.g. learning-rate sweep on the
      same model).

    Always report :func:`dsr_sensitivity` across this range so the
    reader can see how the verdict moves with the assumption.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if trials_sr_var is None:
        trials_sr_var = 1.0
    if trials_sr_var <= 0:
        raise ValueError("trials_sr_var must be > 0")

    if n_trials == 1:
        # With a single trial there's no multiple-comparison inflation;
        # DSR collapses to PSR with the natural zero benchmark.
        expected_max_sr_annual = 0.0
    else:
        # Expected max of N i.i.d. standard normals (Gumbel approximation).
        euler_mascheroni = 0.5772156649015329
        e_max = (
            (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
        expected_max_sr_annual = math.sqrt(trials_sr_var) * float(e_max)

    return probabilistic_sharpe_ratio(
        returns,
        benchmark_sr=expected_max_sr_annual,
        periods_per_year=periods_per_year,
    )


def dsr_sensitivity(
    returns: np.ndarray | Sequence[float],
    *,
    n_trials: int,
    var_grid: Sequence[float] = (1.0, 0.5, 0.25, 0.1),
    periods_per_year: int = PERIODS_PER_YEAR_DEFAULT,
) -> dict[float, float]:
    """Compute Deflated Sharpe across multiple ``trials_sr_var`` assumptions.

    Returns ``{var: dsr_at_var}`` so the reader can see how the
    verdict moves with the cross-trial-variance assumption. Reporting
    a single DSR value at ``var=1.0`` (the worst case) without
    sensitivity reads as p-hacked pessimism.
    """
    return {
        float(v): deflated_sharpe_ratio(
            returns,
            n_trials=n_trials,
            trials_sr_var=float(v),
            periods_per_year=periods_per_year,
        )
        for v in var_grid
    }


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower: float
    upper: float
    samples: np.ndarray


def bootstrap_ci(
    returns: np.ndarray | Sequence[float],
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    block_size: int | None = None,
    method: str = "stationary",
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Percentile bootstrap CI for ``statistic(returns)``.

    For autocorrelated series (typical of daily returns), pass
    ``block_size`` and choose ``method``:

    - ``"stationary"`` (default): Politis-Romano (1994) stationary
      block bootstrap. At each step, with probability ``1/block_size``
      jump to a fresh random index; otherwise advance by one. Block
      lengths are Geometric(1/L); the resampled series itself is
      stationary, which is the property the asymptotic CI math
      assumes.
    - ``"fixed"``: classical circular block bootstrap with non-random
      block length ``block_size``. Faster but the resampled series is
      *not* stationary at the block boundaries.
    - ``"iid"`` (or ``block_size <= 1``): standard non-overlapping
      bootstrap.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        raise ValueError("Need at least two observations to bootstrap.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    if method not in ("stationary", "fixed", "iid"):
        raise ValueError("method must be 'stationary', 'fixed', or 'iid'.")
    rng = rng or np.random.default_rng()

    n = r.size
    samples = np.empty(n_resamples, dtype=float)

    use_block = block_size is not None and block_size > 1 and method != "iid"
    if not use_block:
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            samples[i] = statistic(r[idx])
    elif method == "stationary":
        p = 1.0 / float(block_size)
        for i in range(n_resamples):
            idx = np.empty(n, dtype=np.int64)
            idx[0] = rng.integers(0, n)
            jumps = rng.random(n - 1) < p
            for t in range(1, n):
                idx[t] = rng.integers(0, n) if jumps[t - 1] else (idx[t - 1] + 1) % n
            samples[i] = statistic(r[idx])
    else:  # method == "fixed"
        n_blocks = math.ceil(n / block_size)
        for i in range(n_resamples):
            starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            idx = np.concatenate(
                [np.arange(s, s + block_size) for s in starts]
            )[:n]
            samples[i] = statistic(r[idx])

    point = float(statistic(r))
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return BootstrapResult(point=point, lower=lower, upper=upper, samples=samples)


def holm_correct(p_values: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    """Holm–Bonferroni step-down correction.

    Returns one boolean per p-value: True if the null is rejected at
    family-wise error rate ``alpha``.
    """
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return []
    order = np.argsort(p)
    m = p.size
    rejected = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected.tolist()
