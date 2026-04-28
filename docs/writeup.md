# ML Signals in Markets — An Honest Evaluation

> **Status:** scaffolding. Filled in as the case studies finish.
> Target: ≥3000 words, written like a paper-lite, links to each notebook.

## 1. Motivation

Most "I trained ML on stocks" projects share three sins:

1. **Look-ahead leakage.** Features computed with information from after the prediction date.
2. **Single-split validation.** One train/test split, whatever Sharpe falls out, claim victory.
3. **Cosmetic costs.** Ignore commissions, half-spreads, market impact — the things that turn paper alpha into nothing.

This project is the corrective: three ML-driven signals (gradient-boosted classifier, sequence model, LLM sentiment factor) put through the same honest evaluation harness.

## 2. Methodology

### 2.1 Walk-forward CV with purge & embargo

Every fold is a contiguous time block. Training rows whose label window overlaps the test fold are **purged**; an additional **embargo** buffer separates folds. Implementation: [`src/backtester/eval/walkforward.py`](../src/backtester/eval/walkforward.py); leakage invariant tested in [`tests/test_eval_walkforward.py`](../tests/test_eval_walkforward.py). Reference: López de Prado, *Advances in Financial Machine Learning*, §7.4.

### 2.2 Deflated Sharpe Ratio

The Probabilistic Sharpe Ratio (Bailey & López de Prado 2012) gives `P(true SR > benchmark | observed SR, n, skew, kurt)`. The **Deflated SR** (Bailey & López de Prado 2014) inflates the benchmark to the expected maximum SR across `n_trials` candidates. We default to `n_trials=20` as a placeholder for the implicit hyperparameter / feature-set search. Implementation: [`src/backtester/eval/statistics.py`](../src/backtester/eval/statistics.py).

### 2.3 Realistic costs

`CostModel` charges `commission_bps + half_spread_bps + impact_coef·sqrt(participation)` on `|Δ position|`. Defaults: `EQUITIES_LIQUID = 0.5+1.0 bps` (US blue-chips, retail-broker reality), `CRYPTO_PERP = 4.0+2.0 bps` (Binance USDT taker fees). Implementation: [`src/backtester/eval/costs.py`](../src/backtester/eval/costs.py).

### 2.4 Bootstrap CIs and regime splits

Block-bootstrap (20-day blocks) for autocorrelated daily returns; trend (200-day SMA) and vol (expanding quantile of rolling stdev) regimes. Implementations: [`statistics.bootstrap_ci`](../src/backtester/eval/statistics.py), [`regimes.trend_regimes`](../src/backtester/eval/regimes.py).

## 3. Case study 1 — GBM on US equities

→ [`notebooks/01_gbm_us_equities.ipynb`](../notebooks/01_gbm_us_equities.ipynb)

**Hypothesis.** GBM on technical + macro + cross-sectional features predicts next-day return sign and a daily-rebalanced top/bottom-quintile long/short portfolio survives 1.5 bps round-trip costs.

**Setup.** _(fill in: universe size after yfinance pulls, design-matrix rows, fold dates, train/test sizes)_

**Results.** _(fill in: gross Sharpe, net Sharpe, deflated SR verdict, bootstrap CI, per-regime Sharpe, equity-curve summary)_

**Discussion.** _(fill in: which features had highest permutation importance; behaviour in 2018, 2020, 2022 drawdowns; what failure mode (if any) explains the result)_

## 4. Case study 2 — sequence model on crypto

→ `notebooks/02_lstm_crypto.ipynb` *(week 3)*

## 5. Case study 3 — LLM sentiment factor

→ `notebooks/03_llm_sentiment.ipynb` *(week 4)*

## 6. Cross-cutting findings

_To be written after all three case studies finish. Expected outline:_

- Which evaluation pitfalls bit which model.
- Whether the leakage/cost adjustments killed all candidates equally or selectively.
- What this says about retail ML-trading content more broadly.

## 7. Limitations

- Universe snapshot is hand-curated, not a true point-in-time index membership feed.
- Daily bars only; no intraday microstructure.
- No paper-trading or live-deployment leg.
- Cost model is per-asset-class average — no per-name spread or borrow.

## References

- Bailey, D.H. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk.
- Bailey, D.H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
