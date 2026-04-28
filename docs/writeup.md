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

**Hypothesis.** A `HistGradientBoostingClassifier` on technical + macro + cross-sectional features predicts next-day return sign well enough that a daily-rebalanced top/bottom-quintile long/short portfolio survives 1.5 bps round-trip costs.

**Setup.**
- Universe: 40 liquid US large-caps from `samples/universe_us_liquid.csv`, point-in-time eligibility.
- Period: 2010-01-04 → 2024-12-30 (15 years, 3,773 trading days).
- Features: 5/20/60-day momentum, 20-day vol, RSI(14), MACD line, cross-sectional momentum & vol ranks, VIX level + 5d change, T10Y2Y slope + change, BAA10Y credit spread + change. All leakage-tested.
- Label: 1-day forward return sign.
- Walk-forward: 6 expanding-window folds, 5-day embargo, label-horizon purge. Each fold trains on 18k–111k rows, tests on ~18.5k rows.
- Portfolio: top 20% / bottom 20% of probabilities, equal-weight, dollar-neutral, daily rebalance. Costs `EQUITIES_LIQUID` = 1.5 bps round-trip on book turnover.

**Results.**

| Metric | Value |
| --- | --- |
| Annualised Sharpe (gross) | +0.022 |
| **Annualised Sharpe (net)** | **−0.428** |
| **Deflated SR (n_trials = 20)** | **0.000** |
| Bootstrap 95% CI on Sharpe | [−1.030, +0.216] |
| Approx. annualised return (net) | −7.20% |
| Net Sharpe — bear regime | +0.799 |
| Net Sharpe — bull regime | −0.795 |

**Verdict: fails.** The strategy doesn't clear the deflated-Sharpe threshold by any margin (DSR ≈ 0). The 95% CI on Sharpe straddles zero comfortably. Net of costs the strategy *loses* ~7%/yr.

**Discussion.**

- The gross Sharpe is essentially zero. Costs are not the killer here — there is no signal to begin with.
- The bull/bear asymmetry is the most interesting artefact: net Sharpe is +0.80 in bears, −0.80 in bulls. The model trained mostly on the post-GFC bull market is picking up a short-term mean-reversion pattern that quietly works when markets fall and inverts when they rise. The unconditional Sharpe averages those out to nothing.
- This is the textbook null result: off-the-shelf technical + macro features, a strong off-the-shelf classifier, and an honest evaluation harness produce the most common outcome in retail quant research — *no edge*. Most blog posts get a positive Sharpe here because they (a) overlap forecasts to inflate the magnitude, (b) skip costs, and/or (c) report a single train/test split. Doing it properly removes the illusion.

## 4. Case study 2 — sequence models on Binance USDT perpetuals

→ [`notebooks/02_sequence_models_crypto.ipynb`](../notebooks/02_sequence_models_crypto.ipynb)

**Hypothesis.** A deep sequence model — LSTM (the canonical retail-blog choice) or TCN (dilated causal convolutions; the literature pick) — on a 30-day window of returns + funding-rate features predicts next-day direction on the top-10 USDT-perp pairs well enough that a daily-rebalanced top/bottom-quintile long/short portfolio survives the much higher crypto cost regime (`CRYPTO_PERP` ≈ 6 bps round-trip).

**Setup.**
- Universe: top-10 USDT perpetuals by liquidity (BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, MATIC).
- Period: 2021-01-01 → 2024-12-30 (1,461 daily bars per symbol).
- Features per (date, ticker): 1d/5d/20d momentum, 20-day vol (annualised by √365), RSI(14), MACD line, cross-sectional momentum & vol ranks, funding-rate level + 1d/7d change. **All leakage-tested.**
- Sequence length: 30 days × 11 features = 330 inputs per window; 13,469 (date, ticker) rows after dropping warm-up.
- Walk-forward: 5 expanding folds, label horizon 1, 5-day embargo. Each model trained 5 times on 2.3k–11.4k rows.
- Models: matched architectures (1 LSTM layer / 3 TCN dilation-1-2-4 blocks; 32 hidden units, 6 epochs, batch 1024, Adam @ 1e-3, seed 17).
- Portfolio: top 20% / bottom 20% of probabilities, equal-weight, dollar-neutral. Costs `CRYPTO_PERP` = 6 bps round-trip on book turnover.

**Results.**

| Metric | LSTM | TCN |
| --- | --- | --- |
| Annualised Sharpe (gross) | −1.095 | +0.628 |
| **Annualised Sharpe (net)** | **−1.348** | **+0.138** |
| **Deflated SR (n_trials = 20)** | **0.000** | **0.001** |
| Bootstrap 95% CI on Sharpe | [−2.63, −0.04] | [−0.98, +1.21] |
| Approx. annualised return (net) | −47.97% | +7.18% |
| **Verdict** | **FAIL** | **FAIL** |

**Discussion.**

- **LSTM is catastrophic.** Gross Sharpe is already negative; the LSTM has actively learned to pick the *wrong* names (it loses money before costs even arrive). 47% annualised loss on the long/short book over 3+ OOS years is not noise — that's a model that found a feature/label pattern that consistently inverts out of sample. Most likely failure mode: the recurrent unit overfits to recent regime-specific patterns (the 2021 alt-coin frenzy was in training data; the 2022 collapse was the first OOS fold) and never recovers.
- **TCN looks "promising" gross, then drowns in costs.** Gross Sharpe of +0.63 might tempt an unrigorous reviewer to declare victory. After 6 bps round-trip on a daily-rebalanced book it's +0.14. The 95% CI on the *net* Sharpe is [−0.98, +1.21] — zero is right in the middle. Deflated Sharpe ≈ 0 confirms: we cannot reject the null that this is no better than the best of 20 random models. The "edge" is imaginary.
- **TCN > LSTM** by a wide margin in *gross* terms, which says something real about the architectures even if neither wins after costs. The TCN's deterministic dilated kernels seem to handle the highly non-stationary crypto signal better than the LSTM's recurrent dynamics.
- **The cost gap is the story.** Equity case 1 paid 1.5 bps and lost. Crypto case 2 pays 6 bps. Even when there's a hint of gross signal (TCN), 4× the friction kills it. **This is the central crypto-trading reality that retail content elides:** funding + spread + commission on USDT perps eats anything short of a strong, low-turnover signal.

## 5. Case study 3 — LLM sentiment factor

→ `notebooks/03_llm_sentiment.ipynb` *(week 4)*

## 6. Cross-cutting findings (so far)

| Case | Model | Asset class | Cost regime | Net Sharpe | Deflated SR | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | HistGradientBoosting | US equities | 1.5 bps | −0.428 | 0.000 | FAIL |
| 2a | LSTM | Crypto perps | 6 bps | −1.348 | 0.000 | FAIL |
| 2b | TCN | Crypto perps | 6 bps | +0.138 | 0.001 | FAIL |

Three signals across two asset classes and two model families, all evaluated through the same harness. **Not one clears the deflated-Sharpe bar; not one has a 95% CI strictly above zero net of costs.**

The pattern is consistent with what 50+ years of empirical-finance literature predicts: liquid markets are statistically efficient enough that off-the-shelf ML on widely-known features captures no robust edge. The places where retail trading content claims edge usually exploit one or more of: (a) overlapping forecasts inflating Sharpe magnitude, (b) cost models that ignore real-world friction, (c) single train/test splits, (d) survivorship-biased universes, (e) non-deflated significance tests. This project shuts each of those down in turn — and the candidates collapse.

_Case study 3 (LLM sentiment factor) follows. The expectation now is the same null result; the value is in shutting down one more popular hypothesis under matched conditions._

## 7. Limitations

- Universe snapshot is hand-curated, not a true point-in-time index membership feed.
- Daily bars only; no intraday microstructure.
- No paper-trading or live-deployment leg.
- Cost model is per-asset-class average — no per-name spread or borrow.

## References

- Bailey, D.H. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk.
- Bailey, D.H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
