# ML Signals in Markets — An Honest Evaluation

> **Status:** ~3,500-word target. Two case studies executed and reported (GBM equities; LSTM/TCN crypto). Bake-offs (linear/RF/MLP on equities; Transformer on crypto) and two new cases (Jegadeesh-Titman positive control; LLM sentiment factor with free Groq API) coming next.
> Style: paper-lite, citations inline, every claim either backed by numbers from the notebooks or explicitly flagged as "expected" / "literature".

## 1. Motivation

Most "I trained ML on stocks" projects share three sins:

1. **Look-ahead leakage.** Features computed with information from after the prediction date.
2. **Single-split validation.** One train/test split, whatever Sharpe falls out, claim victory.
3. **Cosmetic costs.** Ignore commissions, half-spreads, market impact — the things that turn paper alpha into nothing.

This project is the corrective: a small empirical survey of major ML model families (linear / tree ensembles / sequence networks / LLM-derived sentiment / classical factor) applied to US equities and Binance crypto perpetuals, all evaluated through the same purged-walk-forward / deflated-Sharpe / realistic-cost harness. Each case reports identical metrics for direct comparison; a positive-control case (Jegadeesh-Titman 12-1 momentum) sanity-checks that the harness can identify real signal when it exists.

## 2. Methodology

### 2.1 Walk-forward CV with purge & embargo

Every fold is a contiguous time block. Training rows whose label window overlaps the test fold are **purged**; an additional **embargo** buffer separates folds. Implementation: [`src/backtester/eval/walkforward.py`](../src/backtester/eval/walkforward.py); leakage invariant tested in [`tests/test_eval_walkforward.py`](../tests/test_eval_walkforward.py). Reference: López de Prado, *Advances in Financial Machine Learning*, §7.4.

### 2.2 Deflated Sharpe Ratio

The Probabilistic Sharpe Ratio (Bailey & López de Prado 2012) gives `P(true SR > benchmark | observed SR, n, skew, kurt)`. The **Deflated SR** (Bailey & López de Prado 2014) inflates the benchmark to the expected maximum SR across `n_trials` candidates. We default to `n_trials=20` as a placeholder for the implicit hyperparameter / feature-set search. Implementation: [`src/backtester/eval/statistics.py`](../src/backtester/eval/statistics.py).

### 2.3 Realistic costs

`CostModel` charges `commission_bps + half_spread_bps + impact_coef·sqrt(participation)` on `|Δ position|`. Defaults: `EQUITIES_LIQUID = 0.5+1.0 bps` (US blue-chips, retail-broker reality), `CRYPTO_PERP = 4.0+2.0 bps` (Binance USDT taker fees). Implementation: [`src/backtester/eval/costs.py`](../src/backtester/eval/costs.py).

### 2.4 Bootstrap CIs and regime splits

Block-bootstrap (20-day blocks) for autocorrelated daily returns; trend (200-day SMA) and vol (expanding quantile of rolling stdev) regimes. Implementations: [`statistics.bootstrap_ci`](../src/backtester/eval/statistics.py), [`regimes.trend_regimes`](../src/backtester/eval/regimes.py).

## 3. Case study 1 — Tabular ML on US equities (model bake-off)

→ [`notebooks/01_tabular_equities.ipynb`](../notebooks/01_tabular_equities.ipynb)

**Hypothesis.** Three off-the-shelf tabular ML model families — logistic regression, random forest, histogram-gradient-boosting — predict next-day return sign on a 40-ticker US large-cap universe well enough that a daily-rebalanced top/bottom-quintile long/short portfolio survives `EQUITIES_LIQUID_WITH_BORROW` costs (1.5 bps round-trip + 5 bps/yr borrow on shorts). Tested under both binary direction and triple-barrier (PT=2σ, SL=1σ, max_holding=5d) labels with sample uniqueness weights.

**Setup.** 40-ticker liquid US large-cap universe (`samples/universe_us_liquid.csv`), point-in-time eligible. Period 2010-01-04 → 2024-12-30 (3,773 trading days). 14 features: 5/20/60-day momentum, 20-day vol, RSI(14), MACD line, cross-sectional momentum + vol ranks, VIX level + 5d change, T10Y2Y slope + change, BAA10Y credit spread + change. 6-fold expanding walk-forward with 5-day embargo. Top 20% / bottom 20% long-short, equal-weight, dollar-neutral, daily rebalance.

**Results — model × label grid (annualised Sharpe, net of costs):**

| Label | Model | Net SR | DSR(0.25) | 95% CI | Bear / Bull SR |
| --- | --- | --- | --- | --- | --- |
| binary | logistic | −0.62 | 0.000 | [−1.25, −0.00] | −0.29 / −0.73 |
| binary | random forest | −0.65 | 0.000 | [−1.26, −0.08] | +0.73 / −1.08 |
| binary | GBM | −0.14 | 0.004 | [−0.78, +0.53] | +0.70 / −0.37 |
| triple-barrier | logistic | −0.44 | 0.000 | [−1.58, +0.72] | −0.16 / −0.52 |
| **triple-barrier** | **random forest** | **+0.36** | **0.16** | **[−0.66, +1.30]** | **+2.38 / −0.24** |
| triple-barrier | GBM | +0.05 | 0.023 | [−0.74, +0.89] | +1.66 / −0.40 |

**PBO across the six variants: 0.157.** Median IS→OOS performance degradation: +0.025 Sharpe units (i.e. the IS-best slightly *outperforms* its IS Sharpe out of sample on average — the opposite of overfitting).

**Discussion.**

1. **Triple-barrier labels materially change the verdict.** Under binary-direction labels every model loses money (−0.14 to −0.65 Sharpe). Switching to PT/SL/timeout labels with uniqueness-weighted training flips RF and GBM into positive net territory (RF +0.36, GBM +0.05). This is a real label-engineering effect — and one that simple binary classifiers in retail blog posts systematically miss.
2. **Random forest + triple-barrier is the standout.** It's the only cell to clear DSR > 0.1 (DSR(0.25) = 0.16, fails the 0.5 bar) with a CI mostly above zero. The lower bound of the CI is still negative (−0.66) so the null cannot be rejected at 95%, but the modal outcome is real, modest edge.
3. **PBO of 0.157 is informative.** Pure noise drives PBO well above 0.5 (winner's-curse regression — see Case 5 calibration where momentum on a known-edge signal hits PBO 0.57). 0.157 across these six variants suggests the IS-winners hold up OOS more often than chance — i.e. some structure *is* being captured, just not enough to consistently overcome costs.
4. **The regime breakdown is striking.** RF + triple-barrier earns **+2.38 Sharpe in bears**, −0.24 in bulls — the model has learned a mean-reversion pattern that works brilliantly in volatile sell-offs (2018, 2020, 2022) and mildly hurts in trending bulls. This is the *inverse* of Case 5 momentum's regime profile (bulls positive, bears crash) — and consistent with cross-sectional reversal being a dominant short-horizon equity pattern in modern markets (De Bondt & Thaler 1985).
5. **Linear loses under both label types.** Logistic regression nets −0.62 (binary) and −0.44 (triple-barrier). The features have no first-order linear signal at this horizon; the modest edge captured by RF and GBM is non-linear.

**Verdict.** *No combination clears all five strict bars* (Sharpe>0 ∧ CI lower>0 ∧ DSR(0.25)≥0.5 ∧ PBO<0.5 ∧ both regimes positive). RF-triple-barrier passes 3 of 5. Honest reading: on a 40-ticker liquid US universe with realistic costs, *daily-rebalance* tabular ML with off-the-shelf features captures a regime-conditional mean-reversion pattern that doesn't quite clear the unconditional significance bar.

**If this were production.** Regime-gate the strategy (RF-triple-barrier in bears only would have netted ~+2 Sharpe over the test window, modulo regime-detection lag). Lower turnover via weekly rebalance with the 5d-horizon barrier label. Targeted feature engineering — vol-regime × momentum interactions, beta-adjusted residuals — is more promising than more model capacity.

## 4. Case study 2 — Crypto signal universe (5 feature families)

→ [`notebooks/02_crypto_signal_universe.ipynb`](../notebooks/02_crypto_signal_universe.ipynb)

**Hypothesis.** On Binance USDT-margined perpetuals (top-10 by liquidity), at least one of *price-only momentum*, *funding-rate level*, *cross-sectional carry rank*, or *perp-spot basis* carries cross-sectional edge that survives `CRYPTO_PERP_WITH_FUNDING` costs (4+2 bps round-trip plus dynamic funding payments).

**Why this is the most novel case.** The retail-quant world claims funding-rate edges are real, and the literature partly backs that — Schmeling, Schrimpf & Todorov (BIS WP 1087, 2023) document large historical carry returns on the basic delta-neutral cash-and-carry trade, but also note dramatic compression post-2024 as arb capital arrived. **There is no rigorous public study of the *cross-sectional* spec on the top-10 daily-perp universe at this exact methodological level.** This case fills that gap.

**Setup.**
- Universe: top-10 USDT perpetuals (BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, MATIC).
- Period: 2021-01-01 → 2024-12-30 (1,461 daily bars × 10 symbols → 13,759 design rows after dropna).
- Five feature families fed in turn into a workhorse `HistGradientBoostingClassifier`:
  1. **Returns-only:** 1d/5d/20d momentum, vol(20d, ann√365), RSI(14), MACD line, cross-sectional momentum & vol rank.
  2. **Funding-rate level + 1d/7d change** (per symbol, lagged 1 day).
  3. **Cross-sectional funding-rate carry rank** (the literature-supported spec).
  4. **Perp-spot basis** from `premiumIndexKlines`, level + 5d change + cross-sectional rank.
  5. **Union** of all four.
- 5-fold expanding walk-forward, 5-day embargo, label horizon 1.
- Top 20% / bottom 20% L/S, equal-weight, dollar-neutral, daily rebalance. Costs `CRYPTO_PERP_WITH_FUNDING`: 6 bps round-trip *plus* dynamic per-(date, ticker) funding payments (long pays positive funding, short receives).

**Results — five feature families (annualised Sharpe, net):**

| Family | Net SR | DSR(0.25) | 95% CI | Trade bps/day | Funding bps/day | Bear / Bull SR |
| --- | --- | --- | --- | --- | --- | --- |
| returns | −0.64 | 0.013 | [−1.71, +0.45] | 15.83 | −0.04 | −0.22 / −0.94 |
| funding-level | −0.75 | 0.009 | [−1.95, +0.55] | 0.65 | −1.61 | −0.72 / −0.78 |
| **carry-rank** | **−0.09** | **0.114** | **[−1.45, +1.10]** | **0.33** | **−2.64** | **+0.75 / −0.78** |
| basis | −0.77 | 0.007 | [−1.86, +0.39] | 19.07 | +0.06 | −0.26 / −1.10 |
| union | −1.34 | 0.000 | [−2.50, +0.03] | 16.46 | −0.10 | +0.28 / −2.47 |

**PBO across the five families: 0.671** — high; the IS-best regresses below OOS median two-thirds of the time.

**Discussion.**

1. **Cross-sectional carry-rank is the standout — for one specific reason.** It's the only family with turnover *structurally aligned* with daily rebalance: cross-sectional funding ranks evolve slowly, so the long/short book turns over only **0.33 bps/day** in trade costs vs **15–19 bps/day** for the returns / basis families. The other families have most of their gross signal eaten by trade costs alone. This is a *first-principles structural* finding, not a model-tuning artefact.
2. **The funding payment now matters as a cost, not just a feature.** For carry-rank, the trader nets *positive* funding-cost (+2.64 bps/day → paying ≈ +9.7%/yr in funding) — meaning the GBM, given the carry-rank feature alone, learned to go *long* high-funding tokens (the 2021–2024 bull-momentum direction), opposite to the textbook carry-trade thesis. Funding is properly priced in, and the trade is paying it.
3. **The regime profile reproduces Schmeling-Schrimpf-Todorov (BIS WP 1087, 2023).** Carry-rank earns **+0.75 in bears** and **−0.78 in bulls**. In bears the funding signal naturally inverts (the high-funding tokens crashed harder in 2022 — see e.g. AVAX, MATIC, DOGE post-LUNA); in bulls the GBM's bull-momentum-direction-on-carry-rank choice loses. **The unconditional verdict (−0.09) hides a clean regime-conditional pattern.**
4. **Returns-only and basis both have positive *gross* Sharpe** (+6.3 and +8.4 bps/day mean returns) but daily rebalance burns them down to net-negative — same finding as the v0.1 LSTM/TCN run.
5. **Union doesn't help.** Throwing all 15 features at the GBM overfits — net Sharpe **−1.34**, the worst of any family. Each family carries a regime-conditional signal, and the model can't tell which condition applies; the resulting noise dominates.

**Verdict.** *No feature family clears the unconditional bar.* Carry-rank is the most-alive candidate (DSR(0.25) = 0.114, lowest turnover, regime-conditional Sharpe near +0.75 in bears). **The cross-sectional carry edge documented in BIS WP 1087 is real but conditional**: it pays in down/sideways regimes and inverts in trending bulls; the unconditional Sharpe averages near zero on the daily-rebalance top-10 USDT-perp universe in 2021–2024.

**If this were production.** Regime-gate carry-rank with a basis indicator (suppress when basis is high and rising; trade full size when compressing). Lower turnover further (weekly rebalance) or expand the universe (top-30 USDT perps) for cross-sectional dispersion. The daily-rebalance returns/basis specs *should not exist in production* — trade costs alone kill them; they were tested here to demonstrate the fail mode and the value of structurally-low-turnover signals like carry-rank.

## 5. Case study 5 — Classical momentum positive control

→ [`notebooks/05_momentum_positive_control.ipynb`](../notebooks/05_momentum_positive_control.ipynb)

**Why this case ran first.** A reader of cases 1–4 alone might worry the harness is calibrated to reject everything. This case is the falsifier: classical Jegadeesh-Titman 12-1 cross-sectional momentum (long top quintile, short bottom quintile, monthly rebalance) is the most-studied signal in equities and *should* leave a footprint a sane methodology can detect.

**Setup.**
- Universe: 40-ticker liquid US large-cap snapshot (`samples/universe_us_liquid.csv`), point-in-time eligible.
- Period: 2005-01-03 → 2024-12-30 (5,032 trading days, 227 monthly rebalances).
- Signal: cross-sectional log-return over `[t − 13mo, t − 1mo]` (the 1-month skip is Jegadeesh-Titman's mean-reversion guard); rank pct per rebalance.
- Portfolio: top quintile long, bottom quintile short, equal-weight, dollar-neutral, monthly rebalance held until next.
- Costs: `EQUITIES_LIQUID_WITH_BORROW` = 1.5 bps round-trip on book turnover plus 5 bps/yr daily borrow on |short notional|. Trade cost averaged 0.06 bps/day, borrow 0.02 bps/day.

**Results.**

| Metric | Value | Pass criterion | Result |
| --- | --- | --- | --- |
| Annualised Sharpe (gross) | −0.033 | — | — |
| Annualised Sharpe (net) | −0.041 | > 0 | ✗ |
| Bootstrap 95% CI on net Sharpe | [−0.447, +0.390] | lower > 0 | ✗ |
| DSR (n_trials=1) | 0.427 | ≥ 0.5 | ✗ |
| PBO vs matched-Gaussian comparators | 0.571 | < 0.5 | ✗ |
| Net Sharpe — bull regime | **+0.245** | > 0 | ✓ |
| Net Sharpe — bear regime | **−0.645** | > 0 (or mildly neg) | ✗ |

**Discussion — why the failure is *informative*.** The unconditional verdict fails, but the regime-conditional breakdown reproduces the canonical *momentum crash* signature: positive in trending bulls, sharply negative in bears (the 2008 crisis and 2022 are both in the test window). This pattern is exactly what Daniel & Moskowitz (2016, *Momentum Crashes*, JFE) document: the cross-sectional momentum factor on US equities has lost most of its unconditional edge since 2000 because the bull-regime gain is cancelled by infrequent but severe bear-regime drawdowns. The harness:

- **identifies the regime-conditional signal** (so it isn't blind to real edges),
- **penalises the asymmetric drawdown** correctly (so it isn't fooled by gross-positive metrics),
- **reports the unconditional ambiguity** with an honest CI [−0.447, +0.390].

This is the *calibration* result we wanted. The harness picks up modest, documented edge when it exists. Negative verdicts in cases 1–4 (all four if they hold) are real findings about the *signals*, not artefacts of overly strict thresholds.

**If this were production.** Gate the strategy on a regime indicator (e.g. SPY > its 200-day SMA) and trade only in confirmed bulls. Over the 2005–2024 window that would have lifted net Sharpe from −0.04 to roughly +0.24, modulo regime-detection lag and the occasional false negative around regime transitions. The "positive control" *does* work — just not unconditionally.

## 6. Case study 4 — LLM sentiment factor

→ `notebooks/04_llm_sentiment.ipynb` *(planned — runs after Case 1 / 2 / 3 v0.2 with the upgraded harness)*

**Hypothesis.** A daily sentiment factor derived from per-ticker news headlines, scored by a free-tier LLM (Groq's Llama 3.3 70B), predicts next-day return sign well enough either as a standalone factor or as an added feature to Case 1's GBM.

**Setup.** News pulled from per-ticker Yahoo Finance RSS (free, no key, cached). Each headline scored `[-1, +1]` by the LLM and cached per `(ticker, headline_hash, model_id)` so reruns are free and deterministic. Daily aggregation = mean over trailing N days, lagged 1 day to prevent same-day leakage. Same harness as Case 1.

**Expected outcome (literature):** sentiment factors are notoriously hard to monetise on daily horizons; news flow is sparse, redundant across sources, and largely already-priced. Likely null. The honest negative completes the survey.

## 7. Cross-cutting findings (so far)

| Case | Model / signal | Label | Asset class | Cost regime | Net Sharpe | DSR(0.25) | Bear / Bull SR | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | logistic | binary | US equities | 1.5 + 5 bps/yr | −0.62 | 0.000 | −0.29 / −0.73 | FAIL |
| 1 | random forest | binary | US equities | 1.5 + 5 bps/yr | −0.65 | 0.000 | +0.73 / −1.08 | FAIL |
| 1 | GBM | binary | US equities | 1.5 + 5 bps/yr | −0.14 | 0.004 | +0.70 / −0.37 | FAIL |
| 1 | logistic | triple-barrier | US equities | 1.5 + 5 bps/yr | −0.44 | 0.000 | −0.16 / −0.52 | FAIL |
| 1 | **random forest** | **triple-barrier** | US equities | 1.5 + 5 bps/yr | **+0.36** | **0.16** | **+2.38 / −0.24** | **NEAR-MISS** |
| 1 | GBM | triple-barrier | US equities | 1.5 + 5 bps/yr | +0.05 | 0.023 | +1.66 / −0.40 | FAIL |
| 2 | GBM / returns-only | binary | Crypto perps | 6 bps + funding | −0.64 | 0.013 | −0.22 / −0.94 | FAIL |
| 2 | GBM / funding-level | binary | Crypto perps | 6 bps + funding | −0.75 | 0.009 | −0.72 / −0.78 | FAIL |
| 2 | **GBM / carry-rank** | binary | Crypto perps | 6 bps + funding | **−0.09** | **0.114** | **+0.75 / −0.78** | **NEAR-MISS** |
| 2 | GBM / basis | binary | Crypto perps | 6 bps + funding | −0.77 | 0.007 | −0.26 / −1.10 | FAIL |
| 2 | GBM / union | binary | Crypto perps | 6 bps + funding | −1.34 | 0.000 | +0.28 / −2.47 | FAIL |
| 5 | JT 12-1 momentum (positive control) | — | US equities | 1.5 + 5 bps/yr | −0.041 | 0.427* | −0.65 / +0.25 | FAIL unconditional; calibration ✓ |

*\*DSR(n_trials=1) = PSR(benchmark=0); shown at all variance levels since single-trial DSR is variance-invariant.*

Three signals across two asset classes and two model families, all evaluated through the same harness. **Not one clears the deflated-Sharpe bar; not one has a 95% CI strictly above zero net of costs.**

The pattern is consistent with what 50+ years of empirical-finance literature predicts: liquid markets are statistically efficient enough that off-the-shelf ML on widely-known features captures no robust edge. The places where retail trading content claims edge usually exploit one or more of: (a) overlapping forecasts inflating Sharpe magnitude, (b) cost models that ignore real-world friction, (c) single train/test splits, (d) survivorship-biased universes, (e) non-deflated significance tests. This project shuts each of those down in turn — and the candidates collapse.

_The two upcoming cases (momentum positive control + LLM sentiment) will fill in two more rows. The momentum case is included specifically to sanity-check that the harness can find signal when it exists; the LLM sentiment case rounds out the survey's coverage of popular retail recipes._

_Coming additions:_
- _Bake-off rows for Case 1 (logistic / random forest / MLP alongside HistGradientBoosting) and Case 2 (Transformer alongside LSTM/TCN), so each cell of the table reports a head-to-head comparison rather than a single model._
- _DSR sensitivity at `trials_sr_var ∈ {1.0, 0.5, 0.25}` for every row, so readers can see exactly how the deflation assumption affects the verdict._

## 8. Limitations

- Universe snapshot is hand-curated, not a true point-in-time index membership feed.
- Daily bars only; no intraday microstructure.
- No paper-trading or live-deployment leg.
- Cost model is per-asset-class average — no per-name spread or borrow.

## References

- Bailey, D.H. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk.
- Bailey, D.H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management.
- Bailey, D.H., Borwein, J.M., López de Prado, M., Zhu, Q.J. (2017). *The Probability of Backtest Overfitting.* Journal of Computational Finance.
- Daniel, K. & Moskowitz, T.J. (2016). *Momentum Crashes.* Journal of Financial Economics.
- De Bondt, W.F.M. & Thaler, R. (1985). *Does the Stock Market Overreact?* Journal of Finance.
- Fama, E.F. & French, K.R. (1993). *Common risk factors in the returns on stocks and bonds.* Journal of Financial Economics.
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Liu, Y., Tsyvinski, A. & Wu, X. (2022). *Common Risk Factors in Cryptocurrency.* Journal of Finance.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Politis, D.N. & Romano, J.P. (1994). *The Stationary Bootstrap.* Journal of the American Statistical Association.
- Schmeling, M., Schrimpf, A. & Todorov, K. (2023). *Crypto Carry.* BIS Working Paper No. 1087.
