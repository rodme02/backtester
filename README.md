# ML Signals in Markets — An Honest Evaluation

[![CI](https://github.com/rodme02/backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/rodme02/backtester/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> An in-depth empirical study of ML techniques applied to US equities and crypto. Every major model family — linear, tree ensembles, sequence models, LLM-derived sentiment, classical factor — tested through the same purged-walk-forward / deflated-Sharpe / realistic-cost harness, with strong pragmatic discussion of why each result is what it is.

## Why this exists

Most "I trained a model to beat the market" projects share three sins:

1. **Look-ahead leakage.** Features computed with information from after the prediction date.
2. **Single-split validation.** Train/test once, report whatever Sharpe came out, claim victory.
3. **Cosmetic costs.** Ignore commissions, half-spreads, and market impact — the things that turn paper alpha into nothing.

This repo is the corrective. Every signal is evaluated through the same harness:

- **Purged & embargoed walk-forward CV** ([Lopez de Prado, AFML §7](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)) — no label leakage between folds.
- **Deflated Sharpe Ratio** ([Bailey & Lopez de Prado 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)) reported with sensitivity over the trial-variance assumption — transparent about how aggressively we're deflating.
- **Bootstrap confidence intervals** on annualised return, Sharpe, and drawdown — point estimates without intervals are vibes.
- **Asset-class-appropriate cost models** — bps-level commission + half-spread + sqrt-impact, calibrated separately for liquid US equities and Binance USDT perpetuals.
- **Per-regime breakdown** — bull/bear, high-vol/low-vol, so a strategy that only "works in 2017" is exposed.
- **Holm–Bonferroni** family-wise correction when comparing models.

The conclusion most signals deserve is "the data doesn't support the claim." This project's framing tells that story honestly — and runs a known positive-control signal (Jegadeesh-Titman 12-1 momentum) through the same harness to prove the methodology can identify edge when it exists.

## Planned scope — comprehensive empirical survey

The repo is being built into a small empirical study spanning model families × asset classes:

| Case | Asset class | Models compared | Status |
| --- | --- | --- | --- |
| 1 | US equities | Logistic regression · Random forest · HistGradientBoosting · (optional) MLP | GBM done; bake-off in progress |
| 2 | Binance USDT perps | LSTM · TCN · Transformer | LSTM/TCN done; Transformer in progress |
| 3 | US equities | Jegadeesh-Titman 12-1 momentum (positive control) | planned |
| 4 | News-covered tickers | LLM-derived sentiment factor (Groq free tier) | planned |

Each case study reports in the same shape: hypothesis → setup → results table → pragmatic discussion → "if this were production." The cross-cutting writeup at [`docs/writeup.md`](docs/writeup.md) ties everything together as a paper-lite (~3,500 words target).

## Status (current)

| Case | Model | Asset | Net Sharpe | Deflated SR | Verdict |
| --- | --- | --- | --- | --- | --- |
| ✅ 1 | HistGradientBoosting | US equities | −0.428 | 0.000 | **FAIL** |
| ✅ 2a | LSTM | Crypto perps | −1.348 | 0.000 | **FAIL** |
| ✅ 2b | TCN | Crypto perps | +0.138 | 0.001 | **FAIL** |
| ⏳ 1' | Logistic + Random forest | US equities | — | — | next |
| ⏳ 2c | Transformer | Crypto perps | — | — | next |
| ⏳ 3 | Jegadeesh-Titman momentum (positive control) | US equities | — | — | next |
| ⏳ 4 | LLM sentiment factor (Groq free) | US equities | — | — | next |

1. ✅ [`notebooks/01_gbm_us_equities.ipynb`](notebooks/01_gbm_us_equities.ipynb) — GBM on technical + macro + cross-sectional features. Off-the-shelf recipe earns no edge; bull/bear Sharpe split (−0.80 / +0.80) suggests the model picked up a short-term mean-reversion pattern that inverts in trending markets.
2. ✅ [`notebooks/02_sequence_models_crypto.ipynb`](notebooks/02_sequence_models_crypto.ipynb) — LSTM and TCN side by side on Binance USDT perpetuals. LSTM is catastrophic out of sample; TCN looks gross-positive but drowns in 6 bps round-trip costs and the net 95% CI [−0.98, +1.21] straddles zero.
3. ⏳ Coming: 4-model bake-off on Case 1 (linear / RF / GBM / MLP), Transformer added to Case 2, Jegadeesh-Titman positive control, LLM sentiment factor.

Full discussion of every case in [`docs/writeup.md`](docs/writeup.md).

## Quickstart

```bash
git clone https://github.com/rodme02/backtester.git
cd backtester
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks,ml]"
pytest
jupyter notebook notebooks/01_gbm_us_equities.ipynb
```

For live data, copy `.env.example` to `.env` and set:

- `FRED_API_KEY` — free at [fred.stlouisfed.org](https://fred.stlouisfed.org/) (macro features)
- `GROQ_API_KEY` — free tier at [groq.com](https://groq.com/) (LLM sentiment, Case 4)
- `OLLAMA_HOST` — optional, for local-LLM fallback on Apple Silicon / Linux

Yahoo Finance and Binance public endpoints need no key.

## Repo layout

```
src/backtester/
  data/         # csv_loader, yfinance, fred, binance, universe, news, llm
  eval/         # walkforward, statistics (with DSR sensitivity), costs, regimes
  features/     # technical, macro, cross_sectional, crypto, sentiment
  models/       # linear, random_forest, gbm, sequence (LSTM/TCN/Transformer)
  portfolio/    # cross-sectional long/short construction + book costs
notebooks/      # 4 case studies (see notebooks/README.md)
docs/writeup.md # paper-lite empirical study (~3,500 words target)
samples/        # bundled OHLCV CSVs + universe snapshot + CI fixtures
tests/          # eval harness, leakage invariants, portfolio, models, data
```

## Tech stack

| Layer       | Choice                                          | Why                                                                                           |
| ----------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Eval harness| numpy, pandas, scipy                            | Pure-Python statistics — readable, auditable, easy for a reviewer to inspect                  |
| Data        | yfinance, FRED, Binance public, Yahoo RSS      | All free; cached on disk; CI runs on bundled fixtures                                          |
| Tabular ML  | scikit-learn (linear, RF, HistGradientBoosting) | Boring, well-understood; project value is in the eval, not model bling                         |
| Sequence ML | PyTorch (LSTM, TCN, Transformer; CPU-only)      | Industry-standard; matched-parameter architectures for a fair head-to-head                     |
| LLM         | Groq free tier (Llama 3.3 70B); Ollama fallback | Free, fast, reproducible; local fallback for offline development                              |
| Tests / lint| pytest + ruff + pre-commit                      | Fast, opinionated, automated                                                                   |
| CI          | GitHub Actions                                  | Lints, tests, *and executes notebooks* (against bundled fixtures) on every push                |

## Honest limitations

- Universe snapshot (`samples/universe_us_liquid.csv`) is hand-curated, not a true point-in-time index-membership feed. Documented as such; replace with a paid source for production research.
- Daily bars only — no intraday microstructure.
- Cost model is per-asset-class average, not per-name spread/borrow.
- LLM sentiment uses a free-tier model — production research would benchmark against more powerful LLMs.

## License

MIT — see [LICENSE](LICENSE).
