# ML Signals in Markets — An Honest Evaluation

[![CI](https://github.com/rodme02/backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/rodme02/backtester/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> An ML engineer's honest study of popular ML-driven trading recipes — gradient-boosted classifiers, sequence models, LLM-driven sentiment — evaluated with the rigour most blog posts skip. Tested across US equities and crypto.

## Why this exists

Most "I trained a model to beat the market" projects share three sins:

1. **Look-ahead leakage.** Features computed with information from after the prediction date.
2. **Single-split validation.** Train/test once, report whatever Sharpe came out, claim victory.
3. **Cosmetic costs.** Ignore commissions, half-spreads, and market impact — the things that turn paper alpha into nothing.

This repo is the corrective. Every signal is evaluated through the same harness:

- **Purged & embargoed walk-forward CV** ([Lopez de Prado, AFML §7](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)) — no label leakage between folds.
- **Deflated Sharpe Ratio** ([Bailey & Lopez de Prado 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)) — the multiple-trial-aware significance test that rejects most lucky strategies.
- **Bootstrap confidence intervals** on annualised return, Sharpe, and drawdown — point estimates without intervals are vibes.
- **Asset-class-appropriate cost models** — bps-level commission + half-spread + sqrt-impact, calibrated separately for liquid US equities and Binance USDT perpetuals.
- **Per-regime breakdown** — bull/bear, high-vol/low-vol, so a strategy that only "works in 2017" is exposed.
- **Holm–Bonferroni** family-wise correction when comparing models.

The conclusion most signals deserve is "the data doesn't support the claim." This project's framing tells that story honestly.

## Status

**🚧 Week 1 of ~4 done.** The honest-evaluation harness is built and tested:

- `eval/walkforward.py` — purged + embargoed CV
- `eval/statistics.py` — PSR, deflated Sharpe, bootstrap, Holm
- `eval/costs.py` — per-asset cost models
- `eval/regimes.py` — trend & vol regime tagging
- `data/{yfinance,fred,binance,universe,alpha_vantage,csv_loader}.py` — cached, point-in-time aware data layer

**Case studies (in progress):**

1. ✅ `notebooks/01_gbm_us_equities.ipynb` — gradient-boosted classifier on technical + macro + cross-sectional features. Built-in walk-forward CV, deflated-Sharpe & bootstrap-CI evaluation, per-regime breakdown vs SPY benchmark.
2. ⏳ `notebooks/02_lstm_crypto.ipynb` — sequence model on Binance USDT perpetuals (week 3).
3. ⏳ `notebooks/03_llm_sentiment.ipynb` — LLM-driven sentiment factor on news headlines (week 4).

…each evaluated through the harness above. The end-of-month deliverable is a long-form writeup at `docs/writeup.md` summarising what survived honest evaluation and what didn't.

## Quickstart

```bash
git clone https://github.com/rodme02/backtester.git
cd backtester
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
pytest
jupyter notebook notebooks/01_gbm_us_equities.ipynb
```

The legacy backtrader engine and Streamlit dashboard remain in the repo as a baseline-strategy playground:

```bash
backtester run --strategy ma_crossover --symbol AAPL
streamlit run dashboard/app.py
```

For live data: copy `.env.example` to `.env`, set `ALPHA_VANTAGE_API_KEY` and/or `FRED_API_KEY`. Yahoo Finance and Binance public endpoints need no key.

## Repo layout

```
src/backtester/
  data/         # csv_loader, alpha_vantage, yfinance, fred, binance, universe
  eval/         # walkforward, statistics, costs, regimes      ← rigour layer
  features/     # (week 2+) leakage-free feature builders
  models/       # (week 2+) GBM, sequence, sentiment-factor wrappers
  strategies/   # backtrader baseline strategies (MA crossover, advanced trend)
  engine/       # backtrader Cerebro wiring used as the realistic simulator
  reporting/    # Sharpe / drawdown / win-rate / equity curve
notebooks/      # (week 2+) one per case study
docs/writeup.md # (week 4) long-form artifact
samples/        # bundled OHLCV CSVs + universe snapshot
tests/          # 32 tests covering the eval harness, data loaders, engine
```

## Tech stack

| Layer       | Choice                                | Why                                                                |
| ----------- | ------------------------------------- | ------------------------------------------------------------------ |
| Eval harness| numpy, pandas, scipy                  | Pure-Python statistics — readable, auditable                       |
| Data        | yfinance, FRED, Binance public, AV    | All free; cached on disk to keep iteration fast and quotas safe    |
| Simulator   | backtrader                            | Mature event-driven engine with bracket orders                     |
| Tests / lint| pytest + ruff                         | Standard, fast, opinionated                                        |
| (Week 2+)   | scikit-learn, lightgbm, torch         | Boring, well-understood; the project's value is in the eval, not the model bling |

## Honest limitations

- Universe snapshot (`samples/universe_us_liquid.csv`) is hand-curated, not a true point-in-time index membership feed. Documented as such; replace with a paid source for production research.
- Daily bars only — no intraday yet.
- The backtrader engine and the eval harness are wired through a returns-series interface; they are not yet *unified* into one runner. That's intentional during the research phase.

## License

MIT — see [LICENSE](LICENSE).
