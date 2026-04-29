# Notebooks

Each notebook is a self-contained case study evaluating one or more ML-driven trading signals through the shared harness in `src/backtester/eval/`. Together they form a small empirical survey across model families and asset classes.

| Notebook | Status | Signals compared | Asset class | Cost regime |
| --- | --- | --- | --- | --- |
| `01_gbm_us_equities.ipynb` | 🟡 GBM done; bake-off in progress | Logistic regression · Random forest · HistGradientBoosting · (optional) MLP | US equities | `EQUITIES_LIQUID` (1.5 bps) |
| `02_sequence_models_crypto.ipynb` | 🟡 LSTM/TCN done; Transformer in progress | LSTM · TCN · Transformer | Binance USDT perps | `CRYPTO_PERP` (6 bps) |
| `03_momentum_positive_control.ipynb` | ⏳ planned | Jegadeesh-Titman 12-1 momentum factor | US equities | `EQUITIES_LIQUID` |
| `04_llm_sentiment.ipynb` | ⏳ planned | LLM-derived sentiment factor (Groq free tier) | News-covered tickers | `EQUITIES_LIQUID` |

The momentum case (notebook 03) is a **positive control** — a classical signal with a documented historical edge — included to prove the harness can identify real signal when it exists.

## Running locally

```bash
pip install -e ".[dev,notebooks,ml,llm]"  # ml = torch; llm = groq + feedparser
cp .env.example .env
# Set FRED_API_KEY (free, https://fred.stlouisfed.org/) and GROQ_API_KEY (free, https://groq.com/)
# Optional: OLLAMA_HOST for local-LLM fallback
jupyter notebook notebooks/01_gbm_us_equities.ipynb
```

`data_cache/` (gitignored) holds yfinance / FRED / Binance / news / LLM pulls keyed by date — same-day re-runs are offline.

For quick smoke tests without live API calls:

```bash
BACKTESTER_FIXTURE_MODE=1 jupyter nbconvert --to notebook --execute notebooks/01_gbm_us_equities.ipynb
```

This is what CI uses to validate the notebooks on every push.

## Reading the verdicts

Each notebook reports the same metrics for every model:

- **Annualised Sharpe — gross and net of costs.** The cost gap usually tells the story.
- **Deflated Sharpe Ratio (DSR), with sensitivity.** Probability the true Sharpe exceeds the *expected best of `n_trials`* candidates, reported across `trials_sr_var ∈ {1.0, 0.5, 0.25}` to be transparent about the multiple-trial assumption. **DSR ≥ 0.95** = clears the bar. Anything less means we cannot reject the null.
- **Bootstrap 95% CI on Sharpe** (20-day block resample). **Strictly above zero** = the point estimate isn't a fluke.
- **Per-regime breakdown** (bull/bear via 200d SMA on the benchmark). A signal that only earns in one regime hasn't generalised.
- **Pragmatic discussion + "if this were production"** — what would we change to give this signal a fair shot in real trading?

A negative result is still a result. Most ML signals fail under honest evaluation, and that finding *is* the contribution. The positive-control case in notebook 03 demonstrates the methodology can identify real (modest, well-documented) edge when it exists.
