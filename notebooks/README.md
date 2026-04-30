# Notebooks

Each notebook is a self-contained case study evaluating one or more ML-driven trading signals through the shared advanced harness in `src/backtester/eval/` and `src/backtester/labels/`. Together they form a small empirical survey across model families, asset classes, and signal types.

| Notebook | Status | Question | Asset class | Cost regime |
| --- | --- | --- | --- | --- |
| `01_tabular_equities.ipynb` | ✅ done — RF/triple-barrier is the lone NEAR-MISS (DSR ≈ 0.16, +2.38 bear SR) | Off-the-shelf tabular ML on equities — which family fails least? | US large-caps | `EQUITIES_LIQUID_WITH_BORROW` |
| `02_crypto_signal_universe.ipynb` | ✅ done — GBM/carry-rank is the NEAR-MISS (DSR ≈ 0.11, +0.75 bear SR) | **Do funding-rate / basis / positioning signals carry tradeable edge?** | Binance USDT perps | `CRYPTO_PERP_WITH_FUNDING` |
| `03_sequence_crypto.ipynb` | ✅ done — LSTM/carry+returns is the closest to PASS (DSR ≈ 0.46, +0.36 net SR) | Sequence architectures (LSTM / TCN / Transformer) on the *best* crypto features | Binance USDT perps | `CRYPTO_PERP_WITH_FUNDING` |
| `04_llm_sentiment.ipynb` | ✅ done — DATA-CONSTRAINED: free yfinance news is rolling ~24h, can't backtest multi-year | Does LLM-derived news sentiment add what price-only features miss? | Equities + crypto | matching per-asset |
| `05_momentum_positive_control.ipynb` | ✅ done — JT 12-1 calibration ✓ (regime-conditional bull edge as expected) | **Can the harness identify Jegadeesh-Titman 12-1 momentum?** | US large-caps | `EQUITIES_LIQUID_WITH_BORROW` |

Case 5 is a positive control: a classical signal with documented historical edge, run through the *exact same* harness as the other cases. If the harness rejects it the methodology is overcalibrated; if it identifies it, the methodology is trustworthy.

## Running locally

```bash
make install
cp .env.example .env
# Set FRED_API_KEY (free) and GROQ_API_KEY (free) for full reproducibility
make notebooks   # executes every notebook end-to-end
```

`data_cache/` (gitignored) holds yfinance / FRED / Binance / news / LLM pulls keyed by date — same-day re-runs are offline.

For CI / smoke testing without live API calls:

```bash
BACKTESTER_FIXTURE_MODE=1 make notebooks
```

This is what GitHub Actions uses to validate the notebooks on every push.

## Reading the verdicts

Each notebook reports the same metric stack:

- **Annualised Sharpe — gross and net of costs.** Cost regime named explicitly. The cost gap usually tells the story.
- **Bootstrap 95% CI on Sharpe** (stationary block bootstrap, Politis-Romano 1994; 20-day mean block). Strictly above zero ⇒ the point estimate isn't a fluke.
- **Deflated Sharpe Ratio with sensitivity** at `trials_sr_var ∈ {1.0, 0.5, 0.25, 0.1}`. Reading a single var=1.0 number is p-hacked pessimism; the table shows how the verdict moves with the trial-correlation assumption. **DSR ≥ 0.95 at any reasonable var** = clears the bar.
- **CPCV path distribution.** With `n_groups=10, k_test=2` we get 9 distinct OOS Sharpe values; we report mean ± std and the median path.
- **Probability of Backtest Overfitting** (Bailey-Borwein-LdP-Zhu 2017). `pbo > 0.5` is bad: the IS-best regresses below OOS median.
- **MDA permutation feature importance** for tabular cases — what the model actually used.
- **Per-regime breakdown** (bull/bear via 200d SMA). A signal that only earns in one regime hasn't generalised.
- **Pragmatic discussion + "if this were production"** — what would we change to give this signal a fair shot in real trading?

A negative result is still a result. Most ML signals fail under honest evaluation, and that finding *is* the contribution. The positive-control case in notebook 05 demonstrates the methodology can identify real (modest, well-documented) edge when it exists.
