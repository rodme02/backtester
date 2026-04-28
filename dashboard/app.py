"""Streamlit dashboard for the backtesting framework.

Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow running without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtester.data import load_samples  # noqa: E402
from backtester.engine import run_backtest  # noqa: E402
from backtester.strategies import REGISTRY  # noqa: E402

st.set_page_config(page_title="Backtester", layout="wide")
st.title("📈 Backtester")
st.caption("Backtrader-powered strategy lab.")

SAMPLES_DIR = ROOT / "samples" / "ohlcv"
AVAILABLE_SYMBOLS = sorted(p.stem for p in SAMPLES_DIR.glob("*.csv"))

with st.sidebar:
    st.header("Configuration")
    strategy_name = st.selectbox("Strategy", sorted(REGISTRY))
    symbol = st.selectbox("Symbol", AVAILABLE_SYMBOLS, index=0)
    cash = st.number_input("Starting cash", value=100_000.0, step=10_000.0, min_value=1000.0)

    st.subheader("Parameters")
    if strategy_name == "ma_crossover":
        params = {
            "fast_period": st.slider("Fast SMA", 5, 100, 20),
            "slow_period": st.slider("Slow SMA", 20, 300, 100),
            "atr_period": st.slider("ATR period", 5, 50, 14),
            "risk_fraction": st.slider("Risk fraction", 0.005, 0.10, 0.02, step=0.005),
            "stop_atr_multiple": st.slider("Stop × ATR", 0.5, 5.0, 2.0, step=0.5),
            "take_profit_multiple": st.slider("Target × ATR", 0.5, 6.0, 3.0, step=0.5),
        }
    else:
        params = {
            "fast_period": st.slider("Fast SMA", 5, 100, 20),
            "slow_period": st.slider("Slow SMA", 20, 300, 100),
            "rsi_min_long": st.slider("RSI min (long)", 30, 70, 50),
            "rsi_max_short": st.slider("RSI max (short)", 30, 70, 50),
            "adx_threshold": st.slider("ADX threshold", 10.0, 40.0, 20.0, step=1.0),
            "risk_fraction": st.slider("Risk fraction", 0.005, 0.05, 0.01, step=0.005),
        }

    run = st.button("Run backtest", type="primary")


@st.cache_data(show_spinner=False)
def _cached_load(symbol: str) -> pd.DataFrame:
    return load_samples(symbol)


@st.cache_data(show_spinner=False)
def _cached_run(strategy_name: str, symbol: str, params_items: tuple, cash: float):
    data = _cached_load(symbol)
    result = run_backtest(REGISTRY[strategy_name], data, params=dict(params_items), cash=cash)
    return result.final_value, dict(result.metrics), result.equity_curve


if run:
    with st.spinner(f"Running {strategy_name} on {symbol}..."):
        final_value, metrics, equity = _cached_run(
            strategy_name, symbol, tuple(sorted(params.items())), cash
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final value", f"${final_value:,.0f}", f"{metrics.get('total_return_pct', 0):.2f}%")
    sharpe = metrics.get("sharpe")
    c2.metric("Sharpe", f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "—")
    dd = metrics.get("max_drawdown_pct")
    c3.metric("Max drawdown", f"{dd:.2f}%" if isinstance(dd, (int, float)) else "—")
    c4.metric("Trades", metrics.get("total_trades", 0))

    if not equity.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(width=2)))
        fig.update_layout(title="Equity curve", xaxis_title="Date", yaxis_title="Portfolio value",
                          height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        running_max = equity.cummax()
        drawdown = (equity / running_max - 1.0) * 100.0
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill="tozeroy",
                                    line=dict(color="crimson"), name="Drawdown"))
        fig_dd.update_layout(title="Drawdown (%)", height=260,
                             margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.info("No trades executed — try wider parameter ranges.")

    with st.expander("Raw metrics"):
        st.json(metrics)
else:
    st.info("Configure parameters in the sidebar and click **Run backtest**.")
