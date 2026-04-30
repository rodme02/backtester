"""Generates a placeholder hero figure for the README.

Once all five case studies run, replace this with a real
results-summary figure. For now we render a methodology diagram so
the README has *something* above the fold.

Usage::

    python scripts/generate_hero_placeholder.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "results.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(
        "ML Signals in Markets — Honest Evaluation Pipeline",
        fontsize=16,
        loc="left",
        pad=20,
    )

    layers = [
        ("Data\n(yfinance · FRED · Binance · Yahoo RSS · LLM)", "#5b8def", 5, 80),
        ("Features\n(technical · macro · cross-sectional · crypto · sentiment)", "#8ed3b1", 5, 60),
        ("Labels\n(binary direction · triple-barrier · uniqueness weights)", "#f5c46e", 5, 40),
        ("Models\n(linear · RF · GBM · LSTM · TCN · Transformer · LLM-derived)", "#f08a8a", 5, 20),
        ("Eval Harness\n(walk-forward · CPCV · deflated SR + sensitivity · PBO · bootstrap CI · MDA · costs w/ borrow & funding · regimes)", "#cba6f0", 35, 50),
    ]
    for label, color, x, y in layers[:4]:
        rect = mpatches.FancyBboxPatch(
            (x, y - 6), 28, 12, boxstyle="round,pad=0.4",
            linewidth=1, edgecolor="#333", facecolor=color, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x + 14, y, label, ha="center", va="center", fontsize=9)
        # Arrow into eval harness
        ax.annotate(
            "", xy=(60, 50), xytext=(33, y),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#666"),
        )

    label, color, x, y = layers[4]
    rect = mpatches.FancyBboxPatch(
        (x + 25, y - 35), 60, 70, boxstyle="round,pad=0.5",
        linewidth=1.5, edgecolor="#222", facecolor=color, alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(x + 55, y, label, ha="center", va="center", fontsize=10, wrap=True)

    fig.text(
        0.02, 0.02,
        "Five case studies, two asset classes, every popular ML recipe — evaluated identically.",
        fontsize=9, style="italic", color="#444",
    )

    plt.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
