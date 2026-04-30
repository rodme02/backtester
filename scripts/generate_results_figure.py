"""Generates the headline results figure for the README.

Hard-coded with the actual numbers from the executed notebooks
(Cases 1, 2, 5 at the v0.2 harness level). When new cases ship,
add rows here.

Usage::

    python scripts/generate_results_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "results.png"


# (label, net_sharpe, ci_low, ci_high, group, near_miss)
# group is used for vertical separators.
ROWS = [
    ("Case 5 — JT 12-1 momentum (positive control)", -0.041, -0.447, 0.390, "control", False),
    # Case 1 — equities tabular bake-off
    ("Case 1 — logistic / binary",        -0.62, -1.25, -0.00, "equities-binary", False),
    ("Case 1 — random forest / binary",   -0.65, -1.26, -0.08, "equities-binary", False),
    ("Case 1 — GBM / binary",             -0.14, -0.78,  0.53, "equities-binary", False),
    ("Case 1 — logistic / triple-barrier", -0.44, -1.58,  0.72, "equities-tb", False),
    ("Case 1 — random forest / triple-barrier", 0.36, -0.66,  1.30, "equities-tb", True),
    ("Case 1 — GBM / triple-barrier",      0.05, -0.74,  0.89, "equities-tb", False),
    # Case 2 — crypto signal universe
    ("Case 2 — returns",                  -0.64, -1.71,  0.45, "crypto", False),
    ("Case 2 — funding-level",            -0.75, -1.95,  0.55, "crypto", False),
    ("Case 2 — carry-rank",               -0.09, -1.45,  1.10, "crypto", True),
    ("Case 2 — basis",                    -0.77, -1.86,  0.39, "crypto", False),
    ("Case 2 — union",                    -1.34, -2.50,  0.03, "crypto", False),
    # Case 3 — sequence models on crypto
    ("Case 3 — LSTM (carry+returns)",      0.36, -1.05,  1.59, "crypto-seq", True),
    ("Case 3 — TCN (carry+returns)",      -1.35, -2.63, -0.20, "crypto-seq", False),
    ("Case 3 — Transformer (carry+returns)", -0.86, -1.76, 0.22, "crypto-seq", False),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 8.4))
    y_positions = np.arange(len(ROWS))
    labels = [r[0] for r in ROWS]
    point = np.array([r[1] for r in ROWS])
    ci_low = np.array([r[2] for r in ROWS])
    ci_high = np.array([r[3] for r in ROWS])

    err_left = point - ci_low
    err_right = ci_high - point

    colors = []
    for r in ROWS:
        if r[5]:
            colors.append("#2e8b57")  # near-miss → green
        elif r[4] == "control":
            colors.append("#888")
        else:
            colors.append("#a6373c")  # fail → red

    ax.errorbar(
        point, y_positions,
        xerr=[err_left, err_right],
        fmt="o", color="black", ecolor="#888",
        elinewidth=1.4, capsize=4, markersize=6, zorder=3,
    )
    for y, p, c in zip(y_positions, point, colors, strict=True):
        ax.scatter([p], [y], s=120, color=c, edgecolors="black",
                   linewidths=0.8, zorder=4)

    ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.axvline(0.5, color="green", linestyle="--", linewidth=0.8,
               alpha=0.4, label="DSR=0.5 reference (loose)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Annualised Sharpe ratio (net of costs) with 95% bootstrap CI", fontsize=10)
    ax.set_xlim(-3.0, 2.0)
    ax.grid(axis="x", alpha=0.3)

    # Group separators
    ax.axhline(0.5, color="#bbb", linewidth=0.6)
    ax.axhline(3.5, color="#bbb", linewidth=0.6)
    ax.axhline(6.5, color="#bbb", linewidth=0.6)

    near_miss_patch = mpatches.Patch(color="#2e8b57", label="Near-miss (regime-conditional edge)")
    fail_patch = mpatches.Patch(color="#a6373c", label="Fail")
    control_patch = mpatches.Patch(color="#888", label="Calibration / control")
    ax.legend(
        handles=[near_miss_patch, fail_patch, control_patch],
        loc="lower right", fontsize=9, framealpha=0.95,
    )

    ax.set_title(
        "ML Signals in Markets — survey results across 16 specifications\n"
        "Three cases are near-misses with regime-conditional edge. "
        "LSTM/carry+returns is closest to passing the unconditional bar.",
        fontsize=12, loc="left", pad=14,
    )

    plt.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
