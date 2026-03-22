"""Bar charts for MLB test results (one subplot per metric).

Saved to notebooks/mlb_results.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = Path(__file__).parents[1] / "notebooks"
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Data: (mae, rmse, win_acc)
# ------------------------------------------------------------------
MODELS = [
    ("GRU",                 3.455, 4.408, 0.554),
    ("Transformer\nSmall",  3.457, 4.414, 0.546),
    ("Transformer\nMedium", 3.457, 4.416, 0.550),
    ("LSTM",                3.468, 4.415, 0.549),
    ("RNN",                 3.469, 4.431, 0.540),
]

BASELINES = [
    ("Team avg",  3.523, 4.459, 0.522),
    ("Home +0.3", 3.526, 4.480, 0.521),
    ("Always 0",  3.538, 4.467, 0.000),
]

METRICS = [
    (1, "Test MAE (runs)", "lower is better", False),
    (2, "Test RMSE (runs)", "lower is better", False),
    (3, "Win Direction Accuracy", "higher is better", True),
]

MODEL_COLORS = [
    "#4e79a7",  # GRU — blue
    "#f28e2b",  # Transformer Small — orange
    "#e15759",  # Transformer Medium — red
    "#59a14f",  # LSTM — green
    "#b07aa1",  # RNN — purple
]
BASELINE_COLORS = ["#bab0ac", "#d3d3d3", "#c0c0c0"]
BASELINE_HATCH = ["//", "\\\\", "xx"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, (mi, ylabel, direction, is_pct) in zip(axes, METRICS):
    labels = [m[0] for m in MODELS]
    vals = [m[mi] for m in MODELS]
    x = np.arange(len(MODELS))

    bars = ax.bar(x, vals, color=MODEL_COLORS, edgecolor="white", linewidth=0.6, zorder=3)

    # Baseline lines — tuple layout: (name, mae, rmse, win_acc) → index mi
    for bi, bl in enumerate(BASELINES):
        bval = bl[mi]
        if bval == 0.0 and is_pct:
            continue  # skip always-0 win acc (0%) — not meaningful visually
        ax.axhline(bval, linestyle="--", linewidth=1.2,
                   color=BASELINE_COLORS[bi], label=f"Baseline: {bl[0]}", zorder=2)

    # Value labels
    for bar, val in zip(bars, vals):
        label = f"{val*100:.1f}%" if is_pct else f"{val:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            label,
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{ylabel}\n({direction})", fontsize=10, fontweight="bold")

    # Zoom y-axis
    all_vals = vals + [bl[mi] for bl in BASELINES if not (is_pct and bl[mi] == 0.0)]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.5
    ax.set_ylim(lo - pad, hi + pad * 2.5)

    if is_pct:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3f}"))

    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper right")

fig.suptitle("MLB Results — All 5 Models vs Baselines (seq_len=45)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out = OUT_DIR / "mlb_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out.name}")
