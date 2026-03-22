"""Grouped bar charts for NBA zero-shot vs fine-tuned transfer results.

For each metric: x-axis = model, groups = zero-shot / fine-tuned.
Saved to notebooks/nba_transfer.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = Path(__file__).parents[1] / "notebooks"
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
MODELS = ["RNN", "LSTM", "GRU", "Transformer\nSmall", "Transformer\nMedium"]

ZERO_SHOT = {
    # mae, rmse, win_acc
    "RNN":                  (11.95, 14.97, 0.642),
    "LSTM":                 (11.99, 15.01, 0.634),
    "GRU":                  (11.94, 14.96, 0.629),
    "Transformer\nSmall":   (11.94, 14.97, 0.640),
    "Transformer\nMedium":  (11.93, 14.97, 0.642),
}

FINE_TUNED = {
    "RNN":                  (10.94, 13.97, 0.636),
    "LSTM":                 (10.88, 13.89, 0.643),
    "GRU":                  (10.79, 13.81, 0.649),
    "Transformer\nSmall":   (10.79, 13.80, 0.646),
    "Transformer\nMedium":  (10.80, 13.82, 0.645),
}

BASELINES = {
    "Home +3.0": (11.96, 15.10, 0.556),
    "Always 0":  (12.16, 15.22, 0.000),
}

METRICS = [
    (0, "Test MAE (pts)", "lower is better", False),
    (1, "Test RMSE (pts)", "lower is better", False),
    (2, "Win Direction Accuracy", "higher is better", True),
]

ZS_COLOR = "#4e79a7"   # blue — zero-shot
FT_COLOR = "#f28e2b"   # orange — fine-tuned
BL_COLORS = ["#bab0ac", "#d3d3d3"]

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

bar_width = 0.35
x = np.arange(len(MODELS))

for ax, (mi, ylabel, direction, is_pct) in zip(axes, METRICS):
    zs_vals = [ZERO_SHOT[m][mi] for m in MODELS]
    ft_vals = [FINE_TUNED[m][mi] for m in MODELS]

    zs_bars = ax.bar(x - bar_width / 2, zs_vals, bar_width,
                     label="Zero-shot (NHL→NBA)", color=ZS_COLOR,
                     edgecolor="white", linewidth=0.5, zorder=3)
    ft_bars = ax.bar(x + bar_width / 2, ft_vals, bar_width,
                     label="Fine-tuned (NHL pre-train + NBA train)", color=FT_COLOR,
                     edgecolor="white", linewidth=0.5, zorder=3)

    # Baseline lines
    for bi, (bname, bvals) in enumerate(BASELINES.items()):
        bval = bvals[mi]
        if is_pct and bval == 0.0:
            continue
        ax.axhline(bval, linestyle="--", linewidth=1.2,
                   color=BL_COLORS[bi], label=f"Baseline: {bname}", zorder=2)

    # Value labels
    for bars, vals in ((zs_bars, zs_vals), (ft_bars, ft_vals)):
        for bar, val in zip(bars, vals):
            label = f"{val*100:.1f}%" if is_pct else f"{val:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                label,
                ha="center", va="bottom", fontsize=7.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{ylabel}\n({direction})", fontsize=10, fontweight="bold")

    all_vals = zs_vals + ft_vals + [
        bvals[mi] for bvals in BASELINES.values()
        if not (is_pct and bvals[mi] == 0.0)
    ]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.5
    ax.set_ylim(lo - pad, hi + pad * 3.5)

    if is_pct:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))

    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper right")

fig.suptitle("NBA Transfer Results — Zero-Shot vs Fine-Tuned (NHL → NBA)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out = OUT_DIR / "nba_transfer.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out.name}")
