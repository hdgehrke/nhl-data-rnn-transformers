"""Line chart: Test MAE vs sequence length for the 5 base NHL models.

seq_len tested: 10, 20, 25, 30, 40, 45
Saved to notebooks/seq_len_sweep.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = Path(__file__).parents[1] / "notebooks"
OUT_DIR.mkdir(exist_ok=True)

SEQ_LENS = [10, 20, 25, 30, 40, 45]

# (label, color, marker, mae_values_in_seq_len_order)
MODELS = [
    ("Transformer Small",  "#f28e2b", "o", [2.220, 2.195, 2.184, 2.176, 2.163, 2.149]),
    ("Transformer Medium", "#e15759", "s", [2.221, 2.199, 2.182, 2.180, 2.157, 2.172]),
    ("GRU",                "#59a14f", "^", [2.224, 2.210, 2.189, 2.179, 2.174, 2.168]),
    ("LSTM",               "#4e79a7", "D", [2.214, 2.210, 2.201, 2.196, 2.175, 2.166]),
    ("RNN",                "#b07aa1", "v", [2.231, 2.208, 2.206, 2.209, 2.230, 2.199]),
]

fig, ax = plt.subplots(figsize=(10, 6))

for label, color, marker, maes in MODELS:
    ax.plot(SEQ_LENS, maes, marker=marker, color=color,
            linewidth=2, markersize=7, label=label, zorder=3)
    # annotate best point
    best_idx = int(np.argmin(maes))
    ax.annotate(
        f"{maes[best_idx]:.3f}",
        xy=(SEQ_LENS[best_idx], maes[best_idx]),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        fontweight="bold",
    )

ax.set_xlabel("Sequence Length (games per team)", fontsize=11)
ax.set_ylabel("Test MAE (goals)", fontsize=11)
ax.set_title("NHL Test MAE vs Sequence Length — Base Models",
             fontsize=13, fontweight="bold")
ax.set_xticks(SEQ_LENS)
ax.set_xticklabels([str(s) for s in SEQ_LENS], fontsize=10)

# Zoom y-axis
all_vals = [v for _, _, _, maes in MODELS for v in maes]
lo, hi = min(all_vals), max(all_vals)
pad = (hi - lo) * 0.4
ax.set_ylim(lo - pad, hi + pad * 1.5)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
ax.grid(alpha=0.3, linestyle="--", zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, loc="upper right")

plt.tight_layout()
out = OUT_DIR / "seq_len_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out.name}")
