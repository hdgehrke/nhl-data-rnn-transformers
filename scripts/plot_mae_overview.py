"""Three-panel MAE overview figure.

Panel 1 (top-left):  All 12 RNN/LSTM/GRU variants × 3 seq_lens — grouped bars
Panel 2 (top-right): All 7 Transformer variants × 3 seq_lens — grouped bars
Panel 3 (bottom):    Family comparison — best/mean/range across variants per
                     family × seq_len as a line + shaded-range chart

Saved to notebooks/mae_overview.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = Path(__file__).parents[1] / "notebooks"
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Data: keyed (config, seq_len_str) → mae
# ------------------------------------------------------------------
MAE: dict[tuple[str, str], float] = {
    ("rnn_base",  "20"): 2.2079, ("rnn_base",  "25"): 2.2055, ("rnn_base",  "45"): 2.1993,
    ("rnn_small", "20"): 2.2009, ("rnn_small", "25"): 2.2004, ("rnn_small", "45"): 2.1865,
    ("rnn_large", "20"): 2.2138, ("rnn_large", "25"): 2.2027, ("rnn_large", "45"): 2.1933,
    ("rnn_deep",  "20"): 2.2068, ("rnn_deep",  "25"): 2.2070, ("rnn_deep",  "45"): 2.2345,

    ("lstm_base",  "20"): 2.2096, ("lstm_base",  "25"): 2.2010, ("lstm_base",  "45"): 2.1664,
    ("lstm_small", "20"): 2.1993, ("lstm_small", "25"): 2.1913, ("lstm_small", "45"): 2.1637,
    ("lstm_large", "20"): 2.2058, ("lstm_large", "25"): 2.2017, ("lstm_large", "45"): 2.1835,
    ("lstm_deep",  "20"): 2.2046, ("lstm_deep",  "25"): 2.2007, ("lstm_deep",  "45"): 2.1785,

    ("gru_base",  "20"): 2.2097, ("gru_base",  "25"): 2.1885, ("gru_base",  "45"): 2.1683,
    ("gru_small", "20"): 2.1978, ("gru_small", "25"): 2.1894, ("gru_small", "45"): 2.1681,
    ("gru_large", "20"): 2.1928, ("gru_large", "25"): 2.1886, ("gru_large", "45"): 2.1856,
    ("gru_deep",  "20"): 2.1986, ("gru_deep",  "25"): 2.1922, ("gru_deep",  "45"): 2.1592,

    ("transformer_small",          "20"): 2.1945, ("transformer_small",          "25"): 2.1841, ("transformer_small",          "45"): 2.1494,
    ("transformer_medium",         "20"): 2.1990, ("transformer_medium",         "25"): 2.1818, ("transformer_medium",         "45"): 2.1718,
    ("transformer_tiny",           "20"): 2.2019, ("transformer_tiny",           "25"): 2.1844, ("transformer_tiny",           "45"): 2.1600,
    ("transformer_small_deep",     "20"): 2.1945, ("transformer_small_deep",     "25"): 2.1887, ("transformer_small_deep",     "45"): 2.1578,
    ("transformer_medium_shallow", "20"): 2.1994, ("transformer_medium_shallow", "25"): 2.1918, ("transformer_medium_shallow", "45"): 2.1487,
    ("transformer_deep",           "20"): 2.1905, ("transformer_deep",           "25"): 2.1850, ("transformer_deep",           "45"): 2.1564,
    ("transformer_large",          "20"): 2.2002, ("transformer_large",          "25"): 2.2071, ("transformer_large",          "45"): 2.1508,
}

SEQ_LENS = ["20", "25", "45"]
SL_COLORS = ["#4e79a7", "#f28e2b", "#59a14f"]   # blue / orange / green

# Family membership
RECURRENT = [
    # (short_label, config_key, family)
    ("base\n(128,2L)",  "rnn_base",  "RNN"),
    ("small\n(64,1L)",  "rnn_small", "RNN"),
    ("large\n(256,2L)", "rnn_large", "RNN"),
    ("deep\n(128,4L)",  "rnn_deep",  "RNN"),
    ("base\n(128,2L)",  "lstm_base",  "LSTM"),
    ("small\n(64,1L)",  "lstm_small", "LSTM"),
    ("large\n(256,2L)", "lstm_large", "LSTM"),
    ("deep\n(128,4L)",  "lstm_deep",  "LSTM"),
    ("base\n(128,2L)",  "gru_base",  "GRU"),
    ("small\n(64,1L)",  "gru_small", "GRU"),
    ("large\n(256,2L)", "gru_large", "GRU"),
    ("deep\n(128,4L)",  "gru_deep",  "GRU"),
]

TRANSFORMERS = [
    ("small\n(d=128,2L)",    "transformer_small"),
    ("medium\n(d=256,4L)",   "transformer_medium"),
    ("tiny\n(d=64,2L)",      "transformer_tiny"),
    ("small-deep\n(d=128,4L)", "transformer_small_deep"),
    ("med-shallow\n(d=256,2L)", "transformer_medium_shallow"),
    ("deep\n(d=256,6L)",     "transformer_deep"),
    ("large\n(d=512,4L)",    "transformer_large"),
]

FAMILY_CONFIGS = {
    "RNN":         ["rnn_base",  "rnn_small",  "rnn_large",  "rnn_deep"],
    "LSTM":        ["lstm_base", "lstm_small", "lstm_large", "lstm_deep"],
    "GRU":         ["gru_base",  "gru_small",  "gru_large",  "gru_deep"],
    "Transformer": [v[1] for v in TRANSFORMERS],
}

FAMILY_COLORS = {
    "RNN":         "#b07aa1",
    "LSTM":        "#4e79a7",
    "GRU":         "#59a14f",
    "Transformer": "#f28e2b",
}

# Family band shading for recurrent panel
BAND_COLORS = {"RNN": "#f9f0ff", "LSTM": "#edf4fb", "GRU": "#edfbf0"}


# ------------------------------------------------------------------
# Helper: draw grouped bar panel (x = models, groups of 3 seq_len bars)
# ------------------------------------------------------------------
def draw_grouped_bar(ax, models, title, band_families=None):
    """
    models: list of (label, cfg_key[, family])
    band_families: optional list of family per model (for background shading)
    """
    n = len(models)
    bw = 0.22          # bar width
    group_w = bw * 3   # width of one model's group
    x = np.arange(n)

    for si, (sl, color) in enumerate(zip(SEQ_LENS, SL_COLORS)):
        offsets = x + (si - 1) * bw
        vals = [MAE[(m[1], sl)] for m in models]
        ax.bar(offsets, vals, width=bw * 0.9, color=color,
               label=f"seq_len = {sl}", edgecolor="white", linewidth=0.4, zorder=3)

    # Family background shading
    if band_families is not None:
        fam_start = 0
        cur_fam = band_families[0]
        for i in range(1, n + 1):
            next_fam = band_families[i] if i < n else None
            if next_fam != cur_fam:
                ax.axvspan(fam_start - 0.5, i - 0.5,
                           color=BAND_COLORS[cur_fam], alpha=0.35, zorder=0)
                # Family label centred in band
                mid = (fam_start + i - 1) / 2
                ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 2.215,
                        cur_fam, ha="center", va="top", fontsize=9,
                        fontweight="bold", color="#555555", zorder=4)
                cur_fam = next_fam
                fam_start = i

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], fontsize=7.5)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_ylabel("Test MAE (goals)", fontsize=9)

    # Zoom y
    all_vals = [MAE[(m[1], sl)] for m in models for sl in SEQ_LENS]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.5
    ax.set_ylim(lo - pad * 0.3, hi + pad * 3.5)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right")


# ------------------------------------------------------------------
# Figure layout: 2 top panels + 1 wide bottom panel
# ------------------------------------------------------------------
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.45, wspace=0.25)

ax_rec  = fig.add_subplot(gs[0, 0])   # top-left: recurrent
ax_trf  = fig.add_subplot(gs[0, 1])   # top-right: transformer
ax_cmp  = fig.add_subplot(gs[1, :])   # bottom-full: comparison

# ------------------------------------------------------------------
# Panel 1: Recurrent
# ------------------------------------------------------------------
band_fams = [m[2] for m in RECURRENT]
draw_grouped_bar(ax_rec, RECURRENT, "RNN / LSTM / GRU — All Variants", band_fams)
# Add family labels manually after y-lim is set
all_rec_vals = [MAE[(m[1], sl)] for m in RECURRENT for sl in SEQ_LENS]
lo_r, hi_r = min(all_rec_vals), max(all_rec_vals)
pad_r = (hi_r - lo_r) * 0.5
y_label = hi_r + pad_r * 3.0
fam_bounds = {"RNN": (0, 4), "LSTM": (4, 8), "GRU": (8, 12)}
for fam, (start, end) in fam_bounds.items():
    mid = (start + end - 1) / 2
    ax_rec.text(mid, y_label, fam, ha="center", va="top",
                fontsize=9, fontweight="bold", color="#444444")
    ax_rec.axvspan(start - 0.5, end - 0.5, color=BAND_COLORS[fam], alpha=0.3, zorder=0)

# ------------------------------------------------------------------
# Panel 2: Transformer
# ------------------------------------------------------------------
draw_grouped_bar(ax_trf, TRANSFORMERS, "Transformer — All Variants")

# ------------------------------------------------------------------
# Panel 3: Family comparison — line + shaded range across variants
# ------------------------------------------------------------------
sl_x = [20, 25, 45]
sl_str = ["20", "25", "45"]

for fam, cfgs in FAMILY_CONFIGS.items():
    color = FAMILY_COLORS[fam]
    best_maes, mean_maes, lo_maes, hi_maes = [], [], [], []
    for sl in sl_str:
        vals = [MAE[(cfg, sl)] for cfg in cfgs]
        best_maes.append(min(vals))
        mean_maes.append(np.mean(vals))
        lo_maes.append(min(vals))
        hi_maes.append(max(vals))

    ax_cmp.plot(sl_x, best_maes, color=color, linewidth=2.2,
                marker="o", markersize=7, label=f"{fam} (best variant)", zorder=4)
    ax_cmp.plot(sl_x, mean_maes, color=color, linewidth=1.2,
                linestyle="--", marker="s", markersize=5,
                alpha=0.7, label=f"{fam} (mean across variants)", zorder=3)
    ax_cmp.fill_between(sl_x, lo_maes, hi_maes,
                        color=color, alpha=0.12, zorder=2)

    # Annotate best value at seq_len=45
    ax_cmp.annotate(
        f"{best_maes[-1]:.4f}",
        xy=(45, best_maes[-1]),
        xytext=(8, 2),
        textcoords="offset points",
        fontsize=8, color=color, fontweight="bold",
    )

ax_cmp.set_xticks(sl_x)
ax_cmp.set_xticklabels(["seq_len = 20", "seq_len = 25", "seq_len = 45"], fontsize=10)
ax_cmp.set_xlabel("Sequence Length", fontsize=10)
ax_cmp.set_ylabel("Test MAE (goals)", fontsize=10)
ax_cmp.set_title(
    "Family Comparison — Best and Mean MAE Across Variants\n"
    "(solid = best variant, dashed = mean, shaded band = min–max range)",
    fontsize=11, fontweight="bold",
)

all_cmp = [MAE[(cfg, sl)] for cfgs in FAMILY_CONFIGS.values() for cfg in cfgs for sl in sl_str]
lo_c, hi_c = min(all_cmp), max(all_cmp)
pad_c = (hi_c - lo_c) * 0.4
ax_cmp.set_ylim(lo_c - pad_c * 0.5, hi_c + pad_c * 2.0)
ax_cmp.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
ax_cmp.grid(alpha=0.3, linestyle="--", zorder=1)
ax_cmp.spines["top"].set_visible(False)
ax_cmp.spines["right"].set_visible(False)
ax_cmp.legend(fontsize=8.5, loc="upper right", ncol=2)

# ------------------------------------------------------------------
# Overall title
# ------------------------------------------------------------------
fig.suptitle("NHL Architecture Sweep — Test MAE Overview",
             fontsize=14, fontweight="bold", y=0.98)

out = OUT_DIR / "mae_overview.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out.name}")
