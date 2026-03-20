"""Generate grouped bar charts for the NHL architecture sweep.

One figure per (family, metric) = 12 figures total.
Each figure has 3 groups (seq_len 20/25/45), one bar per variant.
Saved to notebooks/sweep_<family>_<metric>.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = Path(__file__).parents[1] / "notebooks"
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Data: (mae, rmse, win_acc) keyed by (config, seq_len_str)
# ------------------------------------------------------------------
DATA: dict[tuple[str, str], tuple[float, float, float]] = {
    ("rnn_base", "20"): (2.2079, 2.6086, 0.5720),
    ("rnn_base", "25"): (2.2055, 2.6076, 0.5743),
    ("rnn_base", "45"): (2.1993, 2.5990, 0.5774),
    ("rnn_small", "20"): (2.2009, 2.6012, 0.5678),
    ("rnn_small", "25"): (2.2004, 2.6052, 0.5732),
    ("rnn_small", "45"): (2.1865, 2.5899, 0.5823),
    ("rnn_large", "20"): (2.2138, 2.6030, 0.5720),
    ("rnn_large", "25"): (2.2027, 2.6028, 0.5747),
    ("rnn_large", "45"): (2.1933, 2.5981, 0.5781),
    ("rnn_deep", "20"): (2.2068, 2.6042, 0.5743),
    ("rnn_deep", "25"): (2.2070, 2.6177, 0.5682),
    ("rnn_deep", "45"): (2.2345, 2.6121, 0.5572),

    ("lstm_base", "20"): (2.2096, 2.6026, 0.5694),
    ("lstm_base", "25"): (2.2010, 2.6036, 0.5595),
    ("lstm_base", "45"): (2.1664, 2.5761, 0.5877),
    ("lstm_small", "20"): (2.1993, 2.6030, 0.5762),
    ("lstm_small", "25"): (2.1913, 2.6009, 0.5804),
    ("lstm_small", "45"): (2.1637, 2.5790, 0.5941),
    ("lstm_large", "20"): (2.2058, 2.6114, 0.5655),
    ("lstm_large", "25"): (2.2017, 2.6006, 0.5694),
    ("lstm_large", "45"): (2.1835, 2.5848, 0.5838),
    ("lstm_deep", "20"): (2.2046, 2.5999, 0.5648),
    ("lstm_deep", "25"): (2.2007, 2.6041, 0.5675),
    ("lstm_deep", "45"): (2.1785, 2.5821, 0.5869),

    ("gru_base", "20"): (2.2097, 2.6021, 0.5724),
    ("gru_base", "25"): (2.1885, 2.6027, 0.5716),
    ("gru_base", "45"): (2.1683, 2.5804, 0.5823),
    ("gru_small", "20"): (2.1978, 2.5954, 0.5793),
    ("gru_small", "25"): (2.1894, 2.6031, 0.5686),
    ("gru_small", "45"): (2.1681, 2.5764, 0.5831),
    ("gru_large", "20"): (2.1928, 2.6036, 0.5747),
    ("gru_large", "25"): (2.1886, 2.5991, 0.5709),
    ("gru_large", "45"): (2.1856, 2.5786, 0.5861),
    ("gru_deep", "20"): (2.1986, 2.6034, 0.5636),
    ("gru_deep", "25"): (2.1922, 2.5934, 0.5755),
    ("gru_deep", "45"): (2.1592, 2.5797, 0.5770),

    ("transformer_small", "20"): (2.1945, 2.5940, 0.5762),
    ("transformer_small", "25"): (2.1841, 2.5970, 0.5667),
    ("transformer_small", "45"): (2.1494, 2.5638, 0.5907),
    ("transformer_medium", "20"): (2.1990, 2.5994, 0.5682),
    ("transformer_medium", "25"): (2.1818, 2.5890, 0.5732),
    ("transformer_medium", "45"): (2.1718, 2.5696, 0.5957),
    ("transformer_tiny", "20"): (2.2019, 2.6015, 0.5743),
    ("transformer_tiny", "25"): (2.1844, 2.5901, 0.5736),
    ("transformer_tiny", "45"): (2.1600, 2.5665, 0.5888),
    ("transformer_small_deep", "20"): (2.1945, 2.6020, 0.5701),
    ("transformer_small_deep", "25"): (2.1887, 2.5932, 0.5697),
    ("transformer_small_deep", "45"): (2.1578, 2.5670, 0.5934),
    ("transformer_medium_shallow", "20"): (2.1994, 2.5947, 0.5793),
    ("transformer_medium_shallow", "25"): (2.1918, 2.5899, 0.5777),
    ("transformer_medium_shallow", "45"): (2.1487, 2.5711, 0.5941),
    ("transformer_deep", "20"): (2.1905, 2.6012, 0.5697),
    ("transformer_deep", "25"): (2.1850, 2.5964, 0.5732),
    ("transformer_deep", "45"): (2.1564, 2.5721, 0.5888),
    ("transformer_large", "20"): (2.2002, 2.6023, 0.5777),
    ("transformer_large", "25"): (2.2071, 2.5971, 0.5743),
    ("transformer_large", "45"): (2.1508, 2.5671, 0.5941),
}

# ------------------------------------------------------------------
# Family definitions
# ------------------------------------------------------------------
FAMILIES: dict[str, list[tuple[str, str]]] = {
    "RNN": [
        ("base", "rnn_base"),
        ("small\n(64-dim, 1L)", "rnn_small"),
        ("large\n(256-dim, 2L)", "rnn_large"),
        ("deep\n(128-dim, 4L)", "rnn_deep"),
    ],
    "LSTM": [
        ("base", "lstm_base"),
        ("small\n(64-dim, 1L)", "lstm_small"),
        ("large\n(256-dim, 2L)", "lstm_large"),
        ("deep\n(128-dim, 4L)", "lstm_deep"),
    ],
    "GRU": [
        ("base", "gru_base"),
        ("small\n(64-dim, 1L)", "gru_small"),
        ("large\n(256-dim, 2L)", "gru_large"),
        ("deep\n(128-dim, 4L)", "gru_deep"),
    ],
    "Transformer": [
        ("small\n(d=128, 2L)", "transformer_small"),
        ("medium\n(d=256, 4L)", "transformer_medium"),
        ("tiny\n(d=64, 2L)", "transformer_tiny"),
        ("small-deep\n(d=128, 4L)", "transformer_small_deep"),
        ("med-shallow\n(d=256, 2L)", "transformer_medium_shallow"),
        ("deep\n(d=256, 6L)", "transformer_deep"),
        ("large\n(d=512, 4L)", "transformer_large"),
    ],
}

SEQ_LENS = ["20", "25", "45"]
METRICS = {
    "MAE": (0, "Test MAE (goals)", "lower is better"),
    "RMSE": (1, "Test RMSE (goals)", "lower is better"),
    "Win Accuracy": (2, "Win Direction Accuracy", "higher is better"),
}

COLORS = ["#4e79a7", "#f28e2b", "#59a14f"]  # blue, orange, green per seq_len
SEQ_LABELS = ["seq_len = 20", "seq_len = 25", "seq_len = 45"]

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
for family_name, variants in FAMILIES.items():
    for metric_name, (metric_idx, ylabel, direction) in METRICS.items():
        n_variants = len(variants)
        n_groups = len(SEQ_LENS)
        group_width = 0.7
        bar_width = group_width / n_variants

        fig, ax = plt.subplots(figsize=(max(9, n_variants * 2), 5))

        for vi, (var_label, cfg_key) in enumerate(variants):
            offsets = np.arange(n_groups) + (vi - n_variants / 2 + 0.5) * bar_width
            values = [DATA.get((cfg_key, sl), (None, None, None))[metric_idx] for sl in SEQ_LENS]
            bars = ax.bar(
                offsets,
                [v if v is not None else 0 for v in values],
                width=bar_width * 0.9,
                label=var_label,
                color=plt.cm.tab10(vi / max(n_variants - 1, 1)),
                edgecolor="white",
                linewidth=0.5,
            )
            # Value labels on bars
            for bar, val in zip(bars, values):
                if val is not None:
                    label = f"{val:.3f}" if metric_idx < 2 else f"{val*100:.1f}%"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0005,
                        label,
                        ha="center", va="bottom",
                        fontsize=6.5,
                        rotation=90,
                    )

        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels([f"seq_len = {sl}" for sl in SEQ_LENS], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{family_name} — {metric_name} ({direction})", fontsize=13, fontweight="bold")
        ax.legend(title="Variant", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

        # Zoom y-axis to meaningful range
        all_vals = [DATA.get((cfg, sl), (None,)*3)[metric_idx]
                    for _, cfg in variants for sl in SEQ_LENS
                    if DATA.get((cfg, sl), (None,)*3)[metric_idx] is not None]
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            pad = (hi - lo) * 0.4
            ax.set_ylim(lo - pad, hi + pad * 3.5)

        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:.3f}") if metric_idx < 2
            else ticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%")
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        slug = metric_name.lower().replace(" ", "_")
        family_slug = family_name.lower().replace(" ", "_")
        out = OUT_DIR / f"sweep_{family_slug}_{slug}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out.name}")

print("Done.")
