"""Generate a model architecture reference table image.

Columns: Family | Variant | Type | Hidden/d_model | Layers | Heads | FFN dim | Head MLP | Dropout
Saved to notebooks/model_architecture_table.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path(__file__).parents[1] / "notebooks"
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Model definitions
# Each row: (family, variant, model_type, hidden/d_model, num_layers,
#            nhead, ffn_dim, head_hidden_dim, dropout)
# nhead/ffn_dim = "—" for RNN/LSTM/GRU
# ------------------------------------------------------------------
ROWS = [
    # RNN family
    ("RNN", "base",  "rnn",  128, 2, "—", "—",  128, 0.1),
    ("RNN", "small", "rnn",   64, 1, "—", "—",   64, 0.1),
    ("RNN", "large", "rnn",  256, 2, "—", "—",  256, 0.1),
    ("RNN", "deep",  "rnn",  128, 4, "—", "—",  128, 0.1),
    # LSTM family
    ("LSTM", "base",  "lstm", 128, 2, "—", "—",  128, 0.1),
    ("LSTM", "small", "lstm",  64, 1, "—", "—",   64, 0.1),
    ("LSTM", "large", "lstm", 256, 2, "—", "—",  256, 0.1),
    ("LSTM", "deep",  "lstm", 128, 4, "—", "—",  128, 0.1),
    # GRU family
    ("GRU", "base",  "gru",  128, 2, "—", "—",  128, 0.1),
    ("GRU", "small", "gru",   64, 1, "—", "—",   64, 0.1),
    ("GRU", "large", "gru",  256, 2, "—", "—",  256, 0.1),
    ("GRU", "deep",  "gru",  128, 4, "—", "—",  128, 0.1),
    # Transformer family
    ("Transformer", "small",          "transformer",  128, 2, 4, 256,  128, 0.1),
    ("Transformer", "medium",         "transformer",  256, 4, 8, 512,  256, 0.1),
    ("Transformer", "tiny",           "transformer",   64, 2, 2, 128,   64, 0.1),
    ("Transformer", "small-deep",     "transformer",  128, 4, 4, 256,  128, 0.1),
    ("Transformer", "med-shallow",    "transformer",  256, 2, 8, 512,  256, 0.1),
    ("Transformer", "deep",           "transformer",  256, 6, 8, 512,  256, 0.1),
    ("Transformer", "large",          "transformer",  512, 4, 8,1024,  256, 0.1),
]

HEADERS = [
    "Family", "Variant", "Type",
    "Hidden /\nd_model", "Layers", "Heads\n(attn)", "FFN\ndim",
    "Head\nMLP dim", "Dropout",
]

# Approximate encoder parameter count (shared encoder, counted once)
def approx_params(row):
    family, variant, mtype, hidden, layers, nhead, ffn, head_mlp, drop = row
    input_dim = 28  # feature dim
    if mtype == "rnn":
        # (input + hidden) * hidden * layers (simplified)
        enc = layers * (input_dim + hidden) * hidden
    elif mtype == "lstm":
        # 4 gates
        enc = layers * 4 * (input_dim + hidden) * hidden
    elif mtype == "gru":
        # 3 gates
        enc = layers * 3 * (input_dim + hidden) * hidden
    else:
        # transformer: input proj + num_layers * (attn + ffn)
        input_proj = input_dim * hidden
        attn_per_layer = 4 * hidden * hidden
        ffn_per_layer = 2 * hidden * ffn
        enc = input_proj + layers * (attn_per_layer + ffn_per_layer)
    # MLP head: 2*hidden → head_mlp → 1
    mlp = 2 * hidden * head_mlp + head_mlp
    total = enc + mlp
    if total >= 1_000_000:
        return f"{total/1e6:.2f}M"
    return f"{total/1e3:.0f}K"

# ------------------------------------------------------------------
# Build table data
# ------------------------------------------------------------------
col_data = []
for row in ROWS:
    family, variant, mtype, hidden, layers, nhead, ffn, head_mlp, drop = row
    col_data.append([
        family,
        variant,
        mtype,
        str(hidden),
        str(layers),
        str(nhead),
        str(ffn),
        str(head_mlp),
        f"{drop}",
        approx_params(row),
    ])

headers_full = HEADERS + ["~Params\n(encoder+head)"]

# ------------------------------------------------------------------
# Family color bands
# ------------------------------------------------------------------
FAMILY_COLORS = {
    "RNN":         "#d4e6f1",
    "LSTM":        "#d5f5e3",
    "GRU":         "#fdebd0",
    "Transformer": "#f5eef8",
}

row_colors = []
for row in ROWS:
    c = FAMILY_COLORS[row[0]]
    row_colors.append([c] * len(headers_full))

# ------------------------------------------------------------------
# Draw
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 9))
ax.axis("off")

tbl = ax.table(
    cellText=col_data,
    colLabels=headers_full,
    cellLoc="center",
    loc="center",
    cellColours=row_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.55)

# Style header row
for j in range(len(headers_full)):
    cell = tbl[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Bold family name only on first row of each family group
prev_family = None
for i, row in enumerate(ROWS):
    if row[0] != prev_family:
        tbl[i + 1, 0].set_text_props(fontweight="bold")
        prev_family = row[0]

# Divider lines between families
family_breaks = []
prev = ROWS[0][0]
for i, row in enumerate(ROWS[1:], 1):
    if row[0] != prev:
        family_breaks.append(i)
        prev = row[0]

for break_row in family_breaks:
    for j in range(len(headers_full)):
        tbl[break_row, j].visible_edges = "TBL" if j == 0 else ("TBR" if j == len(headers_full)-1 else "TB")

# Legend
patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
ax.legend(handles=patches, loc="lower center", ncol=4,
          bbox_to_anchor=(0.5, -0.02), fontsize=9, title="Model Family")

ax.set_title(
    "NHL Architecture Sweep — Model Configurations",
    fontsize=14, fontweight="bold", pad=12,
)

plt.tight_layout()
out = OUT_DIR / "model_architecture_table.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out.name}")
