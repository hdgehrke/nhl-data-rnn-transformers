"""Calibration plot: predicted goal differential vs actual win rate in buckets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def calibration_plot(
    preds: np.ndarray,
    labels: np.ndarray,
    n_buckets: int = 10,
    save_path: str | Path | None = None,
    title: str = "Calibration: Predicted Differential vs Win Rate",
) -> None:
    """Plot predicted goal differential buckets vs actual win rate.

    A well-calibrated model should show a monotonically increasing curve.
    """
    # Bucket by predicted differential
    edges = np.percentile(preds, np.linspace(0, 100, n_buckets + 1))
    bucket_centers, win_rates, counts = [], [], []

    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds < hi) if i < n_buckets - 1 else (preds >= lo) & (preds <= hi)
        if mask.sum() == 0:
            continue
        bucket_labels = labels[mask]
        win_rate = (bucket_labels > 0).mean()  # home team won
        bucket_centers.append(float(preds[mask].mean()))
        win_rates.append(float(win_rate))
        counts.append(int(mask.sum()))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(bucket_centers, win_rates, "bo-", label="Win rate")
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Mean Predicted Goal Differential")
    ax1.set_ylabel("Actual Win Rate")
    ax1.set_title(title)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(bucket_centers, counts, width=0.1, alpha=0.3, color="gray", label="# games")
    ax2.set_ylabel("Games in bucket")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved calibration plot to {save_path}")
    else:
        plt.show()
    plt.close()


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table for multiple models/baselines.

    Each entry should have: name, mae, rmse, win_direction_acc
    """
    header = f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'WinAcc':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x.get("mae", float("inf"))):
        print(
            f"{r['name']:<30} {r['mae']:>8.4f} {r['rmse']:>8.4f} {r.get('win_direction_acc', 0):>8.3f}"
        )
