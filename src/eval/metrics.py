"""Evaluation metrics: MAE, RMSE, win-direction accuracy."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.base import BaseSequenceModel
from src.train.trainer import get_device


def mae(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - labels)))


def rmse(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - labels) ** 2)))


def win_direction_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of games where sign(pred) == sign(label). Ties (0) counted as wrong."""
    correct = (np.sign(preds) == np.sign(labels)) & (labels != 0)
    return float(correct.sum() / max((labels != 0).sum(), 1))


def collect_predictions(
    model: BaseSequenceModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq_a, seq_b, mask_a, mask_b, labels = batch
            seq_a = seq_a.to(device)
            seq_b = seq_b.to(device)
            mask_a = mask_a.to(device)
            mask_b = mask_b.to(device)
            preds = model(seq_a, seq_b, mask_a, mask_b)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def evaluate_loader(
    model: BaseSequenceModel,
    loader: DataLoader,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = get_device()
    preds, labels = collect_predictions(model, loader, device)
    return {
        "mae": mae(preds, labels),
        "rmse": rmse(preds, labels),
        "win_direction_acc": win_direction_accuracy(preds, labels),
    }


# ---- Baseline models ----

def baseline_zero(labels: np.ndarray) -> dict:
    """Always predict 0 goal differential."""
    preds = np.zeros_like(labels)
    return {
        "name": "baseline_zero",
        "mae": mae(preds, labels),
        "rmse": rmse(preds, labels),
        "win_direction_acc": win_direction_accuracy(preds, labels),
    }


def baseline_home_advantage(labels: np.ndarray, advantage: float = 0.3) -> dict:
    """Always predict home team wins by `advantage` goals."""
    preds = np.full_like(labels, advantage)
    return {
        "name": f"baseline_home+{advantage}",
        "mae": mae(preds, labels),
        "rmse": rmse(preds, labels),
        "win_direction_acc": win_direction_accuracy(preds, labels),
    }


def baseline_team_avg(
    preds_by_team_diff: np.ndarray,
    labels: np.ndarray,
    name: str = "baseline_team_avg",
) -> dict:
    """Evaluate a team-average-differential baseline."""
    return {
        "name": name,
        "mae": mae(preds_by_team_diff, labels),
        "rmse": rmse(preds_by_team_diff, labels),
        "win_direction_acc": win_direction_accuracy(preds_by_team_diff, labels),
    }
