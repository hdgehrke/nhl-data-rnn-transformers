"""Training loop with MLflow logging and early stopping."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseSequenceModel

CHECKPOINT_DIR = Path(__file__).parents[2] / "mlruns"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_to_device(batch: tuple, device: torch.device) -> tuple:
    return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)


def run_epoch(
    model: BaseSequenceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    is_train: bool,
) -> float:
    model.train(is_train)
    total_loss = 0.0
    n = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            seq_a, seq_b, mask_a, mask_b, labels = _batch_to_device(batch, device)
            preds = model(seq_a, seq_b, mask_a, mask_b)
            loss = criterion(preds, labels)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            n += len(labels)

    return total_loss / n if n > 0 else float("nan")


def train(
    model: BaseSequenceModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    experiment_name: str = "nhl_predictions",
    run_name: str | None = None,
) -> dict:
    """Train a model with early stopping; log to MLflow.

    Returns dict with best_val_loss, run_id, and model state.
    """
    device = get_device()
    print(f"Training on device: {device}")
    model = model.to(device)

    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    epochs = config.get("epochs", 50)
    patience = config.get("patience", 10)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**config, "device": str(device)})

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = run_epoch(model, train_loader, optimizer, criterion, device, is_train=True)
            val_loss = run_epoch(model, val_loader, None, criterion, device, is_train=False)
            elapsed = time.time() - t0

            scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_mse": val_loss},
                step=epoch,
            )

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_MSE={train_loss:.4f} | val_MSE={val_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        mlflow.log_metric("best_val_mse", best_val_loss)
        mlflow.pytorch.log_model(model, "model")

        return {
            "best_val_loss": best_val_loss,
            "run_id": run.info.run_id,
            "history": history,
        }
