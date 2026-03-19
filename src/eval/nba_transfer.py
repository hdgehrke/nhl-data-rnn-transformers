"""Zero-shot and fine-tuned NBA transfer evaluation.

Pipeline:
  1. Load NBA processed data, build sequences, split chronologically
  2. Load NHL-pretrained model weights from MLflow
  3. Zero-shot eval: run NHL model directly on NBA test set
  4. Fine-tune: unfreeze all weights, train on NBA train split, eval on test
  5. Print comparison table vs NBA baselines

Usage:
    python -m src.eval.nba_transfer --nhl-run-id <mlflow_run_id>
    python -m src.eval.nba_transfer --nhl-run-id <mlflow_run_id> --no-finetune
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch

from src.features.sequences import build_examples, chronological_split
from src.features.normalization import normalize_split
from src.features.dataset import make_dataloaders, NHLSequenceDataset
from src.eval.metrics import (
    evaluate_loader,
    baseline_zero,
    baseline_home_advantage,
    baseline_team_avg,
)
from src.eval.calibration import print_comparison_table
from src.train.trainer import get_device, train
from src.fetch.nba_client import nba_season_str

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# NBA chronological split: train 2015–2021, val 2021-22, test 2022–2024
NBA_VAL_DATE = "2021-10-01"
NBA_TEST_DATE = "2022-10-01"
NBA_SEQ_LEN = 45


def load_nba_data(seasons: list[str]) -> tuple[dict, dict, dict, dict]:
    """Load processed NBA parquet files and return train/val/test splits."""
    import pandas as pd

    dfs = []
    for s in seasons:
        path = PROCESSED_DIR / f"nba_games_{s}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
        else:
            print(f"  Missing: {path.name}")

    if not dfs:
        raise FileNotFoundError("No NBA processed data found. Run nba_tokenizer first.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df)} NBA team-game rows across {df['season'].nunique()} seasons")

    data = build_examples(df, seq_len=NBA_SEQ_LEN)
    print(f"  {len(data['labels'])} game examples")

    train_data, val_data, test_data = chronological_split(
        data, data["meta"], val_date=NBA_VAL_DATE, test_date=NBA_TEST_DATE
    )
    print(
        f"  Train: {len(train_data['labels'])} | "
        f"Val: {len(val_data['labels'])} | "
        f"Test: {len(test_data['labels'])}"
    )
    train_data, val_data, test_data, scaler = normalize_split(train_data, val_data, test_data)
    return train_data, val_data, test_data, scaler


def zero_shot_eval(model, test_loader, device) -> dict:
    metrics = evaluate_loader(model, test_loader, device)
    return {"name": "zero_shot_nhl→nba", **metrics}


def finetune_eval(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs: int = 30,
    lr: float = 1e-4,
) -> dict:
    """Fine-tune the NHL model on NBA training data and eval on test."""
    finetune_config = {
        "model_type": "transformer",
        "lr": lr,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "patience": 8,
    }
    result = train(
        model,
        train_loader,
        val_loader,
        finetune_config,
        experiment_name="nba_finetune",
        run_name="transformer_small_nhl→nba_finetune",
    )
    print(f"  Best val MSE after fine-tuning: {result['best_val_loss']:.4f}")
    model = model.to(device)
    metrics = evaluate_loader(model, test_loader, device)
    return {"name": "finetuned_nhl→nba", **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhl-run-id", required=True, help="MLflow run ID of trained NHL model")
    parser.add_argument("--no-finetune", action="store_true", help="Skip fine-tuning step")
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # --- Load NBA data ---
    print("\nLoading NBA data...")
    seasons = [nba_season_str(y) for y in range(2015, 2025)]
    train_data, val_data, test_data, scaler = load_nba_data(seasons)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_data, val_data, test_data, batch_size=256
    )

    # --- Baselines ---
    test_labels = test_data["labels"]
    test_meta = test_data["meta"]
    train_df_proxy = None  # for team-avg baseline we'd need original df

    print("\nBaselines:")
    baselines = [
        baseline_zero(test_labels),
        baseline_home_advantage(test_labels, advantage=3.0),  # NBA home advantage ~3pts
    ]

    # --- Load NHL model ---
    print(f"\nLoading NHL model from run {args.nhl_run_id[:8]}...")
    model = mlflow.pytorch.load_model(
        f"runs:/{args.nhl_run_id}/model", map_location=device
    )
    model = model.to(device)

    # --- Zero-shot ---
    print("\nZero-shot evaluation (no NBA training)...")
    zs = zero_shot_eval(model, test_loader, device)
    print(f"  MAE={zs['mae']:.4f}  RMSE={zs['rmse']:.4f}  WinAcc={zs['win_direction_acc']:.3f}")

    results = baselines + [zs]

    # --- Fine-tune ---
    if not args.no_finetune:
        print(f"\nFine-tuning on NBA train data (lr={args.finetune_lr}, epochs={args.finetune_epochs})...")
        # Reload fresh copy of model for fine-tuning (don't mutate zero-shot model)
        model_ft = mlflow.pytorch.load_model(
            f"runs:/{args.nhl_run_id}/model", map_location=device
        )
        model_ft = model_ft.to(device)
        ft = finetune_eval(
            model_ft, train_loader, val_loader, test_loader, device,
            epochs=args.finetune_epochs, lr=args.finetune_lr,
        )
        print(f"  MAE={ft['mae']:.4f}  RMSE={ft['rmse']:.4f}  WinAcc={ft['win_direction_acc']:.3f}")
        results.append(ft)

    # --- Summary ---
    print("\n=== NBA Transfer Results ===")
    print_comparison_table(results)

    # Log to MLflow
    mlflow.set_experiment("nba_transfer")
    for r in results:
        with mlflow.start_run(run_name=r["name"]):
            mlflow.log_metrics({
                "test_mae": r["mae"],
                "test_rmse": r["rmse"],
                "test_win_direction_acc": r["win_direction_acc"],
            })
            mlflow.log_param("nhl_run_id", args.nhl_run_id)


if __name__ == "__main__":
    main()
