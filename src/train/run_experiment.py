"""CLI entry point to train a model variant from a YAML config.

Usage:
    python -m src.train.run_experiment --config configs/lstm_base.yaml
    python -m src.train.run_experiment --config configs/transformer_small.yaml
"""

from __future__ import annotations

import argparse

import yaml

from src.features.dataset import make_dataloaders
from src.features.normalization import normalize_split
from src.features.schema import FEATURE_DIM
from src.features.sequences import build_examples, chronological_split, load_all_processed
from src.models.registry import build_model
from src.train.trainer import train


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seq-len", type=int, default=None, help="Override seq_len from config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seq_len is not None:
        config["seq_len"] = args.seq_len

    seq_len = config.get("seq_len", 10)
    batch_size = config.get("batch_size", 256)
    val_date = config.get("val_date", "2022-10-01")
    test_date = config.get("test_date", "2023-10-01")
    train_start = config.get("train_start", 2011)
    train_end = config.get("train_end", 2024)

    sport = config.get("sport", "nhl")
    if sport == "nhl":
        from src.fetch.download import season_str
        all_seasons = [season_str(y) for y in range(train_start, train_end + 1)]
        prefix = "games"
    elif sport == "mlb":
        all_seasons = [str(y) for y in range(train_start, train_end + 1)]
        prefix = "mlb_games"
    elif sport == "nba":
        from src.fetch.nba_client import nba_season_str
        all_seasons = [nba_season_str(y) for y in range(train_start, train_end + 1)]
        prefix = "nba_games"
    else:
        raise ValueError(f"Unknown sport: {sport!r}. Expected 'nhl', 'mlb', or 'nba'.")

    print("Loading processed data...")
    df = load_all_processed(all_seasons, prefix=prefix)
    print(f"  {len(df)} team-game rows loaded")

    print(f"Building sequences (seq_len={seq_len})...")
    data = build_examples(df, seq_len=seq_len)
    print(f"  {len(data['labels'])} game examples")

    train_data, val_data, test_data = chronological_split(
        data, data["meta"], val_date=val_date, test_date=test_date
    )
    print(
        f"  Train: {len(train_data['labels'])} | "
        f"Val: {len(val_data['labels'])} | "
        f"Test: {len(test_data['labels'])}"
    )

    train_data, val_data, test_data, scaler = normalize_split(train_data, val_data, test_data)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_data, val_data, test_data, batch_size=batch_size
    )

    feature_dim = FEATURE_DIM
    model = build_model(config, feature_dim=feature_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_type']} | {n_params:,} parameters")

    run_name = config.get("run_name", config["model_type"])
    experiment_name = config.get("experiment", f"nhl_{config['model_type']}")

    result = train(
        model,
        train_loader,
        val_loader,
        config,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    print(f"\nBest val MSE: {result['best_val_loss']:.4f}")
    print(f"MLflow run ID: {result['run_id']}")

    # Evaluate on test set

    from src.eval.metrics import evaluate_loader
    from src.train.trainer import get_device

    device = get_device()
    model = model.to(device)

    if len(test_data["labels"]) == 0:
        print("\nNo test set (full-data training config) — skipping test eval.")
    else:
        test_metrics = evaluate_loader(model, test_loader, device)
        print("\nTest set metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        import mlflow
        with mlflow.start_run(run_id=result["run_id"]):
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == "__main__":
    main()
