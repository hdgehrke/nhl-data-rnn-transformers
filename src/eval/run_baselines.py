"""Compute and log baseline metrics to MLflow.

Baselines:
  1. Always predict 0 (mean differential)
  2. Always predict home team +0.3
  3. Season-to-date team average differential
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.features.sequences import load_all_processed, build_examples, chronological_split
from src.features.normalization import normalize_split
from src.eval.metrics import baseline_zero, baseline_home_advantage, baseline_team_avg
from src.eval.calibration import print_comparison_table
from src.fetch.download import season_str

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
VAL_DATE = "2022-10-01"
TEST_DATE = "2023-10-01"


def compute_team_avg_baseline(
    train_df: pd.DataFrame,
    test_meta: list[dict],
    test_labels: np.ndarray,
) -> dict:
    """Predict each game's differential using each team's season-average differential from train."""
    # Compute per-team average goal differential from training data (home perspective)
    home = train_df[train_df["is_home"] == 1.0][["team_abbrev", "goal_diff"]].copy()
    away = train_df[train_df["is_home"] == 0.0][["team_abbrev", "goal_diff"]].copy()
    # home goal_diff is from home team perspective; away goal_diff is from away perspective
    # combine: each team's avg goal_diff across all their games
    all_diffs = pd.concat([home, away])
    team_avg = all_diffs.groupby("team_abbrev")["goal_diff"].mean().to_dict()
    global_avg = all_diffs["goal_diff"].mean()

    preds = []
    for m in test_meta:
        home_avg = team_avg.get(m["home_team"], global_avg)
        away_avg = team_avg.get(m["away_team"], global_avg)
        # home - away perspective
        pred = (home_avg - away_avg) / 2.0
        preds.append(pred)

    preds = np.array(preds, dtype=np.float32)
    from src.eval.metrics import mae, rmse, win_direction_accuracy
    return {
        "name": "baseline_team_avg",
        "mae": mae(preds, test_labels),
        "rmse": rmse(preds, test_labels),
        "win_direction_acc": win_direction_accuracy(preds, test_labels),
    }


def main() -> None:
    all_seasons = [season_str(y) for y in range(2011, 2026)]

    print("Loading data...")
    df = load_all_processed(all_seasons)
    data = build_examples(df, seq_len=10)
    train_data, val_data, test_data = chronological_split(
        data, data["meta"], val_date=VAL_DATE, test_date=TEST_DATE
    )
    test_labels = test_data["labels"]
    test_meta = test_data["meta"]

    # Reconstruct train dataframe for team-avg baseline
    train_dates = {m["game_id"] for m in train_data["meta"]}
    train_df = df[df["date"] < TEST_DATE].copy()

    print(f"\nTest set: {len(test_labels)} games\n")

    results = [
        baseline_zero(test_labels),
        baseline_home_advantage(test_labels, advantage=0.3),
        compute_team_avg_baseline(train_df, test_meta, test_labels),
    ]

    print_comparison_table(results)

    # Log to MLflow
    mlflow.set_experiment("nhl_baselines")
    for r in results:
        with mlflow.start_run(run_name=r["name"]):
            mlflow.log_metrics({
                "test_mae": r["mae"],
                "test_rmse": r["rmse"],
                "test_win_direction_acc": r["win_direction_acc"],
            })
    print("\nLogged to MLflow experiment: nhl_baselines")


if __name__ == "__main__":
    main()
