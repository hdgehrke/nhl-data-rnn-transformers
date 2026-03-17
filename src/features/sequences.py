"""Build (team_A_seq, team_B_seq, label) training examples from processed game data.

Each example corresponds to one game. Both teams' sequences of the N most recent
games prior to the current game are returned. The label is goal_differential
(team A goals - team B goals from team A's perspective).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.schema import FEATURE_NAMES, FEATURE_DIM
from src.features.normalization import fit_scaler, apply_scaler

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# Metadata columns not included in the feature vector
META_COLS = [
    "sport", "season", "game_id", "team_id", "team_abbrev",
    "opponent_id", "opponent_abbrev", "date", "game_number",
]


def load_all_processed(seasons: list[str]) -> pd.DataFrame:
    dfs = []
    for season in seasons:
        path = PROCESSED_DIR / f"games_{season}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        raise FileNotFoundError("No processed parquet files found. Run tokenizer first.")
    return pd.concat(dfs, ignore_index=True)


def build_team_history(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return dict mapping team_id -> DataFrame of games sorted by date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team_id", "date"])
    return {tid: grp.reset_index(drop=True) for tid, grp in df.groupby("team_id")}


def get_team_sequence(
    history_df: pd.DataFrame,
    before_date: pd.Timestamp,
    seq_len: int,
    feature_cols: list[str],
) -> np.ndarray:
    """Return last seq_len games before before_date as a (seq_len, feature_dim) array.

    Pads with zeros at the front if fewer than seq_len games are available.
    """
    past = history_df[history_df["date"] < before_date]
    past = past.tail(seq_len)
    vectors = past[feature_cols].to_numpy(dtype=np.float32)

    # Zero-pad at the front
    pad_len = seq_len - len(vectors)
    if pad_len > 0:
        padding = np.zeros((pad_len, len(feature_cols)), dtype=np.float32)
        vectors = np.concatenate([padding, vectors], axis=0)

    return vectors  # shape: (seq_len, feature_dim)


def get_padding_mask(
    history_df: pd.DataFrame,
    before_date: pd.Timestamp,
    seq_len: int,
) -> np.ndarray:
    """Return boolean mask (True = padded / invalid) of shape (seq_len,)."""
    n_real = min(len(history_df[history_df["date"] < before_date]), seq_len)
    mask = np.ones(seq_len, dtype=bool)
    mask[seq_len - n_real :] = False  # real games are at the end
    return mask


def build_examples(
    df: pd.DataFrame,
    seq_len: int = 10,
    scaler_params: dict | None = None,
) -> dict:
    """Build all training examples from a processed DataFrame.

    Returns a dict with:
        seq_a: np.ndarray (N, seq_len, feature_dim) — home team sequences
        seq_b: np.ndarray (N, seq_len, feature_dim) — away team sequences
        mask_a: np.ndarray (N, seq_len) bool — padding mask for seq_a
        mask_b: np.ndarray (N, seq_len) bool — padding mask for seq_b
        labels: np.ndarray (N,) — goal differential (home - away)
        meta:   list[dict] — game metadata per example
    """
    feature_cols = FEATURE_NAMES

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    team_history = build_team_history(df)

    # Each game appears twice in df (once per team). Reconstruct game-level view.
    # Build game-level df: one row per game with home_team and away_team info.
    game_cols = ["game_id", "date", "team_id", "team_abbrev", "goals_for", "goals_against"]
    home_df = df[df["is_home"] == 1.0][game_cols].copy()
    away_df = df[df["is_home"] == 0.0][game_cols].copy()

    games = home_df.merge(
        away_df,
        on="game_id",
        suffixes=("_home", "_away"),
    )

    seq_a_list, seq_b_list = [], []
    mask_a_list, mask_b_list = [], []
    labels, meta = [], []

    for _, row in games.iterrows():
        date = row["date_home"]
        home_id = row["team_id_home"]
        away_id = row["team_id_away"]

        hist_home = team_history.get(home_id, pd.DataFrame(columns=["date"] + feature_cols))
        hist_away = team_history.get(away_id, pd.DataFrame(columns=["date"] + feature_cols))

        if not hasattr(hist_home, "empty"):
            hist_home = pd.DataFrame(hist_home)
        if not hasattr(hist_away, "empty"):
            hist_away = pd.DataFrame(hist_away)

        seq_a = get_team_sequence(hist_home, date, seq_len, feature_cols)
        seq_b = get_team_sequence(hist_away, date, seq_len, feature_cols)
        mask_a = get_padding_mask(hist_home, date, seq_len)
        mask_b = get_padding_mask(hist_away, date, seq_len)

        label = float(row["goals_for_home"]) - float(row["goals_for_away"])

        seq_a_list.append(seq_a)
        seq_b_list.append(seq_b)
        mask_a_list.append(mask_a)
        mask_b_list.append(mask_b)
        labels.append(label)
        meta.append({
            "game_id": row["game_id"],
            "date": str(date.date()),
            "home_team": row["team_abbrev_home"],
            "away_team": row["team_abbrev_away"],
            "home_goals": int(row["goals_for_home"]),
            "away_goals": int(row["goals_for_away"]),
        })

    return {
        "seq_a": np.stack(seq_a_list),    # (N, seq_len, feature_dim)
        "seq_b": np.stack(seq_b_list),
        "mask_a": np.stack(mask_a_list),  # (N, seq_len)
        "mask_b": np.stack(mask_b_list),
        "labels": np.array(labels, dtype=np.float32),  # (N,)
        "meta": meta,
    }


def chronological_split(
    data: dict,
    meta: list[dict],
    val_date: str = "2022-10-01",
    test_date: str = "2023-10-01",
) -> tuple[dict, dict, dict]:
    """Split examples chronologically into train/val/test.

    No shuffling — order must be preserved to prevent data leakage.
    """
    dates = np.array([m["date"] for m in meta])
    train_mask = dates < val_date
    val_mask = (dates >= val_date) & (dates < test_date)
    test_mask = dates >= test_date

    def subset(mask):
        return {
            "seq_a": data["seq_a"][mask],
            "seq_b": data["seq_b"][mask],
            "mask_a": data["mask_a"][mask],
            "mask_b": data["mask_b"][mask],
            "labels": data["labels"][mask],
            "meta": [m for m, b in zip(meta, mask) if b],
        }

    return subset(train_mask), subset(val_mask), subset(test_mask)
