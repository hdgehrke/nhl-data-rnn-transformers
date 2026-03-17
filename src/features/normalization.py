"""Per-season z-score normalization for feature vectors.

Scalers are fit on training data only (per season) and applied to val/test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_scaler(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Compute per-feature mean and std from training data.

    Returns a dict with keys 'mean' and 'std' as numpy arrays.
    Std of 0 is replaced with 1 to avoid division by zero.
    """
    data = df[feature_cols].to_numpy(dtype=np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean, "std": std, "feature_cols": feature_cols}


def apply_scaler(arr: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply z-score normalization to a (..., feature_dim) array."""
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    return (arr - mean) / std


def fit_scaler_on_sequences(seq: np.ndarray) -> dict:
    """Fit scaler from a (N, seq_len, feature_dim) sequence array.

    Computes stats across N and seq_len dimensions.
    """
    n, s, d = seq.shape
    flat = seq.reshape(-1, d)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean, "std": std}


def apply_scaler_to_sequences(seq: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply scaler to a (N, seq_len, feature_dim) array."""
    return apply_scaler(seq, scaler)


def normalize_split(train: dict, val: dict, test: dict) -> tuple[dict, dict, dict, dict]:
    """Fit scaler on train sequences, apply to all splits.

    Returns (train, val, test, scaler).
    """
    scaler = fit_scaler_on_sequences(train["seq_a"])

    def norm(split: dict) -> dict:
        return {
            **split,
            "seq_a": apply_scaler_to_sequences(split["seq_a"], scaler),
            "seq_b": apply_scaler_to_sequences(split["seq_b"], scaler),
        }

    return norm(train), norm(val), norm(test), scaler
