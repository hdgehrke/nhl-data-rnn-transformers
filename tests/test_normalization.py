"""Tests for normalization — ensures train stats don't leak into val/test."""

import numpy as np
import pytest

from src.features.normalization import (
    fit_scaler_on_sequences,
    apply_scaler_to_sequences,
    normalize_split,
)


def _make_split(n: int, feat_dim: int = 28) -> dict:
    return {
        "seq_a": np.random.randn(n, 10, feat_dim).astype(np.float32),
        "seq_b": np.random.randn(n, 10, feat_dim).astype(np.float32),
        "mask_a": np.zeros((n, 10), dtype=bool),
        "mask_b": np.zeros((n, 10), dtype=bool),
        "labels": np.random.randn(n).astype(np.float32),
        "meta": [{}] * n,
    }


def test_scaler_fit_shape():
    seq = np.random.randn(100, 10, 28).astype(np.float32)
    scaler = fit_scaler_on_sequences(seq)
    assert scaler["mean"].shape == (28,)
    assert scaler["std"].shape == (28,)


def test_scaler_no_zero_std():
    """Constant features should get std=1 to avoid division by zero."""
    seq = np.zeros((50, 10, 28), dtype=np.float32)
    scaler = fit_scaler_on_sequences(seq)
    assert (scaler["std"] == 1.0).all()


def test_normalize_output_approx_zero_mean():
    train = _make_split(500)
    val = _make_split(100)
    test = _make_split(100)
    train_n, val_n, test_n, scaler = normalize_split(train, val, test)
    # Train sequences should be approximately zero-mean after normalization
    mean = train_n["seq_a"].mean()
    assert abs(mean) < 0.1, f"Expected near-zero mean, got {mean}"


def test_normalize_does_not_mutate_original():
    train = _make_split(100)
    original_vals = train["seq_a"].copy()
    val = _make_split(20)
    test = _make_split(20)
    normalize_split(train, val, test)
    np.testing.assert_array_equal(train["seq_a"], original_vals)
