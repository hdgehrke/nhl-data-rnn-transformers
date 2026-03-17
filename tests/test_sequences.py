"""Tests for sequence building and padding."""

import numpy as np
import pandas as pd
import pytest

from src.features.schema import FEATURE_NAMES, FEATURE_DIM
from src.features.sequences import get_team_sequence, get_padding_mask


def _make_team_df(n_games: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n_games):
        row = {name: float(i) for name in FEATURE_NAMES}
        row["date"] = pd.Timestamp("2022-01-01") + pd.Timedelta(days=i)
        rows.append(row)
    return pd.DataFrame(rows)


def test_sequence_shape_full():
    df = _make_team_df(10)
    cutoff = pd.Timestamp("2022-01-11")  # after all 10 games
    seq = get_team_sequence(df, cutoff, seq_len=10, feature_cols=FEATURE_NAMES)
    assert seq.shape == (10, FEATURE_DIM)
    assert seq.dtype == np.float32


def test_sequence_padding():
    df = _make_team_df(3)
    cutoff = pd.Timestamp("2022-01-04")
    seq = get_team_sequence(df, cutoff, seq_len=10, feature_cols=FEATURE_NAMES)
    assert seq.shape == (10, FEATURE_DIM)
    # First 7 rows should be zeros (padding)
    assert np.all(seq[:7] == 0.0)
    # Last 3 rows should be nonzero (real data from game 0,1,2 — values are i=0..2)
    # game 0 has all features=0, game 1 has features=1, game 2 has features=2
    assert np.all(seq[8] == 1.0) or seq[8, 1] == 1.0  # game index 1


def test_padding_mask_shape():
    df = _make_team_df(3)
    cutoff = pd.Timestamp("2022-01-04")
    mask = get_padding_mask(df, cutoff, seq_len=10)
    assert mask.shape == (10,)
    assert mask.dtype == bool
    assert mask[:7].all()       # padded positions are True
    assert not mask[7:].any()   # real positions are False


def test_sequence_respects_cutoff():
    df = _make_team_df(10)
    # Cutoff after game 5 (index 0..4 = 5 games before cutoff)
    cutoff = pd.Timestamp("2022-01-06")
    seq = get_team_sequence(df, cutoff, seq_len=10, feature_cols=FEATURE_NAMES)
    # 5 real games, 5 padded
    mask = get_padding_mask(df, cutoff, seq_len=10)
    assert mask[:5].all()
    assert not mask[5:].any()


def test_no_data_leakage():
    """Games on or after cutoff date must not appear in sequence."""
    df = _make_team_df(10)
    cutoff = pd.Timestamp("2022-01-06")  # day 5 game is on this date
    seq = get_team_sequence(df, cutoff, seq_len=10, feature_cols=FEATURE_NAMES)
    # day 5 game would have all features = 5.0 — should not appear
    assert not np.any(np.all(seq == 5.0, axis=1))
