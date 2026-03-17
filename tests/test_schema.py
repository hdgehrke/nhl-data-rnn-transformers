"""Tests for GameToken schema and feature vector construction."""

import numpy as np
import pytest

from src.features.schema import GameToken, FEATURE_DIM, FEATURE_NAMES


def make_token(**kwargs) -> GameToken:
    defaults = dict(
        sport="nhl",
        season="20112012",
        game_id="2011020001",
        team_id="10",
        team_abbrev="BOS",
        opponent_id="6",
        opponent_abbrev="TOR",
        date="2011-10-07",
        game_number=1,
    )
    defaults.update(kwargs)
    return GameToken(**defaults)


def test_feature_vector_shape():
    tok = make_token(goals_for=3, goals_against=1)
    vec = tok.to_vector()
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_feature_vector_values():
    tok = make_token(
        goals_for=3,
        goals_against=1,
        shots_for=30,
        shots_against=25,
        is_home=1,
        rest_days=2,
    )
    vec = tok.to_vector()
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    assert vec[idx["goals_for"]] == 3.0
    assert vec[idx["goals_against"]] == 1.0
    assert vec[idx["goal_diff"]] == 2.0
    assert vec[idx["is_home"]] == 1.0
    assert vec[idx["rest_days"]] == 2.0


def test_rest_days_capped_at_7():
    tok = make_token(rest_days=10)
    vec = tok.to_vector()
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert vec[idx["rest_days"]] == 7.0


def test_pp_pct_zero_opps():
    tok = make_token(pp_goals_for=0, pp_opps_for=0)
    assert tok.pp_pct == 0.0


def test_pk_pct_zero_opps():
    tok = make_token(pp_goals_against=0, pp_opps_against=0)
    assert tok.pk_pct == 1.0


def test_shooting_pct_zero_shots():
    tok = make_token(goals_for=0, shots_for=0)
    assert tok.shooting_pct == 0.0


def test_save_pct_zero_shots():
    tok = make_token(goals_against=0, shots_against=0)
    assert tok.save_pct == 1.0


def test_feature_names_count():
    assert len(FEATURE_NAMES) == FEATURE_DIM
