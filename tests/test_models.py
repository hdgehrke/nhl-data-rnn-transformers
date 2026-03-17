"""Smoke tests for all model architectures."""

import torch
import pytest

from src.features.schema import FEATURE_DIM
from src.models.rnn import RNNEncoder
from src.models.lstm import LSTMEncoder
from src.models.gru import GRUEncoder
from src.models.transformer import TransformerEncoder, make_transformer_small, make_transformer_medium
from src.models.registry import build_model

BATCH = 8
SEQ_LEN = 10
FEAT_DIM = FEATURE_DIM


def make_batch():
    seq_a = torch.randn(BATCH, SEQ_LEN, FEAT_DIM)
    seq_b = torch.randn(BATCH, SEQ_LEN, FEAT_DIM)
    mask_a = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask_b = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    # Pad first 3 positions for seq_a
    mask_a[:, :3] = True
    return seq_a, seq_b, mask_a, mask_b


@pytest.mark.parametrize("ModelClass", [RNNEncoder, LSTMEncoder, GRUEncoder])
def test_rnn_family_output_shape(ModelClass):
    model = ModelClass(feature_dim=FEAT_DIM, hidden_dim=64, num_layers=1)
    seq_a, seq_b, mask_a, mask_b = make_batch()
    out = model(seq_a, seq_b, mask_a, mask_b)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


def test_transformer_small_output_shape():
    model = make_transformer_small(FEAT_DIM)
    seq_a, seq_b, mask_a, mask_b = make_batch()
    out = model(seq_a, seq_b, mask_a, mask_b)
    assert out.shape == (BATCH,)


def test_transformer_medium_output_shape():
    model = make_transformer_medium(FEAT_DIM)
    seq_a, seq_b, mask_a, mask_b = make_batch()
    out = model(seq_a, seq_b, mask_a, mask_b)
    assert out.shape == (BATCH,)


def test_transformer_full_padding_mask():
    """All-padded sequences (except CLS) should still produce valid output."""
    model = make_transformer_small(FEAT_DIM)
    seq_a, seq_b, _, _ = make_batch()
    full_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)  # all padded
    out = model(seq_a, seq_b, full_mask, full_mask)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("model_type", ["rnn", "lstm", "gru", "transformer"])
def test_registry_builds_all_models(model_type):
    config = {
        "model_type": model_type,
        "hidden_dim": 64,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 1,
        "ffn_dim": 128,
        "dropout": 0.0,
        "head_hidden_dim": 64,
    }
    model = build_model(config, feature_dim=FEAT_DIM)
    seq_a, seq_b, mask_a, mask_b = make_batch()
    out = model(seq_a, seq_b, mask_a, mask_b)
    assert out.shape == (BATCH,)


def test_shared_encoder_weights():
    """Both teams must use identical encoder weights (no separate parameters)."""
    model = LSTMEncoder(feature_dim=FEAT_DIM, hidden_dim=64, num_layers=1)
    # Forward with swapped A/B should give negated output only from head perspective
    # (not testing that here — just that parameter count is same as single encoder)
    total = sum(p.numel() for p in model.parameters())
    lstm_only = sum(p.numel() for p in model.lstm.parameters())
    head_only = sum(p.numel() for p in model.head.parameters())
    assert total == lstm_only + head_only  # no duplicate encoder params
