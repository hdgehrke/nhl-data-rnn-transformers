"""Model registry — build a model from a config dict."""

from __future__ import annotations

from src.models.rnn import RNNEncoder
from src.models.lstm import LSTMEncoder
from src.models.gru import GRUEncoder
from src.models.transformer import TransformerEncoder
from src.models.base import BaseSequenceModel


def build_model(config: dict, feature_dim: int) -> BaseSequenceModel:
    """Instantiate a model from a config dict.

    Expected config keys:
        model_type: "rnn" | "lstm" | "gru" | "transformer"
        hidden_dim / d_model: int
        num_layers: int
        dropout: float
        (transformer only) nhead, ffn_dim
    """
    mtype = config["model_type"].lower()
    dropout = config.get("dropout", 0.1)
    num_layers = config.get("num_layers", 2)

    if mtype == "rnn":
        return RNNEncoder(
            feature_dim=feature_dim,
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=num_layers,
            dropout=dropout,
            head_hidden_dim=config.get("head_hidden_dim", 128),
        )
    if mtype == "lstm":
        return LSTMEncoder(
            feature_dim=feature_dim,
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=num_layers,
            dropout=dropout,
            head_hidden_dim=config.get("head_hidden_dim", 128),
        )
    if mtype == "gru":
        return GRUEncoder(
            feature_dim=feature_dim,
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=num_layers,
            dropout=dropout,
            head_hidden_dim=config.get("head_hidden_dim", 128),
        )
    if mtype == "transformer":
        return TransformerEncoder(
            feature_dim=feature_dim,
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            num_layers=num_layers,
            ffn_dim=config.get("ffn_dim", 256),
            dropout=dropout,
            head_hidden_dim=config.get("head_hidden_dim", 128),
        )
    raise ValueError(f"Unknown model_type: {mtype!r}")
