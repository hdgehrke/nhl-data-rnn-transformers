"""LSTM sequence encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseSequenceModel


class LSTMEncoder(BaseSequenceModel):
    """Multi-layer LSTM encoder with shared weights for both teams."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ):
        super().__init__(feature_dim, hidden_dim, head_hidden_dim, dropout)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def encode(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(seq)
        return self.dropout(hidden[-1])  # last layer hidden: (batch, hidden_dim)
