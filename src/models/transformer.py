"""Transformer sequence encoder with CLS token.

Supports small (2-4 layers, d_model=128) and medium (4-8 layers, d_model=256).
Uses PyTorch's built-in TransformerEncoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.base import BaseSequenceModel


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(BaseSequenceModel):
    """Transformer-based sequence encoder.

    Prepends a learnable [CLS] token; uses its output as the sequence embedding.
    Shared weights used for both team A and team B.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ):
        super().__init__(feature_dim, d_model, head_hidden_dim, dropout)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def encode(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq:  (batch, seq_len, feature_dim)
            mask: (batch, seq_len) bool — True = padded token
        Returns:
            (batch, d_model)
        """
        batch_size = seq.size(0)

        # Project features to d_model
        x = self.input_proj(seq)              # (B, S, d_model)
        x = self.pos_enc(x)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                   # (B, 1+S, d_model)

        # Extend mask with False for CLS (always valid)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)    # (B, 1+S)

        # TransformerEncoder: src_key_padding_mask=True means IGNORE that position
        out = self.transformer(x, src_key_padding_mask=full_mask)  # (B, 1+S, d_model)
        cls_out = out[:, 0, :]               # (B, d_model) — CLS token output
        return self.dropout_layer(self.norm(cls_out))


def make_transformer_small(feature_dim: int, dropout: float = 0.1) -> TransformerEncoder:
    """2-layer Transformer, d_model=128, nhead=4 — fast training."""
    return TransformerEncoder(
        feature_dim=feature_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        ffn_dim=256,
        dropout=dropout,
        head_hidden_dim=128,
    )


def make_transformer_medium(feature_dim: int, dropout: float = 0.1) -> TransformerEncoder:
    """4-layer Transformer, d_model=256, nhead=8 — more capacity."""
    return TransformerEncoder(
        feature_dim=feature_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        ffn_dim=512,
        dropout=dropout,
        head_hidden_dim=256,
    )
