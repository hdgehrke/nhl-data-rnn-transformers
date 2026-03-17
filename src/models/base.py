"""Shared MLP prediction head and base class for all sequence encoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Takes [embed_a, embed_b, context_features] -> scalar goal_differential."""

    def __init__(
        self,
        embed_dim: int,
        n_context: int = 3,  # home_flag, rest_days_a, rest_days_b
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = embed_dim * 2 + n_context
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        embed_a: torch.Tensor,
        embed_b: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embed_a: (batch, embed_dim)
            embed_b: (batch, embed_dim)
            context: (batch, n_context)
        Returns:
            (batch,) scalar predictions
        """
        x = torch.cat([embed_a, embed_b, context], dim=-1)
        return self.net(x).squeeze(-1)


class BaseSequenceModel(nn.Module):
    """Abstract base class. Subclasses implement encode()."""

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.head = MLPHead(embed_dim, n_context=3, hidden_dim=head_hidden_dim, dropout=dropout)

    def encode(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a sequence to a fixed-size embedding.

        Args:
            seq:  (batch, seq_len, feature_dim)
            mask: (batch, seq_len) bool — True means PADDED (invalid)
        Returns:
            (batch, embed_dim)
        """
        raise NotImplementedError

    def forward(
        self,
        seq_a: torch.Tensor,
        seq_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_a:  (batch, seq_len, feature_dim)
            seq_b:  (batch, seq_len, feature_dim)
            mask_a: (batch, seq_len) bool padding mask
            mask_b: (batch, seq_len) bool padding mask
        Returns:
            (batch,) predicted goal differentials
        """
        embed_a = self.encode(seq_a, mask_a)  # shared weights
        embed_b = self.encode(seq_b, mask_b)

        # Context: derive home_flag, rest_days from last token of seq_a/seq_b
        # Feature indices: is_home=24, rest_days=25 (from schema.FEATURE_NAMES)
        IS_HOME_IDX = 24
        REST_DAYS_IDX = 25

        # Last real token = last non-padded position
        def last_real(seq: torch.Tensor, mask: torch.Tensor, feat_idx: int) -> torch.Tensor:
            # seq: (B, S, F), mask: (B, S) — True=padded
            # find last non-padded index per example
            valid = (~mask).float()  # (B, S)
            # last valid position: find rightmost 1
            idx = (valid * torch.arange(seq.size(1), device=seq.device).float()).argmax(dim=1)
            return seq[torch.arange(seq.size(0), device=seq.device), idx, feat_idx]

        home_flag = last_real(seq_a, mask_a, IS_HOME_IDX).unsqueeze(1)  # next game home flag
        rest_a = last_real(seq_a, mask_a, REST_DAYS_IDX).unsqueeze(1)
        rest_b = last_real(seq_b, mask_b, REST_DAYS_IDX).unsqueeze(1)

        context = torch.cat([home_flag, rest_a, rest_b], dim=1)  # (B, 3)
        return self.head(embed_a, embed_b, context)
