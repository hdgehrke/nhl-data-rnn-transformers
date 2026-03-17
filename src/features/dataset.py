"""PyTorch Dataset for NHL sequence data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NHLSequenceDataset(Dataset):
    """Dataset that yields (seq_a, seq_b, mask_a, mask_b, label) tuples."""

    def __init__(self, data: dict):
        self.seq_a = torch.from_numpy(data["seq_a"]).float()    # (N, seq_len, feat_dim)
        self.seq_b = torch.from_numpy(data["seq_b"]).float()
        self.mask_a = torch.from_numpy(data["mask_a"])           # (N, seq_len) bool
        self.mask_b = torch.from_numpy(data["mask_b"])
        self.labels = torch.from_numpy(data["labels"]).float()   # (N,)
        self.meta = data.get("meta", [{}] * len(self.labels))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.seq_a[idx],
            self.seq_b[idx],
            self.mask_a[idx],
            self.mask_b[idx],
            self.labels[idx],
        )


def make_dataloaders(
    train: dict,
    val: dict,
    test: dict,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = NHLSequenceDataset(train)
    val_ds = NHLSequenceDataset(val)
    test_ds = NHLSequenceDataset(test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader
