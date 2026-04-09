"""PyTorch Dataset for MetaboLM fine-tuning.

Equivalent to the official ``FineTuneDataset`` in MetaboLM_Fine-tuning.py:
each sample returns (expressions, label, eid).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FineTuneDataset(Dataset):
    """Binary classification dataset for a single disease.

    Matches the official implementation exactly:
    - Stores pre-normalised expression values as float32
    - Returns ``(expressions_tensor, label_scalar, eid)``

    Args:
        X: Expression array of shape ``(N, num_metabolites)``.
        y: Binary label array of shape ``(N,)``.
        eids: Participant ID list of length ``N``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, eids: list) -> None:
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.eids = eids

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            self.eids[idx],
        )
