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


class MultiTaskDataset(Dataset):
    """Multi-label dataset for hierarchical multi-task training.

    Returns ``(expressions_168, leaf_labels_16, chapter_labels_6)``
    per sample. Chapter labels are derived from leaf labels: a chapter
    is positive if ANY of its leaf diseases is positive.

    Args:
        X: Expression array of shape ``(N, 168)``.
        leaf_labels: Binary label array of shape ``(N, 16)``.
        disease_to_chapter_idx: Mapping from disease index to chapter index.
        num_chapters: Number of unique chapters (default 6).
    """

    def __init__(
        self,
        X: np.ndarray,
        leaf_labels: np.ndarray,
        disease_to_chapter_idx: dict[int, int],
        num_chapters: int = 6,
    ) -> None:
        self.X = X.astype(np.float32)
        self.leaf_labels = leaf_labels.astype(np.float32)

        # Pre-compute chapter labels: OR over leaf diseases per chapter
        self.chapter_labels = np.zeros(
            (len(leaf_labels), num_chapters), dtype=np.float32
        )
        for disease_idx, chapter_idx in disease_to_chapter_idx.items():
            self.chapter_labels[:, chapter_idx] = np.maximum(
                self.chapter_labels[:, chapter_idx],
                self.leaf_labels[:, disease_idx],
            )

    def __len__(self) -> int:
        return len(self.leaf_labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.leaf_labels[idx], dtype=torch.float32),
            torch.tensor(self.chapter_labels[idx], dtype=torch.float32),
        )
