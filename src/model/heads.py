"""Classification heads for MetaboLM downstream tasks.

SingleTaskHead matches the official fine-tuning code:
    module.classifier = nn.Linear(hidden_size, 1)
Confirmed by reverse-engineering ``best_finetune_model_diabetes.pt``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SingleTaskHead(nn.Module):
    """Single binary classification head (one disease at a time).

    Official architecture::

        pooler_output (768-dim)  ->  Linear(768, 1)  ->  squeeze(-1)  ->  scalar logit

    This is used for E0 reproduction where each disease gets its own model.

    Args:
        hidden_size: Input dimension from backbone pooler (default 768).
    """

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: ``(B, hidden_size)`` from backbone pooler.

        Returns:
            ``(B,)`` logits (one scalar per sample).
        """
        return self.classifier(pooled_output).squeeze(-1)


class HierarchicalMultiTaskHead(nn.Module):
    """Chapter + Leaf joint classification head.

    Architecture::

        pooler_output (768) -> shared_proj (768->256->ReLU->Dropout)
                                    |               |
                              leaf_head (256->16)  chapter_head (256->6)

    Args:
        hidden_size: Input dimension from backbone pooler (default 768).
        proj_size: Shared projection output dimension (default 256).
        num_diseases: Number of leaf disease outputs (default 16).
        num_chapters: Number of ICD-10 chapter outputs (default 6).
        dropout: Dropout rate in shared projection (default 0.1).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        proj_size: int = 256,
        num_diseases: int = 16,
        num_chapters: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_size, proj_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.leaf_head = nn.Linear(proj_size, num_diseases)
        self.chapter_head = nn.Linear(proj_size, num_chapters)

    def forward(
        self, pooled_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pooled_output: ``(B, hidden_size)`` from backbone pooler.

        Returns:
            Tuple of ``(leaf_logits (B, num_diseases), chapter_logits (B, num_chapters))``.
        """
        h = self.shared_proj(pooled_output)
        return self.leaf_head(h), self.chapter_head(h)
