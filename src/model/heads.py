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
