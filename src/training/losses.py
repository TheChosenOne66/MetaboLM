"""Loss functions for MetaboLM fine-tuning.

E0 reproduction uses ``nn.BCEWithLogitsLoss()`` matching the official code:
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_reproduction_loss() -> nn.Module:
    """Return the loss function matching the official fine-tuning code.

    Official: ``nn.BCEWithLogitsLoss()`` with default reduction='mean'.
    No class weighting, no label smoothing.
    """
    return nn.BCEWithLogitsLoss()
