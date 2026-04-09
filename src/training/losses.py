"""Loss functions for MetaboLM fine-tuning.

E0 reproduction: ``nn.BCEWithLogitsLoss()`` (no class weighting).
E1-E4 hierarchical: joint leaf + chapter BCE + hierarchy consistency penalty.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_reproduction_loss() -> nn.Module:
    """Return the loss function matching the official fine-tuning code.

    Official: ``nn.BCEWithLogitsLoss()`` with default reduction='mean'.
    No class weighting, no label smoothing.
    """
    return nn.BCEWithLogitsLoss()


class HierarchicalLoss(nn.Module):
    """Joint hierarchical loss for multi-task classification.

    ``L = L_leaf + lambda_chapter * L_chapter + mu_hierarchy * L_hierarchy``

    - ``L_leaf``: BCEWithLogitsLoss over 16 disease logits (with pos_weight).
    - ``L_chapter``: BCEWithLogitsLoss over 6 chapter logits (with pos_weight).
    - ``L_hierarchy``: Penalises ``P(leaf) > P(parent_chapter)`` violations.
      Computed as ``mean(max(0, sigmoid(leaf_i) - sigmoid(chapter_of_i)))``
      averaged over all leaf-chapter pairs and all samples.

    Args:
        disease_to_chapter_idx: Mapping from disease index (0-15) to chapter
            index (0-5). Obtain from ``endpoints.get_disease_to_chapter_idx()``.
        leaf_pos_weight: Per-disease pos_weight tensor of shape ``(16,)``.
        chapter_pos_weight: Per-chapter pos_weight tensor of shape ``(6,)``.
        lambda_chapter: Weight for chapter loss (default 1.0).
        mu_hierarchy: Weight for hierarchy penalty (default 0.1).
    """

    def __init__(
        self,
        disease_to_chapter_idx: dict[int, int],
        leaf_pos_weight: torch.Tensor,
        chapter_pos_weight: torch.Tensor,
        lambda_chapter: float = 1.0,
        mu_hierarchy: float = 0.1,
    ) -> None:
        super().__init__()
        self.disease_to_chapter_idx = disease_to_chapter_idx
        self.lambda_chapter = lambda_chapter
        self.mu_hierarchy = mu_hierarchy

        self.leaf_criterion = nn.BCEWithLogitsLoss(pos_weight=leaf_pos_weight)
        self.chapter_criterion = nn.BCEWithLogitsLoss(pos_weight=chapter_pos_weight)

        # Pre-compute index tensors for vectorised hierarchy penalty
        leaf_indices = sorted(disease_to_chapter_idx.keys())
        chapter_indices = [disease_to_chapter_idx[i] for i in leaf_indices]
        self.register_buffer(
            "_leaf_idx", torch.tensor(leaf_indices, dtype=torch.long)
        )
        self.register_buffer(
            "_chapter_idx", torch.tensor(chapter_indices, dtype=torch.long)
        )

    def forward(
        self,
        leaf_logits: torch.Tensor,
        chapter_logits: torch.Tensor,
        leaf_labels: torch.Tensor,
        chapter_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute joint loss.

        Args:
            leaf_logits: ``(B, 16)`` raw logits for leaf diseases.
            chapter_logits: ``(B, 6)`` raw logits for chapters.
            leaf_labels: ``(B, 16)`` binary labels.
            chapter_labels: ``(B, 6)`` binary labels.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        l_leaf = self.leaf_criterion(leaf_logits, leaf_labels)
        l_chapter = self.chapter_criterion(chapter_logits, chapter_labels)

        # Hierarchy penalty: penalise P(leaf) > P(parent_chapter)
        leaf_probs = torch.sigmoid(leaf_logits)        # (B, 16)
        chapter_probs = torch.sigmoid(chapter_logits)  # (B, 6)

        # Gather: for each leaf, get its parent chapter probability
        leaf_p = leaf_probs[:, self._leaf_idx]                          # (B, 16)
        parent_p = chapter_probs[:, self._chapter_idx]                  # (B, 16)
        violations = torch.clamp(leaf_p - parent_p, min=0)             # (B, 16)
        l_hierarchy = violations.mean()

        total = l_leaf + self.lambda_chapter * l_chapter + self.mu_hierarchy * l_hierarchy

        components = {
            "loss_leaf": l_leaf.item(),
            "loss_chapter": l_chapter.item(),
            "loss_hierarchy": l_hierarchy.item(),
            "loss_total": total.item(),
        }
        return total, components
