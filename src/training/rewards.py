"""Reward functions for GRPO reinforcement learning (E5-E6).

Two reward signals for optimising multi-task disease prediction beyond SFT:

- **Calibration reward** (E5): Penalises miscalibrated predictions.
  Uses negative Brier score as a per-sample, differentiable proxy for
  Expected Calibration Error (ECE).  GRPO then optimises this via
  group-relative policy gradient, avoiding the need for differentiable
  binning.

- **Hierarchy reward** (E6): Penalises violations of ICD-10 hierarchy
  (P(leaf) > P(parent_chapter)).  While SFT already includes a hierarchy
  penalty in the loss, GRPO can further reduce violations by treating
  violation-free-ness as a reward signal explored through stochastic
  dropout sampling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def calibration_reward(
    leaf_logits: torch.Tensor,
    leaf_labels: torch.Tensor,
    chapter_logits: torch.Tensor | None = None,
    chapter_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample calibration reward (negative Brier score).

    R_cal = - (1/D) * sum_i (sigma(z_i) - y_i)^2

    Lower Brier score = better calibration = higher reward.

    Args:
        leaf_logits: ``(B, 16)`` raw logits.
        leaf_labels: ``(B, 16)`` binary labels.
        chapter_logits: Unused, accepted for uniform interface.
        chapter_labels: Unused, accepted for uniform interface.

    Returns:
        ``(B,)`` reward per sample (non-positive).
    """
    probs = torch.sigmoid(leaf_logits)  # (B, 16)
    brier = (probs - leaf_labels) ** 2  # (B, 16)
    return -brier.mean(dim=-1)          # (B,)


def hierarchy_reward(
    leaf_logits: torch.Tensor,
    leaf_labels: torch.Tensor,
    chapter_logits: torch.Tensor,
    chapter_labels: torch.Tensor,
    leaf_to_chapter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample hierarchy consistency reward.

    R_hier = - (1/D) * sum_i max(0, P(leaf_i) - P(chapter_parent_i))

    Zero when all leaf probs <= parent chapter probs (ideal).

    Args:
        leaf_logits: ``(B, 16)`` raw logits.
        leaf_labels: ``(B, 16)`` binary labels (unused, for interface).
        chapter_logits: ``(B, 6)`` raw chapter logits.
        chapter_labels: ``(B, 6)`` binary labels (unused, for interface).
        leaf_to_chapter: ``(16,)`` int tensor mapping leaf idx → chapter idx.
            Must be set via :func:`set_leaf_to_chapter` before calling.

    Returns:
        ``(B,)`` reward per sample (non-positive).
    """
    if leaf_to_chapter is None:
        raise ValueError(
            "leaf_to_chapter must be provided. "
            "Use set_leaf_to_chapter() or pass explicitly."
        )
    leaf_probs = torch.sigmoid(leaf_logits)        # (B, 16)
    chapter_probs = torch.sigmoid(chapter_logits)  # (B, 6)

    parent_probs = chapter_probs[:, leaf_to_chapter]  # (B, 16)
    violations = torch.clamp(leaf_probs - parent_probs, min=0)  # (B, 16)
    return -violations.mean(dim=-1)  # (B,)


def combined_reward(
    leaf_logits: torch.Tensor,
    leaf_labels: torch.Tensor,
    chapter_logits: torch.Tensor,
    chapter_labels: torch.Tensor,
    leaf_to_chapter: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Weighted combination of calibration and hierarchy rewards.

    R = alpha * R_cal + (1 - alpha) * R_hier

    Args:
        alpha: Weight for calibration reward (default 0.5).

    Returns:
        ``(B,)`` combined reward.
    """
    r_cal = calibration_reward(leaf_logits, leaf_labels)
    r_hier = hierarchy_reward(
        leaf_logits, leaf_labels, chapter_logits, chapter_labels,
        leaf_to_chapter=leaf_to_chapter,
    )
    return alpha * r_cal + (1 - alpha) * r_hier
