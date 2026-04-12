"""Evaluation metrics for MetaboLM disease prediction.

Matches the official fine-tuning code which computes:
  - AUROC (primary, used for best model selection)
  - Accuracy, Precision, Recall, F1 (auxiliary)

Official code uses sklearn with default settings:
    accuracy_score, precision_score(..., zero_division=0),
    recall_score(..., zero_division=0), f1_score(..., zero_division=0),
    roc_auc_score
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    labels: list[float] | np.ndarray,
    probs: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all metrics for a single disease, matching official code.

    Args:
        labels: Ground-truth binary labels (0/1).
        probs: Predicted probabilities (after sigmoid).
        threshold: Classification threshold (official uses 0.5).

    Returns:
        Dict with keys: auc, accuracy, precision, recall, f1.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        # All labels are the same class
        auc = 0.0

    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def compute_multitask_metrics(
    leaf_labels: np.ndarray,
    leaf_probs: np.ndarray,
    disease_names: list[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute per-disease and mean AUROC for multi-task predictions.

    Args:
        leaf_labels: ``(N, 16)`` binary labels.
        leaf_probs: ``(N, 16)`` predicted probabilities.
        disease_names: List of 16 disease names.
        threshold: Classification threshold.

    Returns:
        Dict with ``auc_{disease}``, ``mean_auc``, ``mean_f1``.
    """
    results = {}
    aucs = []
    f1s = []
    for i, name in enumerate(disease_names):
        m = compute_binary_metrics(leaf_labels[:, i], leaf_probs[:, i], threshold)
        results[f"auc_{name}"] = m["auc"]
        results[f"f1_{name}"] = m["f1"]
        if m["auc"] > 0:
            aucs.append(m["auc"])
        f1s.append(m["f1"])

    results["mean_auc"] = float(np.mean(aucs)) if aucs else 0.0
    results["mean_f1"] = float(np.mean(f1s))
    return results


def compute_hierarchy_violation_rate(
    leaf_probs: np.ndarray,
    chapter_probs: np.ndarray,
    disease_to_chapter_idx: dict[int, int],
) -> float:
    """Fraction of (sample, leaf) pairs where P(leaf) > P(parent_chapter).

    Args:
        leaf_probs: ``(N, 16)`` leaf probabilities.
        chapter_probs: ``(N, 6)`` chapter probabilities.
        disease_to_chapter_idx: Mapping disease_idx -> chapter_idx.

    Returns:
        Violation rate in [0, 1].
    """
    total_pairs = 0
    violations = 0
    for d_idx, c_idx in disease_to_chapter_idx.items():
        v = (leaf_probs[:, d_idx] > chapter_probs[:, c_idx] + 1e-7).sum()
        violations += v
        total_pairs += len(leaf_probs)
    return float(violations) / total_pairs if total_pairs > 0 else 0.0
