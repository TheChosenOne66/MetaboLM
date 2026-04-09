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
