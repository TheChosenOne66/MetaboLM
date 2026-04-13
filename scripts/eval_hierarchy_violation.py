"""Evaluate Hierarchy Violation Rate (HVR) for multitask checkpoints (E1-E4)
and an OR-aggregation baseline for E0.

See docs/superpowers/plans/2026-04-13-hierarchy-violation-eval.md for the
full design rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Enable `from src...` imports added in later tasks (e.g. Task 3, Task 4).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def or_aggregate_chapter_probs(
    leaf_probs: np.ndarray,
    disease_to_chapter_idx: dict[int, int],
    n_chapters: int,
) -> np.ndarray:
    """Aggregate per-leaf probabilities into per-chapter probabilities via
    independence-OR: ``P_chap = 1 - ∏(1 - p_leaf for leaf in chapter)``.

    Args:
        leaf_probs: ``(N, n_leaves)`` probabilities in [0, 1].
        disease_to_chapter_idx: ``{leaf_idx: chapter_idx}`` (length n_leaves).
        n_chapters: Number of distinct chapter indices.

    Returns:
        ``(N, n_chapters)`` chapter-level probabilities. The (i, c) cell is
        the probability that *at least one* leaf in chapter ``c`` is positive
        for sample ``i``, assuming the leaves are independent given the
        sample (an oversimplification, but a standard baseline).

    Used as the E0 baseline for hierarchy violation: E0 has no chapter head,
    so we synthesise one this way to get a non-trivial violation rate (vs
    ``max`` which would make violations identically zero).
    """
    n_samples, n_leaves = leaf_probs.shape
    if len(disease_to_chapter_idx) != n_leaves:
        raise ValueError(
            f"disease_to_chapter_idx covers {len(disease_to_chapter_idx)} leaves, "
            f"but leaf_probs has {n_leaves} columns"
        )
    # Initialise chapter "no positive" probability to 1; multiply (1 - p_leaf)
    # for each leaf that lives under that chapter, then take the complement.
    one_minus = np.ones((n_samples, n_chapters), dtype=np.float32)
    for leaf_idx, chap_idx in disease_to_chapter_idx.items():
        one_minus[:, chap_idx] *= (1.0 - leaf_probs[:, leaf_idx])
    return (1.0 - one_minus).astype(np.float32)
