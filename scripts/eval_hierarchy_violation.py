"""Evaluate Hierarchy Violation Rate (HVR) for multitask checkpoints (E1-E4)
and an OR-aggregation baseline for E0.

See docs/superpowers/plans/2026-04-13-hierarchy-violation-eval.md for the
full design rationale.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Enable `from src...` imports added in later tasks (e.g. Task 3, Task 4).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval_hierarchy_violation")


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


def compute_and_persist_hvr(
    leaf_probs: np.ndarray,
    chap_probs: np.ndarray,
    output_path: Path,
    method: str,
    method_details: dict,
    disease_to_chapter_idx: dict[int, int] | None = None,
) -> dict:
    """Compute HVR via the canonical metric and write a JSON sidecar.

    Args:
        leaf_probs: ``(N, 16)`` leaf probabilities in [0, 1].
        chap_probs: ``(N, 6)`` chapter probabilities in [0, 1].
        output_path: Where to write the JSON.
        method: Short identifier (e.g. ``"explicit_chapter_head"`` or
            ``"or_aggregate_baseline"``). Surfaced in the JSON for downstream
            disambiguation.
        method_details: Free-form dict appended to the JSON for provenance
            (model path, predictions source dir, etc.).
        disease_to_chapter_idx: ``{leaf_idx: chapter_idx}`` mapping. Defaults to
            :func:`src.data.endpoints.get_disease_to_chapter_idx` (the canonical
            16-leaf / 6-chapter mapping for this project). Override in tests to
            verify the computation on smaller, hand-checkable inputs.

    Returns:
        The dict that was written to disk (also useful for in-process logging).
    """
    # Lazy import: the metric lives next to training code and pulls in heavier
    # imports we don't want at script-import time.
    from src.training.metrics import compute_hierarchy_violation_rate
    from src.data.endpoints import get_disease_to_chapter_idx

    if leaf_probs.shape[0] != chap_probs.shape[0]:
        raise ValueError(
            f"leaf_probs ({leaf_probs.shape}) and chap_probs ({chap_probs.shape}) "
            "disagree on N (axis 0)."
        )

    if disease_to_chapter_idx is None:
        disease_to_chapter_idx = get_disease_to_chapter_idx()

    rate = compute_hierarchy_violation_rate(
        leaf_probs.astype(np.float64),
        chap_probs.astype(np.float64),
        disease_to_chapter_idx,
    )

    payload = {
        "hierarchy_violation_rate": float(rate),
        "n_samples": int(leaf_probs.shape[0]),
        "n_leaf_chapter_pairs": int(leaf_probs.shape[1]),
        "method": method,
        "method_details": dict(method_details),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Wrote %s — rate=%.4f, n_samples=%d, method=%s",
        output_path, payload["hierarchy_violation_rate"],
        payload["n_samples"], method,
    )
    return payload
