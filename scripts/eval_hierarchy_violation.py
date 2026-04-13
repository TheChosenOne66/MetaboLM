"""Evaluate Hierarchy Violation Rate (HVR) for multitask checkpoints (E1-E4)
and a mean-aggregation baseline for E0.

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


def mean_aggregate_chapter_probs(
    leaf_probs: np.ndarray,
    disease_to_chapter_idx: dict[int, int],
    n_chapters: int,
) -> np.ndarray:
    """Aggregate per-leaf probabilities into per-chapter probabilities via the
    arithmetic mean: ``P_chap = mean(p_leaf for leaf in chapter)``.

    Args:
        leaf_probs: ``(N, n_leaves)`` probabilities in [0, 1].
        disease_to_chapter_idx: ``{leaf_idx: chapter_idx}`` (length n_leaves).
        n_chapters: Number of distinct chapter indices.

    Returns:
        ``(N, n_chapters)`` chapter-level probabilities. The (i, c) cell is
        the average leaf probability for chapter ``c`` on sample ``i``.

    Used as the E0 baseline for hierarchy violation rate. **Why mean and not
    max / OR / noisy-OR**: both ``max(p_leaf)`` and ``1 - prod(1 - p_leaf)``
    (the independence-OR) are *upper bounds* on every leaf prob — by
    construction ``p_leaf <= chap_prob`` always, so the HVR strict
    comparison ``p_leaf > chap_prob + 1e-7`` is never true and the metric
    returns 0.0 identically. The mean is the simplest aggregation that
    *can* be exceeded by individual leaves (any leaf above its sibling
    mean violates), so it produces the non-trivial baseline number we need
    to demonstrate the value of E1's explicit hierarchy loss.
    """
    n_samples, n_leaves = leaf_probs.shape
    if len(disease_to_chapter_idx) != n_leaves:
        raise ValueError(
            f"disease_to_chapter_idx covers {len(disease_to_chapter_idx)} leaves, "
            f"but leaf_probs has {n_leaves} columns"
        )
    sums = np.zeros((n_samples, n_chapters), dtype=np.float32)
    counts = np.zeros(n_chapters, dtype=np.int32)
    for leaf_idx, chap_idx in disease_to_chapter_idx.items():
        sums[:, chap_idx] += leaf_probs[:, leaf_idx]
        counts[chap_idx] += 1
    # Avoid division by zero for chapters with no mapped leaves: leave their
    # column as 0.0 (no leaves means no possible violations from that chapter
    # anyway, so the value is unobservable in HVR).
    safe_counts = np.where(counts > 0, counts, 1).astype(np.float32)
    return (sums / safe_counts).astype(np.float32)


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
            ``"mean_aggregate_baseline"``). Surfaced in the JSON for downstream
            disambiguation.
        method_details: Free-form dict appended to the JSON for provenance
            (model path, predictions source dir, etc.).
        disease_to_chapter_idx: ``{leaf_idx: chapter_idx}`` mapping. Defaults to
            :func:`src.data.endpoints.get_disease_to_chapter_idx` (the canonical
            16-leaf / 6-chapter mapping for this project). Override in tests to
            verify the computation on smaller, hand-checkable inputs.

    Returns:
        The dict that was written to disk (also useful for in-process logging).
        Keys:
          * ``hierarchy_violation_rate`` (float): rate from the canonical metric.
          * ``n_samples`` (int): ``leaf_probs.shape[0]``.
          * ``n_leaf_chapter_pairs`` (int): ``len(disease_to_chapter_idx)`` — the
            number of (leaf, parent_chapter) pairs actually evaluated by the
            canonical metric, NOT ``leaf_probs.shape[1]``. They coincide when
            the full 16-leaf mapping is used; they differ when a caller passes
            a sub-mapping (e.g. the E0 baseline path with a partial eval dir,
            where missing diseases are dropped before this helper is called).
          * ``method`` (str): the ``method`` argument.
          * ``method_details`` (dict): copy of the ``method_details`` argument.
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

    # ``n_leaf_chapter_pairs`` is the number of (leaf, parent_chapter) pairs the
    # canonical metric iterates over (``len(disease_to_chapter_idx)``), NOT the
    # number of leaf columns. They coincide when the full 16-leaf mapping is
    # used; they differ when a caller (e.g. the E0 baseline with a partial eval
    # dir) passes a sub-mapping over the present leaves only.
    payload = {
        "hierarchy_violation_rate": float(rate),
        "n_samples": int(leaf_probs.shape[0]),
        "n_leaf_chapter_pairs": int(len(disease_to_chapter_idx)),
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


def load_e0_leaf_probs(
    eval_dir: Path, disease_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Read per-ckpt predictions under ``eval_dir/<disease>/predictions_ckpt_<D>.csv``
    and return ``(leaf_probs, present_diseases)``.

    Aligns by the ``eid`` column of ``eval_dir/eval_set_labels.csv`` so all
    disease columns share a row order. Diseases without a predictions file
    contribute a column of ``NaN`` — the caller is responsible for handling
    those (typically by dropping them before aggregation).

    Args:
        eval_dir: Output dir produced by ``scripts/eval_e0_global*.py``.
            Must contain ``eval_set_labels.csv`` at the top and optionally
            per-disease subdirs with ``predictions_ckpt_<D>.csv``.
        disease_names: Canonical disease order. Output columns follow this
            order exactly.

    Returns:
        Tuple ``(leaf_probs, present)``:
            - ``leaf_probs``: ``(N, len(disease_names))`` float32 array. Rows
              are aligned to ``eval_set_labels.csv``'s eid order; missing
              disease columns are filled with ``NaN``.
            - ``present``: subset of ``disease_names`` whose predictions file
              was found and successfully read. Order matches the input
              ``disease_names``.
    """
    import pandas as pd

    labels_path = eval_dir / "eval_set_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found — run scripts/eval_e0_global*.py first."
        )

    labels_df = pd.read_csv(labels_path)
    if "eid" not in labels_df.columns:
        raise ValueError(f"{labels_path} missing 'eid' column")
    eids = labels_df["eid"].astype(str).tolist()

    n = len(eids)
    leaf_probs = np.full((n, len(disease_names)), np.nan, dtype=np.float32)
    present: list[str] = []

    eid_to_row = {eid: i for i, eid in enumerate(eids)}
    for col_idx, disease in enumerate(disease_names):
        pred_path = eval_dir / disease / f"predictions_ckpt_{disease}.csv"
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path)
        prob_col = f"prob_ckpt_{disease}"
        if "eid" not in pred_df.columns or prob_col not in pred_df.columns:
            logger.warning(
                "[%s] %s has unexpected schema; skipping",
                disease, pred_path.name,
            )
            continue
        # Iterate via column arrays to avoid pandas' row-wise dtype promotion
        # (int eids become floats and stringify as "1.0", breaking the join).
        pred_eids = pred_df["eid"].astype(str).tolist()
        pred_vals = pred_df[prob_col].astype(float).tolist()
        for eid_str, val in zip(pred_eids, pred_vals):
            i = eid_to_row.get(eid_str)
            if i is None:
                continue
            leaf_probs[i, col_idx] = val
        present.append(disease)
    return leaf_probs, present


def run_e0_baseline_mode(
    eval_dir: Path,
    output_path: Path,
) -> dict:
    """E0 baseline mode: read 16 leaf prob columns from ``eval_dir``, mean-aggregate
    to chapter probs, compute HVR, persist.

    This is the baseline Innovation 2 (hierarchy loss) is supposed to fix.
    We expect a high HVR here because E0's per-disease models have no
    structural prior tying leaf probs to a shared chapter prob. We use mean
    aggregation rather than OR/max because the latter are upper bounds on
    every leaf prob and trivialise HVR to 0.0 — see
    :func:`mean_aggregate_chapter_probs` for the full rationale.

    Args:
        eval_dir: Eval output dir produced by ``scripts/eval_e0_global*.py``.
        output_path: Where to write the HVR sidecar JSON.

    Returns:
        The payload dict written to ``output_path``.
    """
    from src.data.endpoints import (
        get_disease_names, get_disease_to_chapter_idx, get_unique_chapters,
    )

    diseases = get_disease_names()
    full_disease_to_chap = get_disease_to_chapter_idx()
    n_chapters = len(get_unique_chapters())

    leaf_probs, present = load_e0_leaf_probs(eval_dir, diseases)

    # Drop missing diseases entirely so mean aggregation isn't poisoned by NaN.
    # The canonical metric only iterates the mapping we pass it, so limiting
    # the mapping to ``present`` diseases gives the correct "pairs evaluated"
    # count in the persisted payload (via ``compute_and_persist_hvr``'s
    # ``n_leaf_chapter_pairs = len(disease_to_chapter_idx)`` semantics).
    present_idx = [diseases.index(d) for d in present]
    leaf_probs_clean = leaf_probs[:, present_idx]
    sub_disease_to_chap = {
        new_idx: full_disease_to_chap[old_idx]
        for new_idx, old_idx in enumerate(present_idx)
    }

    chap_probs = mean_aggregate_chapter_probs(
        leaf_probs_clean,
        disease_to_chapter_idx=sub_disease_to_chap,
        n_chapters=n_chapters,
    )

    return compute_and_persist_hvr(
        leaf_probs=leaf_probs_clean,
        chap_probs=chap_probs,
        output_path=output_path,
        method="mean_aggregate_baseline",
        method_details={
            "chap_probs_source": (
                "P_chap = mean(p_leaf for leaf in chapter); chosen because OR/max are "
                "upper bounds on the leaves and trivialise HVR. See "
                "mean_aggregate_chapter_probs docstring for rationale."
            ),
            "predictions_source": str(eval_dir),
            "n_present_diseases": len(present),
            "present_diseases": present,
        },
        disease_to_chapter_idx=sub_disease_to_chap,
    )
