#!/usr/bin/env python3
"""Build the 16×16 cross-disease AUROC matrix from on-disk eval outputs.

Pure CPU post-processing: consumes the per-ckpt prediction files written by
``scripts/eval_e0_global.py`` (or its cohort variant) and the top-level
``eval_set_labels.csv``; computes AUROC for every (ckpt, label) pair using
sklearn; writes a wide CSV plus a long-format CSV plus a tiny ASCII heatmap
to the same eval dir.

Why a separate script: the per-ckpt forward pass already happened during
``eval_e0_global*.py``; the resulting probability vectors are on disk under
``<eval_dir>/<ckpt_disease>/predictions_ckpt_<ckpt_disease>.csv``. Off-
diagonal AUCs are just (label, prob) pairs joined on ``eid``, no GPU needed.

Caveat for thesis use
---------------------
Off-diagonal AUCs measure rank correlation between a ckpt's output and a
label it wasn't trained for. They are confounded by **disease comorbidity**:
e.g. T2D and obesity co-occur frequently, so the T2D ckpt's probabilities
will trivially rank obesity-positive samples higher even without "knowing"
about obesity. Treat the off-diagonal as a *transfer / specialisation
diagnostic*, not as a real cross-disease classifier benchmark.

Usage::

    python scripts/cross_disease_matrix.py \
        --eval-dir outputs/E0_global_eval \
        --output-dir outputs/E0_global_eval

    # cohort variant works the same way
    python scripts/cross_disease_matrix.py \
        --eval-dir outputs/E0_global_eval_cohort
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.endpoints import get_disease_names  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("cross_disease_matrix")


def load_labels(eval_dir: Path, disease_names: list[str]) -> pd.DataFrame:
    """Load ``eval_set_labels.csv`` and return a DataFrame indexed by eid."""
    labels_path = eval_dir / "eval_set_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Was it produced by an up-to-date "
            f"eval_e0_global*.py run?"
        )
    df = pd.read_csv(labels_path)
    if "eid" not in df.columns:
        raise ValueError(f"{labels_path} missing 'eid' column")
    label_cols = [f"label_{d}" for d in disease_names]
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{labels_path} missing label columns: {missing[:3]}...")

    df = df.set_index("eid")
    logger.info("Loaded labels: %d samples × %d label columns",
                len(df), len(label_cols))
    return df[label_cols]


def load_predictions(
    eval_dir: Path, ckpt_disease: str
) -> pd.Series | None:
    """Load probabilities for one ckpt; return ``None`` if predictions absent."""
    pred_path = eval_dir / ckpt_disease / f"predictions_ckpt_{ckpt_disease}.csv"
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path)
    expected_col = f"prob_ckpt_{ckpt_disease}"
    if "eid" not in df.columns or expected_col not in df.columns:
        logger.warning(
            "[%s] predictions file %s has unexpected schema (cols=%s); skipping",
            ckpt_disease, pred_path.name, list(df.columns),
        )
        return None
    return df.set_index("eid")[expected_col]


def compute_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """``roc_auc_score`` with NaN sentinel on single-class labels."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def build_matrix(
    eval_dir: Path, disease_names: list[str], labels_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Iterate ckpts × labels, return (auc_matrix, n_pos_lookup).

    ``auc_matrix`` is a ``len(present_ckpts) × 16`` DataFrame. Missing ckpts
    are simply omitted from the rows (the matrix is not padded with NaN
    rows — row presence is itself useful information).
    """
    present_ckpts: list[str] = []
    auc_rows: dict[str, dict[str, float]] = {}
    n_pos_lookup: dict[str, int] = {
        d: int(labels_df[f"label_{d}"].sum()) for d in disease_names
    }
    n_total = len(labels_df)

    for ckpt_disease in disease_names:
        probs = load_predictions(eval_dir, ckpt_disease)
        if probs is None:
            logger.warning("[%s] no predictions on disk — skipping", ckpt_disease)
            continue

        # Align probs to label index; drop any eids the labels file lacks
        # (shouldn't happen if both came from the same run).
        joined = labels_df.join(probs.rename("prob"), how="inner")
        if len(joined) != len(labels_df):
            logger.warning(
                "[%s] eid alignment lost %d rows (labels=%d, joined=%d)",
                ckpt_disease, len(labels_df) - len(joined),
                len(labels_df), len(joined),
            )
        if len(joined) == 0:
            logger.warning("[%s] empty join with labels — skipping", ckpt_disease)
            continue

        present_ckpts.append(ckpt_disease)
        row: dict[str, float] = {}
        for eval_label in disease_names:
            y_true = joined[f"label_{eval_label}"].to_numpy()
            y_score = joined["prob"].to_numpy()
            row[eval_label] = compute_auc_safe(y_true, y_score)
        auc_rows[ckpt_disease] = row

        diag = row[ckpt_disease]
        diag_str = "NaN" if math.isnan(diag) else f"{diag:.4f}"
        off_diag = [
            v for k, v in row.items()
            if k != ckpt_disease and not math.isnan(v)
        ]
        off_mean = sum(off_diag) / len(off_diag) if off_diag else float("nan")
        off_str = "NaN" if math.isnan(off_mean) else f"{off_mean:.4f}"
        logger.info("[%s] diagonal=%s, off-diagonal mean=%s (n=%d)",
                    ckpt_disease, diag_str, off_str, len(off_diag))

    if not present_ckpts:
        logger.error("No ckpts had predictions on disk — nothing to build")
        return pd.DataFrame(), n_pos_lookup

    matrix = pd.DataFrame.from_dict(auc_rows, orient="index", columns=disease_names)
    matrix.index.name = "ckpt_disease"
    return matrix, n_pos_lookup


def write_outputs(
    matrix: pd.DataFrame,
    n_pos_lookup: dict[str, int],
    output_dir: Path,
    n_total: int,
) -> None:
    """Persist the wide matrix CSV, a long-format CSV, and a summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wide matrix (rows = ckpt, cols = eval_label)
    wide_path = output_dir / "cross_disease_matrix.csv"
    matrix.round(4).to_csv(wide_path)
    logger.info("Wrote %s (%d ckpts × %d labels)",
                wide_path, len(matrix), matrix.shape[1])

    # Long format for downstream pivot/plotting
    long_path = output_dir / "cross_disease_matrix_long.csv"
    with open(long_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ckpt_disease", "eval_label", "Val_AUC", "is_diagonal",
            "N_Positive_in_eval_label", "N_Total_eval",
        ])
        for ckpt_disease, row in matrix.iterrows():
            for eval_label, auc in row.items():
                writer.writerow([
                    ckpt_disease, eval_label,
                    "" if math.isnan(auc) else f"{float(auc):.4f}",
                    int(ckpt_disease == eval_label),
                    n_pos_lookup.get(eval_label, ""),
                    n_total,
                ])
    logger.info("Wrote %s (%d rows)", long_path, len(matrix) * matrix.shape[1])

    # Summary JSON: diagonal mean, off-diagonal mean per ckpt
    summary = {
        "n_ckpts_present": int(len(matrix)),
        "n_labels": int(matrix.shape[1]),
        "n_total_eval_samples": int(n_total),
        "diagonal_mean_auc": _safe_mean([
            float(matrix.loc[c, c])
            for c in matrix.index if c in matrix.columns
        ]),
        "per_ckpt": {},
    }
    for ckpt_disease, row in matrix.iterrows():
        diag = (
            float(row[ckpt_disease])
            if ckpt_disease in row.index else float("nan")
        )
        off_vals = [
            float(v) for k, v in row.items()
            if k != ckpt_disease and not (isinstance(v, float) and math.isnan(v))
        ]
        summary["per_ckpt"][ckpt_disease] = {
            "diagonal_auc": _none_if_nan(diag),
            "off_diagonal_mean_auc": _none_if_nan(_safe_mean(off_vals)),
            "n_off_diagonal_valid": len(off_vals),
        }
    summary_path = output_dir / "cross_disease_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_path)


def _safe_mean(values: list[float]) -> float:
    """Mean ignoring NaN; returns NaN on empty input."""
    valid = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return sum(valid) / len(valid) if valid else float("nan")


def _none_if_nan(value: float) -> float | None:
    """Convert NaN to None so JSON output is clean."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def render_ascii_heatmap(matrix: pd.DataFrame) -> str:
    """Render a small text heatmap to the log for quick visual sanity."""
    if matrix.empty:
        return "(empty matrix)"

    # Right-align ckpt names in the row label column
    name_w = max(len(c) for c in matrix.index)

    # Two-character cells: AUC * 100 rounded, e.g. 0.854 -> "85"
    def cell(v: float) -> str:
        if isinstance(v, float) and math.isnan(v):
            return " ·"
        return f"{int(round(v * 100)):2d}"

    lines = []
    # Header: short label codes (first 4 chars or so)
    header_codes = [c[:4] for c in matrix.columns]
    header = " " * (name_w + 1) + " ".join(f"{h:>2}" for h in header_codes)
    lines.append(header)
    for ckpt_disease, row in matrix.iterrows():
        cells = " ".join(cell(v) for v in row.values)
        lines.append(f"{ckpt_disease:>{name_w}} {cells}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the 16×16 cross-disease AUROC matrix from local "
                    "per-ckpt prediction files."
    )
    parser.add_argument(
        "--eval-dir", required=True,
        help="Eval output dir produced by scripts/eval_e0_global*.py "
             "(e.g. outputs/E0_global_eval or outputs/E0_global_eval_cohort).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write cross_disease_matrix.csv etc. Defaults to "
             "``--eval-dir`` (writes alongside the per-ckpt subdirs).",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        logger.error("Eval dir does not exist: %s", eval_dir)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else eval_dir

    disease_names = get_disease_names()
    labels_df = load_labels(eval_dir, disease_names)

    matrix, n_pos_lookup = build_matrix(eval_dir, disease_names, labels_df)
    if matrix.empty:
        return 1

    write_outputs(matrix, n_pos_lookup, output_dir, n_total=len(labels_df))

    # Print ASCII heatmap to log for at-a-glance sanity. Diagonal cells
    # should generally be the largest in their row (a model is typically
    # best at predicting the disease it was trained on); off-diagonal cells
    # being high indicates either real label correlation (comorbidity) or
    # broad metabolic signal shared across diseases.
    logger.info("Cross-disease AUROC matrix (cells = AUC * 100):")
    for line in render_ascii_heatmap(matrix).split("\n"):
        logger.info("  %s", line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
