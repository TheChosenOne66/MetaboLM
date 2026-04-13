#!/usr/bin/env python3
"""Evaluate E0 per-disease checkpoints on the global val.csv — shared preprocessing.

This variant feeds each E0 checkpoint the same globally z-scored ``val.csv``
that E1-E4 consume at evaluation time, i.e. only the normalisation produced
by ``scripts/prepare_data.py`` is applied. The per-disease cohort z-score
that ``scripts/train.py:build_disease_cohort`` fits during training is
intentionally *not* replayed here. The resulting numbers therefore measure
how E0 behaves when forced into the unified E1-E4 preprocessing pipeline
(i.e. under inference-time distribution shift relative to training).

For E0 evaluated under its own trained preprocessing (per-disease cohort
z-score replayed on ``val.csv``), see ``scripts/eval_e0_global_cohort.py``.

Usage::

    python scripts/eval_e0_global.py \
        --config configs/reproduce.yaml \
        --e0-dir outputs/E0_reproduction \
        --output-dir outputs/E0_global_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.biomarkers import get_metabolite_names
from src.data.endpoints import get_disease_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval_e0_global")


def load_val_data(
    val_path: str,
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Load global val.csv and return (eids, X, Y, disease_names).

    eids: list of N participant ids (kept as strings to avoid silent int casts)
    X: (N, 168) float32 — already z-score normalised by prepare_data.py
    Y: (N, 16) float32 — binary labels in canonical disease order
    """
    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{d}" for d in disease_names]

    # Use csv reader to avoid pandas dependency issues
    with open(val_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

    if "eid" not in header:
        raise ValueError(
            f"val.csv missing required 'eid' column; header starts with {header[:5]}..."
        )
    eid_index = header.index("eid")
    feat_indices = [header.index(c) for c in feature_cols]
    label_indices = [header.index(c) for c in label_cols]

    eids: list[str] = []
    X_rows = []
    Y_rows = []
    with open(val_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            eids.append(row[eid_index])
            X_rows.append([float(row[i]) for i in feat_indices])
            Y_rows.append([float(row[i]) for i in label_indices])

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)
    logger.info("Loaded val.csv: %d samples, %d features, %d labels",
                X.shape[0], X.shape[1], Y.shape[1])
    return eids, X, Y, disease_names


def write_eval_set_labels(
    output_dir: str,
    eids: list[str],
    Y: np.ndarray,
    disease_names: list[str],
) -> None:
    """Write the val.csv label matrix once at the top of the eval output dir.

    Stored once (not duplicated under every per-ckpt subdir) so each eval
    output dir is self-contained for downstream analysis without joining
    val.csv, while keeping per-ckpt directories small.
    """
    label_cols = [f"label_{d}" for d in disease_names]
    path = os.path.join(output_dir, "eval_set_labels.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eid"] + label_cols)
        for j, eid in enumerate(eids):
            writer.writerow([eid] + [int(Y[j, k]) for k in range(len(disease_names))])
    logger.info("Wrote %s (%d samples × %d label cols)",
                path, len(eids), len(label_cols))


def load_correlation_matrix(cfg, device: torch.device) -> torch.Tensor:
    """Load padded correlation matrix (169x169).

    Mirrors scripts/train.py:load_correlation_matrix — prefers the prebuilt
    ``.pt`` under ``cfg.data.correlation_matrix_path`` and falls back to the
    reference CSV when the ``.pt`` is unavailable, so this evaluator works in
    the same environments the training path does.
    """
    import pandas as pd

    corr_path = Path(cfg.data.correlation_matrix_path)
    if corr_path.suffix == ".pt" and corr_path.exists():
        return torch.load(str(corr_path), map_location=device)

    csv_path = Path("_reference/Pre-training/metabolomic_correlation_matrix.csv")
    if not csv_path.exists():
        csv_path = Path("_reference_correlation_matrix.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Correlation matrix not found at {corr_path} or fallback CSV paths "
            f"(_reference/Pre-training/metabolomic_correlation_matrix.csv, "
            f"_reference_correlation_matrix.csv)."
        )

    corr_df = pd.read_csv(str(csv_path), index_col=0, encoding="utf-8-sig")
    bias_matrix = torch.tensor(corr_df.values, dtype=torch.float32, device=device)
    # Pad with zeros for [CLS] token at position 0 (matches train path).
    bias_matrix = torch.nn.functional.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)
    return bias_matrix


@torch.no_grad()
def predict_single_disease(
    ckpt_path: str,
    X: np.ndarray,
    bias_matrix: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Load one E0 checkpoint, predict on full val set, return probabilities."""
    from src.model.backbone import MetaboliteBERTModel
    from src.model.heads import SingleTaskHead
    from src.model.wrapper import MetaboLMForClassification

    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location=device)

    # Build model
    backbone = MetaboliteBERTModel(num_metabolites=168)
    head = SingleTaskHead(hidden_size=768)
    model = MetaboLMForClassification(backbone, head)

    # Handle DataParallel prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[new_k] = v
    model.load_state_dict(cleaned, strict=False)

    # Replace bias matrix
    model.metabolite_model.register_buffer("bias_matrix_full", bias_matrix)
    model.to(device)
    model.eval()

    # Predict in batches
    all_probs = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        expressions = torch.tensor(X[start:end], dtype=torch.float32, device=device)
        attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)
        logits, _ = model(expressions, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E0 checkpoints on global val.csv")
    parser.add_argument("--config", default="configs/reproduce.yaml", help="Config file.")
    parser.add_argument("--e0-dir", default="outputs/E0_reproduction", help="E0 output directory.")
    parser.add_argument("--output-dir", default="outputs/E0_global_eval", help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load global val data (already z-score normalised by prepare_data.py)
    eids, X_val, Y_val, disease_names = load_val_data(cfg.data.val_path)

    # Load correlation matrix
    bias_matrix = load_correlation_matrix(cfg, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Output dir contract (subdir = CKPT identity, file names suffixed with the
    # ckpt disease for grep-friendliness):
    #
    #   {output_dir}/
    #     eval_set_labels.csv                      # val.csv labels (one copy)
    #     {ckpt_disease}/                          # one per E0 checkpoint
    #       predictions_ckpt_{ckpt_disease}.csv    # eid, prob
    #       metrics_ckpt_{ckpt_disease}.csv        # ckpt_disease, eval_label, AUC, ...
    #     e0_global_val_metrics.csv                # diagonal aggregate (16 + MEAN)
    #     summary.json
    #
    # Per-ckpt subdirs let parallel runs (e.g. one Nebula task per checkpoint)
    # write without colliding on the aggregate CSV.
    write_eval_set_labels(args.output_dir, eids, Y_val, disease_names)

    # Evaluate each E0 checkpoint on global val
    results = []
    for i, disease_name in enumerate(disease_names):
        ckpt_path = os.path.join(args.e0_dir, disease_name, f"best_finetune_model_{disease_name}.pt")
        if not os.path.exists(ckpt_path):
            logger.warning("[%s] Checkpoint not found: %s — skipping", disease_name, ckpt_path)
            continue

        # Per-ckpt subdir = ckpt identity (matches scripts/train.py convention).
        ckpt_subdir = os.path.join(args.output_dir, disease_name)
        os.makedirs(ckpt_subdir, exist_ok=True)

        logger.info("[%s] Loading checkpoint and predicting on %d samples...", disease_name, len(X_val))
        probs = predict_single_disease(ckpt_path, X_val, bias_matrix, device, args.batch_size)
        probs_flat = np.asarray(probs).reshape(-1).astype(np.float64)

        # Persist per-sample probabilities so any downstream analysis (threshold
        # sweep, calibration, sub-cohort breakdown, 16x16 cross-disease scoring)
        # can run without rerunning GPU forward.
        pred_path = os.path.join(ckpt_subdir, f"predictions_ckpt_{disease_name}.csv")
        with open(pred_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["eid", f"prob_ckpt_{disease_name}"])
            for j, eid in enumerate(eids):
                writer.writerow([eid, f"{probs_flat[j]:.6f}"])

        labels = Y_val[:, i]
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        try:
            auc = float(roc_auc_score(labels, probs_flat))
        except ValueError as e:
            # Typically raised when ``labels`` is single-class on this eval set.
            # Record NaN (rather than 0.0) so downstream aggregation can tell
            # this apart from a legitimate low-but-nonzero AUC.
            logger.warning("[%s] roc_auc_score failed (%s); recording NaN",
                           disease_name, e)
            auc = float("nan")

        # Per-ckpt metrics file. Schema lets us extend to 16x16 cross-disease
        # eval later by appending more rows (one per evaluated label) without
        # touching the schema.
        prev_pct = round(n_pos / len(labels) * 100, 2)
        metrics_path = os.path.join(ckpt_subdir, f"metrics_ckpt_{disease_name}.csv")
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ckpt_disease", "eval_label", "Val_AUC",
                "N_Positive", "N_Negative", "Prevalence_pct",
            ])
            auc_cell = "" if math.isnan(auc) else f"{auc:.4f}"
            writer.writerow([disease_name, disease_name, auc_cell,
                             n_pos, n_neg, prev_pct])

        results.append({
            "Disease": disease_name,
            "Global_Val_AUC": auc if math.isnan(auc) else round(auc, 4),
            "N_Positive": n_pos,
            "N_Negative": n_neg,
            "Prevalence_pct": prev_pct,
        })
        logger.info("[%s] Global val AUC = %s  (pos=%d, neg=%d, prev=%.2f%%)",
                     disease_name,
                     "NaN" if math.isnan(auc) else f"{auc:.4f}",
                     n_pos, n_neg, n_pos / len(labels) * 100)

        # Free GPU memory between diseases
        torch.cuda.empty_cache()

    # Save results
    if results:
        csv_path = os.path.join(args.output_dir, "e0_global_val_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Average only over diseases with a valid (non-NaN) AUC. We deliberately
        # track the excluded set and surface it in the log instead of silently
        # filtering with ``> 0`` (which would conflate a failed roc_auc_score
        # sentinel with a legitimate low AUC).
        valid_aucs = [r["Global_Val_AUC"] for r in results
                      if not (isinstance(r["Global_Val_AUC"], float)
                              and math.isnan(r["Global_Val_AUC"]))]
        excluded = [r["Disease"] for r in results
                    if isinstance(r["Global_Val_AUC"], float)
                    and math.isnan(r["Global_Val_AUC"])]
        if excluded:
            logger.warning(
                "MEAN excludes %d disease(s) with invalid AUC: %s",
                len(excluded), excluded,
            )
        mean_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else float("nan")
        # Aggregate per-disease positives. ``results`` currently holds only the
        # per-disease rows — the MEAN entry is appended *after* the dict literal
        # is fully constructed, so we iterate ``results`` directly here rather
        # than ``results[:-1]`` (which would drop the last disease).
        results.append({
            "Disease": "MEAN",
            "Global_Val_AUC": round(mean_auc, 4),
            "N_Positive": sum(r["N_Positive"] for r in results if "N_Positive" in r),
            "N_Negative": "-",
            "Prevalence_pct": "-",
        })

        # Print summary table
        logger.info("=" * 70)
        logger.info("E0 Global Val Evaluation — Per-Disease AUROC")
        logger.info("=" * 70)
        logger.info("%-20s %12s %12s %8s", "Disease", "Global_AUC", "Original_AUC*", "Delta")
        logger.info("-" * 70)
        # *Original_AUC is from the per-disease balanced eval (LEADERBOARD.md)
        e0_original = {
            "T2D": 0.867, "obesity": 0.757, "hypertension": 0.706,
            "ischemic_heart": 0.738, "atrial_fib": 0.683, "heart_failure": 0.722,
            "rheumatoid": 0.676, "asthma": 0.620, "dementia": 0.652,
            "copd": 0.753, "stroke": 0.651, "parkinsons": 0.611,
            "breast_cancer": 0.687, "colon_cancer": 0.604, "lung_cancer": 0.667,
            "prostate_cancer": 0.777,
        }
        for r in results:
            name = r["Disease"]
            g_auc = r["Global_Val_AUC"]
            orig = e0_original.get(name, None)
            if orig is not None:
                delta = g_auc - orig
                logger.info("%-20s %12.4f %12.4f %+8.4f", name, g_auc, orig, delta)
            else:
                logger.info("%-20s %12.4f %12s %8s", name, g_auc, "-", "-")
        logger.info("=" * 70)

        # Save summary JSON
        summary = {
            "eval_set": "global val.csv",
            "n_samples": len(X_val),
            "mean_auc": mean_auc,
            "per_disease": {r["Disease"]: r["Global_Val_AUC"] for r in results},
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
