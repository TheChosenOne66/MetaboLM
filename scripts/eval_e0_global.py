#!/usr/bin/env python3
"""Evaluate E0 per-disease checkpoints on the global val.csv.

This enables fair comparison with E1-E4 (which are evaluated on the same
global val set).  E0's original metrics were computed on per-disease
balanced subsets with "super-healthy" negatives, making AUROC artificially
easier.  This script re-evaluates each E0 model on the full population.

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


def load_val_data(val_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load global val.csv and return (X, Y, disease_names).

    X: (N, 168) float32 — already z-score normalised by prepare_data.py
    Y: (N, 16) float32 — binary labels
    """
    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{d}" for d in disease_names]

    # Use csv reader to avoid pandas dependency issues
    with open(val_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

    feat_indices = [header.index(c) for c in feature_cols]
    label_indices = [header.index(c) for c in label_cols]

    X_rows = []
    Y_rows = []
    with open(val_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            X_rows.append([float(row[i]) for i in feat_indices])
            Y_rows.append([float(row[i]) for i in label_indices])

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.float32)
    logger.info("Loaded val.csv: %d samples, %d features, %d labels", X.shape[0], X.shape[1], Y.shape[1])
    return X, Y, disease_names


def load_correlation_matrix(cfg, device: torch.device) -> torch.Tensor:
    """Load padded correlation matrix."""
    corr_path = Path(cfg.data.correlation_matrix_path)
    if corr_path.suffix == ".pt" and corr_path.exists():
        return torch.load(str(corr_path), map_location=device)
    raise FileNotFoundError(f"Correlation matrix not found: {corr_path}")


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
    X_val, Y_val, disease_names = load_val_data(cfg.data.val_path)

    # Load correlation matrix
    bias_matrix = load_correlation_matrix(cfg, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each E0 checkpoint on global val
    results = []
    for i, disease_name in enumerate(disease_names):
        ckpt_path = os.path.join(args.e0_dir, disease_name, f"best_finetune_model_{disease_name}.pt")
        if not os.path.exists(ckpt_path):
            logger.warning("[%s] Checkpoint not found: %s — skipping", disease_name, ckpt_path)
            continue

        logger.info("[%s] Loading checkpoint and predicting on %d samples...", disease_name, len(X_val))
        probs = predict_single_disease(ckpt_path, X_val, bias_matrix, device, args.batch_size)

        labels = Y_val[:, i]
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0

        results.append({
            "Disease": disease_name,
            "Global_Val_AUC": round(auc, 4),
            "N_Positive": n_pos,
            "N_Negative": n_neg,
            "Prevalence_pct": round(n_pos / len(labels) * 100, 2),
        })
        logger.info("[%s] Global val AUC = %.4f  (pos=%d, neg=%d, prev=%.2f%%)",
                     disease_name, auc, n_pos, n_neg, n_pos / len(labels) * 100)

        # Free GPU memory between diseases
        torch.cuda.empty_cache()

    # Save results
    if results:
        csv_path = os.path.join(args.output_dir, "e0_global_val_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Compute mean AUC
        aucs = [r["Global_Val_AUC"] for r in results if r["Global_Val_AUC"] > 0]
        mean_auc = sum(aucs) / len(aucs) if aucs else 0
        results.append({
            "Disease": "MEAN",
            "Global_Val_AUC": round(mean_auc, 4),
            "N_Positive": sum(r["N_Positive"] for r in results[:-1] if "N_Positive" in r),
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
