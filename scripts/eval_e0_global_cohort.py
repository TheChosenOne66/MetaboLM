#!/usr/bin/env python3
"""Evaluate E0 per-disease checkpoints on the global val.csv — own pipeline.

This variant replays E0's trained preprocessing before inference, so each
checkpoint sees inputs from the same distribution it was trained on. The
preprocessing pipeline for an E0 checkpoint is::

    raw UKB metabolite values
      -> scripts/prepare_data.py : global z-score (fit on 80% train split)
      -> scripts/train.py : build_disease_cohort(seed=42)
                            -> sample 1:1 disease + healthy
                            -> 80/20 stratified split
                            -> z-score fit on cohort train subset

The global val.csv produced by ``prepare_data.py`` is already at the first
stage. To put it into the distribution the E0 T2D checkpoint (for example)
was trained on, we need the per-disease cohort train mean/std. These are
deterministic given ``seed=42`` and the current ``train.csv`` / ``val.csv``,
so we recompute them here rather than persisting them from training.

Implementation: call ``scripts/train.py:build_disease_cohort`` to get the
deterministic ``eids_train`` list for the cohort, then recompute
``mean/std`` over those rows in the already-globally-z-scored ``df``. This
keeps ``scripts/train.py`` as the single source of truth for cohort
construction — no logic is duplicated.

For E0 evaluated under the E1-E4 shared preprocessing (global z-score
only, no cohort replay), see ``scripts/eval_e0_global.py``.

Usage::

    python scripts/eval_e0_global_cohort.py \
        --config configs/reproduce.yaml \
        --e0-dir outputs/E0_reproduction \
        --output-dir outputs/E0_global_eval_cohort
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
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import build_disease_cohort  # noqa: E402  single source of truth
from src.config import load_config  # noqa: E402
from src.data.biomarkers import get_metabolite_names  # noqa: E402
from src.data.endpoints import get_disease_names  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval_e0_global_cohort")


def load_full_df_and_val(
    train_path: str, val_path: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    """Return (df_concat, X_val, Y_val, feature_cols, label_cols).

    ``df_concat`` mirrors what ``scripts/train.py`` constructs (train + val
    concatenated, already globally z-scored) so ``build_disease_cohort``
    behaves identically to training. ``X_val`` / ``Y_val`` are the global
    val.csv rows we score each checkpoint on.
    """
    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{d}" for d in disease_names]

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    df = pd.concat([train_df, val_df], ignore_index=True)

    X_val = val_df[feature_cols].values.astype(np.float32)
    Y_val = val_df[label_cols].values.astype(np.float32)
    logger.info("Loaded val.csv: %d samples, %d features, %d labels",
                X_val.shape[0], X_val.shape[1], Y_val.shape[1])
    logger.info("Loaded train+val concat for cohort replay: %d rows", len(df))
    return df, X_val, Y_val, feature_cols, label_cols


def compute_cohort_train_stats(
    df: pd.DataFrame,
    disease_name: str,
    label_col: str,
    feature_cols: list[str],
    label_cols: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Recompute the per-disease cohort train mean/std used during E0 training.

    Delegates cohort construction (case+control sampling, stratified split)
    to ``scripts/train.py:build_disease_cohort`` so there is no duplicated
    sampling logic. We then recompute stats from ``df`` in the same globally
    z-scored space the training function operated in, using the train
    ``eid`` list it returned.

    Returns ``None`` if ``build_disease_cohort`` returned ``None`` (cohort
    too small).
    """
    cohort = build_disease_cohort(
        df, disease_name, label_col, feature_cols, label_cols, seed=seed,
    )
    if cohort is None:
        return None

    # build_disease_cohort returns (X_train_norm, y_train, eids_train,
    # X_val_norm, y_val, eids_val). We only need eids_train.
    _, _, eids_train, _, _, _ = cohort

    cohort_train_rows = df[df["eid"].isin(eids_train)]
    if len(cohort_train_rows) != len(eids_train):
        # Sanity check: every cohort-train eid should be present exactly once.
        raise RuntimeError(
            f"[{disease_name}] cohort train rows ({len(cohort_train_rows)}) != "
            f"eids_train ({len(eids_train)}); "
            "df may contain duplicate or missing eids."
        )

    X_cohort_train = cohort_train_rows[feature_cols].values.astype(np.float32)
    # Match scripts/train.py:215-218 (np.mean / np.std defaults, guard zero-std).
    train_mean = np.mean(X_cohort_train, axis=0)
    train_std = np.std(X_cohort_train, axis=0)
    train_std[train_std == 0] = 1.0
    return train_mean.astype(np.float32), train_std.astype(np.float32)


def load_correlation_matrix(cfg, device: torch.device) -> torch.Tensor:
    """Load padded correlation matrix with CSV fallback (mirrors train path)."""
    corr_path = Path(cfg.data.correlation_matrix_path)
    if corr_path.suffix == ".pt" and corr_path.exists():
        return torch.load(str(corr_path), map_location=device)

    csv_path = Path("_reference/Pre-training/metabolomic_correlation_matrix.csv")
    if not csv_path.exists():
        csv_path = Path("_reference_correlation_matrix.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Correlation matrix not found at {corr_path} or fallback CSV paths."
        )

    corr_df = pd.read_csv(str(csv_path), index_col=0, encoding="utf-8-sig")
    bias_matrix = torch.tensor(corr_df.values, dtype=torch.float32, device=device)
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
    """Load one E0 checkpoint, predict on ``X``, return probabilities."""
    from src.model.backbone import MetaboliteBERTModel
    from src.model.heads import SingleTaskHead
    from src.model.wrapper import MetaboLMForClassification

    state_dict = torch.load(ckpt_path, map_location=device)

    backbone = MetaboliteBERTModel(num_metabolites=168)
    head = SingleTaskHead(hidden_size=768)
    model = MetaboLMForClassification(backbone, head)

    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[new_k] = v
    model.load_state_dict(cleaned, strict=False)

    model.metabolite_model.register_buffer("bias_matrix_full", bias_matrix)
    model.to(device)
    model.eval()

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
    parser = argparse.ArgumentParser(
        description="Evaluate E0 checkpoints on global val.csv with per-disease "
                    "cohort z-score replayed (E0's own trained pipeline)."
    )
    parser.add_argument("--config", default="configs/reproduce.yaml")
    parser.add_argument("--e0-dir", default="outputs/E0_reproduction")
    parser.add_argument("--output-dir", default="outputs/E0_global_eval_cohort")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--cohort-seed", type=int, default=42,
        help="Seed passed to build_disease_cohort — must match the value used "
             "at training time (default 42, see scripts/train.py).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    df, X_val, Y_val, feature_cols, label_cols = load_full_df_and_val(
        cfg.data.train_path, cfg.data.val_path,
    )
    disease_names = get_disease_names()

    bias_matrix = load_correlation_matrix(cfg, device)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for i, disease_name in enumerate(disease_names):
        ckpt_path = os.path.join(
            args.e0_dir, disease_name, f"best_finetune_model_{disease_name}.pt",
        )
        if not os.path.exists(ckpt_path):
            logger.warning("[%s] Checkpoint not found: %s — skipping",
                           disease_name, ckpt_path)
            continue

        label_col = f"label_{disease_name}"
        stats = compute_cohort_train_stats(
            df, disease_name, label_col, feature_cols, label_cols,
            seed=args.cohort_seed,
        )
        if stats is None:
            logger.warning("[%s] build_disease_cohort returned None — skipping",
                           disease_name)
            continue
        train_mean, train_std = stats

        # Apply E0's per-disease cohort z-score to the globally z-scored val.csv.
        X_val_d = ((X_val - train_mean) / train_std).astype(np.float32)

        logger.info(
            "[%s] Loading checkpoint and predicting on %d samples with "
            "per-disease cohort stats (mean_abs_max=%.4f, std_range=[%.4f, %.4f])",
            disease_name, len(X_val_d),
            float(np.max(np.abs(train_mean))),
            float(train_std.min()), float(train_std.max()),
        )
        probs = predict_single_disease(ckpt_path, X_val_d, bias_matrix, device,
                                       args.batch_size)

        labels = Y_val[:, i]
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0

        results.append({
            "Disease": disease_name,
            "Global_Val_AUC_Own_Preproc": round(auc, 4),
            "N_Positive": n_pos,
            "N_Negative": n_neg,
            "Prevalence_pct": round(n_pos / len(labels) * 100, 2),
        })
        logger.info("[%s] Global val AUC (own preproc) = %.4f "
                    "(pos=%d, neg=%d, prev=%.2f%%)",
                    disease_name, auc, n_pos, n_neg,
                    n_pos / len(labels) * 100)

        torch.cuda.empty_cache()

    if results:
        csv_path = os.path.join(args.output_dir, "e0_global_val_metrics_own_preproc.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        aucs = [r["Global_Val_AUC_Own_Preproc"] for r in results
                if r["Global_Val_AUC_Own_Preproc"] > 0]
        mean_auc = sum(aucs) / len(aucs) if aucs else 0
        # Aggregate per-disease positives. ``results`` currently holds only the
        # per-disease rows — the MEAN entry is appended *after* the dict literal
        # is fully constructed, so we iterate ``results`` directly here rather
        # than ``results[:-1]`` (which would drop the last disease).
        results.append({
            "Disease": "MEAN",
            "Global_Val_AUC_Own_Preproc": round(mean_auc, 4),
            "N_Positive": sum(r["N_Positive"] for r in results
                              if "N_Positive" in r),
            "N_Negative": "-",
            "Prevalence_pct": "-",
        })

        # Also emit the comparison against the original balanced eval numbers
        # so the log reader can see both preprocessing variants and the
        # original balanced-subset number side-by-side.
        e0_balanced = {
            "T2D": 0.867, "obesity": 0.757, "hypertension": 0.706,
            "ischemic_heart": 0.738, "atrial_fib": 0.683, "heart_failure": 0.722,
            "rheumatoid": 0.676, "asthma": 0.620, "dementia": 0.652,
            "copd": 0.753, "stroke": 0.651, "parkinsons": 0.611,
            "breast_cancer": 0.687, "colon_cancer": 0.604, "lung_cancer": 0.667,
            "prostate_cancer": 0.777,
        }
        logger.info("=" * 78)
        logger.info("E0 Global Val Evaluation — Own preprocessing (per-disease z-score)")
        logger.info("=" * 78)
        logger.info("%-20s %16s %16s %10s", "Disease",
                    "Global_Own_AUC", "Balanced_AUC*", "Delta")
        logger.info("-" * 78)
        for r in results:
            name = r["Disease"]
            g_auc = r["Global_Val_AUC_Own_Preproc"]
            orig = e0_balanced.get(name)
            if orig is not None:
                logger.info("%-20s %16.4f %16.4f %+10.4f",
                            name, g_auc, orig, g_auc - orig)
            else:
                logger.info("%-20s %16.4f %16s %10s", name, g_auc, "-", "-")
        logger.info("=" * 78)

        summary = {
            "eval_set": "global val.csv",
            "preprocessing": "per-disease cohort z-score replayed (seed=%d)" % args.cohort_seed,
            "n_samples": len(X_val),
            "mean_auc": mean_auc,
            "per_disease": {r["Disease"]: r["Global_Val_AUC_Own_Preproc"]
                            for r in results},
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
