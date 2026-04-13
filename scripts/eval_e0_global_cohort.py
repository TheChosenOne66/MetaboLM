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
        --output-dir outputs/E0_global_eval_cohort \
        --ckpt-disease T2D
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
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import build_disease_cohort  # noqa: E402  single source of truth
from src.config import _resolve_output_path, load_config  # noqa: E402
from src.data.biomarkers import get_metabolite_names  # noqa: E402
from src.data.endpoints import get_disease_names  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval_e0_global_cohort")


def load_full_df_and_val(
    train_path: str, val_path: str
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, list[str], list[str]]:
    """Return (df_concat, eids, X_val, Y_val, feature_cols, label_cols).

    ``df_concat`` mirrors what ``scripts/train.py`` constructs (train + val
    concatenated, already globally z-scored) so ``build_disease_cohort``
    behaves identically to training. ``eids``/``X_val``/``Y_val`` are the
    val.csv rows we score each checkpoint on (eids kept as strings to avoid
    silent int casts when written back to CSV).
    """
    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{d}" for d in disease_names]

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    df = pd.concat([train_df, val_df], ignore_index=True)

    eids = [str(e) for e in val_df["eid"].tolist()]
    X_val = val_df[feature_cols].values.astype(np.float32)
    Y_val = val_df[label_cols].values.astype(np.float32)
    logger.info("Loaded val.csv: %d samples, %d features, %d labels",
                X_val.shape[0], X_val.shape[1], Y_val.shape[1])
    logger.info("Loaded train+val concat for cohort replay: %d rows", len(df))
    return df, eids, X_val, Y_val, feature_cols, label_cols


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


def compute_cohort_train_stats(
    df: pd.DataFrame,
    disease_name: str,
    label_col: str,
    feature_cols: list[str],
    label_cols: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Recompute the per-disease cohort train mean/std used during E0 training.

    Delegates cohort construction (case+control sampling, stratified split)
    to ``scripts/train.py:build_disease_cohort`` so there is no duplicated
    sampling logic. We then recompute stats from ``df`` in the same globally
    z-scored space the training function operated in, using the train
    ``eid`` list it returned.

    Returns ``(train_mean, train_std, n_cohort_train)`` or ``None`` if
    ``build_disease_cohort`` returned ``None`` (cohort too small).
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
    return (
        train_mean.astype(np.float32),
        train_std.astype(np.float32),
        len(eids_train),
    )


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
    parser.add_argument(
        "--ckpt-disease",
        required=True,
        help="Disease name whose E0 checkpoint should be evaluated.",
    )
    parser.add_argument(
        "--eval-diseases",
        nargs="+",
        default=None,
        help="Subset of disease labels to evaluate. Defaults to all diseases.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--cohort-seed", type=int, default=None,
        help="Seed passed to build_disease_cohort. Must match the seed used at "
             "training time. If omitted, falls back to ``cfg.seed`` from the "
             "config file (matches scripts/train.py:main, which passes "
             "``cfg.seed`` into build_disease_cohort).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Resolve cohort seed: explicit CLI override > config > error.
    cohort_seed = args.cohort_seed
    if cohort_seed is None:
        cohort_seed = getattr(cfg, "seed", None)
    if cohort_seed is None:
        raise ValueError(
            "Cohort seed could not be resolved: pass --cohort-seed or set "
            "``seed`` in the config file."
        )
    logger.info("Cohort seed (must match training): %d", cohort_seed)

    df, eids, X_val, Y_val, feature_cols, label_cols = load_full_df_and_val(
        cfg.data.train_path, cfg.data.val_path,
    )
    disease_names = get_disease_names()
    disease_to_index = {name: idx for idx, name in enumerate(disease_names)}
    if args.ckpt_disease not in disease_to_index:
        raise ValueError(
            f"--ckpt-disease '{args.ckpt_disease}' is not in canonical diseases: {disease_names}"
        )

    eval_diseases = args.eval_diseases or disease_names
    unknown_eval_diseases = [name for name in eval_diseases if name not in disease_to_index]
    if unknown_eval_diseases:
        raise ValueError(
            f"--eval-diseases contains unknown diseases: {unknown_eval_diseases}"
        )

    bias_matrix = load_correlation_matrix(cfg, device)

    resolved_e0_dir = _resolve_output_path(args.e0_dir)
    resolved_output_dir = _resolve_output_path(args.output_dir)
    os.makedirs(resolved_output_dir, exist_ok=True)
    ckpt_disease = args.ckpt_disease
    ckpt_path = os.path.join(
        resolved_e0_dir, ckpt_disease, f"best_finetune_model_{ckpt_disease}.pt",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_subdir = os.path.join(resolved_output_dir, ckpt_disease)
    os.makedirs(ckpt_subdir, exist_ok=True)

    label_col = f"label_{ckpt_disease}"
    stats = compute_cohort_train_stats(
        df, ckpt_disease, label_col, feature_cols, label_cols,
        seed=cohort_seed,
    )
    if stats is None:
        raise RuntimeError(f"[{ckpt_disease}] build_disease_cohort returned None")
    train_mean, train_std, n_cohort_train = stats

    stats_path = os.path.join(
        ckpt_subdir, f"cohort_stats_ckpt_{ckpt_disease}.json",
    )
    with open(stats_path, "w") as f:
        json.dump({
            "ckpt_disease": ckpt_disease,
            "seed": int(cohort_seed),
            "feature_cols": feature_cols,
            "train_mean": train_mean.tolist(),
            "train_std": train_std.tolist(),
            "n_cohort_train": int(n_cohort_train),
            "build_disease_cohort_source": "scripts/train.py:build_disease_cohort",
        }, f, indent=2)

    X_val_d = ((X_val - train_mean) / train_std).astype(np.float32)

    logger.info(
        "[%s] Loading checkpoint and predicting on %d samples with "
        "per-disease cohort stats (mean_abs_max=%.4f, std_range=[%.4f, %.4f], "
        "e0_dir=%s, output_dir=%s)",
        ckpt_disease, len(X_val_d),
        float(np.max(np.abs(train_mean))),
        float(train_std.min()), float(train_std.max()),
        resolved_e0_dir, resolved_output_dir,
    )
    probs = predict_single_disease(ckpt_path, X_val_d, bias_matrix, device,
                                   args.batch_size)
    probs_flat = np.asarray(probs).reshape(-1).astype(np.float64)

    pred_path = os.path.join(
        ckpt_subdir, f"predictions_ckpt_{ckpt_disease}.csv",
    )
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eid", f"prob_ckpt_{ckpt_disease}"])
        for j, eid in enumerate(eids):
            writer.writerow([eid, f"{probs_flat[j]:.6f}"])

    metrics_path = os.path.join(
        ckpt_subdir, f"metrics_ckpt_{ckpt_disease}.csv",
    )
    rows = []
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ckpt_disease", "eval_label", "Val_AUC",
            "N_Positive", "N_Negative", "Prevalence_pct",
        ])
        for eval_disease in eval_diseases:
            labels = Y_val[:, disease_to_index[eval_disease]]
            n_pos = int(labels.sum())
            n_neg = len(labels) - n_pos

            try:
                auc = float(roc_auc_score(labels, probs_flat))
            except ValueError as e:
                logger.warning("[%s -> %s] roc_auc_score failed (%s); recording NaN",
                               ckpt_disease, eval_disease, e)
                auc = float("nan")

            prev_pct = round(n_pos / len(labels) * 100, 2)
            auc_cell = "" if math.isnan(auc) else f"{auc:.4f}"
            writer.writerow([ckpt_disease, eval_disease, auc_cell,
                             n_pos, n_neg, prev_pct])
            rows.append({
                "eval_label": eval_disease,
                "auc": auc,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "prev_pct": prev_pct,
            })

    logger.info("=" * 78)
    logger.info("E0 Global Val Evaluation — own preprocessing")
    logger.info("Checkpoint disease: %s", ckpt_disease)
    logger.info("=" * 78)
    logger.info("%-20s %16s %8s %8s %10s", "Eval_Label", "Global_AUC", "N_Pos", "N_Neg", "Prev%")
    logger.info("-" * 78)
    for row in rows:
        auc_text = "NaN" if math.isnan(row["auc"]) else f"{row['auc']:.4f}"
        logger.info(
            "%-20s %16s %8d %8d %10.2f",
            row["eval_label"], auc_text, row["n_pos"], row["n_neg"], row["prev_pct"],
        )
    logger.info("=" * 78)
    logger.info("Worker results saved to %s", ckpt_subdir)


if __name__ == "__main__":
    main()
