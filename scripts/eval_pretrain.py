#!/usr/bin/env python3
"""Evaluate MetaboLM pretrained checkpoint via masked metabolite reconstruction.

Verifies that the pretrained backbone loads correctly and produces reasonable
reconstruction metrics on real UK Biobank data before proceeding to fine-tuning.

Usage::

    python scripts/eval_pretrain.py \
        --metabolomics /SPXvePFS/users/jytang/storage_tmp/metabolomics.csv \
        --diagnosis /SPXvePFS/users/jytang/storage_tmp/ICD10_prep/diagnosis_icd10.csv \
        --cause-of-death /SPXvePFS/users/jytang/storage_tmp/ICD10_prep/cause_of_death.csv \
        --checkpoint weights/best_metabolite_bert_model.pt \
        --output-dir outputs/pretrain_eval

Protocol (aligned with official Pretraining.py):
  1. Load metabolomics, select 168 NMR biomarkers (instance 0).
  2. Build healthy cohort (exclude 16 diseases + all cancers).
  3. Split healthy 9:1 → pretraining set (90%).
  4. Split pretraining set 80/20 → train / val.
  5. Median impute + z-score normalize (fit on train).
  6. Load pretrained checkpoint.
  7. Masked reconstruction evaluation on val set (multi-pass).
  8. Report MSE, MAE, R², Accuracy.

Paper reference values (healthy cohort, 15% mask, 200 epochs):
  Val Loss (MSE): ~0.0684
  Val R²:         ~0.9305
  Val Accuracy:   ~0.9533  (tolerance = 0.5)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.biomarkers import get_metabolite_names, select_biomarkers
from src.data.healthy_cohort import build_healthy_eids, split_healthy_cohort
from src.data.preprocessing import impute_missing_simple, zscore_normalize
from src.model.backbone import MetaboliteBERTModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_pretrain")


# ─── Dataset ────────────────────────────────────────────────────────────────

class MetaboliteDataset(Dataset):
    """Simple dataset: returns expression tensor only (no labels)."""

    def __init__(self, expressions: np.ndarray) -> None:
        self.X = expressions.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.X[idx], dtype=torch.float32)


# ─── Masking ────────────────────────────────────────────────────────────────

def mask_expressions(
    expressions: torch.Tensor,
    mask_rate: float,
    random_min: float,
    random_max: float,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """Apply BERT-style masking to expression values.

    Matches official Pretraining.py masking strategy:
      - mask_rate% of positions are selected for masking
      - 80% of masked positions → replace with 0.0
      - 10% of masked positions → replace with random value in [min, max]
      - 10% of masked positions → keep unchanged

    Returns (masked_expressions, mask_indices).
    """
    device = expressions.device
    probability_matrix = torch.full(expressions.size(), mask_rate, device=device)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    if masked_indices.sum().item() == 0:
        # Force at least one mask per sample
        batch_size, num_met = expressions.size()
        rand_pos = torch.randint(0, num_met, (batch_size,), device=device)
        masked_indices[torch.arange(batch_size, device=device), rand_pos] = True

    expressions_masked = expressions.clone()

    replace_prob = torch.rand(masked_indices.sum().item(), device=device)
    mask_as_zero = replace_prob < 0.8
    mask_as_random = (replace_prob >= 0.8) & (replace_prob < 0.9)
    # remaining 10%: keep unchanged

    # Get masked values, apply replacements
    masked_vals = expressions_masked[masked_indices].clone()
    masked_vals[mask_as_zero] = 0.0
    if mask_as_random.sum() > 0:
        random_values = torch.empty(
            mask_as_random.sum().item(), device=device
        ).uniform_(random_min, random_max)
        masked_vals[mask_as_random] = random_values

    expressions_masked[masked_indices] = masked_vals

    return expressions_masked, masked_indices


# ─── Evaluation Loop ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_one_pass(
    model: MetaboliteBERTModel,
    dataloader: DataLoader,
    mask_rate: float,
    random_min: float,
    random_max: float,
    tolerance: float,
    device: torch.device,
) -> dict[str, float]:
    """Run one full evaluation pass over the dataloader.

    Returns dict with keys: loss, mse, mae, r2, accuracy, n_masked.
    """
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    sum_sq_errors = 0.0
    sum_abs_errors = 0.0
    sum_actual = 0.0
    sum_actual_sq = 0.0

    for expressions in dataloader:
        expressions = expressions.to(device)
        attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

        masked_expr, masked_indices = mask_expressions(
            expressions, mask_rate, random_min, random_max
        )

        n_masked = masked_indices.sum().item()
        if n_masked == 0:
            continue

        # Forward: returns (prediction_scores, attentions)
        prediction_scores, _ = model(masked_expr, attention_mask)

        # Compute loss on masked positions only
        pred_masked = prediction_scores[masked_indices]
        actual_masked = expressions[masked_indices]

        loss = criterion(pred_masked, actual_masked)
        total_loss += loss.item()

        # Accuracy: |pred - actual| < tolerance
        correct = (torch.abs(pred_masked - actual_masked) < tolerance).sum().item()
        total_correct += correct
        total_masked += n_masked

        # For MSE / MAE / R²
        errors = pred_masked - actual_masked
        sum_sq_errors += (errors ** 2).sum().item()
        sum_abs_errors += errors.abs().sum().item()
        sum_actual += actual_masked.sum().item()
        sum_actual_sq += (actual_masked ** 2).sum().item()

    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    accuracy = total_correct / total_masked if total_masked > 0 else 0.0

    if total_masked > 0:
        mse = sum_sq_errors / total_masked
        mae = sum_abs_errors / total_masked
        sst = sum_actual_sq - (sum_actual ** 2 / total_masked)
        r2 = 1 - (sum_sq_errors / sst) if sst > 0 else 0.0
    else:
        mse, mae, r2 = 0.0, 0.0, 0.0

    return {
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "accuracy": accuracy,
        "n_masked": total_masked,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MetaboLM pretrained checkpoint (masked reconstruction)."
    )
    parser.add_argument(
        "--metabolomics", required=True,
        help="Path to raw metabolomics.csv (~501K participants).",
    )
    parser.add_argument(
        "--diagnosis", required=True,
        help="Path to diagnosis_icd10.csv (p41270).",
    )
    parser.add_argument(
        "--cause-of-death", required=True,
        help="Path to cause_of_death.csv (p40001).",
    )
    parser.add_argument(
        "--checkpoint", default="weights/best_metabolite_bert_model.pt",
        help="Path to pretrained checkpoint.",
    )
    parser.add_argument("--mask-rate", type=float, default=0.10,
                        help="Mask rate (paper=0.10, code default=0.15).")
    parser.add_argument("--num-passes", type=int, default=5,
                        help="Number of evaluation passes (average out mask randomness).")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Accuracy tolerance: |pred-actual| < tol.")
    parser.add_argument("--output-dir", default="outputs/pretrain_eval")
    parser.add_argument(
        "--skip-healthy-filter", action="store_true",
        help="Skip healthy cohort filtering (use all participants).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load metabolomics and select 168 biomarkers ──────────────────
    logger.info("Loading metabolomics from %s ...", args.metabolomics)
    metab_raw = pd.read_csv(args.metabolomics)
    logger.info("  Raw shape: %s", metab_raw.shape)

    metab = select_biomarkers(metab_raw, instance=0)
    all_eids = set(metab["eid"].astype(int).tolist())
    logger.info("  Selected 168 biomarkers for %d participants.", len(metab))
    del metab_raw

    # ── 2. Build healthy cohort ─────────────────────────────────────────
    if args.skip_healthy_filter:
        logger.info("--skip-healthy-filter: using all %d participants.", len(metab))
        healthy_eids = all_eids
    else:
        logger.info("Building healthy cohort (excluding 16 diseases + all cancers)...")
        healthy_eids = build_healthy_eids(
            args.diagnosis, args.cause_of_death, metabolomics_eids=all_eids,
        )

    # ── 3. Split healthy 9:1 → take 90% as pretraining cohort ──────────
    pretrain_eids, finetune_eids = split_healthy_cohort(
        healthy_eids, pretrain_fraction=0.9, seed=args.seed,
    )
    pretrain_eid_set = set(pretrain_eids)

    # Filter metabolomics to pretraining cohort
    metab_pretrain = metab[metab["eid"].isin(pretrain_eid_set)].copy()
    metab_pretrain = metab_pretrain.reset_index(drop=True)
    logger.info("Pretraining cohort: %d participants.", len(metab_pretrain))
    del metab

    # ── 4. Drop rows with all-NaN metabolites ───────────────────────────
    feature_cols = get_metabolite_names()
    n_present = metab_pretrain[feature_cols].notna().sum(axis=1)
    metab_pretrain = metab_pretrain[n_present >= 1].reset_index(drop=True)
    logger.info("After NaN filter: %d participants.", len(metab_pretrain))

    # ── 5. Train/val split 80/20 ────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        metab_pretrain, test_size=0.2, random_state=args.seed, shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    logger.info("Split: train=%d, val=%d", len(train_df), len(val_df))

    # ── 6. Impute + z-score normalize ───────────────────────────────────
    logger.info("Imputing missing values (median, fit on train)...")
    train_df, val_df = impute_missing_simple(train_df, val_df, feature_cols)

    logger.info("Z-score normalizing (fit on train)...")
    train_df, val_df, means, stds = zscore_normalize(train_df, val_df, feature_cols)

    # Compute random replacement range (matching official code)
    train_vals = train_df[feature_cols].values
    random_min = float(train_vals.min())
    random_max = float(train_vals.max())
    logger.info("Random replacement range: [%.3f, %.3f]", random_min, random_max)

    # ── 7. Build datasets and dataloaders ───────────────────────────────
    train_X = train_df[feature_cols].values
    val_X = val_df[feature_cols].values

    train_dataset = MetaboliteDataset(train_X)
    val_dataset = MetaboliteDataset(val_X)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 8. Load pretrained model ────────────────────────────────────────
    logger.info("Loading pretrained checkpoint from %s ...", args.checkpoint)
    model = MetaboliteBERTModel.load_pretrained(args.checkpoint, num_metabolites=168)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %d parameters (%.1f M).", n_params, n_params / 1e6)

    # ── 9. Run evaluation ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting evaluation: mask_rate=%.2f, tolerance=%.1f, passes=%d",
                args.mask_rate, args.tolerance, args.num_passes)
    logger.info("=" * 60)

    all_results: list[dict] = []

    for pass_idx in range(args.num_passes):
        # Set different seed per pass for mask randomness
        torch.manual_seed(args.seed + pass_idx * 1000)

        t0 = time.time()

        # Evaluate on validation set
        val_metrics = evaluate_one_pass(
            model, val_loader, args.mask_rate, random_min, random_max,
            args.tolerance, device,
        )
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

        # Also evaluate on train set (for reference)
        train_metrics = evaluate_one_pass(
            model, train_loader, args.mask_rate, random_min, random_max,
            args.tolerance, device,
        )
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}

        elapsed = time.time() - t0
        pass_result = {"pass": pass_idx + 1, **train_metrics, **val_metrics}
        all_results.append(pass_result)

        logger.info(
            "Pass %d/%d (%.1fs): "
            "Val MSE=%.4f  MAE=%.4f  R²=%.4f  Acc=%.4f  |  "
            "Train MSE=%.4f  R²=%.4f  Acc=%.4f",
            pass_idx + 1, args.num_passes, elapsed,
            val_metrics["val_mse"], val_metrics["val_mae"],
            val_metrics["val_r2"], val_metrics["val_accuracy"],
            train_metrics["train_mse"], train_metrics["train_r2"],
            train_metrics["train_accuracy"],
        )

    # ── 10. Aggregate results ───────────────────────────────────────────
    metric_keys = [k for k in all_results[0] if k != "pass"]
    summary: dict[str, float] = {}
    for k in metric_keys:
        values = [r[k] for r in all_results]
        summary[f"{k}_mean"] = float(np.mean(values))
        summary[f"{k}_std"] = float(np.std(values))

    logger.info("=" * 60)
    logger.info("SUMMARY (%d passes, mask_rate=%.2f)", args.num_passes, args.mask_rate)
    logger.info("-" * 60)
    logger.info("  Val  MSE:      %.4f ± %.4f", summary["val_mse_mean"], summary["val_mse_std"])
    logger.info("  Val  MAE:      %.4f ± %.4f", summary["val_mae_mean"], summary["val_mae_std"])
    logger.info("  Val  R²:       %.4f ± %.4f", summary["val_r2_mean"], summary["val_r2_std"])
    logger.info("  Val  Accuracy: %.4f ± %.4f", summary["val_accuracy_mean"], summary["val_accuracy_std"])
    logger.info("-" * 60)
    logger.info("  Train MSE:      %.4f ± %.4f", summary["train_mse_mean"], summary["train_mse_std"])
    logger.info("  Train R²:       %.4f ± %.4f", summary["train_r2_mean"], summary["train_r2_std"])
    logger.info("  Train Accuracy: %.4f ± %.4f", summary["train_accuracy_mean"], summary["train_accuracy_std"])
    logger.info("-" * 60)
    logger.info("Paper reference (healthy, 15%% mask): MSE~0.0684, R²~0.9305, Acc~0.9533")
    logger.info("=" * 60)

    # ── 11. Save outputs ────────────────────────────────────────────────
    # Per-pass results
    results_path = out_dir / "pretrain_eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "checkpoint": args.checkpoint,
                "mask_rate": args.mask_rate,
                "tolerance": args.tolerance,
                "num_passes": args.num_passes,
                "seed": args.seed,
                "skip_healthy_filter": args.skip_healthy_filter,
                "batch_size": args.batch_size,
                "n_params": n_params,
            },
            "data_stats": {
                "n_healthy_total": len(healthy_eids),
                "n_pretrain_cohort": len(pretrain_eids),
                "n_finetune_ctrl": len(finetune_eids),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "random_min": random_min,
                "random_max": random_max,
            },
            "per_pass": all_results,
            "summary": summary,
        }, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Human-readable summary
    summary_path = out_dir / "pretrain_eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write("MetaboLM Pretrained Checkpoint Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint:     {args.checkpoint}\n")
        f.write(f"Mask rate:      {args.mask_rate}\n")
        f.write(f"Tolerance:      {args.tolerance}\n")
        f.write(f"Passes:         {args.num_passes}\n")
        f.write(f"Healthy filter: {'OFF' if args.skip_healthy_filter else 'ON'}\n")
        f.write(f"Healthy total:  {len(healthy_eids)}\n")
        f.write(f"Pretrain set:   {len(pretrain_eids)}\n")
        f.write(f"Train / Val:    {len(train_df)} / {len(val_df)}\n\n")
        f.write("Validation Metrics (mean ± std)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  MSE:      {summary['val_mse_mean']:.4f} ± {summary['val_mse_std']:.4f}\n")
        f.write(f"  MAE:      {summary['val_mae_mean']:.4f} ± {summary['val_mae_std']:.4f}\n")
        f.write(f"  R²:       {summary['val_r2_mean']:.4f} ± {summary['val_r2_std']:.4f}\n")
        f.write(f"  Accuracy: {summary['val_accuracy_mean']:.4f} ± {summary['val_accuracy_std']:.4f}\n\n")
        f.write("Paper Reference (healthy, 15% mask)\n")
        f.write("-" * 50 + "\n")
        f.write("  MSE:      ~0.0684\n")
        f.write("  R²:       ~0.9305\n")
        f.write("  Accuracy: ~0.9533\n")
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
