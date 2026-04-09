#!/usr/bin/env python3
"""End-to-end data preparation: raw UK Biobank -> train/val CSVs + tensors.

Usage::

    python scripts/prepare_data.py --config configs/base.yaml

Steps:
 1. Load raw metabolomics and select 168 NMR biomarkers.
 2. Build per-participant baseline dates from blood-collection timestamps.
 3. Parse ICD-10 diagnoses and build incident/prevalent labels for 16 endpoints.
 4. Merge features + labels, split train / val.
 5. Impute missing values (median, fitted on train).
 6. Z-score normalize (fitted on train).
 7. Save processed CSVs, correlation-matrix tensor, and cohort statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

# Allow imports from project root when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.biomarkers import get_metabolite_names, select_biomarkers
from src.data.cohort import build_baseline_dates, build_cohort, build_diagnosis_labels
from src.data.endpoints import DISEASE_ENDPOINTS, get_disease_names
from src.data.preprocessing import impute_missing_simple, zscore_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fine-tuning cohort from raw UKB data.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--correlation-matrix",
        default="/SPXvePFS/users/jytang/metabolm_posttrain/_reference_correlation_matrix.csv",
        help="Path to reference correlation-matrix CSV (168x168).",
    )
    parser.add_argument("--skip-labels", action="store_true", help="Skip diagnosis labelling (for quick testing).")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── 1. Load metabolomics and select biomarkers ──────────────────────
    logger.info("Loading raw metabolomics from %s ...", cfg.data.raw_metabolomics_path)
    metab_raw = pd.read_csv(cfg.data.raw_metabolomics_path)
    logger.info("  Raw shape: %s", metab_raw.shape)

    metab = select_biomarkers(metab_raw, instance=0)
    logger.info("  Selected 168 biomarkers for %d participants.", len(metab))
    del metab_raw  # free ~1.2 GB

    # ── 2. Build baseline dates ─────────────────────────────────────────
    logger.info("Building baseline dates from %s ...", cfg.data.raw_time_blood_path)
    baseline_dates = build_baseline_dates(cfg.data.raw_time_blood_path)

    # ── 3. Build diagnosis labels ───────────────────────────────────────
    if args.skip_labels:
        logger.info("--skip-labels: creating dummy zero labels for all endpoints.")
        labels = baseline_dates[["eid"]].copy()
        for ep in DISEASE_ENDPOINTS:
            labels[f"label_{ep.name}"] = 0
        labels["is_prevalent_any"] = False
    else:
        logger.info("Building diagnosis labels for %d endpoints ...", len(DISEASE_ENDPOINTS))
        labels = build_diagnosis_labels(
            cfg.data.raw_diagnoses_path,
            cfg.data.raw_first_occur_path,
            cfg.data.raw_cause_of_death_path,
            baseline_dates,
            DISEASE_ENDPOINTS,
        )

    # ── 4. Build cohort (merge + train/val split) ──────────────────────
    logger.info("Building cohort (merge + split) ...")
    train_df, val_df = build_cohort(metab, labels, seed=cfg.seed)
    del metab, labels  # free memory

    # ── 5. Impute missing values ────────────────────────────────────────
    feature_cols = get_metabolite_names()
    logger.info("Imputing missing values (%d features) ...", len(feature_cols))
    train_df, val_df = impute_missing_simple(train_df, val_df, feature_cols)

    # ── 6. Z-score normalize ───────────────────────────────────────────
    logger.info("Z-score normalizing ...")
    train_df, val_df, means, stds = zscore_normalize(train_df, val_df, feature_cols)

    # ── 7. Save outputs ────────────────────────────────────────────────
    out_dir = Path(cfg.data.train_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(cfg.data.train_path)
    val_path = Path(cfg.data.val_path)

    logger.info("Saving train CSV to %s ...", train_path)
    train_df.to_csv(train_path, index=False)

    logger.info("Saving val CSV to %s ...", val_path)
    val_df.to_csv(val_path, index=False)

    # Save normalization stats for potential later use
    norm_stats = pd.DataFrame({"mean": means, "std": stds})
    norm_stats_path = out_dir / "normalization_stats.csv"
    norm_stats.to_csv(norm_stats_path)
    logger.info("Saved normalization stats to %s", norm_stats_path)

    # ── Correlation matrix tensor ──────────────────────────────────────
    corr_csv = cfg.data.correlation_matrix_path
    corr_path = Path(corr_csv)

    # If config points to a .pt file, build it from the reference CSV
    if corr_path.suffix == ".pt":
        ref_csv = args.correlation_matrix
        logger.info("Building correlation-matrix tensor from %s ...", ref_csv)
        corr_df = pd.read_csv(ref_csv, index_col=0, encoding="utf-8-sig")
        # Pad with zeros for [CLS] token: position 0 is CLS
        n = len(corr_df) + 1  # 169 x 169
        padded = torch.zeros(n, n)
        padded[1:, 1:] = torch.tensor(corr_df.values, dtype=torch.float32)
        torch.save(padded, str(corr_path))
        logger.info("Saved %dx%d correlation-matrix tensor to %s", n, n, corr_path)
    elif corr_path.suffix == ".csv":
        # Config points directly to a CSV — load and convert
        logger.info("Loading correlation matrix from %s ...", corr_path)
        corr_df = pd.read_csv(str(corr_path), index_col=0, encoding="utf-8-sig")
        n = len(corr_df) + 1
        padded = torch.zeros(n, n)
        padded[1:, 1:] = torch.tensor(corr_df.values, dtype=torch.float32)
        pt_path = out_dir / "correlation_matrix.pt"
        torch.save(padded, str(pt_path))
        logger.info("Saved %dx%d correlation-matrix tensor to %s", n, n, pt_path)

    # ── Cohort statistics ──────────────────────────────────────────────
    label_cols = [f"label_{ep.name}" for ep in DISEASE_ENDPOINTS]
    stats: dict = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "num_features": len(feature_cols),
        "num_endpoints": len(DISEASE_ENDPOINTS),
    }
    for col in label_cols:
        if col in train_df.columns:
            stats[f"train_{col}_positive"] = int(train_df[col].sum())
            stats[f"train_{col}_prevalence"] = round(float(train_df[col].mean()), 6)
            stats[f"val_{col}_positive"] = int(val_df[col].sum())
            stats[f"val_{col}_prevalence"] = round(float(val_df[col].mean()), 6)

    stats_path = out_dir / "cohort_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved cohort statistics to %s", stats_path)

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DONE  Train: %d   Val: %d   Features: %d", len(train_df), len(val_df), len(feature_cols))
    for col in label_cols:
        if col in train_df.columns:
            logger.info(
                "  %-25s  train=%5d (%.3f%%)  val=%5d (%.3f%%)",
                col,
                stats[f"train_{col}_positive"],
                stats[f"train_{col}_prevalence"] * 100,
                stats[f"val_{col}_positive"],
                stats[f"val_{col}_prevalence"] * 100,
            )
    logger.info("Output directory: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
