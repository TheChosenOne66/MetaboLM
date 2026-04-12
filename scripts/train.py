#!/usr/bin/env python3
"""Per-disease fine-tuning script for MetaboLM E0 reproduction.

Matches the official MetaboLM_Fine-tuning.py orchestration:
  1. Load expression data + correlation matrix
  2. For each of 16 diseases:
     a. Build balanced cohort (1:1 disease : healthy controls)
     b. Split 80/20 stratified
     c. Z-score normalise (fitted on train split)
     d. Fine-tune backbone + Linear(768,1) head
     e. Save best model (by val AUC) + predictions + metrics
  3. Output summary table

Usage::

    python scripts/train.py --config configs/reproduce.yaml

Equivalence notes vs official code:
  - Architecture: MetaboliteBERTModel + nn.Linear(768,1) via pooler_output  [identical]
  - Hyperparams: lr=2e-5, epochs=40, bs=512, AdamW, cosine+warmup(10%)     [identical]
  - Loss: BCEWithLogitsLoss, default reduction                              [identical]
  - Model selection: best validation AUROC                                  [identical]
  - Data: per-disease balanced 1:1 cohort, per-disease z-score              [identical]
  - Official uses no missing-value imputation in training code; we accept
    that prepare_data.py may have imputed upstream (negligible impact).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.biomarkers import get_metabolite_names
from src.data.dataset import FineTuneDataset
from src.data.endpoints import DISEASE_ENDPOINTS, get_disease_names
from src.model.backbone import MetaboliteBERTModel
from src.model.heads import SingleTaskHead
from src.model.wrapper import MetaboLMForClassification
from src.training.sft_trainer import SFTTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train")


def load_expression_and_labels(cfg) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load expression data with labels.

    Tries two data sources in order:
      1. Pre-processed train+val CSVs (from prepare_data.py) — already has
         eid + 168 metabolites + 16 label columns. If these exist, combine
         them back into a single DataFrame (per-disease splitting is done here).
      2. Raw metabolomics CSV — requires separate label construction.

    Returns:
        (expression_df, feature_cols, label_cols)
    """
    train_path = Path(cfg.data.train_path)
    val_path = Path(cfg.data.val_path)

    if train_path.exists() and val_path.exists():
        logger.info("Loading pre-processed data from %s and %s", train_path, val_path)
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        # Combine for per-disease re-splitting (matching official flow)
        df = pd.concat([train_df, val_df], ignore_index=True)
        del train_df, val_df
    else:
        raise FileNotFoundError(
            f"Processed data not found at {train_path} / {val_path}. "
            "Run scripts/prepare_data.py first."
        )

    feature_cols = get_metabolite_names()
    label_cols = [f"label_{name}" for name in get_disease_names()]

    # Verify columns exist
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features[:5]}...")

    missing_labels = [c for c in label_cols if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels[:5]}...")

    logger.info(
        "Loaded %d participants, %d features, %d label columns",
        len(df), len(feature_cols), len(label_cols),
    )
    return df, feature_cols, label_cols


def load_correlation_matrix(cfg, device: torch.device) -> torch.Tensor:
    """Load padded correlation matrix (169x169) for the backbone."""
    corr_path = Path(cfg.data.correlation_matrix_path)

    if corr_path.suffix == ".pt" and corr_path.exists():
        bias_matrix = torch.load(str(corr_path), map_location=device)
    else:
        # Fallback: load from reference CSV and pad
        csv_path = Path("_reference/Pre-training/metabolomic_correlation_matrix.csv")
        if not csv_path.exists():
            csv_path = Path("_reference_correlation_matrix.csv")
        corr_df = pd.read_csv(str(csv_path), index_col=0, encoding="utf-8-sig")
        bias_matrix = torch.tensor(corr_df.values, dtype=torch.float32, device=device)
        # Pad with zeros for [CLS] token at position 0
        bias_matrix = torch.nn.functional.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)

    logger.info("Correlation matrix shape: %s", bias_matrix.shape)
    return bias_matrix


def build_disease_cohort(
    df: pd.DataFrame,
    disease_name: str,
    label_col: str,
    feature_cols: list[str],
    all_label_cols: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray, list]:
    """Build balanced 1:1 cohort for a single disease, matching official code.

    Official logic::

        df_disease = read_csv('{disease}_labels.csv')  # disease eids
        df_healthy = read_csv('healthy_eids_10.csv')    # healthy controls
        # remove overlap
        healthy_available = healthy_eids - disease_eids
        # sample to match disease count
        df_healthy = df_healthy.sample(n=disease_count, random_state=42)
        # merge with expression data, split 80/20 stratified
        X_train, X_val = train_test_split(X, y, test_size=0.2, stratify=y)
        # z-score on train
        mean, std = X_train.mean(0), X_train.std(0)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

    Args:
        df: Full expression DataFrame with eid + features + labels.
        disease_name: For logging.
        label_col: Label column name (e.g. 'label_T2D').
        feature_cols: List of 168 metabolite column names.
        all_label_cols: All 16 label columns (to define "healthy").
        seed: Random seed.

    Returns:
        (X_train, y_train, eids_train, X_val, y_val, eids_val)
        All feature arrays are z-score normalised (fitted on train).
    """
    # Disease cases
    disease_mask = df[label_col] == 1
    df_disease = df[disease_mask]
    disease_count = len(df_disease)

    # Healthy controls: no positive label for ANY of the 16 diseases
    healthy_mask = (df[all_label_cols].sum(axis=1) == 0)
    df_healthy_all = df[healthy_mask]

    # Remove any overlap (shouldn't exist, but match official safety check)
    df_healthy_available = df_healthy_all[
        ~df_healthy_all["eid"].isin(df_disease["eid"])
    ]

    # Sample healthy to match disease count (1:1 ratio, matching official)
    if len(df_healthy_available) < disease_count:
        logger.warning(
            "[%s] Available healthy (%d) < disease (%d). Using all.",
            disease_name, len(df_healthy_available), disease_count,
        )
        df_healthy = df_healthy_available
    else:
        df_healthy = df_healthy_available.sample(n=disease_count, random_state=seed)

    # Combine
    df_cohort = pd.concat(
        [df_disease[["eid"] + feature_cols + [label_col]],
         df_healthy[["eid"] + feature_cols + [label_col]]],
        ignore_index=True,
    )

    logger.info(
        "[%s] Cohort: %d disease + %d healthy = %d total",
        disease_name, disease_count, len(df_healthy), len(df_cohort),
    )

    X = df_cohort[feature_cols].values
    y = df_cohort[label_col].values
    eids = df_cohort["eid"].tolist()

    if len(df_cohort) < 50:
        logger.warning("[%s] Too few samples (%d), skipping.", disease_name, len(df_cohort))
        return None

    # Split 80/20 stratified (matching official: test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val, eids_train, eids_val = train_test_split(
        X, y, eids, test_size=0.2, random_state=seed, stratify=y,
    )

    # Z-score normalise on train set (matching official code exactly)
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1.0
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std

    logger.info(
        "[%s] Train: %d (disease=%d), Val: %d (disease=%d)",
        disease_name,
        len(y_train), int(y_train.sum()),
        len(y_val), int(y_val.sum()),
    )

    return X_train, y_train, eids_train, X_val, y_val, eids_val


def train_single_disease(
    disease_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    eids_train: list,
    X_val: np.ndarray,
    y_val: np.ndarray,
    eids_val: list,
    pretrained_ckpt: str,
    bias_matrix: torch.Tensor,
    trainer_config: TrainerConfig,
    output_dir: str,
    device: torch.device,
    num_metabolites: int = 168,
) -> dict:
    """Train a single-disease model matching the official code."""
    disease_output_dir = os.path.join(output_dir, disease_name)
    os.makedirs(disease_output_dir, exist_ok=True)

    # Create dataloaders (matching official: batch_size=512, shuffle=True/False, num_workers=4)
    train_ds = FineTuneDataset(X_train, y_train, eids_train)
    val_ds = FineTuneDataset(X_val, y_val, eids_val)

    batch_size = 512  # official default; overridable via config in future
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Create model (matching official: fresh backbone per disease, load pretrained)
    # Use load_pretrained() to correctly strip DataParallel 'module.' prefix
    backbone = MetaboliteBERTModel.load_pretrained(pretrained_ckpt, num_metabolites=num_metabolites)
    # Replace the bias_matrix_full buffer with the one loaded from our correlation matrix
    backbone.register_buffer("bias_matrix_full", bias_matrix)

    head = SingleTaskHead(hidden_size=768)
    model = MetaboLMForClassification(backbone, head)
    model.to(device)

    # Train
    trainer = SFTTrainer(model, train_loader, val_loader, trainer_config, device)
    best_auc, best_epoch, best_state, val_labels, val_probs = trainer.train(disease_name)

    # Save best model (matching official: torch.save(best_model_state, path))
    if best_state is not None:
        model_path = os.path.join(disease_output_dir, f"best_finetune_model_{disease_name}.pt")
        torch.save(best_state, model_path)
        logger.info("[%s] Best model saved to %s", disease_name, model_path)

    # Save predictions (matching official CSV format)
    val_preds = [1 if p >= 0.5 else 0 for p in val_probs]
    results_df = pd.DataFrame({
        "eid": eids_val,
        "True_Label": val_labels,
        "Predicted_Probability": val_probs,
        "Predicted_Label": val_preds,
    })
    results_path = os.path.join(disease_output_dir, f"finetune_predictions_val_{disease_name}.csv")
    results_df.to_csv(results_path, index=False)

    # Compute final metrics on best model's predictions
    from src.training.metrics import compute_binary_metrics
    final_metrics = compute_binary_metrics(val_labels, val_probs)

    # Save per-disease summary (matching official format)
    summary = {
        "Disease": disease_name,
        "Best_Epoch": best_epoch,
        "Val_AUC": best_auc,
        "Val_Acc": final_metrics["accuracy"],
        "Val_Precision": final_metrics["precision"],
        "Val_Recall": final_metrics["recall"],
        "Val_F1": final_metrics["f1"],
        "Num_Disease": int(y_train.sum()) + int(y_val.sum()),
        "Num_Healthy": len(y_train) - int(y_train.sum()) + len(y_val) - int(y_val.sum()),
    }
    metrics_path = os.path.join(disease_output_dir, f"finetune_metrics_{disease_name}.csv")
    pd.DataFrame([summary]).to_csv(metrics_path, index=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaboLM per-disease fine-tuning (E0 reproduction)")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--diseases", nargs="*", default=None,
                        help="Subset of disease names to train (default: all 16).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Set seed (matching official set_seed(42) exactly)
    import random
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # Load data
    df, feature_cols, label_cols = load_expression_and_labels(cfg)
    bias_matrix = load_correlation_matrix(cfg, device)

    # Determine pretrained checkpoint path
    pretrained_ckpt = cfg.model.pretrained_ckpt
    if not Path(pretrained_ckpt).exists():
        pretrained_ckpt = os.path.join(
            os.path.dirname(__file__), "..", pretrained_ckpt
        )
    logger.info("Pretrained checkpoint: %s", pretrained_ckpt)

    # Training config (matching official hyperparameters)
    trainer_config = TrainerConfig(
        learning_rate=cfg.training.learning_rate,
        num_epochs=cfg.training.num_epochs,
        warmup_ratio=cfg.training.warmup_ratio,
    )

    # Select diseases to train
    disease_names = get_disease_names()
    if args.diseases:
        disease_names = [d for d in disease_names if d in args.diseases]
    logger.info("Training %d diseases: %s", len(disease_names), disease_names)

    # Output directory
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Per-disease training loop (matching official main() flow)
    all_summaries = []
    for disease_name in disease_names:
        label_col = f"label_{disease_name}"

        cohort = build_disease_cohort(
            df, disease_name, label_col, feature_cols, label_cols, seed=cfg.seed,
        )
        if cohort is None:
            continue

        X_train, y_train, eids_train, X_val, y_val, eids_val = cohort

        summary = train_single_disease(
            disease_name=disease_name,
            X_train=X_train, y_train=y_train, eids_train=eids_train,
            X_val=X_val, y_val=y_val, eids_val=eids_val,
            pretrained_ckpt=pretrained_ckpt,
            bias_matrix=bias_matrix,
            trainer_config=trainer_config,
            output_dir=output_dir,
            device=device,
        )
        all_summaries.append(summary)

    # Save summary table (matching official finetune_summary_metrics.csv)
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(output_dir, "finetune_summary_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info("Summary saved to %s", summary_path)

        # Print AUC table
        logger.info("=" * 60)
        logger.info("RESULTS: Per-disease AUROC")
        logger.info("=" * 60)
        for _, row in summary_df.iterrows():
            logger.info("  %-20s  AUC = %.4f", row["Disease"], row["Val_AUC"])
        logger.info("  %-20s  AUC = %.4f", "MEAN", summary_df["Val_AUC"].mean())
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
