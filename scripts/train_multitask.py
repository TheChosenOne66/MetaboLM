#!/usr/bin/env python3
"""Multi-task hierarchical fine-tuning for MetaboLM (E1-E4).

Unlike E0 (per-disease independent models), this trains a SINGLE model
that jointly predicts 16 diseases + 6 ICD-10 chapters with hierarchical
loss and optional parameter-efficient fine-tuning.

Usage::

    python scripts/train_multitask.py --config configs/sft_full_ft.yaml      # E1
    python scripts/train_multitask.py --config configs/sft_head_only.yaml    # E2
    python scripts/train_multitask.py --config configs/sft_adapter.yaml      # E3
    python scripts/train_multitask.py --config configs/sft_lora.yaml         # E4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.biomarkers import get_metabolite_names
from src.data.dataset import MultiTaskDataset
from src.data.endpoints import (
    get_disease_names,
    get_disease_to_chapter_idx,
    get_unique_chapters,
)
from src.model.backbone import MetaboliteBERTModel
from src.model.heads import HierarchicalMultiTaskHead
from src.model.wrapper import MetaboLMForClassification
from src.training.losses import HierarchicalLoss
from src.training.metrics import compute_multitask_metrics, compute_hierarchy_violation_rate
from src.training.sft_trainer import MultiTaskSFTTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_multitask")


def compute_pos_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute pos_weight = N_neg / N_pos per column.

    For columns with zero positives, uses weight 1.0 to avoid division by zero.
    """
    n_samples = labels.shape[0]
    n_pos = labels.sum(axis=0)  # (num_labels,)
    n_neg = n_samples - n_pos
    weights = np.where(n_pos > 0, n_neg / n_pos, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MetaboLM multi-task hierarchical fine-tuning (E1-E4)"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Experiment: %s | Device: %s", cfg.experiment, device)

    # Seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # Load data
    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{name}" for name in disease_names]
    d2c = get_disease_to_chapter_idx()
    num_chapters = len(get_unique_chapters())

    train_df = pd.read_csv(cfg.data.train_path)
    val_df = pd.read_csv(cfg.data.val_path)
    logger.info("Train: %d, Val: %d", len(train_df), len(val_df))

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    y_train = train_df[label_cols].values
    y_val = val_df[label_cols].values

    # Compute pos_weight from training set
    leaf_pos_weight = compute_pos_weights(y_train).to(device)
    logger.info("Leaf pos_weights: %s", leaf_pos_weight.cpu().numpy().round(1))

    # Compute chapter labels + pos_weight
    train_ds = MultiTaskDataset(X_train, y_train, d2c, num_chapters)
    val_ds = MultiTaskDataset(X_val, y_val, d2c, num_chapters)
    chapter_pos_weight = compute_pos_weights(train_ds.chapter_labels).to(device)
    logger.info("Chapter pos_weights: %s", chapter_pos_weight.cpu().numpy().round(1))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Load correlation matrix
    corr_path = Path(cfg.data.correlation_matrix_path)
    if corr_path.suffix == ".pt" and corr_path.exists():
        bias_matrix = torch.load(str(corr_path), map_location=device)
    else:
        csv_path = Path("_reference/Pre-training/metabolomic_correlation_matrix.csv")
        corr_df = pd.read_csv(str(csv_path), index_col=0, encoding="utf-8-sig")
        bias_matrix = torch.tensor(corr_df.values, dtype=torch.float32, device=device)
        bias_matrix = torch.nn.functional.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)
    logger.info("Correlation matrix: %s", bias_matrix.shape)

    # Build model
    backbone = MetaboliteBERTModel.load_pretrained(
        cfg.model.pretrained_ckpt, num_metabolites=cfg.data.num_metabolites
    )
    backbone.register_buffer("bias_matrix_full", bias_matrix)

    head = HierarchicalMultiTaskHead(
        hidden_size=cfg.model.hidden_size,
        proj_size=cfg.model.proj_size,
        num_diseases=cfg.model.num_diseases,
        num_chapters=num_chapters,
    )

    model = MetaboLMForClassification(
        metabolite_model=backbone,
        head=head,
        freeze_strategy=cfg.model.freeze_strategy,
        adapter_bottleneck=cfg.model.adapter_bottleneck,
        lora_rank=cfg.model.lora_rank,
    )
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %s | Trainable: %d / %d (%.2f%%)",
        cfg.model.freeze_strategy, trainable, total, 100 * trainable / total,
    )

    # Loss
    loss_fn = HierarchicalLoss(
        disease_to_chapter_idx=d2c,
        leaf_pos_weight=leaf_pos_weight,
        chapter_pos_weight=chapter_pos_weight,
        lambda_chapter=cfg.training.lambda_chapter,
        mu_hierarchy=cfg.training.mu_hierarchy,
    )

    # Trainer
    trainer_config = TrainerConfig(
        learning_rate=cfg.training.learning_rate,
        num_epochs=cfg.training.num_epochs,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
    )
    trainer = MultiTaskSFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=trainer_config,
        device=device,
        disease_names=disease_names,
    )

    # Train
    best_auc, best_epoch, best_state, val_labels, val_probs = trainer.train()

    # Save results
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if best_state is not None:
        model_path = os.path.join(output_dir, "best_model.pt")
        torch.save(best_state, model_path)
        logger.info("Best model saved to %s", model_path)

    # Save per-disease AUC table
    metrics = compute_multitask_metrics(val_labels, val_probs, disease_names)
    metrics_rows = []
    for name in disease_names:
        metrics_rows.append({
            "Disease": name,
            "Val_AUC": metrics.get(f"auc_{name}", 0),
            "Val_F1": metrics.get(f"f1_{name}", 0),
        })
    metrics_rows.append({
        "Disease": "MEAN",
        "Val_AUC": metrics["mean_auc"],
        "Val_F1": metrics["mean_f1"],
    })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(output_dir, "multitask_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Save summary
    summary = {
        "experiment": cfg.experiment,
        "freeze_strategy": cfg.model.freeze_strategy,
        "best_epoch": best_epoch,
        "best_mean_auc": best_auc,
        "trainable_params": trainable,
        "total_params": total,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS: %s (freeze=%s)", cfg.experiment, cfg.model.freeze_strategy)
    logger.info("=" * 60)
    for _, row in metrics_df.iterrows():
        logger.info("  %-20s  AUC = %.4f", row["Disease"], row["Val_AUC"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
