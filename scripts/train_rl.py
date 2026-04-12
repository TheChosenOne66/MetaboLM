#!/usr/bin/env python3
"""GRPO reinforcement learning for MetaboLM (E5-E6).

Starts from the best SFT checkpoint (E1-E4) and optimises a reward signal
via Group Relative Policy Optimisation.  A frozen copy of the SFT model
serves as the reference policy for KL regularisation.

Usage::

    python scripts/train_rl.py --config configs/rl_calibration.yaml   # E5
    python scripts/train_rl.py --config configs/rl_hierarchy.yaml     # E6
"""

from __future__ import annotations

import argparse
import copy
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
from src.training.grpo_trainer import GRPOConfig, GRPOTrainer
from src.training.metrics import (
    compute_multitask_metrics,
    compute_hierarchy_violation_rate,
)
from src.training.rewards import calibration_reward, hierarchy_reward, combined_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_rl")


def _build_model(cfg, device, bias_matrix):
    """Build a MetaboLMForClassification with hierarchical head."""
    num_chapters = len(get_unique_chapters())
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
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MetaboLM GRPO reinforcement learning (E5-E6)"
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

    train_ds = MultiTaskDataset(X_train, y_train, d2c, num_chapters)
    val_ds = MultiTaskDataset(X_val, y_val, d2c, num_chapters)

    # Smaller batch size for RL (K forward passes per batch)
    rl_batch_size = cfg.training.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=rl_batch_size,
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

    # --- Build current policy (load from SFT checkpoint) ---
    model = _build_model(cfg, device, bias_matrix)
    sft_ckpt = cfg.model.sft_ckpt
    if sft_ckpt and Path(sft_ckpt).exists():
        logger.info("Loading SFT checkpoint: %s", sft_ckpt)
        state = torch.load(sft_ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        logger.warning(
            "No SFT checkpoint at '%s', starting from pretrained backbone",
            sft_ckpt,
        )
    model.to(device)

    # --- Build reference policy (frozen copy) ---
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Policy: freeze=%s | Trainable: %d / %d (%.2f%%)",
        cfg.model.freeze_strategy, trainable, total, 100 * trainable / total,
    )

    # --- Reward function ---
    leaf_to_chapter_tensor = torch.tensor(
        [d2c[i] for i in range(len(disease_names))],
        dtype=torch.long,
        device=device,
    )

    reward_type = cfg.training.reward
    if reward_type == "calibration":
        reward_fn = calibration_reward
    elif reward_type == "hierarchy":
        def reward_fn(ll, ly, cl, cy):
            return hierarchy_reward(
                ll, ly, cl, cy, leaf_to_chapter=leaf_to_chapter_tensor
            )
    elif reward_type == "combined":
        def reward_fn(ll, ly, cl, cy):
            return combined_reward(
                ll, ly, cl, cy,
                leaf_to_chapter=leaf_to_chapter_tensor,
                alpha=0.5,
            )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
    logger.info("Reward: %s", reward_type)

    # --- GRPO config ---
    grpo_config = GRPOConfig(
        learning_rate=cfg.training.learning_rate,
        num_epochs=cfg.training.num_epochs,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        group_size=cfg.training.group_size,
        kl_coeff=cfg.training.kl_coeff,
    )

    # --- Train ---
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        train_loader=train_loader,
        val_loader=val_loader,
        reward_fn=reward_fn,
        config=grpo_config,
        device=device,
        disease_names=disease_names,
        disease_to_chapter_idx=d2c,
    )

    best_auc, best_epoch, best_state, val_labels, val_probs = trainer.train()

    # --- Save results ---
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if best_state is not None:
        model_path = os.path.join(output_dir, "best_model.pt")
        torch.save(best_state, model_path)
        logger.info("Best model saved to %s", model_path)

    # Per-disease metrics
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
    metrics_df.to_csv(os.path.join(output_dir, "multitask_metrics.csv"), index=False)

    # Summary
    summary = {
        "experiment": cfg.experiment,
        "reward": reward_type,
        "freeze_strategy": cfg.model.freeze_strategy,
        "group_size": grpo_config.group_size,
        "kl_coeff": grpo_config.kl_coeff,
        "best_epoch": best_epoch,
        "best_mean_auc": best_auc,
        "trainable_params": trainable,
        "total_params": total,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESULTS: %s (reward=%s)", cfg.experiment, reward_type)
    logger.info("=" * 60)
    for _, row in metrics_df.iterrows():
        logger.info("  %-20s  AUC = %.4f", row["Disease"], row["Val_AUC"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
