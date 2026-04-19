#!/usr/bin/env python3
"""Multi-task hierarchical fine-tuning for MetaboLM (E1-E4), HF Trainer + DDP.

Single-node single-GPU::

    python scripts/train_multitask.py --config configs/sft_full_ft.yaml

Single-node multi-GPU (DDP, no DeepSpeed)::

    torchrun --standalone --nproc_per_node=8 scripts/train_multitask.py \
        --config configs/sft_full_ft.yaml

Strict alignment with the single-card E1 baseline is enforced by:
  - ``per_device_train_batch_size = cfg.training.batch_size // world_size``
    so the effective **global** batch stays 512 regardless of num GPUs.
  - ``bf16=False, fp16=False`` — FP32 throughout, matching the original
    sft_trainer run.
  - ``deepspeed=None`` — plain DDP (HF Trainer uses Accelerate + NCCL under
    the hood); ZeRO-2 is unnecessary for 85M params on 97 GB GPUs and
    would introduce a second precision/optimizer-state code path.
  - ``seed=cfg.seed`` uniformly across all ranks so model init is identical
    on every replica; ``DistributedSampler`` handles the per-rank data
    split deterministically.
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
import torch.distributed as dist
from transformers import TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.biomarkers import get_metabolite_names
from src.data.dataset import MultiTaskDictDataset
from src.data.endpoints import (
    get_disease_names,
    get_disease_to_chapter_idx,
    get_unique_chapters,
)
from src.model.backbone import MetaboliteBERTModel
from src.model.heads import HierarchicalMultiTaskHead
from src.model.wrapper import MetaboLMForClassification
from src.training.distributed import is_main_process
from src.training.hf_trainer import MetaboLMSFTTrainer
from src.training.losses import HierarchicalLoss
from src.training.metrics import compute_multitask_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_multitask")


def compute_pos_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute ``pos_weight = N_neg / N_pos`` per column (1.0 for zero-pos cols)."""
    n_samples = labels.shape[0]
    n_pos = labels.sum(axis=0)
    n_neg = n_samples - n_pos
    weights = np.where(n_pos > 0, n_neg / n_pos, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MetaboLM multi-task hierarchical fine-tuning (E1-E4) — HF Trainer + DDP"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args, _remaining = parser.parse_known_args()

    cfg = load_config(args.config)

    # ── Distributed detection ──
    # HF Trainer + torchrun sets RANK/WORLD_SIZE; single-GPU runs leave them unset.
    # We do NOT call init_process_group here — TrainingArguments handles that.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0

    if is_main:
        logger.info(
            "Experiment: %s | World size: %d | Per-device BS: %d (global %d)",
            cfg.experiment,
            world_size,
            cfg.training.batch_size // max(1, world_size),
            cfg.training.batch_size,
        )

    # Seed (same across all ranks so model init is identical)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # ── Load data ──
    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{name}" for name in disease_names]
    d2c = get_disease_to_chapter_idx()
    num_chapters = len(get_unique_chapters())

    train_df = pd.read_csv(cfg.data.train_path)
    val_df = pd.read_csv(cfg.data.val_path)
    if is_main:
        logger.info("Train: %d, Val: %d", len(train_df), len(val_df))

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    y_train = train_df[label_cols].values
    y_val = val_df[label_cols].values

    # Dict datasets for HF Trainer
    train_ds = MultiTaskDictDataset(X_train, y_train, d2c, num_chapters)
    val_ds = MultiTaskDictDataset(X_val, y_val, d2c, num_chapters)

    # pos_weight on CPU; HF Trainer / our loss_fn move to device on first forward
    leaf_pos_weight = compute_pos_weights(y_train)
    chapter_pos_weight = compute_pos_weights(train_ds.chapter_labels)
    if is_main:
        logger.info("Leaf pos_weights: %s", leaf_pos_weight.numpy().round(1))
        logger.info("Chapter pos_weights: %s", chapter_pos_weight.numpy().round(1))

    # ── Load correlation matrix (CPU; model.to(device) moves the buffer) ──
    corr_path = Path(cfg.data.correlation_matrix_path)
    if corr_path.suffix == ".pt" and corr_path.exists():
        bias_matrix = torch.load(str(corr_path), map_location="cpu", weights_only=False)
    else:
        csv_path = Path("_reference/Pre-training/metabolomic_correlation_matrix.csv")
        corr_df = pd.read_csv(str(csv_path), index_col=0, encoding="utf-8-sig")
        bias_matrix = torch.tensor(corr_df.values, dtype=torch.float32)
        bias_matrix = torch.nn.functional.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)
    if is_main:
        logger.info("Correlation matrix: %s", bias_matrix.shape)

    # ── Build model ──
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info(
            "Model: %s | Trainable: %d / %d (%.2f%%)",
            cfg.model.freeze_strategy, trainable, total, 100 * trainable / total,
        )

    # ── Loss (CPU; trainer.compute_loss moves it to device on first call) ──
    loss_fn = HierarchicalLoss(
        disease_to_chapter_idx=d2c,
        leaf_pos_weight=leaf_pos_weight,
        chapter_pos_weight=chapter_pos_weight,
        lambda_chapter=cfg.training.lambda_chapter,
        mu_hierarchy=cfg.training.mu_hierarchy,
    )

    # ── TrainingArguments (strict alignment: global bs=512, FP32, plain DDP) ──
    per_device_bs = cfg.training.batch_size // max(1, world_size)
    if per_device_bs * max(1, world_size) != cfg.training.batch_size:
        raise ValueError(
            f"Global batch_size={cfg.training.batch_size} is not divisible by "
            f"world_size={world_size}. Adjust world_size or config.batch_size."
        )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=False,                                # strict FP32
        fp16=False,                                # strict FP32
        deepspeed=None,                            # plain DDP, no ZeRO
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mean_auc",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,               # keep dict keys
        seed=cfg.seed,                             # same on every rank
        report_to="none",
        ddp_find_unused_parameters=False,          # custom model has no unused params
        disable_tqdm=not is_main,
    )

    # ── Trainer ──
    trainer = MetaboLMSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss_fn=loss_fn,
        disease_names=disease_names,
    )
    # compute_metrics is a bound method, set after construction.
    trainer.compute_metrics = trainer.compute_metrics_fn

    # ── Train ──
    trainer.train()

    # ── Save final artefacts (rank 0 only) ──
    if is_main_process():
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Per-disease predictions (best model already loaded via load_best_model_at_end=True)
        preds = trainer.predict(val_ds)
        leaf_probs = preds.predictions
        leaf_labels = preds.label_ids
        metrics = compute_multitask_metrics(leaf_labels, leaf_probs, disease_names)

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
        metrics_df.to_csv(os.path.join(cfg.output_dir, "multitask_metrics.csv"), index=False)

        # Save unwrapped state_dict so external tooling (eval scripts, leaderboard)
        # can load without needing the Trainer/DDP wrapper.
        unwrapped = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        torch.save(unwrapped.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))

        # Recover best epoch from Trainer log history
        best_epoch = -1
        best_auc = -1.0
        for record in trainer.state.log_history:
            if "eval_mean_auc" in record and record["eval_mean_auc"] > best_auc:
                best_auc = record["eval_mean_auc"]
                best_epoch = int(record.get("epoch", -1))

        summary = {
            "experiment": cfg.experiment,
            "freeze_strategy": cfg.model.freeze_strategy,
            "best_epoch": best_epoch,
            "best_mean_auc": float(metrics["mean_auc"]),
            "trainable_params": int(trainable),
            "total_params": int(total),
            "world_size": world_size,
            "per_device_batch_size": per_device_bs,
            "global_batch_size": cfg.training.batch_size,
        }
        with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 60)
        logger.info(
            "RESULTS: %s (freeze=%s, world_size=%d)",
            cfg.experiment, cfg.model.freeze_strategy, world_size,
        )
        logger.info("=" * 60)
        for _, row in metrics_df.iterrows():
            logger.info("  %-20s  AUC = %.4f", row["Disease"], row["Val_AUC"])
        logger.info("=" * 60)

    # Best-effort cleanup (HF Trainer usually destroys the pg on exit)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
