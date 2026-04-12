#!/usr/bin/env python3
"""Quick smoke test: verify E1-E4 SFT and E5-E6 GRPO run end-to-end.

Uses only 256 samples, 2 epochs. Takes ~1-2 minutes on GPU.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.biomarkers import get_metabolite_names
from src.data.dataset import MultiTaskDataset
from src.data.endpoints import (
    get_disease_names, get_disease_to_chapter_idx, get_unique_chapters,
)
from src.model.backbone import MetaboliteBERTModel
from src.model.heads import HierarchicalMultiTaskHead
from src.model.wrapper import MetaboLMForClassification
from src.training.losses import HierarchicalLoss
from src.training.sft_trainer import MultiTaskSFTTrainer, TrainerConfig
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig
from src.training.rewards import calibration_reward, hierarchy_reward


RUNTIME_ROOT = Path(os.environ.get("METABOLM_RUNTIME_ROOT", "."))


def build_model(device, freeze_strategy="none"):
    corr_path = RUNTIME_ROOT / "data/processed/correlation_matrix.pt"
    bias_matrix = torch.load(str(corr_path), map_location=device, weights_only=False)

    backbone = MetaboliteBERTModel.load_pretrained(
        str(RUNTIME_ROOT / "weights/best_metabolite_bert_model.pt"), num_metabolites=168
    )
    backbone.register_buffer("bias_matrix_full", bias_matrix)

    head = HierarchicalMultiTaskHead(768, 256, 16, len(get_unique_chapters()))
    model = MetaboLMForClassification(
        backbone, head, freeze_strategy=freeze_strategy
    )
    model.to(device)
    return model


def load_small_data(n=256):
    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{name}" for name in disease_names]
    d2c = get_disease_to_chapter_idx()
    num_chapters = len(get_unique_chapters())

    df = pd.read_csv(RUNTIME_ROOT / "data/processed/val.csv", nrows=n)
    X = df[feature_cols].values
    y = df[label_cols].values
    ds = MultiTaskDataset(X, y, d2c, num_chapters)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    return loader, disease_names, d2c


def compute_pos_weights(loader, device):
    all_leaf = []
    all_chap = []
    for _, leaf, chap in loader:
        all_leaf.append(leaf)
        all_chap.append(chap)
    leaf = torch.cat(all_leaf).numpy()
    chap = torch.cat(all_chap).numpy()

    def pw(labels):
        n = labels.shape[0]
        n_pos = labels.sum(axis=0)
        n_neg = n - n_pos
        return torch.tensor(np.where(n_pos > 0, n_neg / n_pos, 1.0), dtype=torch.float32).to(device)

    return pw(leaf), pw(chap)


def test_sft(strategy_name, freeze_strategy, device, loader, disease_names, d2c):
    print(f"\n{'='*60}")
    print(f"SFT Smoke Test: {strategy_name} (freeze={freeze_strategy})")
    print(f"{'='*60}")

    model = build_model(device, freeze_strategy)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    leaf_pw, chap_pw = compute_pos_weights(loader, device)
    loss_fn = HierarchicalLoss(d2c, leaf_pw, chap_pw, lambda_chapter=1.0, mu_hierarchy=0.1)
    config = TrainerConfig(learning_rate=2e-5, num_epochs=2, warmup_ratio=0.1)

    trainer = MultiTaskSFTTrainer(
        model, loader, loader, loss_fn, config, device, disease_names
    )
    best_auc, best_epoch, _, _, _ = trainer.train()
    print(f"  Result: Mean AUC = {best_auc:.4f} (epoch {best_epoch})")
    return best_auc


def test_grpo(reward_name, reward_fn, device, loader, disease_names, d2c):
    print(f"\n{'='*60}")
    print(f"GRPO Smoke Test: {reward_name} reward")
    print(f"{'='*60}")

    model = build_model(device, "none")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    config = GRPOConfig(
        learning_rate=1e-5, num_epochs=2, group_size=2, kl_coeff=0.1
    )

    trainer = GRPOTrainer(
        model, ref_model, loader, loader,
        reward_fn, config, device, disease_names, d2c,
    )
    best_auc, best_epoch, _, _, _ = trainer.train()
    print(f"  Result: Mean AUC = {best_auc:.4f} (epoch {best_epoch})")
    return best_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader, disease_names, d2c = load_small_data(256)
    print(f"Data: 256 samples, batch_size=64")

    results = {}

    # SFT tests
    for name, strategy in [
        ("E1 Full FT", "none"),
        ("E2 Head Only", "head_only"),
        ("E3 Adapter", "adapter"),
        ("E4 LoRA", "lora"),
    ]:
        results[name] = test_sft(name, strategy, device, loader, disease_names, d2c)
        torch.cuda.empty_cache()

    # GRPO tests
    leaf_to_chapter = torch.tensor(
        [d2c[i] for i in range(16)], dtype=torch.long, device=device
    )

    results["E5 GRPO Cal"] = test_grpo(
        "calibration", calibration_reward, device, loader, disease_names, d2c
    )
    torch.cuda.empty_cache()

    def hier_fn(ll, ly, cl, cy):
        return hierarchy_reward(ll, ly, cl, cy, leaf_to_chapter=leaf_to_chapter)

    results["E6 GRPO Hier"] = test_grpo(
        "hierarchy", hier_fn, device, loader, disease_names, d2c
    )

    # Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for name, auc in results.items():
        print(f"  {name:20s}  Mean AUC = {auc:.4f}")
    print(f"{'='*60}")
    print("All smoke tests PASSED" if all(v > 0 for v in results.values()) else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
