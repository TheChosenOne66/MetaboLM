"""SFT training loop for MetaboLM fine-tuning.

Matches the official MetaboLM_Fine-tuning.py training loop:
  - AdamW optimizer
  - Cosine schedule with 10% linear warmup
  - BCEWithLogitsLoss
  - Best model selection by validation AUROC
  - Per-epoch: full train pass, full val pass, log metrics

Official hyperparameters (from recovered Fine-tuning code):
  lr = 2e-5, epochs = 40, batch_size = 512, warmup_ratio = 0.1
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from .metrics import compute_binary_metrics
from .losses import HierarchicalLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Training hyperparameters matching the official fine-tuning code."""

    learning_rate: float = 2e-5
    num_epochs: int = 40
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01  # AdamW default


class SFTTrainer:
    """Per-disease fine-tuning trainer matching the official code.

    Official training loop structure (per epoch)::

        # Train
        model.train()
        for batch in train_loader:
            logits, _ = model(expressions, attention_mask)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step(); scheduler.step()

        # Validate
        model.eval()
        for batch in val_loader:
            logits, _ = model(expressions, attention_mask)
            loss = criterion(logits, labels)
            probs = sigmoid(logits)
        val_auc = roc_auc_score(all_labels, all_probs)

        # Save best by val AUC
        if val_auc > best_val_auc:
            best_model_state = model.state_dict()

    Args:
        model: ``MetaboLMForClassification`` instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training hyperparameters.
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss: matches official nn.BCEWithLogitsLoss()
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer: matches official AdamW(model.parameters(), lr=learning_rate)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler: matches official get_cosine_schedule_with_warmup
        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(config.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

    def train_epoch(self) -> tuple[float, dict[str, float]]:
        """Run one training epoch. Returns (avg_loss, metrics_dict).

        Matches official loop: zero_grad -> forward -> loss -> backward ->
        step optimizer -> step scheduler.
        """
        self.model.train()
        epoch_loss = 0.0
        all_labels: list[float] = []
        all_probs: list[float] = []

        progress = tqdm(self.train_loader, desc="Train", leave=False)
        for expressions, labels, _eids in progress:
            expressions = expressions.to(self.device)
            labels = labels.to(self.device)

            # Official: attention_mask = torch.ones(expressions.size(), ...)
            attention_mask = torch.ones(
                expressions.size(), dtype=torch.long, device=self.device
            )

            self.optimizer.zero_grad()
            logits, _ = self.model(expressions, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item() * expressions.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(self.train_loader.dataset)
        metrics = compute_binary_metrics(all_labels, all_probs)
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self) -> tuple[float, dict[str, float], list[float], list[float]]:
        """Run validation. Returns (avg_loss, metrics_dict, all_labels, all_probs).

        Matches official validation loop exactly.
        """
        self.model.eval()
        epoch_loss = 0.0
        all_labels: list[float] = []
        all_probs: list[float] = []

        for expressions, labels, _eids in self.val_loader:
            expressions = expressions.to(self.device)
            labels = labels.to(self.device)
            attention_mask = torch.ones(
                expressions.size(), dtype=torch.long, device=self.device
            )

            logits, _ = self.model(expressions, attention_mask)
            loss = self.criterion(logits, labels)

            epoch_loss += loss.item() * expressions.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        avg_loss = epoch_loss / len(self.val_loader.dataset)
        metrics = compute_binary_metrics(all_labels, all_probs)
        return avg_loss, metrics, all_labels, all_probs

    def train(
        self, disease_name: str = ""
    ) -> tuple[float, int, Optional[dict], list[float], list[float]]:
        """Full training loop with best-model tracking by validation AUC.

        Matches official logic::

            best_val_auc = 0.0
            for epoch in range(num_epochs):
                train_one_epoch()
                val_metrics = evaluate()
                if val_auc > best_val_auc:
                    best_model_state = model.state_dict()

        Args:
            disease_name: For logging only.

        Returns:
            Tuple of (best_val_auc, best_epoch, best_model_state_dict,
            final_val_labels, final_val_probs).
        """
        tag = f"[{disease_name}] " if disease_name else ""
        best_val_auc = 0.0
        best_epoch = -1
        best_model_state: Optional[dict] = None
        final_val_labels: list[float] = []
        final_val_probs: list[float] = []

        for epoch in range(self.config.num_epochs):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics, val_labels, val_probs = self.evaluate()

            logger.info(
                "%sEpoch %d/%d: "
                "Train Loss: %.4f, Train Acc: %.4f, Train F1: %.4f | "
                "Val Loss: %.4f, Val Acc: %.4f, Val F1: %.4f, Val AUC: %.4f",
                tag,
                epoch + 1,
                self.config.num_epochs,
                train_loss,
                train_metrics["accuracy"],
                train_metrics["f1"],
                val_loss,
                val_metrics["accuracy"],
                val_metrics["f1"],
                val_metrics["auc"],
            )

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(self.model.state_dict())
                final_val_labels = val_labels
                final_val_probs = val_probs
                logger.info(
                    "%sNew best AUC: %.4f at epoch %d",
                    tag,
                    best_val_auc,
                    best_epoch,
                )

        logger.info(
            "%sBest Val AUC: %.4f at epoch %d",
            tag,
            best_val_auc,
            best_epoch,
        )
        return best_val_auc, best_epoch, best_model_state, final_val_labels, final_val_probs


class MultiTaskSFTTrainer:
    """Multi-task hierarchical fine-tuning trainer.

    Unlike ``SFTTrainer`` which trains per-disease models, this trains a
    single model on all 16 diseases + 6 chapters simultaneously using
    ``HierarchicalLoss`` and tracks mean AUROC for model selection.

    Args:
        model: ``MetaboLMForClassification`` with ``HierarchicalMultiTaskHead``.
        train_loader: DataLoader yielding ``(expressions, leaf_labels, chapter_labels)``.
        val_loader: Validation DataLoader.
        loss_fn: ``HierarchicalLoss`` instance.
        config: Training hyperparameters.
        device: Torch device.
        disease_names: List of 16 disease names for per-disease metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: HierarchicalLoss,
        config: TrainerConfig,
        device: torch.device,
        disease_names: list[str],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.disease_names = disease_names

        # Only optimise trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(config.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

    def train_epoch(self) -> tuple[float, dict[str, float]]:
        """One training epoch. Returns (avg_loss, metrics)."""
        self.model.train()
        epoch_loss = 0.0
        all_leaf_labels = []
        all_leaf_probs = []
        n_samples = 0

        progress = tqdm(self.train_loader, desc="Train", leave=False)
        for expressions, leaf_labels, chapter_labels in progress:
            expressions = expressions.to(self.device)
            leaf_labels = leaf_labels.to(self.device)
            chapter_labels = chapter_labels.to(self.device)
            attention_mask = torch.ones(
                expressions.size(), dtype=torch.long, device=self.device
            )

            self.optimizer.zero_grad()
            (leaf_logits, chapter_logits), _ = self.model(expressions, attention_mask)
            loss, components = self.loss_fn(
                leaf_logits, chapter_logits, leaf_labels, chapter_labels
            )
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            bs = expressions.size(0)
            epoch_loss += loss.item() * bs
            n_samples += bs
            all_leaf_probs.append(torch.sigmoid(leaf_logits).detach().cpu().numpy())
            all_leaf_labels.append(leaf_labels.detach().cpu().numpy())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / n_samples
        all_leaf_labels_np = np.concatenate(all_leaf_labels, axis=0)
        all_leaf_probs_np = np.concatenate(all_leaf_probs, axis=0)

        from .metrics import compute_multitask_metrics
        metrics = compute_multitask_metrics(
            all_leaf_labels_np, all_leaf_probs_np, self.disease_names
        )
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validation pass. Returns (avg_loss, metrics, leaf_labels, leaf_probs, chap_labels, chap_probs)."""
        self.model.eval()
        epoch_loss = 0.0
        n_samples = 0
        all_leaf_labels = []
        all_leaf_probs = []
        all_chap_labels = []
        all_chap_probs = []

        for expressions, leaf_labels, chapter_labels in self.val_loader:
            expressions = expressions.to(self.device)
            leaf_labels = leaf_labels.to(self.device)
            chapter_labels = chapter_labels.to(self.device)
            attention_mask = torch.ones(
                expressions.size(), dtype=torch.long, device=self.device
            )

            (leaf_logits, chapter_logits), _ = self.model(expressions, attention_mask)
            loss, _ = self.loss_fn(
                leaf_logits, chapter_logits, leaf_labels, chapter_labels
            )

            bs = expressions.size(0)
            epoch_loss += loss.item() * bs
            n_samples += bs
            all_leaf_probs.append(torch.sigmoid(leaf_logits).cpu().numpy())
            all_leaf_labels.append(leaf_labels.cpu().numpy())
            all_chap_probs.append(torch.sigmoid(chapter_logits).cpu().numpy())
            all_chap_labels.append(chapter_labels.cpu().numpy())

        avg_loss = epoch_loss / n_samples
        leaf_labels_np = np.concatenate(all_leaf_labels, axis=0)
        leaf_probs_np = np.concatenate(all_leaf_probs, axis=0)
        chap_labels_np = np.concatenate(all_chap_labels, axis=0)
        chap_probs_np = np.concatenate(all_chap_probs, axis=0)

        from .metrics import compute_multitask_metrics, compute_hierarchy_violation_rate
        from ..data.endpoints import get_disease_to_chapter_idx
        metrics = compute_multitask_metrics(
            leaf_labels_np, leaf_probs_np, self.disease_names
        )
        metrics["hierarchy_violation_rate"] = compute_hierarchy_violation_rate(
            leaf_probs_np, chap_probs_np, get_disease_to_chapter_idx()
        )
        return avg_loss, metrics, leaf_labels_np, leaf_probs_np, chap_labels_np, chap_probs_np

    def train(self) -> tuple[float, int, dict | None, np.ndarray, np.ndarray]:
        """Full training loop. Model selection by mean AUROC.

        Returns:
            (best_mean_auc, best_epoch, best_model_state, val_leaf_labels, val_leaf_probs)
        """
        best_mean_auc = 0.0
        best_epoch = -1
        best_model_state = None
        best_leaf_labels = np.array([])
        best_leaf_probs = np.array([])

        for epoch in range(self.config.num_epochs):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics, vl_labels, vl_probs, _, _ = self.evaluate()

            logger.info(
                "Epoch %d/%d: Train Loss: %.4f, Train Mean AUC: %.4f | "
                "Val Loss: %.4f, Val Mean AUC: %.4f, Hierarchy Violation: %.4f",
                epoch + 1, self.config.num_epochs,
                train_loss, train_metrics.get("mean_auc", 0),
                val_loss, val_metrics.get("mean_auc", 0),
                val_metrics.get("hierarchy_violation_rate", 0),
            )

            current_auc = val_metrics.get("mean_auc", 0)
            if current_auc > best_mean_auc:
                best_mean_auc = current_auc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_leaf_labels = vl_labels
                best_leaf_probs = vl_probs
                logger.info(
                    "New best Mean AUC: %.4f at epoch %d", best_mean_auc, best_epoch
                )

        logger.info("Best Mean AUC: %.4f at epoch %d", best_mean_auc, best_epoch)
        return best_mean_auc, best_epoch, best_model_state, best_leaf_labels, best_leaf_probs
