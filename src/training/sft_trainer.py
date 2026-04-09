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
