"""Group Relative Policy Optimisation (GRPO) trainer for MetaboLM.

GRPO optimises a reward signal via policy gradient without a value network.
Stochasticity comes from MC dropout: K forward passes per input with
different dropout masks yield a *group* of predictions.  Advantages are
computed by normalising rewards within each group, providing low-variance
gradient estimates.

Algorithm (per batch)::

    1.  model.train()            # dropout ON
    2.  for k in 1..K:
            logits_k = model(x)  # different dropout mask each time
            r_k = reward(logits_k, y)
    3.  advantages = (r - mean(r)) / (std(r) + eps)   per-sample across K
    4.  L_pg = - mean_k( advantage_k * log_prob_k )
    5.  L_kl = 0.5 * mean( (logit_current - logit_ref)^2 )
    6.  L = L_pg + beta * L_kl
    7.  backward + step

Reference policy (π_ref) is the frozen SFT checkpoint.  KL divergence is
approximated with a Gaussian assumption on logit space.
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from .metrics import compute_multitask_metrics, compute_hierarchy_violation_rate

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO trainer for multi-task disease prediction.

    Args:
        model: Current policy (initialised from SFT checkpoint).
        ref_model: Frozen reference policy (SFT checkpoint, eval mode).
        train_loader: Training DataLoader yielding
            ``(expressions, leaf_labels, chapter_labels)``.
        val_loader: Validation DataLoader.
        reward_fn: Callable ``(leaf_logits, leaf_labels, chapter_logits,
            chapter_labels) -> (B,)`` per-sample reward.
        config: :class:`GRPOConfig` hyperparameters.
        device: Torch device.
        disease_names: List of 16 disease names.
        disease_to_chapter_idx: Mapping for hierarchy violation metric.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        reward_fn: Callable,
        config: "GRPOConfig",
        device: torch.device,
        disease_names: list[str],
        disease_to_chapter_idx: dict[int, int],
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.reward_fn = reward_fn
        self.config = config
        self.device = device
        self.disease_names = disease_names
        self.disease_to_chapter_idx = disease_to_chapter_idx

        # Freeze reference model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optimiser: only trainable params of current policy
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(config.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

    def _compute_kl(
        self,
        current_leaf: torch.Tensor,
        current_chap: torch.Tensor,
        ref_leaf: torch.Tensor,
        ref_chap: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate KL divergence in logit space (Gaussian assumption).

        KL ≈ 0.5 * mean( (z_current - z_ref)^2 )

        This penalises the policy for drifting too far from the SFT
        reference in logit space, regardless of label.
        """
        kl_leaf = 0.5 * (current_leaf - ref_leaf).pow(2).mean()
        kl_chap = 0.5 * (current_chap - ref_chap).pow(2).mean()
        return kl_leaf + kl_chap

    def train_epoch(self) -> tuple[float, float, float]:
        """One GRPO training epoch.

        Returns:
            Tuple of (avg_pg_loss, avg_kl, avg_reward).
        """
        self.model.train()
        total_pg_loss = 0.0
        total_kl = 0.0
        total_reward = 0.0
        n_batches = 0

        progress = tqdm(self.train_loader, desc="GRPO Train", leave=False)
        for expressions, leaf_labels, chapter_labels in progress:
            expressions = expressions.to(self.device)
            leaf_labels = leaf_labels.to(self.device)
            chapter_labels = chapter_labels.to(self.device)
            attention_mask = torch.ones(
                expressions.size(), dtype=torch.long, device=self.device
            )

            K = self.config.group_size
            group_leaf_logits = []
            group_chap_logits = []
            group_rewards = []

            # --- Sample K forward passes with different dropout masks ---
            for _ in range(K):
                (leaf_logits, chap_logits), _ = self.model(
                    expressions, attention_mask
                )
                group_leaf_logits.append(leaf_logits)
                group_chap_logits.append(chap_logits)

                with torch.no_grad():
                    r = self.reward_fn(
                        leaf_logits, leaf_labels, chap_logits, chapter_labels
                    )  # (B,)
                group_rewards.append(r)

            # Stack: (K, B)
            rewards = torch.stack(group_rewards, dim=0)  # (K, B)

            # --- Group-normalised advantages ---
            r_mean = rewards.mean(dim=0, keepdim=True)   # (1, B)
            r_std = rewards.std(dim=0, keepdim=True) + 1e-8
            advantages = (rewards - r_mean) / r_std       # (K, B)

            # --- Policy gradient loss ---
            pg_loss = torch.tensor(0.0, device=self.device)
            for k in range(K):
                # Log-likelihood of ground-truth labels under current policy
                log_prob = -F.binary_cross_entropy_with_logits(
                    group_leaf_logits[k], leaf_labels, reduction="none"
                ).sum(dim=-1)  # (B,)
                pg_loss = pg_loss - (advantages[k].detach() * log_prob).mean()
            pg_loss = pg_loss / K

            # --- KL penalty against reference policy ---
            with torch.no_grad():
                (ref_leaf, ref_chap), _ = self.ref_model(
                    expressions, attention_mask
                )
            # Use the mean of group logits for KL
            mean_leaf = torch.stack(group_leaf_logits).mean(dim=0)
            mean_chap = torch.stack(group_chap_logits).mean(dim=0)
            kl = self._compute_kl(mean_leaf, mean_chap, ref_leaf, ref_chap)

            # --- Total loss ---
            loss = pg_loss + self.config.kl_coeff * kl

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self.optimizer.step()
            self.scheduler.step()

            total_pg_loss += pg_loss.item()
            total_kl += kl.item()
            total_reward += rewards.mean().item()
            n_batches += 1

            progress.set_postfix(
                pg=f"{pg_loss.item():.4f}",
                kl=f"{kl.item():.4f}",
                r=f"{rewards.mean().item():.4f}",
            )

        n = max(n_batches, 1)
        return total_pg_loss / n, total_kl / n, total_reward / n

    @torch.no_grad()
    def evaluate(self) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """Validation pass (deterministic, dropout OFF).

        Returns:
            Tuple of (metrics_dict, leaf_labels, leaf_probs).
        """
        self.model.eval()
        all_leaf_labels = []
        all_leaf_probs = []
        all_chap_probs = []
        all_chap_labels = []
        total_reward = 0.0
        n_samples = 0

        for expressions, leaf_labels, chapter_labels in self.val_loader:
            expressions = expressions.to(self.device)
            leaf_labels = leaf_labels.to(self.device)
            chapter_labels = chapter_labels.to(self.device)
            attention_mask = torch.ones(
                expressions.size(), dtype=torch.long, device=self.device
            )

            (leaf_logits, chap_logits), _ = self.model(
                expressions, attention_mask
            )
            r = self.reward_fn(
                leaf_logits, leaf_labels, chap_logits, chapter_labels
            )
            bs = expressions.size(0)
            total_reward += r.sum().item()
            n_samples += bs

            all_leaf_probs.append(torch.sigmoid(leaf_logits).cpu().numpy())
            all_leaf_labels.append(leaf_labels.cpu().numpy())
            all_chap_probs.append(torch.sigmoid(chap_logits).cpu().numpy())
            all_chap_labels.append(chapter_labels.cpu().numpy())

        leaf_labels_np = np.concatenate(all_leaf_labels)
        leaf_probs_np = np.concatenate(all_leaf_probs)
        chap_probs_np = np.concatenate(all_chap_probs)

        metrics = compute_multitask_metrics(
            leaf_labels_np, leaf_probs_np, self.disease_names
        )
        metrics["hierarchy_violation_rate"] = compute_hierarchy_violation_rate(
            leaf_probs_np, chap_probs_np, self.disease_to_chapter_idx
        )
        metrics["avg_reward"] = total_reward / max(n_samples, 1)

        return metrics, leaf_labels_np, leaf_probs_np

    def train(self) -> tuple[float, int, dict | None, np.ndarray, np.ndarray]:
        """Full GRPO training loop. Model selection by mean AUROC.

        Returns:
            (best_mean_auc, best_epoch, best_state, val_labels, val_probs)
        """
        best_mean_auc = 0.0
        best_epoch = -1
        best_state = None
        best_labels = np.array([])
        best_probs = np.array([])

        for epoch in range(self.config.num_epochs):
            pg_loss, kl, train_reward = self.train_epoch()
            metrics, val_labels, val_probs = self.evaluate()

            logger.info(
                "Epoch %d/%d: PG Loss: %.4f, KL: %.4f, "
                "Train Reward: %.4f | Val Mean AUC: %.4f, "
                "Val Reward: %.4f, Hier Viol: %.4f",
                epoch + 1,
                self.config.num_epochs,
                pg_loss,
                kl,
                train_reward,
                metrics.get("mean_auc", 0),
                metrics.get("avg_reward", 0),
                metrics.get("hierarchy_violation_rate", 0),
            )

            current_auc = metrics.get("mean_auc", 0)
            if current_auc > best_mean_auc:
                best_mean_auc = current_auc
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                best_labels = val_labels
                best_probs = val_probs
                logger.info(
                    "New best Mean AUC: %.4f at epoch %d",
                    best_mean_auc,
                    best_epoch,
                )

        logger.info(
            "Best Mean AUC: %.4f at epoch %d", best_mean_auc, best_epoch
        )
        return best_mean_auc, best_epoch, best_state, best_labels, best_probs


from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO training."""

    learning_rate: float = 1e-5
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    group_size: int = 4
    kl_coeff: float = 0.1
