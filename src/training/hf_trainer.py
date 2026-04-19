"""HuggingFace Trainer subclass for MetaboLM hierarchical multi-task SFT (E1-E4).

Only :class:`MetaboLMSFTTrainer` is included; the single-task and GRPO
trainers from the multi-GPU prototype are intentionally omitted — EXP-000
touches neither E0 (per-disease) nor E5/E6 (GRPO) paths.

Strict-alignment note: this trainer does not set ``bf16`` / ``fp16`` /
``deepspeed`` itself; those are controlled by :class:`transformers.TrainingArguments`
in the caller. For EXP-000 we pass ``bf16=False, fp16=False, deepspeed=None``
to keep FP32 + plain DDP semantics equivalent to the single-GPU baseline.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import Trainer

from .losses import HierarchicalLoss
from .metrics import compute_multitask_metrics

logger = logging.getLogger(__name__)


class MetaboLMSFTTrainer(Trainer):
    """HF Trainer for hierarchical multi-task SFT (E1-E4).

    Overrides ``compute_loss`` to call :class:`HierarchicalLoss` on the
    ``(leaf_logits, chapter_logits)`` model output, and ``prediction_step``
    to emit leaf probabilities for AUROC computation.

    Args:
        loss_fn: ``HierarchicalLoss`` instance (constructed on CPU; moved
            to the correct device on first forward).
        disease_names: List of 16 disease names for metric reporting.
    """

    def __init__(
        self,
        *args: Any,
        loss_fn: HierarchicalLoss | None = None,
        disease_names: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if loss_fn is None:
            raise ValueError("MetaboLMSFTTrainer requires loss_fn=HierarchicalLoss(...)")
        self.loss_fn = loss_fn
        self.disease_names = disease_names or []

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        expressions = inputs["expressions"]
        leaf_labels = inputs["leaf_labels"]
        chapter_labels = inputs["chapter_labels"]

        attention_mask = torch.ones(
            expressions.size(), dtype=torch.long, device=expressions.device
        )
        (leaf_logits, chapter_logits), _ = model(expressions, attention_mask)

        # Lazy-move pos_weight buffers to the right device. Once they are
        # on-device the second call is a no-op.
        self.loss_fn.to(leaf_logits.device)

        loss, _components = self.loss_fn(
            leaf_logits, chapter_logits, leaf_labels, chapter_labels
        )

        if return_outputs:
            return loss, {
                "leaf_logits": leaf_logits,
                "chapter_logits": chapter_logits,
                "leaf_labels": leaf_labels,
                "chapter_labels": chapter_labels,
            }
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        with torch.no_grad():
            expressions = inputs["expressions"].to(self.args.device)
            leaf_labels = inputs["leaf_labels"].to(self.args.device)
            chapter_labels = inputs["chapter_labels"].to(self.args.device)

            attention_mask = torch.ones(
                expressions.size(), dtype=torch.long, device=expressions.device
            )
            (leaf_logits, chapter_logits), _ = model(expressions, attention_mask)

            self.loss_fn.to(leaf_logits.device)
            loss, _ = self.loss_fn(
                leaf_logits, chapter_logits, leaf_labels, chapter_labels
            )

        if prediction_loss_only:
            return (loss, None, None)

        leaf_probs = torch.sigmoid(leaf_logits)
        return (loss, leaf_probs, leaf_labels)

    def compute_metrics_fn(self, eval_pred):
        """Pass this bound method to ``TrainingArguments.compute_metrics`` so HF
        Trainer calls it with ``(predictions, label_ids)`` as numpy arrays."""
        probs, labels = eval_pred
        metrics = compute_multitask_metrics(labels, probs, self.disease_names)
        return {"mean_auc": metrics["mean_auc"], "mean_f1": metrics["mean_f1"]}
