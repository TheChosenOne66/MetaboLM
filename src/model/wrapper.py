"""Model wrapper: backbone + classification head + freeze strategies.

Supports four freeze strategies for parameter-efficient fine-tuning:
  - none: full fine-tune, all parameters trainable
  - head_only: freeze entire backbone, train only the classification head
  - adapter: freeze backbone, inject AdapterLayer per Transformer layer
  - lora: freeze backbone, inject LoRA on Q/V attention projections

For E0 (single-task), this matches the official forward path exactly.
For E1-E4 (multi-task), the head returns (leaf_logits, chapter_logits).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .adapters import AdapterLayer, LoRALinear
from .backbone import MetaboliteBERTModel


class MetaboLMForClassification(nn.Module):
    """Backbone + classification head for disease prediction.

    Args:
        metabolite_model: Pretrained ``MetaboliteBERTModel`` backbone.
        head: Classification head (``SingleTaskHead`` or ``HierarchicalMultiTaskHead``).
        freeze_strategy: One of ``"none"``, ``"head_only"``, ``"adapter"``, ``"lora"``.
        adapter_bottleneck: Bottleneck dim for adapter (default 64).
        lora_rank: Rank for LoRA (default 8).
    """

    def __init__(
        self,
        metabolite_model: MetaboliteBERTModel,
        head: nn.Module,
        freeze_strategy: str = "none",
        adapter_bottleneck: int = 64,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        self.metabolite_model = metabolite_model
        self.head = head
        self.freeze_strategy = freeze_strategy
        self._apply_freeze_strategy(adapter_bottleneck, lora_rank)

    def _apply_freeze_strategy(
        self, adapter_bottleneck: int, lora_rank: int
    ) -> None:
        """Freeze backbone parameters and optionally inject adapters or LoRA."""
        if self.freeze_strategy == "none":
            return  # All params trainable

        # All other strategies freeze the backbone
        for param in self.metabolite_model.parameters():
            param.requires_grad = False

        if self.freeze_strategy == "head_only":
            return  # Only head is trainable

        elif self.freeze_strategy == "adapter":
            hidden_size = self.metabolite_model.hidden_size
            for layer in self.metabolite_model.bert.encoder.layer:
                layer.adapter = AdapterLayer(hidden_size, adapter_bottleneck)

        elif self.freeze_strategy == "lora":
            for layer in self.metabolite_model.bert.encoder.layer:
                attn = layer.attention.self
                attn.query = LoRALinear(attn.query, lora_rank)
                attn.value = LoRALinear(attn.value, lora_rank)

        else:
            raise ValueError(f"Unknown freeze_strategy: {self.freeze_strategy}")

    def forward(
        self,
        expressions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Optional[Tuple[torch.Tensor, ...]],
    ]:
        """Forward pass.

        Args:
            expressions: ``(B, num_metabolites)`` expression values.
            attention_mask: ``(B, num_metabolites)`` attention mask.

        Returns:
            Tuple of ``(head_output, attentions)``.
            - head_output: ``(B,)`` for SingleTaskHead,
              ``(leaf_logits, chapter_logits)`` for HierarchicalMultiTaskHead.
            - attentions: tuple of per-layer attention weights, or None.
        """
        outputs = self.metabolite_model._embed_and_encode(
            expressions,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        pooled_output = outputs["pooler_output"]
        head_output = self.head(pooled_output)
        return head_output, outputs["attentions"]
