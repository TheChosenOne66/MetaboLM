"""Model wrapper: backbone + classification head.

Matches the official ``MetaboliteBERTForClassification`` forward path:
  expressions -> Hadamard embed -> prepend [CLS] -> BERT encoder
  -> AdaptiveAvgPool1d (pooler_output) -> classifier head -> logits

Key equivalence notes (vs official MetaboLM_Fine-tuning.py):
- Uses ``pooler_output`` (avg pool over ALL tokens), NOT ``[CLS]`` hidden state.
- The official code manually re-implements embedding in the wrapper forward;
  we delegate to ``backbone._embed_and_encode()`` which is numerically identical.
- State dict key prefix: ``self.metabolite_model.*`` for backbone params,
  matching the official ``module.metabolite_model.*`` (minus DataParallel prefix).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .backbone import MetaboliteBERTModel


class MetaboLMForClassification(nn.Module):
    """Backbone + classification head for disease prediction.

    Args:
        metabolite_model: Pretrained ``MetaboliteBERTModel`` backbone.
        head: Classification head (``SingleTaskHead`` or future variants).
    """

    def __init__(
        self,
        metabolite_model: MetaboliteBERTModel,
        head: nn.Module,
    ) -> None:
        super().__init__()
        # Name matches official checkpoint key structure: module.metabolite_model.*
        self.metabolite_model = metabolite_model
        self.head = head

    def forward(
        self,
        expressions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Forward pass matching official fine-tuning code.

        Official code path::

            expr_embeds = expressions.unsqueeze(-1) * expr_weight + expr_bias
            embeddings = cat([cls_token, expr_embeds], dim=1)
            outputs = bert(inputs_embeds=embeddings, ...)
            pooled_output = outputs['pooler_output']
            logits = classifier(pooled_output).squeeze(-1)

        Our equivalent path::

            outputs = metabolite_model._embed_and_encode(expressions, ...)
            pooled_output = outputs['pooler_output']
            logits = head(pooled_output)

        Args:
            expressions: ``(B, num_metabolites)`` expression values.
            attention_mask: ``(B, num_metabolites)`` attention mask (all-ones).

        Returns:
            Tuple of ``(logits, attentions)``.
            - logits: ``(B,)`` for SingleTaskHead, ``(B, D)`` for multi-task.
            - attentions: tuple of per-layer attention weight tensors, or None.
        """
        outputs = self.metabolite_model._embed_and_encode(
            expressions,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        pooled_output = outputs["pooler_output"]
        logits = self.head(pooled_output)
        return logits, outputs["attentions"]
