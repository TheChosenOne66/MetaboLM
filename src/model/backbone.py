"""
MetaboLM BERT Backbone

Ported from the official Pretraining.py. This module provides the core BERT
encoder with metabolite-correlation-aware attention for metabolomics data.

Key design choices (preserved from original):
- No positional encoding (metabolomics data has no inherent order)
- Correlation matrix injected into attention via learnable bias_coef (init 0.01)
- Hadamard (element-wise) expression embedding: scalar * weight + bias per metabolite
- [CLS] token prepended for global information summarization
- Xavier initialization for all linear layers
- Masking strategy for pretraining: 80% zero, 10% random, 10% unchanged
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig


class CustomBertSelfAttention(nn.Module):
    """Multi-head self-attention with correlation bias injection."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.num_attention_heads: int = config.num_attention_heads
        self.attention_head_size: int = config.hidden_size // config.num_attention_heads
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder: bool = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        bias_matrix_chunk: Optional[torch.Tensor] = None,
        bias_coef: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        else:
            past_key_value = None

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if bias_matrix_chunk is not None and bias_coef is not None:
            bias = bias_matrix_chunk.unsqueeze(0).unsqueeze(0) * bias_coef
            bias = bias.expand(attention_scores.size())
            attention_scores = attention_scores + bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)

        return outputs


class CustomBertAttention(nn.Module):
    """Self-attention + output projection + residual + LayerNorm."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.self = CustomBertSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        bias_matrix_chunk: Optional[torch.Tensor] = None,
        bias_coef: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef,
        )

        attention_output = self.output(self_outputs[0])
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class CustomBertLayer(nn.Module):
    """Transformer layer: attention + FFN (ReLU) + residual + LayerNorm."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = CustomBertAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.is_decoder: bool = config.is_decoder
        # Optional adapter, set externally for parameter-efficient fine-tuning
        self.adapter = None
        if self.is_decoder:
            self.crossattention = CustomBertAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        bias_matrix_chunk: Optional[torch.Tensor] = None,
        bias_coef: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                bias_matrix_chunk=bias_matrix_chunk,
                bias_coef=bias_coef,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.relu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + attention_output)
        if self.adapter is not None:
            layer_output = self.adapter(layer_output)
        outputs = (layer_output,) + outputs

        return outputs


class CustomBertEncoder(nn.Module):
    """Stack of CustomBertLayer modules."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [CustomBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[List[Optional[torch.Tensor]]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        bias_matrix_chunk: Optional[torch.Tensor] = None,
        bias_coef: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, Any], Tuple[torch.Tensor, ...]]:
        all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = () if output_hidden_states else None
        all_attentions: Optional[Tuple[torch.Tensor, ...]] = () if output_attentions else None
        next_decoder_cache: Optional[Tuple[Any, ...]] = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                bias_matrix_chunk=bias_matrix_chunk,
                bias_coef=bias_coef,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_decoder_cache, all_hidden_states, all_attentions]
                if v is not None
            )
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }


class CustomBertModel(nn.Module):
    """BERT model with encoder + adaptive avg pooling, Xavier init."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.encoder = CustomBertEncoder(config)
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.config = config
        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        return (1.0 - encoder_attention_mask) * -10000.0

    def get_head_mask(
        self,
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
    ) -> List[Optional[torch.Tensor]]:
        if head_mask is not None:
            head_mask = torch.tensor(head_mask, dtype=torch.float32)
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).expand(num_hidden_layers, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1)
            return head_mask
        return [None] * num_hidden_layers

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        bias_matrix_chunk: Optional[torch.Tensor] = None,
        bias_coef: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, Any], Tuple[torch.Tensor, ...]]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        batch_size, seq_length = input_shape[0], input_shape[1]
        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    (encoder_batch_size, encoder_sequence_length), device=device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = inputs_embeds
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef,
        )

        sequence_output = encoder_outputs["last_hidden_state"]
        pooled_output = (
            self.pooler(sequence_output.transpose(1, 2)).squeeze(-1)
            if self.pooler is not None
            else None
        )

        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + tuple(
                v
                for v in [
                    encoder_outputs["past_key_values"],
                    encoder_outputs["hidden_states"],
                    encoder_outputs["attentions"],
                ]
                if v is not None
            )

        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "past_key_values": encoder_outputs["past_key_values"],
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
        }


class MetaboliteBERTModel(nn.Module):
    """
    MetaboLM: BERT backbone for metabolomics data.

    Expression embedding uses Hadamard (element-wise) mapping:
        embed_i = expression_i * weight_i + bias_i   (per metabolite)

    A learnable [CLS] token is prepended for global representation.
    The correlation matrix is injected into every attention layer via
    a learnable ``bias_coef`` (initialized to 0.01).

    Masking strategy for pretraining reference:
        - 80% of masked positions: replace with zero
        - 10% of masked positions: replace with random value
        - 10% of masked positions: keep unchanged

    Args:
        hidden_size: Hidden dimension (default 768).
        num_layers: Number of transformer layers (default 12).
        num_metabolites: Number of metabolites (default 168).
        bias_matrix: Correlation/bias matrix of shape
            ``(num_metabolites+1, num_metabolites+1)`` (padded with [CLS]).
            Registered as a non-trainable buffer.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_metabolites: int = 168,
        bias_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_metabolites = num_metabolites
        self.hidden_size = hidden_size

        config = BertConfig(
            vocab_size=num_metabolites,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=num_metabolites + 1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
        )
        self.bert = CustomBertModel(config)

        # Hadamard embedding: scalar * weight + bias per metabolite
        self.expr_weight = nn.Parameter(torch.ones(1, num_metabolites, hidden_size))
        self.expr_bias = nn.Parameter(torch.zeros(1, num_metabolites, hidden_size))

        # [CLS] token for global information summarization
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Output projection for masked prediction (scalar per metabolite)
        self.output_layer = nn.Linear(hidden_size, 1)

        # Correlation matrix as non-trainable buffer (same as official)
        if bias_matrix is not None:
            self.register_buffer("bias_matrix_full", bias_matrix)

        # Learnable coefficient for correlation bias injection
        self.bias_coef = nn.Parameter(torch.tensor(0.01))

    def _embed_and_encode(
        self,
        expressions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Dict[str, Any]:
        """Shared logic: embed expressions, prepend [CLS], run encoder.

        Args:
            expressions: Raw expression values of shape ``(B, num_metabolites)``.
            attention_mask: Optional mask of shape ``(B, num_metabolites)``.
                If *None*, all positions are attended to.
            output_attentions: Whether to return attention weights.

        Returns:
            Dictionary of BERT outputs (from ``CustomBertModel.forward``).
        """
        batch_size = expressions.size(0)

        # Hadamard embedding
        expr_embeds = (
            expressions.unsqueeze(-1) * self.expr_weight + self.expr_bias
        )  # (B, num_metabolites, hidden_size)

        # Prepend [CLS]
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)  # (B, 1, hidden_size)
        embeddings = torch.cat((cls_tokens, expr_embeds), dim=1)  # (B, num_metabolites+1, hidden_size)

        # Build attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, self.num_metabolites, device=expressions.device
            )
        new_attention_mask = torch.cat(
            (torch.ones(batch_size, 1, device=attention_mask.device), attention_mask),
            dim=1,
        )

        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=new_attention_mask,
            output_attentions=output_attentions,
            bias_matrix_chunk=self.bias_matrix_full,
            bias_coef=self.bias_coef,
        )
        return outputs

    def forward(
        self,
        expressions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Forward pass matching the official pretraining interface.

        Args:
            expressions: ``(B, num_metabolites)`` expression values.
            attention_mask: ``(B, num_metabolites)`` attention mask.

        Returns:
            Tuple of (prediction_scores ``(B, num_metabolites)``, attentions).
        """
        outputs = self._embed_and_encode(
            expressions,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        prediction_scores = self.output_layer(
            outputs["last_hidden_state"][:, 1:, :]
        ).squeeze(-1)
        attentions = outputs["attentions"]
        return prediction_scores, attentions

    def get_cls_embedding(
        self,
        expressions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the [CLS] representation (768-dim) for downstream tasks.

        Args:
            expressions: ``(B, num_metabolites)`` raw expression values.
            attention_mask: Optional ``(B, num_metabolites)`` mask.

        Returns:
            ``(B, hidden_size)`` [CLS] hidden state.
        """
        outputs = self._embed_and_encode(
            expressions, attention_mask=attention_mask
        )
        # [CLS] is at position 0
        cls_hidden_state = outputs["last_hidden_state"][:, 0, :]
        return cls_hidden_state

    def get_pooled_output(
        self,
        expressions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Return the pooled output for downstream classification.

        Uses ``AdaptiveAvgPool1d`` over all token hidden states (CLS + 168
        metabolites), matching the official fine-tuning code which classifies
        on ``pooler_output`` rather than the ``[CLS]`` hidden state.

        Args:
            expressions: ``(B, num_metabolites)`` expression values.
            attention_mask: Optional ``(B, num_metabolites)`` mask.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of ``(pooled_output (B, hidden_size), attentions or None)``.
        """
        outputs = self._embed_and_encode(
            expressions,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        return outputs["pooler_output"], outputs.get("attentions")

    @classmethod
    def load_pretrained(
        cls,
        ckpt_path: str,
        num_metabolites: int = 168,
    ) -> "MetaboliteBERTModel":
        """Load a pretrained checkpoint.

        Handles:
        - Loading checkpoint from disk
        - Extracting ``bias_matrix_full`` and passing to constructor
        - Stripping ``module.`` prefix from DataParallel keys

        Args:
            ckpt_path: Path to the checkpoint file.
            num_metabolites: Number of metabolites (must match checkpoint).

        Returns:
            A ``MetaboliteBERTModel`` with loaded weights.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip `module.` prefix from DataParallel keys
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
            cleaned_state_dict[new_key] = value

        # Extract bias_matrix_full for constructor
        bias_matrix = cleaned_state_dict.get("bias_matrix_full")

        model = cls(
            num_metabolites=num_metabolites,
            bias_matrix=bias_matrix,
        )

        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            import logging
            logging.getLogger(__name__).info(
                f"Missing keys when loading pretrained weights: {missing}"
            )
        if unexpected:
            import logging
            logging.getLogger(__name__).info(
                f"Unexpected keys when loading pretrained weights: {unexpected}"
            )

        return model
