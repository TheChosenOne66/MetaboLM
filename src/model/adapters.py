"""Parameter-efficient fine-tuning modules: Adapter and LoRA.

AdapterLayer: Bottleneck adapter (Houlsby et al. 2019) inserted after each
    Transformer layer's FFN block. Residual connection ensures init ≈ identity.

LoRALinear: Low-Rank Adaptation (Hu et al. 2021) wrapping an existing nn.Linear.
    Original weights frozen; only rank-r matrices A, B are trainable.
    Init: A ~ N(0, 1/r), B = 0 → output starts identical to original.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    """Bottleneck adapter with residual connection.

    ``x -> down(768->bottleneck) -> GELU -> up(bottleneck->768) -> + x``

    Args:
        hidden_size: Input/output dimension (default 768).
        bottleneck_size: Bottleneck dimension (default 64).
    """

    def __init__(self, hidden_size: int = 768, bottleneck_size: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear.

    Wraps an existing Linear layer and adds trainable low-rank matrices::

        y = W_original @ x + b  +  x @ A @ B

    Original weights are frozen. A is initialized with small random values
    (scaled by 1/rank), B is initialized to zero, so the LoRA contribution
    starts at zero (output identical to original).

    Args:
        original: The nn.Linear layer to wrap.
        rank: Rank of the low-rank decomposition (default 8).
    """

    def __init__(self, original: nn.Linear, rank: int = 8) -> None:
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_features = original.in_features
        out_features = original.out_features
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + x @ self.lora_A @ self.lora_B
