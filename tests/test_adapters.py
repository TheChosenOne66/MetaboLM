import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/SPXvePFS/users/jytang/metabolm_posttrain")
from src.model.adapters import AdapterLayer, LoRALinear


def test_adapter_forward_shape():
    """Adapter preserves input shape (B, seq, 768)."""
    adapter = AdapterLayer(hidden_size=768, bottleneck_size=64)
    x = torch.randn(4, 169, 768)
    out = adapter(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("test_adapter_forward_shape PASSED")


def test_adapter_residual():
    """At init (small weights), output ≈ input due to residual."""
    adapter = AdapterLayer(hidden_size=768, bottleneck_size=64)
    x = torch.randn(2, 169, 768)
    out = adapter(x)
    diff = (out - x).abs().mean().item()
    assert diff < 1.0, f"Residual not working, mean diff = {diff}"
    print("test_adapter_residual PASSED")


def test_adapter_param_count():
    """One adapter: 768*64 + 64 + 64*768 + 768 = 99,136."""
    adapter = AdapterLayer(hidden_size=768, bottleneck_size=64)
    total = sum(p.numel() for p in adapter.parameters())
    print(f"AdapterLayer params: {total:,}")
    assert total == 768 * 64 + 64 + 64 * 768 + 768, f"Unexpected: {total}"
    print("test_adapter_param_count PASSED")


def test_lora_forward_shape():
    """LoRA wraps Linear and preserves output shape."""
    original = nn.Linear(768, 768)
    lora = LoRALinear(original, rank=8)
    x = torch.randn(4, 169, 768)
    out = lora(x)
    assert out.shape == (4, 169, 768), f"Expected (4,169,768), got {out.shape}"
    print("test_lora_forward_shape PASSED")


def test_lora_init_identity():
    """At init, LoRA output equals original output (B initialized to zero)."""
    original = nn.Linear(768, 768)
    lora = LoRALinear(original, rank=8)
    x = torch.randn(2, 768)
    with torch.no_grad():
        out_orig = original(x)
        out_lora = lora(x)
    diff = (out_orig - out_lora).abs().max().item()
    assert diff < 1e-6, f"LoRA should start as identity, max diff = {diff}"
    print("test_lora_init_identity PASSED")


def test_lora_freezes_original():
    """Original weights are frozen, only lora_A and lora_B are trainable."""
    original = nn.Linear(768, 768)
    lora = LoRALinear(original, rank=8)
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    frozen = [n for n, p in lora.named_parameters() if not p.requires_grad]
    assert "lora_A" in trainable, f"lora_A should be trainable, trainable={trainable}"
    assert "lora_B" in trainable, f"lora_B should be trainable, trainable={trainable}"
    assert any("original" in n for n in frozen), f"Original should be frozen, frozen={frozen}"
    print("test_lora_freezes_original PASSED")


def test_lora_param_count():
    """LoRA rank-8 on 768->768: 768*8 + 8*768 = 12,288 trainable."""
    original = nn.Linear(768, 768)
    lora = LoRALinear(original, rank=8)
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    print(f"LoRA trainable params: {trainable:,}")
    assert trainable == 768 * 8 + 8 * 768, f"Unexpected: {trainable}"
    print("test_lora_param_count PASSED")


if __name__ == "__main__":
    test_adapter_forward_shape()
    test_adapter_residual()
    test_adapter_param_count()
    test_lora_forward_shape()
    test_lora_init_identity()
    test_lora_freezes_original()
    test_lora_param_count()
    print("All adapter/LoRA tests passed!")
