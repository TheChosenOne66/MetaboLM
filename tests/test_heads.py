import torch
import sys
sys.path.insert(0, "/SPXvePFS/users/jytang/metabolm_posttrain")
from src.model.heads import HierarchicalMultiTaskHead


def test_hierarchical_head_shapes():
    """Leaf logits (B,16), chapter logits (B,6)."""
    head = HierarchicalMultiTaskHead(hidden_size=768, proj_size=256,
                                      num_diseases=16, num_chapters=6)
    x = torch.randn(4, 768)
    leaf_logits, chapter_logits = head(x)
    assert leaf_logits.shape == (4, 16), f"Expected (4,16), got {leaf_logits.shape}"
    assert chapter_logits.shape == (4, 6), f"Expected (4,6), got {chapter_logits.shape}"
    print("test_hierarchical_head_shapes PASSED")


def test_hierarchical_head_param_count():
    """Shared proj + two heads: roughly 768*256 + 256 + 256*16 + 16 + 256*6 + 6 ~ 203K."""
    head = HierarchicalMultiTaskHead(hidden_size=768, proj_size=256,
                                      num_diseases=16, num_chapters=6)
    total = sum(p.numel() for p in head.parameters())
    print(f"HierarchicalMultiTaskHead params: {total:,}")
    assert 200_000 < total < 210_000, f"Unexpected param count: {total}"
    print("test_hierarchical_head_param_count PASSED")


def test_hierarchical_head_gradient_flow():
    """Both outputs receive gradients through shared projection."""
    head = HierarchicalMultiTaskHead(hidden_size=768, proj_size=256,
                                      num_diseases=16, num_chapters=6)
    x = torch.randn(2, 768, requires_grad=True)
    leaf, chapter = head(x)
    loss = leaf.sum() + chapter.sum()
    loss.backward()
    assert x.grad is not None
    assert head.shared_proj[0].weight.grad is not None
    print("test_hierarchical_head_gradient_flow PASSED")


if __name__ == "__main__":
    test_hierarchical_head_shapes()
    test_hierarchical_head_param_count()
    test_hierarchical_head_gradient_flow()
    print("All hierarchical head tests passed!")
