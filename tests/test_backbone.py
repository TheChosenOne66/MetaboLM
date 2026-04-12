import torch

from src.model.backbone import MetaboliteBERTModel


def test_forward_shape():
    """forward(expressions, attention_mask) matches official interface."""
    bias_matrix = torch.randn(169, 169)  # 168 + 1 for [CLS]
    model = MetaboliteBERTModel(num_metabolites=168, bias_matrix=bias_matrix)
    expressions = torch.randn(4, 168)
    mask = torch.ones(4, 168)
    pred, attn = model(expressions, mask)
    assert pred.shape == (4, 168), f"Expected (4, 168), got {pred.shape}"
    assert len(attn) == 12, f"Expected 12 attention layers, got {len(attn)}"
    print("test_forward_shape PASSED")


def test_cls_embedding_shape():
    """get_cls_embedding returns (B, 768) [CLS] hidden state."""
    bias_matrix = torch.randn(169, 169)
    model = MetaboliteBERTModel(num_metabolites=168, bias_matrix=bias_matrix)
    expressions = torch.randn(4, 168)
    cls_emb = model.get_cls_embedding(expressions)
    assert cls_emb.shape == (4, 768), f"Expected (4, 768), got {cls_emb.shape}"
    print("test_cls_embedding_shape PASSED")


def test_param_count():
    model = MetaboliteBERTModel(num_metabolites=168)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    # 12-layer BERT-base-like should be roughly 85M params
    assert total > 50_000_000, f"Too few params: {total}"
    print("test_param_count PASSED")


if __name__ == "__main__":
    test_forward_shape()
    test_cls_embedding_shape()
    test_param_count()
    print("All backbone tests passed!")
