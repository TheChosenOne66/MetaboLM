"""Tests for GRPO trainer and reward functions."""

import pytest
import torch
import numpy as np

from src.training.rewards import calibration_reward, hierarchy_reward, combined_reward
from src.training.grpo_trainer import GRPOConfig


# ── Reward function tests ─────────────────────────────────────────────


class TestCalibrationReward:
    def test_perfect_predictions(self):
        """Perfect predictions should give reward close to 0."""
        logits = torch.tensor([[10.0, -10.0, 10.0]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
        r = calibration_reward(logits, labels)
        assert r.shape == (1,)
        assert r.item() > -0.01  # Near zero (perfect)

    def test_worst_predictions(self):
        """Completely wrong predictions should give very negative reward."""
        logits = torch.tensor([[-10.0, 10.0, -10.0]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
        r = calibration_reward(logits, labels)
        assert r.item() < -0.5

    def test_batch_shape(self):
        """Should return (B,) for batch inputs."""
        logits = torch.randn(32, 16)
        labels = torch.randint(0, 2, (32, 16), dtype=torch.float32)
        r = calibration_reward(logits, labels)
        assert r.shape == (32,)

    def test_reward_is_nonpositive(self):
        """Brier-based reward is always <= 0."""
        logits = torch.randn(100, 16)
        labels = torch.randint(0, 2, (100, 16), dtype=torch.float32)
        r = calibration_reward(logits, labels)
        assert (r <= 1e-7).all()


class TestHierarchyReward:
    @pytest.fixture
    def leaf_to_chapter(self):
        # Simplified: 4 diseases, 2 chapters
        return torch.tensor([0, 0, 1, 1], dtype=torch.long)

    def test_no_violations(self, leaf_to_chapter):
        """When all P(leaf) <= P(chapter), reward should be 0."""
        leaf_logits = torch.tensor([[-5.0, -5.0, -5.0, -5.0]])  # small probs
        chapter_logits = torch.tensor([[5.0, 5.0]])  # large probs
        labels = torch.zeros(1, 4)
        chap_labels = torch.zeros(1, 2)
        r = hierarchy_reward(
            leaf_logits, labels, chapter_logits, chap_labels,
            leaf_to_chapter=leaf_to_chapter,
        )
        assert r.item() == pytest.approx(0.0, abs=1e-5)

    def test_violations_give_negative_reward(self, leaf_to_chapter):
        """When P(leaf) > P(chapter), reward should be negative."""
        leaf_logits = torch.tensor([[5.0, 5.0, 5.0, 5.0]])  # large probs
        chapter_logits = torch.tensor([[-5.0, -5.0]])  # small probs
        labels = torch.zeros(1, 4)
        chap_labels = torch.zeros(1, 2)
        r = hierarchy_reward(
            leaf_logits, labels, chapter_logits, chap_labels,
            leaf_to_chapter=leaf_to_chapter,
        )
        assert r.item() < -0.1

    def test_requires_leaf_to_chapter(self):
        """Should raise if leaf_to_chapter not provided."""
        with pytest.raises(ValueError):
            hierarchy_reward(
                torch.randn(1, 4), torch.zeros(1, 4),
                torch.randn(1, 2), torch.zeros(1, 2),
            )


class TestCombinedReward:
    def test_shape(self):
        leaf_to_chapter = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        r = combined_reward(
            torch.randn(8, 4), torch.randint(0, 2, (8, 4), dtype=torch.float32),
            torch.randn(8, 2), torch.randint(0, 2, (8, 2), dtype=torch.float32),
            leaf_to_chapter, alpha=0.5,
        )
        assert r.shape == (8,)


# ── GRPOConfig tests ──────────────────────────────────────────────────


class TestGRPOConfig:
    def test_defaults(self):
        cfg = GRPOConfig()
        assert cfg.group_size == 4
        assert cfg.kl_coeff == 0.1
        assert cfg.learning_rate == 1e-5

    def test_custom(self):
        cfg = GRPOConfig(group_size=8, kl_coeff=0.05)
        assert cfg.group_size == 8
        assert cfg.kl_coeff == 0.05


# ── Integration: GRPOTrainer with tiny model ─────────────────────────


class TestGRPOTrainerIntegration:
    @pytest.fixture
    def setup(self):
        """Build a minimal model + trainer for integration testing."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        from src.model.backbone import MetaboliteBERTModel
        from src.model.heads import HierarchicalMultiTaskHead
        from src.model.wrapper import MetaboLMForClassification
        from src.training.grpo_trainer import GRPOTrainer, GRPOConfig
        from src.training.rewards import calibration_reward
        from src.data.endpoints import get_disease_to_chapter_idx
        from torch.utils.data import DataLoader, TensorDataset
        import copy

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tiny synthetic data
        B, D = 32, 168
        X = torch.randn(B, D)
        leaf_labels = torch.randint(0, 2, (B, 16), dtype=torch.float32)
        chap_labels = torch.randint(0, 2, (B, 6), dtype=torch.float32)

        ds = TensorDataset(X, leaf_labels, chap_labels)
        loader = DataLoader(ds, batch_size=16)

        # Model
        backbone = MetaboliteBERTModel(num_metabolites=168)
        # Register dummy bias matrix (169x169: 168 metabolites + [CLS])
        bias_matrix = torch.zeros(169, 169, device=device)
        backbone.register_buffer("bias_matrix_full", bias_matrix)
        head = HierarchicalMultiTaskHead(768, 256, 16, 6)
        model = MetaboLMForClassification(backbone, head, freeze_strategy="none")
        model.to(device)

        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.to(device)

        config = GRPOConfig(
            learning_rate=1e-4,
            num_epochs=2,
            group_size=2,
            kl_coeff=0.1,
        )

        d2c = get_disease_to_chapter_idx()
        disease_names = [f"d{i}" for i in range(16)]

        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            train_loader=loader,
            val_loader=loader,
            reward_fn=calibration_reward,
            config=config,
            device=device,
            disease_names=disease_names,
            disease_to_chapter_idx=d2c,
        )
        return trainer

    def test_train_epoch_returns_metrics(self, setup):
        """Training epoch should return pg_loss, kl, reward."""
        pg_loss, kl, reward = setup.train_epoch()
        assert isinstance(pg_loss, float)
        assert isinstance(kl, float)
        assert isinstance(reward, float)

    def test_evaluate_returns_metrics(self, setup):
        """Evaluate should return metrics dict with mean_auc."""
        metrics, labels, probs = setup.evaluate()
        assert "mean_auc" in metrics
        assert labels.shape[1] == 16
        assert probs.shape[1] == 16

    def test_full_training_loop(self, setup):
        """Full train() should complete and return best AUC."""
        best_auc, best_epoch, state, labels, probs = setup.train()
        assert best_epoch >= 1
        assert isinstance(best_auc, float)
        assert state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
