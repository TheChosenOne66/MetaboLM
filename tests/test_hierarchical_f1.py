"""Unit tests for compute_hierarchical_f1 (Kiritchenko et al. 2005).

Hand-computed examples on a minimal 2-level hierarchy:
  chapter_A (idx 0) → disease_0, disease_1
  chapter_B (idx 1) → disease_2

disease_to_chapter_idx = {0: 0, 1: 0, 2: 1}
n_chapters = 2
Augmented node indices: diseases 0,1,2; chapters 3 (=3+0), 4 (=3+1).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)
sys.path.insert(0, str(Path(_project_root) / "scripts"))

from eval_hierarchical_f1 import (
    _validate_state_dict_load,
    compute_hierarchical_f1,
    youden_optimal_thresholds,
)

D2C = {0: 0, 1: 0, 2: 1}  # 3 diseases, 2 chapters
N_CHAP = 2


def test_perfect_predictions():
    """All predictions correct → hF1 = flat_F1 = 1.0."""
    labels = np.array([[1, 0, 1],
                       [0, 1, 0]])
    preds = labels.copy()
    r = compute_hierarchical_f1(preds, labels, D2C, N_CHAP)
    assert r["hF1"] == pytest.approx(1.0)
    assert r["flat_F1"] == pytest.approx(1.0)
    assert r["delta_hF1_flat_F1"] == pytest.approx(0.0)


def test_same_chapter_false_positive():
    """FP in the same chapter as a TP → hF1 > flat_F1.

    Sample: true={disease_0}, pred={disease_0, disease_1}
      flat_pred = {0, 1}, flat_true = {0}
      flat_P = 1/2, flat_R = 1/1, flat_F1 = 2/3

      aug_pred = {0, 1, chap_A} = {0, 1, 3}   (both map to chap_A)
      aug_true = {0, chap_A}    = {0, 3}
      hP = |{0, 3}| / |{0, 1, 3}| = 2/3
      hR = |{0, 3}| / |{0, 3}|    = 2/2 = 1.0
      hF1 = 2 * (2/3) * 1 / (2/3 + 1) = (4/3) / (5/3) = 4/5 = 0.8
    """
    labels = np.array([[1, 0, 0]])
    preds = np.array([[1, 1, 0]])
    r = compute_hierarchical_f1(preds, labels, D2C, N_CHAP)
    assert r["flat_F1"] == pytest.approx(2 / 3, abs=1e-6)
    assert r["hF1"] == pytest.approx(4 / 5, abs=1e-6)
    assert r["hF1"] > r["flat_F1"]


def test_cross_chapter_false_positive():
    """FP in a different chapter → hF1 = flat_F1 (no partial credit).

    Sample: true={disease_0}, pred={disease_0, disease_2}
      flat_pred = {0, 2}, flat_true = {0}
      flat_P = 1/2, flat_R = 1/1, flat_F1 = 2/3

      aug_pred = {0, chap_A, 2, chap_B} = {0, 3, 2, 4}  (4 elements)
      aug_true = {0, chap_A}             = {0, 3}         (2 elements)
      hP = |{0, 3}| / |{0, 3, 2, 4}| = 2/4 = 0.5
      hR = |{0, 3}| / |{0, 3}|        = 2/2 = 1.0
      hF1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 1.0 / 1.5 = 2/3
    """
    labels = np.array([[1, 0, 0]])
    preds = np.array([[1, 0, 1]])
    r = compute_hierarchical_f1(preds, labels, D2C, N_CHAP)
    assert r["flat_F1"] == pytest.approx(2 / 3, abs=1e-6)
    assert r["hF1"] == pytest.approx(2 / 3, abs=1e-6)
    assert r["delta_hF1_flat_F1"] == pytest.approx(0.0, abs=1e-6)


def test_all_wrong_cross_chapter():
    """Predict only cross-chapter diseases (no TP overlap at all).

    Sample: true={disease_2 (chap_B)}, pred={disease_0 (chap_A)}
      flat_pred = {0}, flat_true = {2}
      flat_P = 0, flat_R = 0, flat_F1 = 0

      aug_pred = {0, chap_A} = {0, 3}
      aug_true = {2, chap_B} = {2, 4}
      hP = 0, hR = 0, hF1 = 0
    """
    labels = np.array([[0, 0, 1]])
    preds = np.array([[1, 0, 0]])
    r = compute_hierarchical_f1(preds, labels, D2C, N_CHAP)
    assert r["flat_F1"] == pytest.approx(0.0)
    assert r["hF1"] == pytest.approx(0.0)


def test_micro_averaging_across_samples():
    """Two samples: one with same-chapter FP, one with cross-chapter FP.

    Sample 1: true={0}, pred={0, 1}  (same-chap FP)
      aug_pred = {0, 1, 3}, aug_true = {0, 3}
      intersect = {0, 3} → 2; |aug_pred| = 3; |aug_true| = 2
      flat: TP=1, pred=2, true=1

    Sample 2: true={0}, pred={0, 2}  (cross-chap FP)
      aug_pred = {0, 3, 2, 4}, aug_true = {0, 3}
      intersect = {0, 3} → 2; |aug_pred| = 4; |aug_true| = 2
      flat: TP=1, pred=2, true=1

    Micro hP = (2+2)/(3+4) = 4/7
    Micro hR = (2+2)/(2+2) = 4/4 = 1.0
    Micro hF1 = 2 * (4/7) * 1 / (4/7 + 1) = (8/7) / (11/7) = 8/11

    Micro flat_P = 2/4 = 0.5
    Micro flat_R = 2/2 = 1.0
    Micro flat_F1 = 2/3
    """
    labels = np.array([[1, 0, 0],
                       [1, 0, 0]])
    preds = np.array([[1, 1, 0],
                      [1, 0, 1]])
    r = compute_hierarchical_f1(preds, labels, D2C, N_CHAP)
    assert r["hF1"] == pytest.approx(8 / 11, abs=1e-6)
    assert r["flat_F1"] == pytest.approx(2 / 3, abs=1e-6)
    assert r["hF1"] > r["flat_F1"]


def test_no_predictions_no_labels():
    """All zeros → hF1 = 0 (no predictions, no labels to match)."""
    labels = np.array([[0, 0, 0]])
    preds = np.array([[0, 0, 0]])
    r = compute_hierarchical_f1(preds, labels, D2C, N_CHAP)
    assert r["hF1"] == pytest.approx(0.0)
    assert r["flat_F1"] == pytest.approx(0.0)


def test_youden_thresholds_are_finite_on_normal_data():
    """Regular two-class column → threshold is a finite value in (0, 1)."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=(200, 3)).astype(np.int32)
    # Force column 2 to have a strong positive signal so ROC is well-defined.
    labels[:, 2] = (rng.random(200) < 0.3).astype(np.int32)
    probs = rng.random((200, 3))
    probs[:, 2] = np.where(labels[:, 2] == 1, rng.uniform(0.6, 1.0, 200), rng.uniform(0.0, 0.5, 200))

    ths = youden_optimal_thresholds(probs, labels)
    assert ths.shape == (3,)
    assert np.all(np.isfinite(ths)), f"Thresholds contain non-finite values: {ths}"


def test_youden_thresholds_single_class_column_falls_back_to_0_5():
    """Single-class column (all positives or all negatives) → threshold = 0.5, no inf."""
    # Column 0: mixed; Column 1: all zeros (no positives); Column 2: all ones (no negatives).
    labels = np.zeros((100, 3), dtype=np.int32)
    labels[:50, 0] = 1
    labels[:, 2] = 1
    rng = np.random.default_rng(1)
    probs = rng.random((100, 3))

    ths = youden_optimal_thresholds(probs, labels)
    assert np.all(np.isfinite(ths)), f"Thresholds contain non-finite values: {ths}"
    assert ths[1] == 0.5, "Single-class (all-negative) column should fall back to 0.5"
    assert ths[2] == 0.5, "Single-class (all-positive) column should fall back to 0.5"


def test_youden_thresholds_constant_scores_no_inf():
    """Constant predicted scores (Youden ties at zero) → threshold is finite, not inf."""
    labels = np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.int32).T  # (8, 1)
    probs = np.full((8, 1), 0.5, dtype=np.float32)  # every score identical

    ths = youden_optimal_thresholds(probs, labels)
    assert np.isfinite(ths[0]), (
        "argmax over ties must not pick the roc_curve +inf sentinel; "
        f"got threshold = {ths[0]}"
    )


# ── _validate_state_dict_load ────────────────────────────────────────────


class _FakeLoadResult:
    def __init__(self, missing_keys, unexpected_keys):
        self.missing_keys = list(missing_keys)
        self.unexpected_keys = list(unexpected_keys)


def test_validate_load_clean_passes():
    _validate_state_dict_load(_FakeLoadResult([], []), context="test")  # no raise


def test_validate_load_unexpected_keys_only_warns(caplog):
    # Unexpected keys (e.g., MLM head) are non-fatal — should only log a warning.
    with caplog.at_level("WARNING", logger="eval_hierarchical_f1"):
        _validate_state_dict_load(
            _FakeLoadResult(missing_keys=[], unexpected_keys=["output_layer.weight", "output_layer.bias"]),
            context="test",
        )
    assert any("unexpected" in r.message.lower() for r in caplog.records)


def test_validate_load_critical_missing_raises():
    # Missing head parameters → critical config mismatch → must raise.
    with pytest.raises(RuntimeError, match="missing .* critical parameter"):
        _validate_state_dict_load(
            _FakeLoadResult(
                missing_keys=["head.leaf_classifier.weight", "head.leaf_classifier.bias"],
                unexpected_keys=[],
            ),
            context="test",
        )


def test_validate_load_allowed_prefix_suppresses_missing():
    # When a prefix is whitelisted, its missing keys are not critical.
    _validate_state_dict_load(
        _FakeLoadResult(
            missing_keys=["metabolite_model.bias_matrix_full"],
            unexpected_keys=[],
        ),
        context="test",
        allowed_missing_prefixes=("metabolite_model.bias_matrix_full",),
    )  # no raise


def test_full_16_disease_mapping():
    """Verify the function works with the real 16-disease / 6-chapter mapping."""
    from src.data.endpoints import get_disease_to_chapter_idx, get_unique_chapters
    d2c = get_disease_to_chapter_idx()
    n_chap = len(get_unique_chapters())

    np.random.seed(42)
    labels = (np.random.rand(100, 16) > 0.9).astype(np.int32)
    preds = (np.random.rand(100, 16) > 0.8).astype(np.int32)

    r = compute_hierarchical_f1(preds, labels, d2c, n_chap)
    assert 0.0 <= r["hF1"] <= 1.0
    assert 0.0 <= r["flat_F1"] <= 1.0
    assert r["n_samples"] == 100
    # With random preds, hF1 should be >= flat_F1 on average
    # (some same-chapter FPs get partial credit)
