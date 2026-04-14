"""Tests for scripts/eval_hierarchy_violation.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_hierarchy_violation as ehv


def test_mean_aggregate_chapter_probs_basic():
    """Two samples × 3 leaves; chapter 0 covers leaves [0, 2]; chapter 1 covers leaf [1]."""
    leaf_probs = np.array(
        [
            [0.0, 0.5, 0.0],   # sample 0: chap0 = mean(0, 0) = 0; chap1 = 0.5
            [0.5, 0.5, 0.5],   # sample 1: chap0 = mean(0.5, 0.5) = 0.5; chap1 = 0.5
        ],
        dtype=np.float32,
    )
    chap_probs = ehv.mean_aggregate_chapter_probs(
        leaf_probs,
        disease_to_chapter_idx={0: 0, 1: 1, 2: 0},
        n_chapters=2,
    )
    assert chap_probs.shape == (2, 2)
    assert np.allclose(chap_probs[:, 0], [0.0, 0.5])
    assert np.allclose(chap_probs[:, 1], [0.5, 0.5])


def test_mean_aggregate_chapter_probs_all_zero():
    """All-zero leaves → all-zero chapters."""
    leaf_probs = np.zeros((3, 4), dtype=np.float32)
    chap_probs = ehv.mean_aggregate_chapter_probs(
        leaf_probs,
        disease_to_chapter_idx={0: 0, 1: 0, 2: 1, 3: 1},
        n_chapters=2,
    )
    assert chap_probs.shape == (3, 2)
    assert np.allclose(chap_probs, 0.0)


def test_mean_aggregate_chapter_probs_violation_possible():
    """Critical contrast with OR/max: a single high leaf in a multi-leaf chapter
    can produce a chapter prob LOWER than itself (its siblings drag down the mean).
    This is the property that makes mean a non-trivial HVR baseline.
    """
    # Chapter 0 covers leaves [0, 1, 2]. Leaf 0 is high (0.9), the others zero.
    # Mean = 0.3, so leaf 0 (0.9) > chap (0.3) — violation possible.
    leaf_probs = np.array([[0.9, 0.0, 0.0]], dtype=np.float32)
    chap_probs = ehv.mean_aggregate_chapter_probs(
        leaf_probs,
        disease_to_chapter_idx={0: 0, 1: 0, 2: 0},
        n_chapters=1,
    )
    assert np.isclose(chap_probs[0, 0], 0.3)
    # Sanity: leaf 0 > chap 0 (this is the property that breaks for OR/max).
    assert leaf_probs[0, 0] > chap_probs[0, 0] + 1e-7


def test_compute_and_persist_hvr_basic(tmp_path):
    # 2 samples, 4 leaves under 2 chapters {leaf 0,1 -> chap 0; leaf 2,3 -> chap 1}.
    # Sample 0: leaf=[0.9, 0.1, 0.2, 0.0], chap=[0.5, 0.5]
    #   -> leaf 0 (0.9) > chap 0 (0.5)  : VIOLATION
    #   -> leaf 1 (0.1) > chap 0 (0.5)  : OK
    #   -> leaf 2 (0.2) > chap 1 (0.5)  : OK
    #   -> leaf 3 (0.0) > chap 1 (0.5)  : OK
    # Sample 1: leaf=[0.0, 0.0, 0.6, 0.0], chap=[0.5, 0.5]
    #   -> leaf 0 (0.0) > chap 0 (0.5)  : OK
    #   -> leaf 1 (0.0) > chap 0 (0.5)  : OK
    #   -> leaf 2 (0.6) > chap 1 (0.5)  : VIOLATION
    #   -> leaf 3 (0.0) > chap 1 (0.5)  : OK
    # Expected violations: 2 / 8 = 0.25
    leaf_probs = np.array(
        [[0.9, 0.1, 0.2, 0.0], [0.0, 0.0, 0.6, 0.0]],
        dtype=np.float32,
    )
    chap_probs = np.array(
        [[0.5, 0.5], [0.5, 0.5]],
        dtype=np.float32,
    )
    out_path = tmp_path / "hierarchy_violation.json"

    result = ehv.compute_and_persist_hvr(
        leaf_probs=leaf_probs,
        chap_probs=chap_probs,
        output_path=out_path,
        method="unit_test",
        method_details={"test_id": "compute_and_persist_basic"},
        disease_to_chapter_idx={0: 0, 1: 0, 2: 1, 3: 1},  # toy mapping for this test
    )

    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written == result
    assert written["method"] == "unit_test"
    assert written["method_details"]["test_id"] == "compute_and_persist_basic"
    assert written["n_samples"] == 2
    assert written["n_leaf_chapter_pairs"] == 4
    assert pytest.approx(written["hierarchy_violation_rate"], abs=1e-6) == 0.25


def test_compute_and_persist_hvr_partial_mapping(tmp_path):
    """``n_leaf_chapter_pairs`` tracks the mapping, not leaf_probs column count.

    If a caller provides a sub-mapping (e.g. Task 3's E0-OR with some
    missing diseases), only the mapped leaves contribute pairs — the
    persisted field must reflect that.
    """
    # 2 samples × 4 leaf columns, but mapping only covers 2 of them.
    leaf_probs = np.array(
        [[0.9, 0.1, 0.9, 0.9], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    chap_probs = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    out_path = tmp_path / "hierarchy_violation_partial.json"

    payload = ehv.compute_and_persist_hvr(
        leaf_probs=leaf_probs,
        chap_probs=chap_probs,
        output_path=out_path,
        method="unit_test",
        method_details={},
        # Only leaves 0 and 1 are mapped; leaves 2 and 3 are deliberately
        # ignored (simulating Task 3's "only present diseases" case).
        disease_to_chapter_idx={0: 0, 1: 0},
    )

    # Pair count is len(mapping) = 2, NOT leaf_probs.shape[1] = 4.
    assert payload["n_leaf_chapter_pairs"] == 2
    # Sample 0 violates on leaf 0 (0.9 > 0.5); sample 1 violates on
    # nothing. Total 1 / 4 = 0.25 (2 samples × 2 mapped pairs).
    assert pytest.approx(payload["hierarchy_violation_rate"], abs=1e-6) == 0.25


FIXTURES = Path(__file__).parent / "fixtures" / "hierarchy_violation"


def test_load_e0_leaf_probs_partial_dir():
    """Loader returns a (3, 16) matrix; missing diseases are NaN columns."""
    from src.data.endpoints import get_disease_names
    diseases = get_disease_names()

    leaf_probs, present = ehv.load_e0_leaf_probs(
        FIXTURES / "eval_dir", diseases,
    )
    assert leaf_probs.shape == (3, 16)
    assert "T2D" in present and "obesity" in present
    assert len(present) == 2

    t2d_idx = diseases.index("T2D")
    obesity_idx = diseases.index("obesity")
    hypertension_idx = diseases.index("hypertension")

    # Present columns: exact values from the fixture
    np.testing.assert_allclose(
        leaf_probs[:, t2d_idx], [0.9, 0.1, 0.8], atol=1e-6,
    )
    np.testing.assert_allclose(
        leaf_probs[:, obesity_idx], [0.2, 0.7, 0.6], atol=1e-6,
    )
    # Missing columns: NaN
    assert np.all(np.isnan(leaf_probs[:, hypertension_idx]))


def test_load_e0_leaf_probs_skips_disease_with_no_eid_overlap(tmp_path):
    """A prediction file with eids that don't match labels → disease NOT present.

    Codex P2 on PR #5 (round 2): previously such a file made it into
    ``present`` with an all-NaN column. Then HVR's NaN comparisons read as
    non-violations and silently deflate the rate.
    """
    from src.data.endpoints import get_disease_names
    diseases = get_disease_names()

    eval_dir = tmp_path / "stale_dir"
    eval_dir.mkdir()
    (eval_dir / "eval_set_labels.csv").write_text("eid\n1\n2\n3\n")

    t2d_dir = eval_dir / "T2D"
    t2d_dir.mkdir()
    # Predictions for eids 999/998/997 — not a single one overlaps with
    # labels (1/2/3). Simulates stale artifact or dtype drift.
    (t2d_dir / "predictions_ckpt_T2D.csv").write_text(
        "eid,prob_ckpt_T2D\n999,0.5\n998,0.6\n997,0.7\n"
    )

    leaf_probs, present = ehv.load_e0_leaf_probs(eval_dir, diseases)
    assert "T2D" not in present
    # The T2D column should still be all NaN (loader initialises NaN and we
    # never aligned any row), but since T2D is not in ``present``, it'll be
    # dropped by run_e0_baseline_mode before it can deflate HVR.
    t2d_idx = diseases.index("T2D")
    assert np.all(np.isnan(leaf_probs[:, t2d_idx]))


def test_run_e0_baseline_mode_partial_alignment_raises(tmp_path):
    """Partial eid alignment leaves NaNs → must raise, not write a misleading rate.

    Setup: labels has eids 1/2/3; T2D predictions align only eid 1 (covers
    1/3 rows). obesity predictions align all 3. After ``load_e0_leaf_probs``
    drops nothing (both diseases had ≥1 aligned row), ``run_e0_baseline_mode``
    still has 2 NaN cells in the T2D column and must refuse to proceed.
    """
    eval_dir = tmp_path / "partial_dir"
    eval_dir.mkdir()
    (eval_dir / "eval_set_labels.csv").write_text("eid\n1\n2\n3\n")

    t2d_dir = eval_dir / "T2D"
    t2d_dir.mkdir()
    (t2d_dir / "predictions_ckpt_T2D.csv").write_text(
        # Only eid 1 aligns; eids 4/5 are unknown.
        "eid,prob_ckpt_T2D\n1,0.9\n4,0.1\n5,0.2\n"
    )
    obesity_dir = eval_dir / "obesity"
    obesity_dir.mkdir()
    (obesity_dir / "predictions_ckpt_obesity.csv").write_text(
        "eid,prob_ckpt_obesity\n1,0.3\n2,0.4\n3,0.5\n"
    )

    out_path = tmp_path / "should_not_be_written.json"
    with pytest.raises(RuntimeError, match="NaN cells remain in leaf_probs"):
        ehv.run_e0_baseline_mode(eval_dir, out_path)
    assert not out_path.exists()


def test_run_e0_baseline_mode_empty_dir_raises(tmp_path):
    """When NO predictions are found, refuse to write a fake HVR=0.0.

    Codex P2 on PR #5: previously the script silently produced
    ``hierarchy_violation_rate: 0.0`` over zero evaluated pairs, which
    renders indistinguishably from a legitimately-zero rate.
    """
    # Empty eval dir: labels CSV exists but NO per-disease predictions.
    eval_dir = tmp_path / "empty_eval_dir"
    eval_dir.mkdir()
    (eval_dir / "eval_set_labels.csv").write_text("eid\n1\n2\n3\n")

    out_path = tmp_path / "should_not_be_written.json"
    with pytest.raises(RuntimeError, match="No disease predictions loaded"):
        ehv.run_e0_baseline_mode(eval_dir, out_path)
    # And we really should NOT have written the output file.
    assert not out_path.exists()


def test_run_e0_baseline_mode_partial_dir(tmp_path):
    """End-to-end on the partial fixture: only T2D + obesity contribute.

    With T2D and obesity both in chapter_04, mean-aggregation gives:
        chapter_04 prob[i] = mean(p_T2D[i], p_obesity[i])
    Sample 0: mean(0.9, 0.2) = 0.55 ; T2D=0.9 > 0.55 VIOLATION ; obesity=0.2 OK
    Sample 1: mean(0.1, 0.7) = 0.4  ; T2D=0.1 OK ; obesity=0.7 > 0.4 VIOLATION
    Sample 2: mean(0.8, 0.6) = 0.7  ; T2D=0.8 > 0.7 VIOLATION ; obesity=0.6 OK
    Expected violations from T2D / obesity rows = 3 / 6 = 0.5
    All other 14 leaves have NaN probs → must be dropped (not counted).
    """
    out_path = tmp_path / "hierarchy_violation_baseline.json"
    payload = ehv.run_e0_baseline_mode(FIXTURES / "eval_dir", out_path)

    assert out_path.exists()
    assert payload["method"] == "mean_aggregate_baseline"
    assert payload["n_samples"] == 3
    # Only the 2 present diseases count toward the leaf-chapter pair total.
    assert payload["n_leaf_chapter_pairs"] == 2
    assert pytest.approx(payload["hierarchy_violation_rate"], abs=1e-6) == 0.5
    # Provenance: predictions_source path is recorded
    assert "eval_dir" in payload["method_details"]["predictions_source"]


# ── Critical-key predicates for ckpt load validation (codex P1 on PR #6) ─

def test_critical_missing_keys_flags_backbone_params():
    """Backbone params MUST be flagged as critical — this was the P1 bug.

    Previously the filter only caught head / adapter / LoRA keys, so a
    ckpt missing ``metabolite_model.expr_*`` or ``cls_token`` (backbone
    parameters) was silently accepted, and HVR would run against random
    backbone weights.
    """
    missing = [
        "metabolite_model.expr_embedding.weight",  # Backbone
        "metabolite_model.cls_token",              # Backbone
        "metabolite_model.bert.encoder.layer.5.attention.output.dense.weight",
        "head.leaf_head.weight",                   # Head
        "metabolite_model.bias_matrix_full",       # Buffer — NOT critical
    ]
    critical = ehv._critical_missing_keys(missing)
    assert "metabolite_model.expr_embedding.weight" in critical
    assert "metabolite_model.cls_token" in critical
    assert "head.leaf_head.weight" in critical
    assert "metabolite_model.bias_matrix_full" not in critical


def test_critical_missing_keys_empty_when_only_bias_matrix():
    """Only ``bias_matrix_full`` missing → no critical misses (legit load)."""
    critical = ehv._critical_missing_keys([
        "metabolite_model.bias_matrix_full",
    ])
    assert critical == []


def test_critical_unexpected_keys_flags_unexpected_head_and_encoder():
    """Unexpected head / encoder-layer keys flagged (freeze-strategy mismatch)."""
    unexpected = [
        "head.leaf_head.weight",
        "metabolite_model.bert.encoder.layer.3.adapter.down.weight",
        "metabolite_model.bias_matrix_full",  # Buffer — NOT critical
        "some.unrelated.key",                 # Not head/encoder — NOT critical
    ]
    critical = ehv._critical_unexpected_keys(unexpected)
    assert "head.leaf_head.weight" in critical
    assert any("adapter" in k for k in critical)
    assert "metabolite_model.bias_matrix_full" not in critical
    assert "some.unrelated.key" not in critical
