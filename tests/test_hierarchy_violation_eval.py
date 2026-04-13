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


def test_or_aggregate_chapter_probs_basic():
    """Two samples × 3 leaves; chapter 0 covers leaves [0, 2]; chapter 1 covers leaf [1]."""
    leaf_probs = np.array(
        [
            [0.0, 0.5, 0.0],   # sample 0: chap0 = 1 - (1-0)*(1-0) = 0; chap1 = 0.5
            [0.5, 0.5, 0.5],   # sample 1: chap0 = 1 - 0.5*0.5    = 0.75; chap1 = 0.5
        ],
        dtype=np.float32,
    )
    chap_probs = ehv.or_aggregate_chapter_probs(
        leaf_probs,
        disease_to_chapter_idx={0: 0, 1: 1, 2: 0},
        n_chapters=2,
    )
    assert chap_probs.shape == (2, 2)
    assert np.allclose(chap_probs[:, 0], [0.0, 0.75])
    assert np.allclose(chap_probs[:, 1], [0.5, 0.5])


def test_or_aggregate_chapter_probs_all_zero():
    """All-zero leaves → all-zero chapters."""
    leaf_probs = np.zeros((3, 4), dtype=np.float32)
    chap_probs = ehv.or_aggregate_chapter_probs(
        leaf_probs,
        disease_to_chapter_idx={0: 0, 1: 0, 2: 1, 3: 1},
        n_chapters=2,
    )
    assert chap_probs.shape == (3, 2)
    assert np.allclose(chap_probs, 0.0)


def test_or_aggregate_chapter_probs_all_one():
    """A single leaf at 1.0 saturates its chapter to 1.0."""
    leaf_probs = np.array([[1.0, 0.3, 0.2, 0.1]], dtype=np.float32)
    chap_probs = ehv.or_aggregate_chapter_probs(
        leaf_probs,
        disease_to_chapter_idx={0: 0, 1: 0, 2: 1, 3: 1},
        n_chapters=2,
    )
    assert np.isclose(chap_probs[0, 0], 1.0)
    # chap 1 = 1 - (1-0.2)(1-0.1) = 1 - 0.8*0.9 = 0.28
    assert np.isclose(chap_probs[0, 1], 0.28)


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
