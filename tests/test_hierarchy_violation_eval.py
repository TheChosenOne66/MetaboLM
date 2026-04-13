"""Tests for scripts/eval_hierarchy_violation.py."""

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
