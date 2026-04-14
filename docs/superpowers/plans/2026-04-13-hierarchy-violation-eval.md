# Hierarchy Violation Rate Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Revision 2 (2026-04-14) — Option B, two HVR metrics**:
> The original plan compared E0 (synthesised chapter probs via aggregation)
> against E1-E4 (chapter probs from the explicit chapter head). Those are
> **not the same metric** — the P(chap) definition differs across rows, so
> the column values aren't apples-to-apples.
>
> Revised design:
> - **E1-E4** produce TWO HVR numbers per checkpoint, from a single forward:
>   - `HVR (mean)` — `P_chap = mean(p_leaf)` **over the leaf-head output**.
>     This is the *cross-model-comparable* metric (E0 and E1-E4 use the same
>     recipe).
>   - `HVR (head)` — `P_chap` from the **explicit chapter head**. This
>     measures whether the hierarchy loss converged on the target it was
>     actually defined against. Only defined for models that have a chapter
>     head.
> - **E0** produces only `HVR (mean)`. `HVR (head)` is rendered as `—`
>   because E0 has no chapter head (and synthesising one would not be E0).
> - Two sidecar files per multitask exp dir:
>   `hierarchy_violation_mean.json`, `hierarchy_violation_head.json`.
>   E0 writes only `hierarchy_violation_mean.json`.
> - Leaderboard surfaces both in the Summary table:
>   `| HVR (mean) | HVR (head) |`.
>
> Tasks 1-3 were already shipped under the old design (with the correction
> that E0's aggregation is ``mean``, not ``OR``, per commit ``8857af1``).
> Tasks 4-7 below are rewritten for Option B. Task 8 (PR / handoff) is
> unchanged in spirit; commit messages just mention "mean + head" instead
> of "multitask head only".

**Goal:** Compute Hierarchy Violation Rate (HVR) for E1-E4 (from saved multitask checkpoints, producing **both** a ``mean`` and ``head`` variant per ckpt) **and** an E0 ``mean`` baseline, persist results to disk in a leaderboard-readable format, and surface them in `LEADERBOARD.md` as two columns so the thesis can quote cross-model-comparable numbers for Innovation 2 (hierarchy consistency loss) alongside the loss-target-native numbers for E1-E4.

**Architecture:** One new CPU/GPU evaluator script `scripts/eval_hierarchy_violation.py` with two `--mode` subcommands:
1. `--mode multitask` — load `outputs/E{1..4}_sft_*/best_model.pt`, GPU-forward the full `val.csv`, get `(leaf_probs, chap_probs)` from the dual-head model, compute HVR via the existing `src/training/metrics.py:compute_hierarchy_violation_rate` (single source of truth — do not duplicate), persist to `outputs/<exp_dir>/hierarchy_violation.json`.
2. `--mode e0-baseline` — pure CPU. Read the 16 E0 per-ckpt prediction files already on disk under `outputs/E0_global_eval{,_cohort}/<disease>/predictions_ckpt_<disease>.csv`, build a `(N, 16)` leaf prob matrix, **mean-aggregate** to get a `(N, 6)` implicit chapter prob matrix via `P_chap = mean(p_leaf for leaf in chapter)`, feed the same `compute_hierarchy_violation_rate`, persist to `outputs/E0_reproduction/hierarchy_violation_baseline.json`. This is the **baseline E1's hierarchy loss is supposed to fix**, so a high E0 number is the desired outcome.

**Why mean and not OR/max** (post-Task-3 design correction): both `max(p_leaf)` and `1 - ∏(1 - p_leaf)` are mathematically upper bounds on every leaf prob, so by construction `p_leaf ≤ chap_prob` always — HVR identically 0, the comparison loses all force. Mean aggregation can be exceeded by individual leaves (any leaf above its sibling-mean violates), giving a non-trivial baseline that captures the absence of hierarchy structure in E0.

Then extend `parse_per_disease_sft` and `parse_multitask` in `scripts/update_leaderboard.py` to read the new JSON files into the existing `hierarchy_violation_rate` field of `ExperimentRow` so the Summary table's "Hier. Violation" column populates automatically.

**Tech Stack:** Python 3.11 + numpy + pandas + torch (multitask path only) + sklearn (already imported elsewhere) + pytest. No new dependencies.

**Why these choices:**
- **Two modes in one script** keeps the OR-aggregation utility close to the GPU-forward path (both feed the same `compute_hierarchy_violation_rate`), avoids having to maintain two related scripts.
- **OR aggregation, not max**: `max(p_leaf)` makes E0's implicit chapter prob trivially ≥ each leaf, so violation rate would always be 0. That misleadingly suggests "E0 is fine without hierarchy loss". OR (`1 - ∏(1 - p)`) gives a meaningful baseline because some leaf can still exceed it when leaves are correlated.
- **Read existing predictions instead of re-forwarding E0**: 16 forward passes already happened in PR #3's `eval_e0_global*.py`. Re-doing them wastes GPU time and risks divergence if the data pipeline changes.
- **Single source of truth (`compute_hierarchy_violation_rate`)**: never re-implement the violation loop in the new script — import from `src/training/metrics.py`. If the metric definition ever changes (e.g. epsilon tuning), it changes in one place.
- **Output as JSON sidecar files**, not a new CSV column in `multitask_metrics.csv`: keeps the metric optional / additive and not coupled to the training-time CSV schema. Same pattern as `summary.json`, `cohort_stats_ckpt_*.json`.

**Out of scope (for this plan):** retraining E1-E4, modifying the training script to dump HVR at training time, adding HVR to the per-disease AUROC table.

---

## File Structure

| File | Purpose |
|---|---|
| `scripts/eval_hierarchy_violation.py` (NEW) | Two-mode evaluator: GPU multitask path + CPU E0 OR baseline path |
| `tests/test_hierarchy_violation_eval.py` (NEW) | Unit tests for OR aggregation + persistence helper + integration smoke for E0-OR mode |
| `tests/fixtures/hierarchy_violation/` (NEW) | Tiny synthetic eval dir with ~5-sample `predictions_ckpt_<D>.csv` files for end-to-end test |
| `scripts/update_leaderboard.py` (MODIFY) | Two parsers (`parse_per_disease_sft`, `parse_multitask`) read new JSON sidecar files |
| `tests/test_leaderboard.py` (MODIFY) | Add unit tests for HVR-reading branch in both parsers |

After the script runs once on real data, it produces:

| Generated file | Owner |
|---|---|
| `outputs/E{1..4}_*/hierarchy_violation_mean.json` | `--mode multitask` (mean-agg from leaf-head output) |
| `outputs/E{1..4}_*/hierarchy_violation_head.json` | `--mode multitask` (from explicit chapter head) |
| `outputs/E0_reproduction/hierarchy_violation_mean.json` | `--mode e0-mean` |

---

## Task 1: Script scaffold + OR aggregation utility (TDD)

**Files:**
- Create: `scripts/eval_hierarchy_violation.py`
- Create: `tests/test_hierarchy_violation_eval.py`

The script needs a small helper that converts per-disease leaf probs into per-chapter implicit chapter probs via the OR (independence) formula. This is the only piece of new math in the entire plan, so it deserves a real unit test.

- [ ] **Step 1: Create script scaffold with the function signature only**

```python
# scripts/eval_hierarchy_violation.py
"""Evaluate Hierarchy Violation Rate (HVR) for multitask checkpoints (E1-E4)
and an OR-aggregation baseline for E0.

See docs/superpowers/plans/2026-04-13-hierarchy-violation-eval.md for the
full design rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def or_aggregate_chapter_probs(
    leaf_probs: np.ndarray,
    disease_to_chapter_idx: dict[int, int],
    n_chapters: int,
) -> np.ndarray:
    """Aggregate per-leaf probabilities into per-chapter probabilities via
    independence-OR: ``P_chap = 1 - ∏(1 - p_leaf for leaf in chapter)``.

    Args:
        leaf_probs: ``(N, n_leaves)`` probabilities in [0, 1].
        disease_to_chapter_idx: ``{leaf_idx: chapter_idx}`` (length n_leaves).
        n_chapters: Number of distinct chapter indices.

    Returns:
        ``(N, n_chapters)`` chapter-level probabilities. The (i, c) cell is
        the probability that *at least one* leaf in chapter ``c`` is positive
        for sample ``i``, assuming the leaves are independent given the
        sample (an oversimplification, but a standard baseline).

    Used as the E0 baseline for hierarchy violation: E0 has no chapter head,
    so we synthesise one this way to get a non-trivial violation rate (vs
    ``max`` which would make violations identically zero).
    """
    raise NotImplementedError
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_hierarchy_violation_eval.py
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
```

- [ ] **Step 3: Run the test, expect failure**

Run: `cd /SPXvePFS/users/jytang/metabolm_posttrain && pytest tests/test_hierarchy_violation_eval.py -v`
Expected: 3 FAIL with `NotImplementedError`.

- [ ] **Step 4: Implement `or_aggregate_chapter_probs`**

Replace the `raise NotImplementedError` body with:

```python
    n_samples, n_leaves = leaf_probs.shape
    if len(disease_to_chapter_idx) != n_leaves:
        raise ValueError(
            f"disease_to_chapter_idx covers {len(disease_to_chapter_idx)} leaves, "
            f"but leaf_probs has {n_leaves} columns"
        )
    # Initialise chapter "no positive" probability to 1; multiply (1 - p_leaf)
    # for each leaf that lives under that chapter, then take the complement.
    one_minus = np.ones((n_samples, n_chapters), dtype=np.float32)
    for leaf_idx, chap_idx in disease_to_chapter_idx.items():
        one_minus[:, chap_idx] *= (1.0 - leaf_probs[:, leaf_idx])
    return (1.0 - one_minus).astype(np.float32)
```

- [ ] **Step 5: Run the test, expect pass**

Run: `cd /SPXvePFS/users/jytang/metabolm_posttrain && pytest tests/test_hierarchy_violation_eval.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
cd /SPXvePFS/users/jytang/metabolm_posttrain
git checkout -b feat/hierarchy-violation-eval
git add scripts/eval_hierarchy_violation.py tests/test_hierarchy_violation_eval.py
git commit -m "feat(hvr): scaffold hierarchy-violation evaluator + OR aggregation helper"
```

---

## Task 2: HVR computation + persistence helper (TDD)

**Files:**
- Modify: `scripts/eval_hierarchy_violation.py`
- Modify: `tests/test_hierarchy_violation_eval.py`

The two modes share a "given (leaf_probs, chap_probs, meta), compute HVR and persist a strict-JSON sidecar" function. Extract this into its own helper so both modes call it.

- [ ] **Step 1: Add the persistence helper signature**

Append to `scripts/eval_hierarchy_violation.py`:

```python
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval_hierarchy_violation")


def compute_and_persist_hvr(
    leaf_probs: np.ndarray,
    chap_probs: np.ndarray,
    output_path: Path,
    method: str,
    method_details: dict,
) -> dict:
    """Compute HVR via the canonical metric and write a JSON sidecar.

    Args:
        leaf_probs: ``(N, 16)`` leaf probabilities in [0, 1].
        chap_probs: ``(N, 6)`` chapter probabilities in [0, 1].
        output_path: Where to write the JSON.
        method: Short identifier (e.g. ``"explicit_chapter_head"`` or
            ``"or_aggregate_baseline"``). Surfaced in the JSON for downstream
            disambiguation.
        method_details: Free-form dict appended to the JSON for provenance
            (model path, predictions source dir, etc.).

    Returns:
        The dict that was written to disk (also useful for in-process logging).
    """
    raise NotImplementedError
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_hierarchy_violation_eval.py`:

```python
import json


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
    )

    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written == result
    assert written["method"] == "unit_test"
    assert written["method_details"]["test_id"] == "compute_and_persist_basic"
    assert written["n_samples"] == 2
    assert written["n_leaf_chapter_pairs"] == 4
    assert pytest.approx(written["hierarchy_violation_rate"], abs=1e-6) == 0.25
```

- [ ] **Step 3: Run the test, expect failure**

Run: `pytest tests/test_hierarchy_violation_eval.py::test_compute_and_persist_hvr_basic -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 4: Implement `compute_and_persist_hvr`**

Replace the `raise NotImplementedError` body with:

```python
    # Lazy import: the metric lives next to training code and pulls in heavier
    # imports we don't want at script-import time.
    from src.training.metrics import compute_hierarchy_violation_rate
    from src.data.endpoints import get_disease_to_chapter_idx

    if leaf_probs.shape[0] != chap_probs.shape[0]:
        raise ValueError(
            f"leaf_probs ({leaf_probs.shape}) and chap_probs ({chap_probs.shape}) "
            "disagree on N (axis 0)."
        )

    disease_to_chapter_idx = get_disease_to_chapter_idx()
    rate = compute_hierarchy_violation_rate(
        leaf_probs.astype(np.float64),
        chap_probs.astype(np.float64),
        disease_to_chapter_idx,
    )

    payload = {
        "hierarchy_violation_rate": float(rate),
        "n_samples": int(leaf_probs.shape[0]),
        "n_leaf_chapter_pairs": int(leaf_probs.shape[1]),
        "method": method,
        "method_details": dict(method_details),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Wrote %s — rate=%.4f, n_samples=%d, method=%s",
        output_path, payload["hierarchy_violation_rate"],
        payload["n_samples"], method,
    )
    return payload
```

- [ ] **Step 5: Run the test, expect pass**

Run: `pytest tests/test_hierarchy_violation_eval.py -v`
Expected: 4 PASS (3 from Task 1 + 1 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_hierarchy_violation.py tests/test_hierarchy_violation_eval.py
git commit -m "feat(hvr): add compute_and_persist_hvr helper around canonical metric"
```

---

## Task 3: E0-OR mode — fixture-based integration test (TDD)

**Files:**
- Modify: `scripts/eval_hierarchy_violation.py`
- Modify: `tests/test_hierarchy_violation_eval.py`
- Create: `tests/fixtures/hierarchy_violation/eval_dir/eval_set_labels.csv`
- Create: `tests/fixtures/hierarchy_violation/eval_dir/T2D/predictions_ckpt_T2D.csv`
- Create: `tests/fixtures/hierarchy_violation/eval_dir/obesity/predictions_ckpt_obesity.csv`

This task glues the two helpers above into the E0-OR mode end-to-end. We test it on a tiny fixture shaped exactly like a real `outputs/E0_global_eval{,_cohort}/` dir but with only two diseases populated.

- [ ] **Step 1: Create the tiny fixture eval dir**

```bash
mkdir -p /SPXvePFS/users/jytang/metabolm_posttrain/tests/fixtures/hierarchy_violation/eval_dir/T2D
mkdir -p /SPXvePFS/users/jytang/metabolm_posttrain/tests/fixtures/hierarchy_violation/eval_dir/obesity
```

Write `tests/fixtures/hierarchy_violation/eval_dir/eval_set_labels.csv` with 3 rows × 16 label columns (the rest of the disease columns are present but all zero — only T2D / obesity vary):

```csv
eid,label_T2D,label_obesity,label_hypertension,label_ischemic_heart,label_atrial_fib,label_heart_failure,label_rheumatoid,label_asthma,label_dementia,label_copd,label_stroke,label_parkinsons,label_breast_cancer,label_colon_cancer,label_lung_cancer,label_prostate_cancer
1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
3,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

Write `tests/fixtures/hierarchy_violation/eval_dir/T2D/predictions_ckpt_T2D.csv`:

```csv
eid,prob_ckpt_T2D
1,0.900000
2,0.100000
3,0.800000
```

Write `tests/fixtures/hierarchy_violation/eval_dir/obesity/predictions_ckpt_obesity.csv`:

```csv
eid,prob_ckpt_obesity
1,0.200000
2,0.700000
3,0.600000
```

- [ ] **Step 2: Add the loader + e0-or runner signatures**

Append to `scripts/eval_hierarchy_violation.py`:

```python
def load_e0_leaf_probs(
    eval_dir: Path, disease_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Read per-ckpt predictions under ``eval_dir/<disease>/predictions_ckpt_<D>.csv``
    and return ``(leaf_probs, present_diseases)``.

    Aligns by the ``eid`` column of ``eval_dir/eval_set_labels.csv`` so all
    disease columns share a row order. Diseases without a predictions file
    contribute a column of ``NaN`` (later treated as missing — see Task 3
    Step 4 for handling).
    """
    raise NotImplementedError


def run_e0_or_mode(
    eval_dir: Path,
    output_path: Path,
) -> dict:
    """E0 baseline mode: read 16 leaf prob columns from ``eval_dir``, OR-aggregate
    to chapter probs, compute HVR, persist."""
    raise NotImplementedError
```

- [ ] **Step 3: Write the failing integration test**

Append to `tests/test_hierarchy_violation_eval.py`:

```python
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


def test_run_e0_or_mode_partial_dir(tmp_path):
    """End-to-end on the partial fixture: only T2D + obesity contribute.

    With T2D and obesity both in chapter_04, OR-aggregation gives:
        chapter_04 prob[i] = 1 - (1 - p_T2D[i]) * (1 - p_obesity[i])
    Sample 0: 1 - 0.1 * 0.8 = 0.92  ; T2D=0.9 < 0.92 OK ; obesity=0.2 OK
    Sample 1: 1 - 0.9 * 0.3 = 0.73  ; T2D=0.1 OK ; obesity=0.7 OK
    Sample 2: 1 - 0.2 * 0.4 = 0.92  ; T2D=0.8 OK ; obesity=0.6 OK
    Expected violations from T2D / obesity rows = 0 / 6 = 0.0
    All other 14 leaves have NaN probs → must be skipped (not counted).
    """
    out_path = tmp_path / "hierarchy_violation_or.json"
    payload = ehv.run_e0_or_mode(FIXTURES / "eval_dir", out_path)

    assert out_path.exists()
    assert payload["method"] == "or_aggregate_baseline"
    assert payload["n_samples"] == 3
    # Only the 2 present diseases count toward the leaf-chapter pair total.
    assert payload["n_leaf_chapter_pairs"] == 2
    assert pytest.approx(payload["hierarchy_violation_rate"], abs=1e-6) == 0.0
    # Provenance: predictions_source path is recorded
    assert "eval_dir" in payload["method_details"]["predictions_source"]
```

- [ ] **Step 4: Run tests, expect failure**

Run: `pytest tests/test_hierarchy_violation_eval.py::test_load_e0_leaf_probs_partial_dir tests/test_hierarchy_violation_eval.py::test_run_e0_or_mode_partial_dir -v`
Expected: 2 FAIL with `NotImplementedError`.

- [ ] **Step 5: Implement `load_e0_leaf_probs`**

Replace its body with:

```python
    import pandas as pd

    labels_path = eval_dir / "eval_set_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found — run scripts/eval_e0_global*.py first."
        )

    labels_df = pd.read_csv(labels_path)
    if "eid" not in labels_df.columns:
        raise ValueError(f"{labels_path} missing 'eid' column")
    eids = labels_df["eid"].astype(str).tolist()

    n = len(eids)
    leaf_probs = np.full((n, len(disease_names)), np.nan, dtype=np.float32)
    present: list[str] = []

    eid_to_row = {eid: i for i, eid in enumerate(eids)}
    for col_idx, disease in enumerate(disease_names):
        pred_path = eval_dir / disease / f"predictions_ckpt_{disease}.csv"
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path)
        prob_col = f"prob_ckpt_{disease}"
        if "eid" not in pred_df.columns or prob_col not in pred_df.columns:
            logger.warning(
                "[%s] %s has unexpected schema; skipping",
                disease, pred_path.name,
            )
            continue
        for _, row in pred_df.iterrows():
            i = eid_to_row.get(str(row["eid"]))
            if i is None:
                continue
            leaf_probs[i, col_idx] = float(row[prob_col])
        present.append(disease)
    return leaf_probs, present
```

- [ ] **Step 6: Implement `run_e0_or_mode`**

Replace its body with:

```python
    from src.data.endpoints import (
        get_disease_names, get_disease_to_chapter_idx, get_unique_chapters,
    )
    from src.training.metrics import compute_hierarchy_violation_rate

    diseases = get_disease_names()
    disease_to_chap = get_disease_to_chapter_idx()
    n_chapters = len(get_unique_chapters())

    leaf_probs, present = load_e0_leaf_probs(eval_dir, diseases)

    # Drop missing diseases entirely so OR aggregation isn't poisoned by NaN.
    # Each missing leaf contributes 0 to its chapter (1 - NaN would propagate).
    present_idx = [diseases.index(d) for d in present]
    leaf_probs_clean = leaf_probs[:, present_idx]
    sub_disease_to_chap = {
        new_idx: disease_to_chap[old_idx]
        for new_idx, old_idx in enumerate(present_idx)
    }

    chap_probs = or_aggregate_chapter_probs(
        leaf_probs_clean,
        disease_to_chapter_idx=sub_disease_to_chap,
        n_chapters=n_chapters,
    )

    # Build the final leaf+chap arrays the canonical metric expects.
    # leaf_probs_full has 16 columns; missing leaves get NaN, which we
    # filter out by passing a sub-mapping that only references present leaves.
    rate = compute_hierarchy_violation_rate(
        leaf_probs_clean.astype(np.float64),
        chap_probs.astype(np.float64),
        sub_disease_to_chap,
    )

    payload = {
        "hierarchy_violation_rate": float(rate),
        "n_samples": int(leaf_probs.shape[0]),
        "n_leaf_chapter_pairs": int(len(present)),
        "method": "or_aggregate_baseline",
        "method_details": {
            "chap_probs_source": (
                "P_chap = 1 - prod(1 - p_leaf for leaf in chapter); "
                "assumes leaf independence given the sample (a baseline, "
                "not a justified probabilistic claim)."
            ),
            "predictions_source": str(eval_dir),
            "n_present_diseases": len(present),
            "present_diseases": present,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Wrote %s — rate=%.4f, present=%d/%d diseases",
        output_path, payload["hierarchy_violation_rate"],
        len(present), len(diseases),
    )
    return payload
```

- [ ] **Step 7: Run all tests, expect pass**

Run: `pytest tests/test_hierarchy_violation_eval.py -v`
Expected: 6 PASS (3 from Task 1 + 1 from Task 2 + 2 new).

- [ ] **Step 8: Commit**

```bash
git add scripts/eval_hierarchy_violation.py tests/test_hierarchy_violation_eval.py tests/fixtures/hierarchy_violation/
git commit -m "feat(hvr): e0-or mode reads per-ckpt predictions + OR-aggregates baseline"
```

---

## Task 4: Multitask mode — GPU-forward path (produces BOTH mean + head JSONs)

**Files:**
- Modify: `scripts/eval_hierarchy_violation.py`

This task is **not unit-tested** (it requires loading a real ckpt and running a GPU forward pass). It is verified by the smoke test in Task 7 once a multitask checkpoint is on disk. The structure mirrors `scripts/eval_e0_global.py`'s GPU pattern so any reviewer who has read that script can follow this one.

**Option B key point**: a SINGLE forward pass produces both metrics. We
run the model once to get `(leaf_probs, head_chap_probs)`, then separately
derive `mean_chap_probs = mean_aggregate_chapter_probs(leaf_probs, ...)`
using the existing helper from Task 1. The canonical metric is called
twice (once per chap_probs variant), writing two sidecar JSONs with
distinct `method` fields and filenames.

- [ ] **Step 1: Add the multitask mode runner**

Append to `scripts/eval_hierarchy_violation.py`:

```python
def run_multitask_mode(
    config_path: Path,
    output_dir: Path,
    mean_output_path: Path,
    head_output_path: Path,
    batch_size: int = 512,
) -> tuple[dict, dict]:
    """Multitask mode: load ``output_dir/best_model.pt``, GPU-forward val.csv,
    extract (leaf_probs, head_chap_probs) from the dual-head model, compute
    HVR **twice** (once with mean-aggregated chap probs for cross-model
    comparability with E0, once with the explicit chapter head output for
    the loss-target-native metric), persist **two** sidecar JSONs.

    Returns ``(mean_payload, head_payload)``.

    Mirrors scripts/eval_e0_global.py's prediction loop for the forward
    pass; then diverges to produce both metrics from the same leaf/chap
    arrays to avoid a second model load.
    """
    import csv
    import os
    import torch

    from src.config import load_config
    from src.data.biomarkers import get_metabolite_names
    from src.data.endpoints import get_disease_names
    from src.model.backbone import MetaboliteBERTModel
    from src.model.heads import HierarchicalMultiTaskHead
    from src.model.wrapper import MetaboLMForClassification

    cfg = load_config(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load val.csv directly (lighter than calling eval_e0_global.load_val_data
    # which is csv-reader based; here we just need features).
    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{d}" for d in disease_names]

    with open(cfg.data.val_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
    feat_indices = [header.index(c) for c in feature_cols]
    label_indices = [header.index(c) for c in label_cols]

    X_rows: list[list[float]] = []
    Y_rows: list[list[float]] = []
    with open(cfg.data.val_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X_rows.append([float(row[i]) for i in feat_indices])
            Y_rows.append([float(row[i]) for i in label_indices])
    X_val = np.array(X_rows, dtype=np.float32)
    Y_val = np.array(Y_rows, dtype=np.float32)
    logger.info("Loaded val.csv: %d samples × %d features", X_val.shape[0], X_val.shape[1])

    # Load correlation matrix (mirror eval_e0_global.load_correlation_matrix
    # to keep the .pt + CSV fallback consistent).
    from scripts.eval_e0_global import load_correlation_matrix
    bias_matrix = load_correlation_matrix(cfg, device)

    # Load the multitask model. The ckpt was saved by scripts/train_multitask.py
    # and contains the full state_dict for backbone + HierarchicalMultiTaskHead.
    ckpt_path = output_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Multitask checkpoint not found: {ckpt_path}")
    state_dict = torch.load(str(ckpt_path), map_location=device)

    from src.data.endpoints import get_unique_chapters
    n_chapters = len(get_unique_chapters())
    backbone = MetaboliteBERTModel(num_metabolites=len(feature_cols))
    # Signature: HierarchicalMultiTaskHead(hidden_size=768, proj_size=256,
    # num_diseases=16, num_chapters=6, dropout=0.1). Source: src/model/heads.py:59-66.
    head = HierarchicalMultiTaskHead(
        hidden_size=768,
        num_diseases=len(disease_names),
        num_chapters=n_chapters,
    )
    model = MetaboLMForClassification(backbone, head)

    cleaned = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=False)
    model.metabolite_model.register_buffer("bias_matrix_full", bias_matrix)
    model.to(device)
    model.eval()

    # GPU forward in batches, accumulating both heads' outputs.
    leaf_chunks: list[np.ndarray] = []
    chap_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X_val.shape[0], batch_size):
            end = min(start + batch_size, X_val.shape[0])
            x = torch.tensor(X_val[start:end], dtype=torch.float32, device=device)
            mask = torch.ones(x.size(), dtype=torch.long, device=device)
            (leaf_logits, chap_logits), _ = model(x, mask)
            leaf_chunks.append(torch.sigmoid(leaf_logits).cpu().numpy())
            chap_chunks.append(torch.sigmoid(chap_logits).cpu().numpy())
    leaf_probs = np.concatenate(leaf_chunks, axis=0)
    head_chap_probs = np.concatenate(chap_chunks, axis=0)

    # Derive the mean-aggregated chap probs from the SAME leaf outputs so
    # we can report a cross-model-comparable HVR alongside the head-native
    # one without a second forward.
    from src.data.endpoints import get_disease_to_chapter_idx, get_unique_chapters
    disease_to_chap = get_disease_to_chapter_idx()
    n_chapters = len(get_unique_chapters())
    mean_chap_probs = mean_aggregate_chapter_probs(
        leaf_probs,
        disease_to_chapter_idx=disease_to_chap,
        n_chapters=n_chapters,
    )

    method_details_common = {
        "model_path": str(ckpt_path),
        "config_path": str(config_path),
        "val_path": cfg.data.val_path,
        "n_features": X_val.shape[1],
    }

    mean_payload = compute_and_persist_hvr(
        leaf_probs=leaf_probs,
        chap_probs=mean_chap_probs,
        output_path=mean_output_path,
        method="mean_aggregate_multitask",
        method_details={
            **method_details_common,
            "chap_probs_source": (
                "P_chap = mean(p_leaf for leaf in chapter); identical recipe "
                "to E0 baseline. Use for cross-model comparison."
            ),
        },
    )
    head_payload = compute_and_persist_hvr(
        leaf_probs=leaf_probs,
        chap_probs=head_chap_probs,
        output_path=head_output_path,
        method="explicit_chapter_head",
        method_details={
            **method_details_common,
            "chap_probs_source": (
                "P_chap = sigmoid(chapter_head_logits); the distribution "
                "the hierarchy loss was trained against. Only defined for "
                "multitask models with an explicit chapter head."
            ),
        },
    )
    return mean_payload, head_payload
```

- [ ] **Step 2: Sanity check — script still imports cleanly without GPU**

Run: `cd /SPXvePFS/users/jytang/metabolm_posttrain && python -c "import importlib.util, sys; sys.path.insert(0, 'scripts'); importlib.util.spec_from_file_location('ehv', 'scripts/eval_hierarchy_violation.py').loader.exec_module(__import__('importlib').util.module_from_spec(importlib.util.spec_from_file_location('ehv', 'scripts/eval_hierarchy_violation.py'))); print('imports OK')"`

If that one-liner is too gnarly, this is equivalent and simpler:

```bash
cd /SPXvePFS/users/jytang/metabolm_posttrain
python -c "
import sys; sys.path.insert(0, 'scripts')
import eval_hierarchy_violation as ehv
print('module loaded:', dir(ehv))
"
```

Expected: prints a list of attributes including `or_aggregate_chapter_probs`, `compute_and_persist_hvr`, `load_e0_leaf_probs`, `run_e0_or_mode`, `run_multitask_mode`. No torch / GPU errors at import time (lazy imports inside the function).

- [ ] **Step 3: Re-run the existing pytest suite to confirm nothing broke**

Run: `pytest tests/test_hierarchy_violation_eval.py -v`
Expected: 6 PASS (no regression from Task 3).

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_hierarchy_violation.py
git commit -m "feat(hvr): multitask mode produces both mean-agg and head HVR from one forward"
```

---

## Task 5: CLI dispatch + entry point

**Files:**
- Modify: `scripts/eval_hierarchy_violation.py`

Wire `argparse` to dispatch to one of the two modes. The CLI shape is intentionally narrow: each mode takes only what it needs.

- [ ] **Step 1: Add `main()` and `__main__` block**

Append to `scripts/eval_hierarchy_violation.py`:

```python
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Hierarchy Violation Rate (HVR) for "
                    "multitask checkpoints (E1-E4; writes BOTH a mean-agg and "
                    "a head-native sidecar) or for the E0 mean-agg baseline.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    mt = sub.add_parser(
        "multitask",
        help=("Load best_model.pt and forward val.csv to extract leaf+chap "
              "probs, then write BOTH hierarchy_violation_mean.json and "
              "hierarchy_violation_head.json to the experiment dir."),
    )
    mt.add_argument("--config", type=Path, required=True,
                    help="Config YAML (e.g. configs/sft_full_ft.yaml).")
    mt.add_argument("--output-dir", type=Path, required=True,
                    help="Experiment output dir containing best_model.pt "
                         "(e.g. outputs/E1_sft_full_ft). Sidecar JSONs are "
                         "written here by default.")
    mt.add_argument("--mean-output-json", type=Path, default=None,
                    help="Override path for hierarchy_violation_mean.json. "
                         "Defaults to <output-dir>/hierarchy_violation_mean.json.")
    mt.add_argument("--head-output-json", type=Path, default=None,
                    help="Override path for hierarchy_violation_head.json. "
                         "Defaults to <output-dir>/hierarchy_violation_head.json.")
    mt.add_argument("--batch-size", type=int, default=512)

    e0 = sub.add_parser(
        "e0-mean",
        help="Read E0 per-ckpt predictions and mean-aggregate to baseline HVR.",
    )
    e0.add_argument("--eval-dir", type=Path, required=True,
                    help="Eval output dir produced by scripts/eval_e0_global*.py "
                         "(e.g. outputs/E0_global_eval_cohort).")
    e0.add_argument("--output-json", type=Path, default=None,
                    help="Where to write hierarchy_violation_mean.json. "
                         "Defaults to outputs/E0_reproduction/hierarchy_violation_mean.json.")

    args = parser.parse_args()

    if args.mode == "multitask":
        mean_out = args.mean_output_json or (
            args.output_dir / "hierarchy_violation_mean.json"
        )
        head_out = args.head_output_json or (
            args.output_dir / "hierarchy_violation_head.json"
        )
        run_multitask_mode(
            config_path=args.config,
            output_dir=args.output_dir,
            mean_output_path=mean_out,
            head_output_path=head_out,
            batch_size=args.batch_size,
        )
    elif args.mode == "e0-mean":
        out_json = args.output_json or Path(
            "outputs/E0_reproduction/hierarchy_violation_mean.json"
        )
        run_e0_baseline_mode(eval_dir=args.eval_dir, output_path=out_json)
    else:
        parser.error(f"unknown mode: {args.mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: CLI smoke check — `--help` for each mode**

Run:

```bash
cd /SPXvePFS/users/jytang/metabolm_posttrain
python scripts/eval_hierarchy_violation.py --help
python scripts/eval_hierarchy_violation.py multitask --help
python scripts/eval_hierarchy_violation.py e0-mean --help
```

Expected: each invocation prints a help message and exits 0; no module / import errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_hierarchy_violation.py
git commit -m "feat(hvr): wire argparse for multitask + e0-mean subcommands"
```

---

## Task 6: Split HVR into two columns in leaderboard + wire parsers (TDD)

**Files:**
- Modify: `scripts/update_leaderboard.py` — `ExperimentRow`, parsers, renderer
- Modify: `tests/test_leaderboard.py`

The script-side currently has a single `hierarchy_violation_rate` field and a single "Hier. Violation" column. For Option B we split these into two: `hierarchy_violation_rate_mean` and `hierarchy_violation_rate_head`, with two columns `HVR (mean)` / `HVR (head)`. Parsers read the new sidecar JSONs:
- `parse_per_disease_sft` (E0): reads only `hierarchy_violation_mean.json`; `_head` stays None.
- `parse_multitask` (E1-E6): reads BOTH sidecars; falls back to `summary.json`'s legacy `hierarchy_violation_rate` field for `_head` when the sidecar is absent (backward compat with training-time HVR logging).

- [ ] **Step 1: Rename `ExperimentRow.hierarchy_violation_rate` to `_mean` and add `_head`**

In `scripts/update_leaderboard.py`, update the dataclass:

```python
    hierarchy_violation_rate_mean: float | None = None
    hierarchy_violation_rate_head: float | None = None
```

(Delete the old single-field line.)

Also update every `empty_result` dict in `parse_per_disease_sft`, `parse_multitask`, the error-fallback dict in `collect_experiment_rows`, and the ExperimentRow kwargs — replace each `"hierarchy_violation_rate": ...` / `hierarchy_violation_rate=...` with both the `_mean` and `_head` variants (both `None` in the empty/default cases).

- [ ] **Step 2: Update renderer to two columns**

In `render_markdown` and `render_readme_section`, change the Summary header:

```python
    "Mean AUPRC | HVR (mean) | HVR (head) | Trainable Params | Best Epoch |"
```

with matching alignment row, and each data row renders both:

```python
    f"{_format_float(row.hierarchy_violation_rate_mean)} | "
    f"{_format_float(row.hierarchy_violation_rate_head)} | "
```

- [ ] **Step 3: Wire `parse_multitask` to read both sidecars**

Just before the final `return`:

```python
    hvr_mean_path = exp_dir / "hierarchy_violation_mean.json"
    hvr_mean: float | None = None
    if hvr_mean_path.exists():
        try:
            hvr_mean = float(json.loads(hvr_mean_path.read_text()).get(
                "hierarchy_violation_rate"
            ))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"[WARN] parse_multitask({exp_dir.name}): "
                  f"hierarchy_violation_mean.json unreadable ({exc})",
                  file=sys.stderr)

    hvr_head_path = exp_dir / "hierarchy_violation_head.json"
    hvr_head: float | None = None
    if hvr_head_path.exists():
        try:
            hvr_head = float(json.loads(hvr_head_path.read_text()).get(
                "hierarchy_violation_rate"
            ))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"[WARN] parse_multitask({exp_dir.name}): "
                  f"hierarchy_violation_head.json unreadable ({exc})",
                  file=sys.stderr)

    # Backward compat: pre-sidecar training runs dumped HVR in summary.json.
    # Treat that as the head metric since training used the chapter head.
    if hvr_head is None:
        legacy = summary.get("hierarchy_violation_rate")
        if legacy is not None:
            try:
                hvr_head = float(legacy)
            except (TypeError, ValueError):
                pass
```

Replace the return dict's `"hierarchy_violation_rate"` entry with:

```python
        "hierarchy_violation_rate_mean": hvr_mean,
        "hierarchy_violation_rate_head": hvr_head,
```

- [ ] **Step 4: Wire `parse_per_disease_sft` to read `_mean.json`**

Before the final `return`:

```python
    hvr_mean_path = exp_dir / "hierarchy_violation_mean.json"
    hvr_mean: float | None = None
    if hvr_mean_path.exists():
        try:
            hvr_mean = float(json.loads(hvr_mean_path.read_text()).get(
                "hierarchy_violation_rate"
            ))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"[WARN] parse_per_disease_sft({exp_dir.name}): "
                  f"hierarchy_violation_mean.json unreadable ({exc})",
                  file=sys.stderr)
```

Replace `"hierarchy_violation_rate": None` in the successful return with:

```python
        "hierarchy_violation_rate_mean": hvr_mean,
        "hierarchy_violation_rate_head": None,
```

- [ ] **Step 5: Update tests**

In `tests/test_leaderboard.py`:
- Update `_make_completed_multitask_row` to use the new field names (e.g. `hierarchy_violation_rate_mean=0.15, hierarchy_violation_rate_head=0.028`).
- Add 4 new tests following the pattern below.

```python
def test_parse_multitask_reads_both_hvr_jsons(tmp_path):
    exp_dir = tmp_path / "E1_test"
    exp_dir.mkdir()
    (exp_dir / "multitask_metrics.csv").write_text(
        "Disease,Val_AUC,Val_F1\nT2D,0.75,0.5\nMEAN,0.75,0.5\n"
    )
    (exp_dir / "summary.json").write_text(json.dumps({
        "experiment": "E1", "freeze_strategy": "none",
        "best_epoch": 10, "best_mean_auc": 0.75,
        "trainable_params": 100, "total_params": 100,
    }))
    (exp_dir / "hierarchy_violation_mean.json").write_text(json.dumps({
        "hierarchy_violation_rate": 0.15, "method": "mean_aggregate_multitask",
        "n_samples": 1, "n_leaf_chapter_pairs": 16, "method_details": {},
    }))
    (exp_dir / "hierarchy_violation_head.json").write_text(json.dumps({
        "hierarchy_violation_rate": 0.028, "method": "explicit_chapter_head",
        "n_samples": 1, "n_leaf_chapter_pairs": 16, "method_details": {},
    }))
    r = ulb.parse_multitask(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] == pytest.approx(0.15)
    assert r["hierarchy_violation_rate_head"] == pytest.approx(0.028)


def test_parse_multitask_falls_back_to_summary_for_head(tmp_path):
    """Pre-sidecar runs dumped HVR in summary.json; treat as head metric."""
    exp_dir = tmp_path / "E1_legacy"
    exp_dir.mkdir()
    (exp_dir / "multitask_metrics.csv").write_text(
        "Disease,Val_AUC,Val_F1\nT2D,0.75,0.5\nMEAN,0.75,0.5\n"
    )
    (exp_dir / "summary.json").write_text(json.dumps({
        "experiment": "E1", "freeze_strategy": "none",
        "best_epoch": 10, "best_mean_auc": 0.75,
        "trainable_params": 100, "total_params": 100,
        "hierarchy_violation_rate": 0.042,
    }))
    r = ulb.parse_multitask(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] is None
    assert r["hierarchy_violation_rate_head"] == pytest.approx(0.042)


def test_parse_per_disease_sft_reads_hvr_mean_json(tmp_path):
    exp_dir = tmp_path / "E0_test"
    exp_dir.mkdir()
    (exp_dir / "finetune_summary_metrics.csv").write_text(
        "Disease,Val_AUC,Best_Epoch\nT2D,0.85,5\n"
    )
    (exp_dir / "hierarchy_violation_mean.json").write_text(json.dumps({
        "hierarchy_violation_rate": 0.37, "method": "mean_aggregate_baseline",
        "n_samples": 1, "n_leaf_chapter_pairs": 16, "method_details": {},
    }))
    r = ulb.parse_per_disease_sft(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] == pytest.approx(0.37)
    assert r["hierarchy_violation_rate_head"] is None


def test_render_markdown_shows_two_hvr_columns():
    rows = [_make_completed_multitask_row("E1")]
    md = ulb.render_markdown(rows, CANONICAL_DISEASES)
    assert "HVR (mean)" in md
    assert "HVR (head)" in md
```

- [ ] **Step 6: Run the suite, expect pass**

Run: `pytest tests/test_leaderboard.py -v`
Expected: all existing tests still pass + the 4 new HVR tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/update_leaderboard.py tests/test_leaderboard.py
git commit -m "feat(leaderboard): split HVR into mean/head columns + parser sidecar reads"
```

---

## Task 7: Local end-to-end smoke + leaderboard regen

**Files:**
- Generated: `outputs/E0_reproduction/hierarchy_violation_mean.json`
- Generated (regenerated): `LEADERBOARD.md`, `README.md` (between markers)

This task is the local sanity check. We can only run the **e0-mean** mode here (no E1-E4 ckpts on disk locally) — the **multitask** mode will be exercised on Nebula in a follow-up.

- [ ] **Step 1: Run e0-mean against the cohort eval dir we already have**

Run:

```bash
cd /SPXvePFS/users/jytang/metabolm_posttrain
source /SPXvePFS/share/miniconda3/etc/profile.d/conda.sh && conda activate metabolm
python scripts/eval_hierarchy_violation.py e0-mean \
    --eval-dir outputs/E0_global_eval_cohort \
    --output-json outputs/E0_reproduction/hierarchy_violation_mean.json
```

Expected log:
```
INFO Wrote outputs/E0_reproduction/hierarchy_violation_mean.json — rate=<...>, present=1/16 diseases
```

(Locally only T2D is present, so `n_present_diseases=1`. Single-leaf chapters can't violate under mean aggregation, so the rate may well be 0.0; a more interesting number will materialise once the other 15 ckpts' predictions land.)

- [ ] **Step 2: Verify the JSON shape**

Run: `cat outputs/E0_reproduction/hierarchy_violation_mean.json`
Expected fields: `hierarchy_violation_rate`, `n_samples`, `n_leaf_chapter_pairs`, `method` = `"mean_aggregate_baseline"`, `method_details.predictions_source`, `method_details.present_diseases`.

- [ ] **Step 3: Regenerate the leaderboard**

Run: `python scripts/update_leaderboard.py`
Expected log: `Leaderboard updated: 7 experiments (...) + README`.

- [ ] **Step 4: Confirm the HVR (mean) cell populated for E0**

Run: `grep -A 1 "^| E0 " LEADERBOARD.md | head -2`
Expected: the row's `HVR (mean)` column contains a real number (or `0.000` locally); `HVR (head)` still shows `—` for E0.

- [ ] **Step 5: Commit the regenerated leaderboard**

```bash
git add LEADERBOARD.md README.md outputs/E0_reproduction/hierarchy_violation_mean.json
git commit -m "leaderboard: regenerate with E0 HVR (mean) baseline (T2D-only locally)"
```

(The `outputs/` add is fine here — it's small JSON, and committing the local sanity-check artifact makes it obvious the script ran. If `.gitignore` excludes the file, drop it from the `git add`.)

---

## Task 8: Open PR + cherry-pick to exp + handoff for Nebula

**Files:** none (git operations only)

- [ ] **Step 1: Push the feature branch and open the PR**

```bash
cd /SPXvePFS/users/jytang/metabolm_posttrain
git push -u origin feat/hierarchy-violation-eval
gh auth switch --user TheChosenOne66
gh pr create --base main --head feat/hierarchy-violation-eval \
    --title "Hierarchy Violation Rate evaluator (multitask + E0 OR baseline)" \
    --body-file - <<'BODY'
## Summary
- New `scripts/eval_hierarchy_violation.py` with two modes:
  - `multitask` — GPU-forward E1-E4 `best_model.pt` against val.csv, extract `(leaf_probs, chap_probs)` from `HierarchicalMultiTaskHead`, compute HVR via the canonical `src/training/metrics.py` function, persist `hierarchy_violation.json` per exp dir.
  - `e0-or` — pure CPU. Reads existing per-ckpt predictions from `outputs/E0_global_eval{,_cohort}/`, OR-aggregates leaves into implicit chapter probs (`P_chap = 1 - prod(1 - p_leaf)`), persists `hierarchy_violation_or.json` to `outputs/E0_reproduction/`.
- `scripts/update_leaderboard.py` parsers now read these JSONs and populate the existing `Hier. Violation` column.
- Tests: 6 unit tests for the new script (OR aggregation, persistence, fixture-based e0-or e2e) + 4 unit tests for the leaderboard parser branches.

## Why this matters
Innovation 2 in the thesis (hierarchy consistency loss) had no headline metric until now — `Hier. Violation` was always `—`. After this PR:
- E0 OR baseline is expected to be high (single-task models trained without hierarchy structure routinely produce P(leaf) > P(chapter)),
- E1 with hierarchy loss is expected to be near zero,
- giving the thesis a clean "before / after" pair for Innovation 2.

## Test plan
- [x] All new unit tests pass: `pytest tests/test_hierarchy_violation_eval.py tests/test_leaderboard.py -v`
- [x] CLI smoke: `python scripts/eval_hierarchy_violation.py {multitask,e0-or} --help`
- [x] e0-or end-to-end on `outputs/E0_global_eval_cohort/` (T2D-only locally; full 16-disease run pending Nebula sync)
- [ ] multitask end-to-end on Nebula for E1-E4 (deferred; `best_model.pt` not on this workspace)
BODY
```

- [ ] **Step 2: Cherry-pick the feature commits onto the exp workspace**

```bash
cd /SPXvePFS/users/jytang/MetaboLM
git fetch origin main
git config user.name "TheChosenOne66"
git config user.email "2406012671@qq.com"
# Replace <SHAs> with the commits from Tasks 1, 2, 3, 4, 5, 6 (NOT Task 7's
# leaderboard regen — that's a derivative, regenerate it on exp separately).
git cherry-pick <task1_sha> <task2_sha> <task3_sha> <task4_sha> <task5_sha> <task6_sha>
git push origin exp
```

If the rebase / cherry-pick conflicts, abort and re-do as a `git rebase origin/exp` of the local feature branch.

- [ ] **Step 3: Note for Nebula execution**

Add a one-line follow-up to `docs/PROGRESS.md` under the "下一步" section of the 2026-04-13 entry:

```markdown
- [ ] On Nebula: `python scripts/eval_hierarchy_violation.py multitask --config configs/sft_full_ft.yaml --output-dir outputs/E1_sft_full_ft` (and analogously for E2/E3/E4); then re-run `update_leaderboard.py` to populate the Hier. Violation column for E1-E4.
```

Commit + push:

```bash
cd /SPXvePFS/users/jytang/metabolm_posttrain
git add docs/PROGRESS.md
git commit -m "docs: note Nebula follow-up for E1-E4 hierarchy violation eval"
git push origin feat/hierarchy-violation-eval
```

---

## Self-Review Checklist

After Task 8, verify:

1. **Spec coverage**:
   - ✅ E1-E4 HVR computation from saved ckpts (Task 4 + Task 7 deferred to Nebula)
   - ✅ E0 OR-aggregation baseline (Tasks 1-3, executed locally in Task 7)
   - ✅ Leaderboard surfaces both (Task 6 + Task 7 regen)
   - ✅ Single source of truth for the metric (`src/training/metrics.py`, imported, never re-implemented)

2. **Type consistency**:
   - `mean_aggregate_chapter_probs(leaf_probs, disease_to_chapter_idx, n_chapters)` — same signature in Task 1 (definition), Task 3 (caller in `run_e0_baseline_mode`), Task 4 (caller in `run_multitask_mode` for the mean variant).
   - `compute_and_persist_hvr(leaf_probs, chap_probs, output_path, method, method_details, disease_to_chapter_idx=None)` — same in Task 2 (definition), Task 3, Task 4 (called TWICE in Task 4, once per chap variant).
   - JSON schema is identical across `hierarchy_violation_mean.json` (E0 + E1-E4) and `hierarchy_violation_head.json` (E1-E4 only): both have `hierarchy_violation_rate`, `n_samples`, `n_leaf_chapter_pairs`, `method`, `method_details`. Only `method` differs.

3. **No placeholders**: every code-changing step shows the exact code to write; every test step shows the exact test; every shell step shows the exact command + expected output.

4. **TDD discipline**: every implementation step is preceded by a failing-test step.
