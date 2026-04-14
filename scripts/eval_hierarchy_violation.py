"""Evaluate Hierarchy Violation Rate (HVR) for multitask checkpoints (E1-E4)
and a mean-aggregation baseline for E0.

See docs/superpowers/plans/2026-04-13-hierarchy-violation-eval.md for the
full design rationale.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Enable `from src...` imports added in later tasks (e.g. Task 3, Task 4).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval_hierarchy_violation")


def mean_aggregate_chapter_probs(
    leaf_probs: np.ndarray,
    disease_to_chapter_idx: dict[int, int],
    n_chapters: int,
) -> np.ndarray:
    """Aggregate per-leaf probabilities into per-chapter probabilities via the
    arithmetic mean: ``P_chap = mean(p_leaf for leaf in chapter)``.

    Args:
        leaf_probs: ``(N, n_leaves)`` probabilities in [0, 1].
        disease_to_chapter_idx: ``{leaf_idx: chapter_idx}`` (length n_leaves).
        n_chapters: Number of distinct chapter indices.

    Returns:
        ``(N, n_chapters)`` chapter-level probabilities. The (i, c) cell is
        the average leaf probability for chapter ``c`` on sample ``i``.

    Used as the E0 baseline for hierarchy violation rate. **Why mean and not
    max / OR / noisy-OR**: both ``max(p_leaf)`` and ``1 - prod(1 - p_leaf)``
    (the independence-OR) are *upper bounds* on every leaf prob — by
    construction ``p_leaf <= chap_prob`` always, so the HVR strict
    comparison ``p_leaf > chap_prob + 1e-7`` is never true and the metric
    returns 0.0 identically. The mean is the simplest aggregation that
    *can* be exceeded by individual leaves (any leaf above its sibling
    mean violates), so it produces the non-trivial baseline number we need
    to demonstrate the value of E1's explicit hierarchy loss.
    """
    n_samples, n_leaves = leaf_probs.shape
    if len(disease_to_chapter_idx) != n_leaves:
        raise ValueError(
            f"disease_to_chapter_idx covers {len(disease_to_chapter_idx)} leaves, "
            f"but leaf_probs has {n_leaves} columns"
        )
    sums = np.zeros((n_samples, n_chapters), dtype=np.float32)
    counts = np.zeros(n_chapters, dtype=np.int32)
    for leaf_idx, chap_idx in disease_to_chapter_idx.items():
        sums[:, chap_idx] += leaf_probs[:, leaf_idx]
        counts[chap_idx] += 1
    # Avoid division by zero for chapters with no mapped leaves: leave their
    # column as 0.0 (no leaves means no possible violations from that chapter
    # anyway, so the value is unobservable in HVR).
    safe_counts = np.where(counts > 0, counts, 1).astype(np.float32)
    return (sums / safe_counts).astype(np.float32)


def compute_and_persist_hvr(
    leaf_probs: np.ndarray,
    chap_probs: np.ndarray,
    output_path: Path,
    method: str,
    method_details: dict,
    disease_to_chapter_idx: dict[int, int] | None = None,
) -> dict:
    """Compute HVR via the canonical metric and write a JSON sidecar.

    Args:
        leaf_probs: ``(N, 16)`` leaf probabilities in [0, 1].
        chap_probs: ``(N, 6)`` chapter probabilities in [0, 1].
        output_path: Where to write the JSON.
        method: Short identifier (e.g. ``"explicit_chapter_head"`` or
            ``"mean_aggregate_baseline"``). Surfaced in the JSON for downstream
            disambiguation.
        method_details: Free-form dict appended to the JSON for provenance
            (model path, predictions source dir, etc.).
        disease_to_chapter_idx: ``{leaf_idx: chapter_idx}`` mapping. Defaults to
            :func:`src.data.endpoints.get_disease_to_chapter_idx` (the canonical
            16-leaf / 6-chapter mapping for this project). Override in tests to
            verify the computation on smaller, hand-checkable inputs.

    Returns:
        The dict that was written to disk (also useful for in-process logging).
        Keys:
          * ``hierarchy_violation_rate`` (float): rate from the canonical metric.
          * ``n_samples`` (int): ``leaf_probs.shape[0]``.
          * ``n_leaf_chapter_pairs`` (int): ``len(disease_to_chapter_idx)`` — the
            number of (leaf, parent_chapter) pairs actually evaluated by the
            canonical metric, NOT ``leaf_probs.shape[1]``. They coincide when
            the full 16-leaf mapping is used; they differ when a caller passes
            a sub-mapping (e.g. the E0 baseline path with a partial eval dir,
            where missing diseases are dropped before this helper is called).
          * ``method`` (str): the ``method`` argument.
          * ``method_details`` (dict): copy of the ``method_details`` argument.
    """
    # Lazy import: the metric lives next to training code and pulls in heavier
    # imports we don't want at script-import time.
    from src.training.metrics import compute_hierarchy_violation_rate
    from src.data.endpoints import get_disease_to_chapter_idx

    if leaf_probs.shape[0] != chap_probs.shape[0]:
        raise ValueError(
            f"leaf_probs ({leaf_probs.shape}) and chap_probs ({chap_probs.shape}) "
            "disagree on N (axis 0)."
        )

    if disease_to_chapter_idx is None:
        disease_to_chapter_idx = get_disease_to_chapter_idx()

    rate = compute_hierarchy_violation_rate(
        leaf_probs.astype(np.float64),
        chap_probs.astype(np.float64),
        disease_to_chapter_idx,
    )

    # ``n_leaf_chapter_pairs`` is the number of (leaf, parent_chapter) pairs the
    # canonical metric iterates over (``len(disease_to_chapter_idx)``), NOT the
    # number of leaf columns. They coincide when the full 16-leaf mapping is
    # used; they differ when a caller (e.g. the E0 baseline with a partial eval
    # dir) passes a sub-mapping over the present leaves only.
    payload = {
        "hierarchy_violation_rate": float(rate),
        "n_samples": int(leaf_probs.shape[0]),
        "n_leaf_chapter_pairs": int(len(disease_to_chapter_idx)),
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


def load_e0_leaf_probs(
    eval_dir: Path, disease_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Read per-ckpt predictions under ``eval_dir/<disease>/predictions_ckpt_<D>.csv``
    and return ``(leaf_probs, present_diseases)``.

    Aligns by the ``eid`` column of ``eval_dir/eval_set_labels.csv`` so all
    disease columns share a row order. Diseases without a predictions file
    contribute a column of ``NaN`` — the caller is responsible for handling
    those (typically by dropping them before aggregation).

    Args:
        eval_dir: Output dir produced by ``scripts/eval_e0_global*.py``.
            Must contain ``eval_set_labels.csv`` at the top and optionally
            per-disease subdirs with ``predictions_ckpt_<D>.csv``.
        disease_names: Canonical disease order. Output columns follow this
            order exactly.

    Returns:
        Tuple ``(leaf_probs, present)``:
            - ``leaf_probs``: ``(N, len(disease_names))`` float32 array. Rows
              are aligned to ``eval_set_labels.csv``'s eid order; missing
              disease columns are filled with ``NaN``.
            - ``present``: subset of ``disease_names`` whose predictions file
              was found and successfully read. Order matches the input
              ``disease_names``.
    """
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
        # Iterate via column arrays to avoid pandas' row-wise dtype promotion
        # (int eids become floats and stringify as "1.0", breaking the join).
        pred_eids = pred_df["eid"].astype(str).tolist()
        pred_vals = pred_df[prob_col].astype(float).tolist()
        n_aligned = 0
        for eid_str, val in zip(pred_eids, pred_vals):
            i = eid_to_row.get(eid_str)
            if i is None:
                continue
            leaf_probs[i, col_idx] = val
            n_aligned += 1
        # Mark the disease as present ONLY if at least one prediction row
        # actually aligned with eval_set_labels.csv. A stale/format-drifted
        # file (e.g. eids as ``1.0`` instead of ``1``) would otherwise slip
        # through as an all-NaN column, and the canonical metric would treat
        # every NaN > P_chap comparison as False — silently deflating HVR.
        # Codex P2 on PR #5 (2026-04-14 round 2).
        if n_aligned == 0:
            logger.warning(
                "[%s] %s has %d rows but NONE aligned with eval_set_labels.csv "
                "(eid mismatch — stale file or dtype drift?); skipping disease",
                disease, pred_path.name, len(pred_eids),
            )
            continue
        if n_aligned < len(eids):
            logger.warning(
                "[%s] only %d/%d eids in eval_set_labels.csv got a prediction; "
                "unaligned rows will be NaN and will fail the no-NaN check "
                "in run_e0_baseline_mode",
                disease, n_aligned, len(eids),
            )
        present.append(disease)
    return leaf_probs, present


def run_e0_baseline_mode(
    eval_dir: Path,
    output_path: Path,
) -> dict:
    """E0 baseline mode: read 16 leaf prob columns from ``eval_dir``, mean-aggregate
    to chapter probs, compute HVR, persist.

    This is the baseline Innovation 2 (hierarchy loss) is supposed to fix.
    We expect a high HVR here because E0's per-disease models have no
    structural prior tying leaf probs to a shared chapter prob. We use mean
    aggregation rather than OR/max because the latter are upper bounds on
    every leaf prob and trivialise HVR to 0.0 — see
    :func:`mean_aggregate_chapter_probs` for the full rationale.

    Args:
        eval_dir: Eval output dir produced by ``scripts/eval_e0_global*.py``.
        output_path: Where to write the HVR sidecar JSON.

    Returns:
        The payload dict written to ``output_path``.
    """
    from src.data.endpoints import (
        get_disease_names, get_disease_to_chapter_idx, get_unique_chapters,
    )

    diseases = get_disease_names()
    full_disease_to_chap = get_disease_to_chapter_idx()
    n_chapters = len(get_unique_chapters())

    leaf_probs, present = load_e0_leaf_probs(eval_dir, diseases)

    # Fail loudly when NO disease predictions were found (e.g. eval_dir points
    # at a stale / empty directory, or all predictions_ckpt_*.csv files have
    # schema issues and were skipped). Previously this path produced
    # ``hierarchy_violation_rate: 0.0`` over zero evaluated pairs, which
    # renders indistinguishably from a legitimately-zero rate in the
    # leaderboard. See codex review on PR #5.
    if not present:
        raise RuntimeError(
            f"No disease predictions loaded from {eval_dir}. "
            "Expected at least one <disease>/predictions_ckpt_<disease>.csv "
            "file with columns (eid, prob_ckpt_<disease>). "
            "Refusing to write a meaningless HVR=0.0 metric — run "
            "scripts/eval_e0_global*.py first, or point --eval-dir at a "
            "populated directory."
        )

    # Drop missing diseases entirely so mean aggregation isn't poisoned by NaN.
    # The canonical metric only iterates the mapping we pass it, so limiting
    # the mapping to ``present`` diseases gives the correct "pairs evaluated"
    # count in the persisted payload (via ``compute_and_persist_hvr``'s
    # ``n_leaf_chapter_pairs = len(disease_to_chapter_idx)`` semantics).
    present_idx = [diseases.index(d) for d in present]
    leaf_probs_clean = leaf_probs[:, present_idx]
    # Defense-in-depth against the same "silent NaN deflates HVR" class of
    # bug flagged by codex P2 (round 2). load_e0_leaf_probs drops diseases
    # with zero aligned eids; partial alignment still leaves NaN cells in
    # otherwise-present columns. Either way, a NaN reaching the canonical
    # metric turns into ``NaN > P_chap == False`` and misleadingly lowers
    # the rate — refuse to proceed.
    if np.isnan(leaf_probs_clean).any():
        n_nan = int(np.isnan(leaf_probs_clean).sum())
        bad_disease_idx = np.where(np.isnan(leaf_probs_clean).any(axis=0))[0]
        bad_diseases = [present[i] for i in bad_disease_idx]
        raise RuntimeError(
            f"{n_nan} NaN cells remain in leaf_probs after dropping "
            f"missing-prediction diseases; affected diseases: {bad_diseases}. "
            "This indicates partial eid alignment — some eids in "
            "eval_set_labels.csv have no corresponding prediction row. "
            "Refusing to write HVR: NaN comparisons silently read as "
            "non-violations and deflate the metric. Fix the upstream eval "
            "run so every eid gets a prediction, or drop the mismatched "
            "eids from eval_set_labels.csv."
        )
    sub_disease_to_chap = {
        new_idx: full_disease_to_chap[old_idx]
        for new_idx, old_idx in enumerate(present_idx)
    }

    chap_probs = mean_aggregate_chapter_probs(
        leaf_probs_clean,
        disease_to_chapter_idx=sub_disease_to_chap,
        n_chapters=n_chapters,
    )

    return compute_and_persist_hvr(
        leaf_probs=leaf_probs_clean,
        chap_probs=chap_probs,
        output_path=output_path,
        method="mean_aggregate_baseline",
        method_details={
            "chap_probs_source": (
                "P_chap = mean(p_leaf for leaf in chapter); chosen because OR/max are "
                "upper bounds on the leaves and trivialise HVR. See "
                "mean_aggregate_chapter_probs docstring for rationale."
            ),
            "predictions_source": str(eval_dir),
            "n_present_diseases": len(present),
            "present_diseases": present,
        },
        disease_to_chapter_idx=sub_disease_to_chap,
    )


def run_multitask_mode(
    config_path: Path,
    output_dir: Path,
    mean_output_path: Path,
    head_output_path: Path,
    batch_size: int = 512,
) -> tuple[dict, dict]:
    """Multitask mode: load ``output_dir/best_model.pt``, GPU-forward val.csv,
    extract ``(leaf_probs, head_chap_probs)`` from the dual-head model,
    compute HVR **twice** (once with mean-aggregated chap probs for
    cross-model comparability with E0, once with the explicit chapter head
    output for the loss-target-native metric), persist **two** sidecar JSONs.

    Returns ``(mean_payload, head_payload)``.
    """
    import csv
    import torch

    from src.config import load_config
    from src.data.biomarkers import get_metabolite_names
    from src.data.endpoints import (
        get_disease_names,
        get_disease_to_chapter_idx,
        get_unique_chapters,
    )
    from src.model.backbone import MetaboliteBERTModel
    from src.model.heads import HierarchicalMultiTaskHead
    from src.model.wrapper import MetaboLMForClassification

    cfg = load_config(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{d}" for d in disease_names]

    # Stream val.csv into two float32 arrays (features + labels are tiny here).
    with open(cfg.data.val_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
    feat_indices = [header.index(c) for c in feature_cols]
    # Label columns are optional (not used here, but we keep the shape check).
    missing_labels = [c for c in label_cols if c not in header]
    if missing_labels:
        logger.warning(
            "val.csv missing label columns: %s — HVR only needs features, "
            "continuing without them.", missing_labels[:3]
        )

    X_rows: list[list[float]] = []
    with open(cfg.data.val_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X_rows.append([float(row[i]) for i in feat_indices])
    X_val = np.array(X_rows, dtype=np.float32)
    logger.info(
        "Loaded val.csv: %d samples × %d features",
        X_val.shape[0], X_val.shape[1],
    )

    # Reuse eval_e0_global's correlation matrix loader (.pt + CSV fallback) so
    # the bias matrix resolves the same way the training pipeline does.
    # scripts/ is not a package, so use the file-based import pattern.
    import importlib.util
    _ee0_spec = importlib.util.spec_from_file_location(
        "eval_e0_global",
        str(Path(__file__).resolve().parent / "eval_e0_global.py"),
    )
    _ee0 = importlib.util.module_from_spec(_ee0_spec)
    _ee0_spec.loader.exec_module(_ee0)
    bias_matrix = _ee0.load_correlation_matrix(cfg, device)

    ckpt_path = output_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Multitask checkpoint not found: {ckpt_path}")
    state_dict = torch.load(str(ckpt_path), map_location=device)

    n_chapters = len(get_unique_chapters())
    backbone = MetaboliteBERTModel(num_metabolites=len(feature_cols))
    # Read head dimensions from the same cfg the ckpt was trained against.
    # Hardcoded 768/256 previously → any experiment that overrode
    # cfg.model.hidden_size or cfg.model.proj_size would fail to load with
    # a size-mismatch. Codex P2 on PR #5 (round 3).
    head = HierarchicalMultiTaskHead(
        hidden_size=cfg.model.hidden_size,
        proj_size=cfg.model.proj_size,
        num_diseases=len(disease_names),
        num_chapters=n_chapters,
    )
    # IMPORTANT: instantiate the wrapper with the SAME freeze_strategy /
    # adapter_bottleneck / lora_rank the checkpoint was trained with,
    # otherwise adapter modules are never injected (they'd be silently
    # dropped by strict=False) and LoRA's wrapped q/v projections
    # (``*.original.*`` keys) don't exist on the plain ``nn.Linear`` sides
    # of the backbone. HVR would then be computed against an effectively
    # pretrained-only model. See codex review on PR #5.
    model = MetaboLMForClassification(
        backbone,
        head,
        freeze_strategy=cfg.model.freeze_strategy,
        adapter_bottleneck=cfg.model.adapter_bottleneck,
        lora_rank=cfg.model.lora_rank,
    )
    logger.info(
        "Model: freeze_strategy=%s adapter_bottleneck=%d lora_rank=%d",
        cfg.model.freeze_strategy,
        cfg.model.adapter_bottleneck,
        cfg.model.lora_rank,
    )

    cleaned = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    # strict=False is still appropriate because the checkpoint may or may
    # not include a ``bias_matrix_full`` buffer (we register it below), but
    # we now verify that the core head + adapter/LoRA keys actually
    # matched, so a stale config + ckpt mismatch is loud rather than
    # silent.
    load_report = model.load_state_dict(cleaned, strict=False)
    unexpected_critical = [
        k for k in load_report.unexpected_keys
        if k.startswith(("head.", "metabolite_model.bert.encoder.layer"))
        and "bias_matrix_full" not in k
    ]
    if unexpected_critical:
        raise RuntimeError(
            f"Unexpected keys in checkpoint that do not match the configured "
            f"freeze_strategy={cfg.model.freeze_strategy!r}: "
            f"{unexpected_critical[:5]}{'...' if len(unexpected_critical) > 5 else ''}. "
            "Double-check --config matches the strategy used at training time."
        )
    missing_critical = [
        k for k in load_report.missing_keys
        if k.startswith("head.")
        or ".adapter." in k
        or (".query.lora_" in k or ".value.lora_" in k)
    ]
    if missing_critical:
        raise RuntimeError(
            f"Checkpoint is missing critical parameters for "
            f"freeze_strategy={cfg.model.freeze_strategy!r}: "
            f"{missing_critical[:5]}{'...' if len(missing_critical) > 5 else ''}. "
            "Training config and eval config likely disagree."
        )
    model.metabolite_model.register_buffer("bias_matrix_full", bias_matrix)
    model.to(device)
    model.eval()

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

    # Derive the mean-aggregated chap probs from the SAME leaf outputs so we
    # get the cross-model-comparable metric alongside the head-native one
    # without a second forward.
    disease_to_chap = get_disease_to_chapter_idx()
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
                "P_chap = mean(p_leaf for leaf in chapter); identical recipe to "
                "E0 baseline. Use for cross-model comparison."
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
                "P_chap = sigmoid(chapter_head_logits); the distribution the "
                "hierarchy loss was trained against. Only defined for "
                "multitask models with an explicit chapter head."
            ),
        },
    )
    return mean_payload, head_payload


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute Hierarchy Violation Rate (HVR). Two modes: "
            "`multitask` (E1-E4, writes mean-agg + head-native sidecars in "
            "one forward) and `e0-mean` (E0 baseline from per-ckpt CSVs)."
        ),
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    mt = sub.add_parser(
        "multitask",
        help=(
            "Load best_model.pt and forward val.csv. Writes BOTH "
            "hierarchy_violation_mean.json and hierarchy_violation_head.json."
        ),
    )
    mt.add_argument(
        "--config", type=Path, required=True,
        help="Config YAML (e.g. configs/sft_full_ft.yaml).",
    )
    mt.add_argument(
        "--output-dir", type=Path, required=True,
        help=(
            "Experiment output dir containing best_model.pt "
            "(e.g. outputs/E1_sft_full_ft). Sidecar JSONs are written here "
            "by default."
        ),
    )
    mt.add_argument(
        "--mean-output-json", type=Path, default=None,
        help=(
            "Override path for hierarchy_violation_mean.json. "
            "Defaults to <output-dir>/hierarchy_violation_mean.json."
        ),
    )
    mt.add_argument(
        "--head-output-json", type=Path, default=None,
        help=(
            "Override path for hierarchy_violation_head.json. "
            "Defaults to <output-dir>/hierarchy_violation_head.json."
        ),
    )
    mt.add_argument("--batch-size", type=int, default=512)

    e0 = sub.add_parser(
        "e0-mean",
        help="Read E0 per-ckpt predictions and mean-aggregate to baseline HVR.",
    )
    e0.add_argument(
        "--eval-dir", type=Path, required=True,
        help=(
            "Eval output dir produced by scripts/eval_e0_global*.py "
            "(e.g. outputs/E0_global_eval_cohort)."
        ),
    )
    e0.add_argument(
        "--output-json", type=Path, default=None,
        help=(
            "Where to write hierarchy_violation_mean.json. Defaults to "
            "outputs/E0_reproduction/hierarchy_violation_mean.json."
        ),
    )

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
