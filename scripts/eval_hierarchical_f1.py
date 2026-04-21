"""Evaluate Hierarchical F-measure (hF₁) for MetaboLM experiments.

Implements Kiritchenko et al. (2005) hierarchical precision / recall / F1
using ancestor-set augmentation on the 2-level ICD-10 hierarchy
(root → chapter → disease).

Two modes:
  - ``multitask``: loads a multitask checkpoint (E1-E4 / μ sweep),
    forwards val.csv, binarizes predictions, computes hF₁.
  - ``e0``: loads 16 per-disease E0 checkpoints, assembles a full
    (N, 16) prediction matrix, computes hF₁.

Reference:
  Kiritchenko, S., Matwin, S., & Famili, A. F. (2005).
  "Functional annotation of genes using hierarchical text categorization."
  BioLINK SIG Workshop.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval_hierarchical_f1")


# Buffer re-registered on the model AFTER load (see forward_*), so it legitimately
# appears as an "unexpected" key at load time when the training checkpoint saved it.
# Any OTHER unexpected key (LoRA / adapter wrappers like ``*.original.*`` /
# ``*.lora_A`` when loading into a non-LoRA / non-adapter model) must fail loudly.
_MULTITASK_ALLOWED_UNEXPECTED: tuple[str, ...] = (
    "metabolite_model.bias_matrix_full",
)
_E0_ALLOWED_UNEXPECTED: tuple[str, ...] = (
    "metabolite_model.bias_matrix_full",
)


# ── Output path resolver (METABOLM_OUTPUT_BASE remapping) ────────────────


def _resolve_output_arg(path: Path | None) -> Path | None:
    """Optional variant: ``None`` passes through."""
    if path is None:
        return None
    from src.config import _resolve_output_path

    return Path(_resolve_output_path(str(path)))


def _resolve_required_output_arg(path: str | Path) -> Path:
    """Required variant used for CLI args that must exist on disk."""
    from src.config import _resolve_output_path

    return Path(_resolve_output_path(str(path)))


# ── Core metric ──────────────────────────────────────────────────────────


def compute_hierarchical_f1(
    preds_binary: np.ndarray,
    labels_binary: np.ndarray,
    disease_to_chapter_idx: dict[int, int],
    n_chapters: int,
) -> dict:
    """Kiritchenko et al. (2005) hierarchical Precision / Recall / F1.

    For each sample, expand the predicted label set and true label set
    with their chapter ancestors, then compute micro-averaged P / R / F1
    over the augmented sets.

    In the 2-level tree (root → chapter → disease):
      - Each positive disease d adds its parent chapter to the augmented set.
      - Chapters are represented as ``n_diseases + chapter_idx`` to avoid
        index collision with leaf disease indices.

    Args:
        preds_binary: ``(N, n_diseases)`` binary predictions (0 or 1).
        labels_binary: ``(N, n_diseases)`` binary ground truth (0 or 1).
        disease_to_chapter_idx: ``{disease_idx: chapter_idx}``.
        n_chapters: Number of distinct chapters.

    Returns:
        Dict with keys: ``hP``, ``hR``, ``hF1``, ``flat_P``, ``flat_R``,
        ``flat_F1``, ``delta_hF1_flat_F1``, ``n_samples``.
    """
    n_samples, n_diseases = preds_binary.shape
    assert labels_binary.shape == preds_binary.shape

    # Chapter nodes are indexed as n_diseases + chapter_idx
    chapter_offset = n_diseases

    def _augment(binary_row: np.ndarray) -> set[int]:
        """Expand a binary label/pred vector to a set including ancestors."""
        s: set[int] = set()
        for d_idx in range(n_diseases):
            if binary_row[d_idx]:
                s.add(d_idx)
                c_idx = disease_to_chapter_idx.get(d_idx)
                if c_idx is not None:
                    s.add(chapter_offset + c_idx)
        return s

    # Accumulators for micro-averaging
    sum_intersect = 0
    sum_aug_pred = 0
    sum_aug_true = 0
    # Flat metric accumulators
    sum_flat_tp = 0
    sum_flat_pred = 0
    sum_flat_true = 0

    for i in range(n_samples):
        aug_pred = _augment(preds_binary[i])
        aug_true = _augment(labels_binary[i])

        sum_intersect += len(aug_pred & aug_true)
        sum_aug_pred += len(aug_pred)
        sum_aug_true += len(aug_true)

        # Flat (leaf-only) sets for comparison
        flat_pred = set(d for d in range(n_diseases) if preds_binary[i, d])
        flat_true = set(d for d in range(n_diseases) if labels_binary[i, d])
        sum_flat_tp += len(flat_pred & flat_true)
        sum_flat_pred += len(flat_pred)
        sum_flat_true += len(flat_true)

    hP = sum_intersect / sum_aug_pred if sum_aug_pred > 0 else 0.0
    hR = sum_intersect / sum_aug_true if sum_aug_true > 0 else 0.0
    hF1 = 2 * hP * hR / (hP + hR) if (hP + hR) > 0 else 0.0

    flat_P = sum_flat_tp / sum_flat_pred if sum_flat_pred > 0 else 0.0
    flat_R = sum_flat_tp / sum_flat_true if sum_flat_true > 0 else 0.0
    flat_F1 = 2 * flat_P * flat_R / (flat_P + flat_R) if (flat_P + flat_R) > 0 else 0.0

    return {
        "hP": float(hP),
        "hR": float(hR),
        "hF1": float(hF1),
        "flat_P": float(flat_P),
        "flat_R": float(flat_R),
        "flat_F1": float(flat_F1),
        "delta_hF1_flat_F1": float(hF1 - flat_F1),
        "n_samples": int(n_samples),
    }


def youden_optimal_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-column Youden-optimal thresholds (maximize sensitivity + specificity - 1).

    Args:
        probs: ``(N, D)`` predicted probabilities.
        labels: ``(N, D)`` binary ground truth.

    Returns:
        ``(D,)`` array of finite thresholds. Degenerate columns (single-class,
        or no finite threshold from ``roc_curve``) fall back to ``0.5``.
    """
    from sklearn.metrics import roc_curve

    n_cols = probs.shape[1]
    thresholds = np.full(n_cols, 0.5)
    for j in range(n_cols):
        col_labels = labels[:, j]
        n_pos = int(col_labels.sum())
        n_neg = int(len(col_labels) - n_pos)
        # roc_curve is undefined when only one class is present in y_true.
        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Column %d has only one class (pos=%d, neg=%d) — falling back to threshold 0.5.",
                j, n_pos, n_neg,
            )
            continue

        fpr, tpr, ths = roc_curve(col_labels, probs[:, j])
        # sklearn's roc_curve prepends an artificial threshold of +inf (since 0.21)
        # so that the curve starts at (0, 0). argmax-with-ties defaults to index 0,
        # which would otherwise pick that +inf, making the whole column predict 0
        # and serializing non-standard Infinity values to JSON.
        finite = np.isfinite(ths)
        if not finite.any():
            logger.warning(
                "Column %d: roc_curve returned no finite thresholds — falling back to 0.5.", j,
            )
            continue
        fpr_f, tpr_f, ths_f = fpr[finite], tpr[finite], ths[finite]
        youden = tpr_f - fpr_f
        best_idx = int(np.argmax(youden))
        thresholds[j] = float(ths_f[best_idx])

    return thresholds


# ── Checkpoint load validation ───────────────────────────────────────────


def _validate_state_dict_load(
    load_result,
    context: str,
    allowed_missing_prefixes: tuple[str, ...] = (),
    allowed_unexpected_prefixes: tuple[str, ...] = (),
) -> None:
    """Fail fast on critical ``load_state_dict(strict=False)`` mismatches.

    Silent ``strict=False`` loads can leave head / adapter / LoRA parameters
    uninitialized when the supplied ``--config`` disagrees with the checkpoint,
    and still produce plausible-looking hF1 numbers. We refuse to score in that
    case. Mismatches are checked in both directions:

    - **missing** — parameter the model expects but the checkpoint does not
      provide. Critical unless whitelisted via ``allowed_missing_prefixes``.
    - **unexpected** — parameter the checkpoint carries but the model does not
      declare. Critical unless whitelisted via ``allowed_unexpected_prefixes``.
      This catches e.g. loading a LoRA-trained checkpoint against a non-LoRA
      config: the LoRA wrappers / ``*.original.*`` keys become unexpected and
      would otherwise be silently ignored while the base weights are only
      half-initialized.
    """
    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])

    def _filter(keys, prefixes):
        return [k for k in keys if not any(k.startswith(p) for p in prefixes)]

    critical_missing = _filter(missing, allowed_missing_prefixes)
    critical_unexpected = _filter(unexpected, allowed_unexpected_prefixes)

    errors: list[str] = []
    if critical_missing:
        preview = critical_missing[:10]
        more = f" ... (+{len(critical_missing) - 10} more)" if len(critical_missing) > 10 else ""
        errors.append(
            f"missing {len(critical_missing)} parameter(s): {preview}{more}"
        )
    if critical_unexpected:
        preview = critical_unexpected[:10]
        more = f" ... (+{len(critical_unexpected) - 10} more)" if len(critical_unexpected) > 10 else ""
        errors.append(
            f"unexpected {len(critical_unexpected)} parameter(s): {preview}{more}"
        )
    if errors:
        joined = "; ".join(errors)
        raise RuntimeError(
            f"[{context}] Checkpoint load mismatch — {joined}. Likely --config "
            f"mismatch with the checkpoint (freeze_strategy / adapter_bottleneck / "
            f"lora_rank). Refusing to score — numbers would be misleading."
        )

    # Whitelisted unexpected keys (e.g. bias_matrix_full buffer registered
    # post-load) are still worth surfacing so the operator knows they were
    # dropped on purpose.
    benign_unexpected = [k for k in unexpected if k not in critical_unexpected]
    if benign_unexpected:
        logger.info(
            "[%s] Ignoring %d whitelisted unexpected key(s): %s",
            context, len(benign_unexpected), benign_unexpected[:10],
        )


# ── Data loading utilities ───────────────────────────────────────────────


def load_val_features_and_labels(val_path: str | Path):
    """Load val.csv → (X_features, Y_labels, disease_names).

    Returns:
        X: ``(N, 168)`` float32 feature array.
        Y: ``(N, 16)`` int32 label array.
    """
    from src.data.biomarkers import get_metabolite_names
    from src.data.endpoints import get_disease_names

    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    label_cols = [f"label_{d}" for d in disease_names]

    with open(val_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

    feat_indices = [header.index(c) for c in feature_cols]
    label_indices = [header.index(c) for c in label_cols]

    X_rows, Y_rows = [], []
    with open(val_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            X_rows.append([float(row[i]) for i in feat_indices])
            Y_rows.append([int(float(row[i])) for i in label_indices])

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.int32)
    logger.info("Loaded val.csv: %d samples × %d features, %d labels",
                X.shape[0], X.shape[1], Y.shape[1])
    return X, Y, disease_names


def forward_multitask_model(
    config_path: Path,
    ckpt_path: Path,
    X_val: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Load a multitask checkpoint and forward val features → leaf_probs (N, 16)."""
    import torch
    import importlib.util

    from src.config import load_config
    from src.data.biomarkers import get_metabolite_names
    from src.data.endpoints import get_disease_names, get_unique_chapters
    from src.model.backbone import MetaboliteBERTModel
    from src.model.heads import HierarchicalMultiTaskHead
    from src.model.wrapper import MetaboLMForClassification

    cfg = load_config(str(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_cols = get_metabolite_names()
    disease_names = get_disease_names()
    n_chapters = len(get_unique_chapters())

    # Load correlation matrix for attention bias
    _ee0_spec = importlib.util.spec_from_file_location(
        "eval_e0_global",
        str(Path(__file__).resolve().parent / "eval_e0_global.py"),
    )
    _ee0 = importlib.util.module_from_spec(_ee0_spec)
    _ee0_spec.loader.exec_module(_ee0)
    bias_matrix = _ee0.load_correlation_matrix(cfg, device)

    # Build model
    backbone = MetaboliteBERTModel(num_metabolites=len(feature_cols))
    head = HierarchicalMultiTaskHead(
        hidden_size=cfg.model.hidden_size,
        proj_size=cfg.model.proj_size,
        num_diseases=len(disease_names),
        num_chapters=n_chapters,
    )
    model = MetaboLMForClassification(
        backbone, head,
        freeze_strategy=cfg.model.freeze_strategy,
        adapter_bottleneck=cfg.model.adapter_bottleneck,
        lora_rank=cfg.model.lora_rank,
    )

    # Load checkpoint
    state_dict = torch.load(str(ckpt_path), map_location=device)
    cleaned = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    load_result = model.load_state_dict(cleaned, strict=False)
    _validate_state_dict_load(
        load_result,
        context=f"multitask / {ckpt_path.name} / config={config_path.name}",
        allowed_unexpected_prefixes=_MULTITASK_ALLOWED_UNEXPECTED,
    )
    model.metabolite_model.register_buffer("bias_matrix_full", bias_matrix)
    model.to(device)
    model.eval()

    # Forward pass
    leaf_chunks = []
    with torch.no_grad():
        for start in range(0, X_val.shape[0], batch_size):
            end = min(start + batch_size, X_val.shape[0])
            x = torch.tensor(X_val[start:end], dtype=torch.float32, device=device)
            mask = torch.ones(x.size(), dtype=torch.long, device=device)
            (leaf_logits, _chap_logits), _ = model(x, mask)
            leaf_chunks.append(torch.sigmoid(leaf_logits).cpu().numpy())

    return np.concatenate(leaf_chunks, axis=0)


def forward_e0_models(
    e0_dir: Path,
    X_val: np.ndarray,
    val_path: str | Path,
    batch_size: int = 512,
) -> np.ndarray:
    """Load 16 per-disease E0 checkpoints and assemble (N, 16) leaf_probs.

    Each per-disease model is a single-task classifier producing 1 probability.
    """
    import torch

    from src.config import load_config
    from src.data.biomarkers import get_metabolite_names
    from src.data.endpoints import get_disease_names
    from src.model.backbone import MetaboliteBERTModel
    from src.model.heads import SingleTaskHead
    from src.model.wrapper import MetaboLMForClassification
    import importlib.util

    disease_names = get_disease_names()
    feature_cols = get_metabolite_names()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = X_val.shape[0]
    leaf_probs = np.zeros((n_samples, len(disease_names)), dtype=np.float32)

    # Load bias matrix using reproduce config
    reproduce_cfg_path = Path(__file__).resolve().parent.parent / "configs" / "reproduce.yaml"
    if reproduce_cfg_path.exists():
        cfg = load_config(str(reproduce_cfg_path))
    else:
        # fallback: use any sft config just for the correlation matrix path
        cfg = load_config(str(
            Path(__file__).resolve().parent.parent / "configs" / "sft_full_ft.yaml"
        ))

    _ee0_spec = importlib.util.spec_from_file_location(
        "eval_e0_global",
        str(Path(__file__).resolve().parent / "eval_e0_global.py"),
    )
    _ee0 = importlib.util.module_from_spec(_ee0_spec)
    _ee0_spec.loader.exec_module(_ee0)
    bias_matrix = _ee0.load_correlation_matrix(cfg, device)

    for d_idx, disease in enumerate(disease_names):
        ckpt_path = e0_dir / disease / f"best_finetune_model_{disease}.pt"
        if not ckpt_path.exists():
            logger.warning("E0 checkpoint not found: %s — column will be NaN", ckpt_path)
            leaf_probs[:, d_idx] = np.nan
            continue

        backbone = MetaboliteBERTModel(num_metabolites=len(feature_cols))
        head = SingleTaskHead(hidden_size=768)
        model = MetaboLMForClassification(backbone, head, freeze_strategy="none")
        state_dict = torch.load(str(ckpt_path), map_location=device)
        cleaned = {
            (k.replace("module.", "") if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        load_result = model.load_state_dict(cleaned, strict=False)
        _validate_state_dict_load(
            load_result,
            context=f"e0 / {disease} / {ckpt_path.name}",
            allowed_unexpected_prefixes=_E0_ALLOWED_UNEXPECTED,
        )
        model.metabolite_model.register_buffer("bias_matrix_full", bias_matrix)
        model.to(device)
        model.eval()

        chunks = []
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x = torch.tensor(X_val[start:end], dtype=torch.float32, device=device)
                mask = torch.ones(x.size(), dtype=torch.long, device=device)
                logits, _ = model(x, mask)
                chunks.append(torch.sigmoid(logits).cpu().numpy().ravel())
        leaf_probs[:, d_idx] = np.concatenate(chunks)
        logger.info("  [%s] forwarded %d samples", disease, n_samples)

    return leaf_probs


# ── Persist results ──────────────────────────────────────────────────────


def persist_hf1(
    results: dict,
    output_path: Path,
    threshold_method: str,
    thresholds: np.ndarray | float,
    disease_names: list[str],
    method_details: dict,
) -> dict:
    """Write hierarchical F1 results to a JSON sidecar."""
    if isinstance(thresholds, np.ndarray):
        thresh_dict = {d: float(t) for d, t in zip(disease_names, thresholds)}
    else:
        thresh_dict = {d: float(thresholds) for d in disease_names}

    payload = {
        **results,
        "threshold_method": threshold_method,
        "per_disease_thresholds": thresh_dict,
        "method_details": method_details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Wrote %s — hF1=%.4f, flat_F1=%.4f, delta=%.4f",
        output_path, payload["hF1"], payload["flat_F1"],
        payload["delta_hF1_flat_F1"],
    )
    return payload


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    import argparse

    from src.data.endpoints import get_disease_names, get_disease_to_chapter_idx, get_unique_chapters

    parser = argparse.ArgumentParser(
        description="Compute Hierarchical F-measure (Kiritchenko 2005) for MetaboLM experiments.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── multitask mode ──
    mt = sub.add_parser(
        "multitask",
        help="Load a multitask checkpoint (E1-E4 / μ sweep), forward val.csv, compute hF₁.",
    )
    mt.add_argument("--config", type=Path, required=True,
                    help="Config YAML (e.g. configs/sft_full_ft.yaml).")
    mt.add_argument("--output-dir", type=Path, required=True,
                    help="Experiment output dir containing best_model.pt.")
    mt.add_argument("--threshold", type=str, default="0.5",
                    choices=["0.5", "youden"],
                    help="Binarization strategy: '0.5' (default) or 'youden'.")
    mt.add_argument("--output-json", type=Path, default=None,
                    help="Override output path (default: <output-dir>/hierarchical_f1.json).")
    mt.add_argument("--batch-size", type=int, default=512)

    # ── e0 mode ──
    e0 = sub.add_parser(
        "e0",
        help="Load 16 E0 per-disease checkpoints, assemble predictions, compute hF₁.",
    )
    e0.add_argument("--e0-dir", type=Path, required=True,
                    help="E0 reproduction dir (e.g. outputs/E0_reproduction/) "
                         "with per-disease subdirs each containing best_finetune_model_{D}.pt.")
    e0.add_argument("--threshold", type=str, default="0.5",
                    choices=["0.5", "youden"],
                    help="Binarization strategy.")
    e0.add_argument("--output-json", type=Path, default=None,
                    help="Override output path (default: <e0-dir>/hierarchical_f1.json).")
    e0.add_argument("--batch-size", type=int, default=512)

    args = parser.parse_args()

    disease_names = get_disease_names()
    disease_to_chap = get_disease_to_chapter_idx()
    n_chapters = len(get_unique_chapters())

    if args.mode == "multitask":
        from src.config import load_config
        cfg = load_config(str(args.config))

        # Respect METABOLM_OUTPUT_BASE remapping used by the rest of the
        # training / eval scripts — otherwise plain "outputs/..." args point
        # to the wrong filesystem location and produce false "checkpoint not
        # found" errors.
        output_dir = _resolve_required_output_arg(args.output_dir)
        ckpt_path = output_dir / "best_model.pt"
        if not ckpt_path.exists():
            logger.error("Checkpoint not found: %s", ckpt_path)
            return 1

        X_val, Y_val, _ = load_val_features_and_labels(cfg.data.val_path)
        leaf_probs = forward_multitask_model(
            args.config, ckpt_path, X_val, args.batch_size,
        )

        if args.threshold == "youden":
            thresholds = youden_optimal_thresholds(leaf_probs, Y_val)
            preds_binary = (leaf_probs >= thresholds[np.newaxis, :]).astype(np.int32)
            thresh_method = "youden"
        else:
            thresholds = 0.5
            preds_binary = (leaf_probs >= 0.5).astype(np.int32)
            thresh_method = "fixed_0.5"

        results = compute_hierarchical_f1(preds_binary, Y_val, disease_to_chap, n_chapters)

        out_path = _resolve_output_arg(args.output_json) or (
            output_dir / "hierarchical_f1.json"
        )
        persist_hf1(results, out_path, thresh_method, thresholds, disease_names, {
            "mode": "multitask",
            "config_path": str(args.config),
            "checkpoint_path": str(ckpt_path),
        })

    elif args.mode == "e0":
        # Use sft_full_ft config for val_path (same val.csv for all experiments)
        sft_cfg_path = Path(__file__).resolve().parent.parent / "configs" / "sft_full_ft.yaml"
        from src.config import load_config
        cfg = load_config(str(sft_cfg_path))

        # See note above — respect METABOLM_OUTPUT_BASE remapping.
        e0_dir = _resolve_required_output_arg(args.e0_dir)

        X_val, Y_val, _ = load_val_features_and_labels(cfg.data.val_path)
        leaf_probs = forward_e0_models(
            e0_dir, X_val, cfg.data.val_path, args.batch_size,
        )

        # Check for missing diseases
        nan_cols = np.isnan(leaf_probs).any(axis=0)
        if nan_cols.any():
            missing = [disease_names[i] for i in range(len(disease_names)) if nan_cols[i]]
            logger.error("Missing E0 checkpoints for: %s — cannot compute hF₁", missing)
            return 1

        if args.threshold == "youden":
            thresholds = youden_optimal_thresholds(leaf_probs, Y_val)
            preds_binary = (leaf_probs >= thresholds[np.newaxis, :]).astype(np.int32)
            thresh_method = "youden"
        else:
            thresholds = 0.5
            preds_binary = (leaf_probs >= 0.5).astype(np.int32)
            thresh_method = "fixed_0.5"

        results = compute_hierarchical_f1(preds_binary, Y_val, disease_to_chap, n_chapters)

        out_path = _resolve_output_arg(args.output_json) or (
            e0_dir / "hierarchical_f1.json"
        )
        persist_hf1(results, out_path, thresh_method, thresholds, disease_names, {
            "mode": "e0",
            "e0_dir": str(e0_dir),
        })

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
