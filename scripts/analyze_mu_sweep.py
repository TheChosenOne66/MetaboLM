#!/usr/bin/env python3
"""Aggregate EXP-000 μ-sweep results into a single summary.

For each ``<sweep_dir>/mu_*/`` subdirectory:
  1. Load ``summary.json`` → Mean AUROC, best epoch, trainable params
  2. Load ``multitask_metrics.csv`` → per-disease Val AUC
  3. Load ``hierarchy_violation_mean.json`` + ``hierarchy_violation_head.json``
     (written by ``scripts/eval_hierarchy_violation.py multitask``)
  4. Re-forward ``val.csv`` through ``best_model.pt`` to extract
     ``(leaf_probs, chap_probs)`` and compute **per-chapter HVR** by calling
     :func:`compute_hierarchy_violation_rate` with a sub-mapping restricted
     to each ICD-10 chapter. This uses the canonical HVR metric unchanged.

Outputs:
  - ``<sweep_dir>/summary.md``    — Markdown table + Tier A/B/C verdict
  - ``<sweep_dir>/summary.csv``   — Machine-readable table (one row per μ)
  - ``<sweep_dir>/mu_response_curve.png``  — Matplotlib plot of
    (μ, HVR_head, HVR_mean, Mean AUROC)

Usage::

    python scripts/analyze_mu_sweep.py --sweep-dir outputs/E1_mu_sweep

Runs on CPU or single GPU (whichever ``torch.device`` resolves).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from src.training.metrics import compute_hierarchy_violation_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("analyze_mu_sweep")

_MU_RE = re.compile(r"^mu_(?P<mu>[\d.]+)$")


def _forward_val(
    best_model_pt: Path, config_path: Path, device: torch.device, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Run one forward pass over val.csv with ``best_model.pt``.

    Returns ``(leaf_probs_N16, chap_probs_N6)`` as float32 numpy arrays.
    """
    cfg = load_config(str(config_path))
    feature_cols = get_metabolite_names()

    # Load val features only (labels not needed for HVR)
    with open(cfg.data.val_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
    feat_idx = [header.index(c) for c in feature_cols]
    rows: list[list[float]] = []
    with open(cfg.data.val_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([float(row[i]) for i in feat_idx])
    X = np.asarray(rows, dtype=np.float32)
    logger.info("Loaded val.csv: %d × %d", X.shape[0], X.shape[1])

    # Correlation matrix — same loader as training
    corr_path = Path(cfg.data.correlation_matrix_path)
    if corr_path.suffix == ".pt" and corr_path.exists():
        bias_matrix = torch.load(str(corr_path), map_location=device, weights_only=False)
    else:
        raise FileNotFoundError(f"Correlation matrix missing: {corr_path}")

    # Re-build the same model topology, load ckpt
    backbone = MetaboliteBERTModel(num_metabolites=cfg.data.num_metabolites)
    backbone.register_buffer("bias_matrix_full", bias_matrix)
    head = HierarchicalMultiTaskHead(
        hidden_size=cfg.model.hidden_size,
        proj_size=cfg.model.proj_size,
        num_diseases=cfg.model.num_diseases,
        num_chapters=len(get_unique_chapters()),
    )
    model = MetaboLMForClassification(
        metabolite_model=backbone,
        head=head,
        freeze_strategy="none",  # inference only; freeze doesn't matter
        adapter_bottleneck=cfg.model.adapter_bottleneck,
        lora_rank=cfg.model.lora_rank,
    )
    state = torch.load(str(best_model_pt), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)  # strict=False: ignore backbone bias_matrix_full mismatch if ckpt carries it
    model.to(device).eval()

    leaf_probs_list: list[np.ndarray] = []
    chap_probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            mask = torch.ones(xb.size(), dtype=torch.long, device=device)
            (leaf_logits, chap_logits), _ = model(xb, mask)
            leaf_probs_list.append(torch.sigmoid(leaf_logits).cpu().numpy())
            chap_probs_list.append(torch.sigmoid(chap_logits).cpu().numpy())
    leaf_probs = np.concatenate(leaf_probs_list, axis=0)
    chap_probs = np.concatenate(chap_probs_list, axis=0)
    return leaf_probs, chap_probs


def _per_chapter_hvr(
    leaf_probs: np.ndarray,
    chap_probs: np.ndarray,
    disease_to_chapter_idx: dict[int, int],
) -> dict[int, float]:
    """Restrict the canonical HVR metric to one chapter at a time.

    Uses :func:`compute_hierarchy_violation_rate` with a sub-mapping containing
    only the leaves of that chapter — no algorithmic change to the metric.
    """
    by_chap: dict[int, dict[int, int]] = defaultdict(dict)
    for leaf_idx, chap_idx in disease_to_chapter_idx.items():
        by_chap[chap_idx][leaf_idx] = chap_idx
    return {
        chap_idx: compute_hierarchy_violation_rate(
            leaf_probs.astype(np.float64),
            chap_probs.astype(np.float64),
            sub_mapping,
        )
        for chap_idx, sub_mapping in by_chap.items()
    }


def _load_run(run_dir: Path, config_dir: Path, device: torch.device) -> dict:
    """Pull Mean AUROC, HVR mean/head, and per-chapter HVR out of one μ dir."""
    mu_match = _MU_RE.match(run_dir.name)
    if not mu_match:
        raise ValueError(f"Unexpected dir name: {run_dir.name}")
    mu = float(mu_match.group("mu"))

    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "multitask_metrics.csv"
    hvr_mean_path = run_dir / "hierarchy_violation_mean.json"
    hvr_head_path = run_dir / "hierarchy_violation_head.json"
    ckpt = run_dir / "best_model.pt"

    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    mean_auc = float(summary.get("best_mean_auc", float("nan")))
    best_epoch = summary.get("best_epoch")

    hvr_mean = (
        json.loads(hvr_mean_path.read_text())["hierarchy_violation_rate"]
        if hvr_mean_path.exists() else float("nan")
    )
    hvr_head = (
        json.loads(hvr_head_path.read_text())["hierarchy_violation_rate"]
        if hvr_head_path.exists() else float("nan")
    )

    per_chapter: dict[int, float] = {}
    if ckpt.exists():
        # Locate the training config for this μ. Convention: configs/sft_full_ft_mu_{mu}.yaml.
        # Float → string must match how configs were named (see M5).
        mu_s = f"{mu:g}"  # "0" or "0.5" or "10" — preserve original literals
        # Handle "10.0" / "1.0" etc: configs use literal "10.0", not "10". Try both.
        cfg_candidates = [
            config_dir / f"sft_full_ft_mu_{mu}.yaml",
            config_dir / f"sft_full_ft_mu_{mu_s}.yaml",
        ]
        cfg = next((c for c in cfg_candidates if c.exists()), None)
        if cfg is None:
            logger.warning("No config match for μ=%s under %s — skipping per-chapter HVR", mu, config_dir)
        else:
            logger.info("Re-forwarding val.csv for μ=%s using %s", mu, cfg)
            leaf_probs, chap_probs = _forward_val(ckpt, cfg, device=device)
            per_chapter = _per_chapter_hvr(leaf_probs, chap_probs, get_disease_to_chapter_idx())
    else:
        logger.warning("No best_model.pt in %s — per-chapter HVR unavailable", run_dir)

    return {
        "mu": mu,
        "mean_auc": mean_auc,
        "best_epoch": best_epoch,
        "hvr_mean": hvr_mean,
        "hvr_head": hvr_head,
        "per_chapter_hvr": per_chapter,
    }


def _tier_verdict(records: list[dict], e0_auc: float = 0.671) -> tuple[str, str]:
    """Apply rescue plan §4 Tier A/B/C rule to the sweep records."""
    # Tier A: exists μ* with HVR_head < 0.05 ∧ HVR_mean < 0.15 ∧ mean_auc ≥ 0.670
    #         ∧ ≥ 4/6 per-chapter HVR < 0.1
    tier_a = [
        r for r in records
        if r["hvr_head"] < 0.05
        and r["hvr_mean"] < 0.15
        and r["mean_auc"] >= e0_auc
        and sum(1 for v in r["per_chapter_hvr"].values() if v < 0.1) >= 4
    ]
    if tier_a:
        best = min(tier_a, key=lambda r: r["hvr_head"])
        return "A", f"μ* = {best['mu']} satisfies all Tier A conditions"

    # Tier B: Pareto — HVR_head < 0.1 ∧ AUROC drop ≤ 0.02 at ≥ 2 points
    pareto = [
        r for r in records if r["hvr_head"] < 0.1 and (0.680 - r["mean_auc"]) <= 0.02
    ]
    if len(pareto) >= 2:
        best = min(pareto, key=lambda r: r["hvr_head"])
        return "B", f"recommended μ_rec = {best['mu']} on HVR-AUROC Pareto"

    # Tier C: all runs have HVR_head >= 0.2
    if all(r["hvr_head"] >= 0.2 for r in records if not np.isnan(r["hvr_head"])):
        return "C", "no configuration achieves HVR_head < 0.2 — proceed to EXP-001 (mean-aggregated leaf target)"

    return "mixed", "intermediate result — see per-run numbers; most likely route EXP-001"


def _write_summary(records: list[dict], sweep_dir: Path) -> None:
    # CSV
    records_sorted = sorted(records, key=lambda r: r["mu"])
    chap_names = get_unique_chapters()
    rows = []
    for r in records_sorted:
        row = {
            "mu": r["mu"],
            "mean_auc": r["mean_auc"],
            "best_epoch": r["best_epoch"],
            "hvr_mean": r["hvr_mean"],
            "hvr_head": r["hvr_head"],
        }
        for idx, name in enumerate(chap_names):
            row[f"hvr_chap_{name}"] = r["per_chapter_hvr"].get(idx, float("nan"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(sweep_dir / "summary.csv", index=False)
    logger.info("Wrote %s", sweep_dir / "summary.csv")

    # Tier verdict
    tier, detail = _tier_verdict(records_sorted)

    # Markdown
    lines: list[str] = []
    lines.append("# EXP-000 μ Sweep Summary")
    lines.append("")
    lines.append(f"**Verdict**: Tier **{tier}** — {detail}")
    lines.append("")
    lines.append("## Main table")
    lines.append("")
    lines.append("| μ | Mean AUROC | HVR_mean | HVR_head | Best epoch |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    for r in records_sorted:
        lines.append(
            f"| {r['mu']} | {r['mean_auc']:.4f} | {r['hvr_mean']:.4f} | "
            f"{r['hvr_head']:.4f} | {r['best_epoch']} |"
        )
    lines.append("")
    lines.append("## Per-chapter HVR (head variant)")
    lines.append("")
    lines.append("| μ | " + " | ".join(chap_names) + " |")
    lines.append("|:---:|" + "|".join([":---:"] * len(chap_names)) + "|")
    for r in records_sorted:
        cells = [
            f"{r['per_chapter_hvr'].get(idx, float('nan')):.4f}" for idx in range(len(chap_names))
        ]
        lines.append(f"| {r['mu']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append(f"Generated by `{Path(__file__).name}` from `{sweep_dir}/mu_*/`.")
    (sweep_dir / "summary.md").write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", sweep_dir / "summary.md")

    # Plot — μ on log axis, AUROC + HVR on twin y-axes
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mus = [r["mu"] for r in records_sorted]
        aurocs = [r["mean_auc"] for r in records_sorted]
        hvr_m = [r["hvr_mean"] for r in records_sorted]
        hvr_h = [r["hvr_head"] for r in records_sorted]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        # Shift μ=0 to a tiny positive value so log scale works but the label stays "0"
        mus_plot = [m if m > 0 else 1e-3 for m in mus]
        ax1.plot(mus_plot, aurocs, "o-", color="C0", label="Mean AUROC")
        ax1.set_xscale("log")
        ax1.set_xlabel("μ_hierarchy")
        ax1.set_ylabel("Mean AUROC", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.axhline(0.671, ls=":", color="gray", label="E0 fair baseline")

        ax2 = ax1.twinx()
        ax2.plot(mus_plot, hvr_m, "s--", color="C1", label="HVR (mean)")
        ax2.plot(mus_plot, hvr_h, "^--", color="C2", label="HVR (head)")
        ax2.set_ylabel("HVR", color="black")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

        plt.title("EXP-000 μ Sweep: HVR-AUROC response")
        plt.tight_layout()
        plt.savefig(sweep_dir / "mu_response_curve.png", dpi=120)
        logger.info("Wrote %s", sweep_dir / "mu_response_curve.png")
    except Exception as exc:
        logger.warning("Plot skipped: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir", required=True, type=Path,
        help="Directory containing mu_*/ subdirs (e.g. outputs/E1_mu_sweep)",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=Path("configs"),
        help="Where to find sft_full_ft_mu_*.yaml (default: configs)",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device override (default: cuda if available)",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    run_dirs = sorted(d for d in args.sweep_dir.iterdir() if d.is_dir() and _MU_RE.match(d.name))
    if not run_dirs:
        raise SystemExit(f"No mu_* subdirs found in {args.sweep_dir}")
    logger.info("Found %d runs: %s", len(run_dirs), [d.name for d in run_dirs])

    records = [_load_run(rd, args.config_dir, device) for rd in run_dirs]
    _write_summary(records, args.sweep_dir)


if __name__ == "__main__":
    main()
