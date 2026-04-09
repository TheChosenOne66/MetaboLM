#!/usr/bin/env python3
"""Compare results across all MetaboLM experiments (E0-E6).

Loads per-disease metrics from each experiment's output directory and
generates:
  1. A unified comparison CSV with per-disease AUC across experiments.
  2. A summary table with mean AUC, hierarchy violation rate, and param count.
  3. Ablation analysis for hierarchy loss components (if E1 variants exist).

Usage::

    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --output-dir outputs/comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.endpoints import get_disease_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("compare")

# Experiment registry: (name, output_dir, metrics_file_pattern)
EXPERIMENTS = [
    ("E0", "outputs/E0_reproduction", "e0"),
    ("E1 (Full FT)", "outputs/E1_sft_full_ft", "multitask"),
    ("E2 (Head Only)", "outputs/E2_sft_head_only", "multitask"),
    ("E3 (Adapter)", "outputs/E3_sft_adapter", "multitask"),
    ("E4 (LoRA)", "outputs/E4_sft_lora", "multitask"),
    ("E5 (GRPO Cal)", "outputs/E5_grpo_calibration", "multitask"),
    ("E6 (GRPO Hier)", "outputs/E6_grpo_hierarchy", "multitask"),
]


def load_e0_metrics(output_dir: str) -> dict[str, float] | None:
    """Load E0 per-disease AUC from finetune_summary_metrics.csv."""
    csv_path = os.path.join(output_dir, "finetune_summary_metrics.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        result[row["Disease"]] = row["Val_AUC"]
    # Compute mean if not present
    disease_names = get_disease_names()
    aucs = [result[d] for d in disease_names if d in result and result[d] > 0]
    result["MEAN"] = sum(aucs) / len(aucs) if aucs else 0.0
    return result


def load_multitask_metrics(output_dir: str) -> dict[str, float] | None:
    """Load E1-E6 per-disease AUC from multitask_metrics.csv."""
    csv_path = os.path.join(output_dir, "multitask_metrics.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        result[row["Disease"]] = row["Val_AUC"]
    return result


def load_summary(output_dir: str) -> dict | None:
    """Load summary.json for param counts and metadata."""
    json_path = os.path.join(output_dir, "summary.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MetaboLM experiments")
    parser.add_argument(
        "--output-dir",
        default="outputs/comparison",
        help="Directory for comparison outputs.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    disease_names = get_disease_names()

    # --- Load all available results ---
    all_metrics = {}
    all_summaries = {}
    for exp_name, exp_dir, exp_type in EXPERIMENTS:
        if exp_type == "e0":
            metrics = load_e0_metrics(exp_dir)
        else:
            metrics = load_multitask_metrics(exp_dir)

        if metrics is not None:
            all_metrics[exp_name] = metrics
            logger.info("Loaded: %s (%d diseases)", exp_name, len(metrics) - 1)
        else:
            logger.warning("Not found: %s (%s)", exp_name, exp_dir)

        summary = load_summary(exp_dir)
        if summary:
            all_summaries[exp_name] = summary

    if not all_metrics:
        logger.error("No experiment results found. Run training first.")
        return

    # --- Per-disease AUC comparison table ---
    rows = []
    for disease in disease_names + ["MEAN"]:
        row = {"Disease": disease}
        for exp_name in all_metrics:
            row[exp_name] = all_metrics[exp_name].get(disease, "")
        rows.append(row)
    comparison_df = pd.DataFrame(rows)

    # Add delta columns (vs E0 baseline if available)
    if "E0" in all_metrics:
        for exp_name in all_metrics:
            if exp_name == "E0":
                continue
            delta_col = f"Δ({exp_name} - E0)"
            deltas = []
            for disease in disease_names + ["MEAN"]:
                e0_val = all_metrics["E0"].get(disease, None)
                exp_val = all_metrics[exp_name].get(disease, None)
                if isinstance(e0_val, (int, float)) and isinstance(exp_val, (int, float)):
                    deltas.append(exp_val - e0_val)
                else:
                    deltas.append("")
            comparison_df[delta_col] = deltas

    comparison_path = os.path.join(args.output_dir, "per_disease_auc.csv")
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Per-disease AUC table saved: %s", comparison_path)

    # --- Summary table ---
    summary_rows = []
    for exp_name in all_metrics:
        row = {
            "Experiment": exp_name,
            "Mean AUC": all_metrics[exp_name].get("MEAN", ""),
        }
        if exp_name in all_summaries:
            s = all_summaries[exp_name]
            row["Trainable Params"] = s.get("trainable_params", "")
            row["Total Params"] = s.get("total_params", "")
            row["Best Epoch"] = s.get("best_epoch", "")
            row["Freeze Strategy"] = s.get("freeze_strategy", "")
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary table saved: %s", summary_path)

    # --- Print comparison ---
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
