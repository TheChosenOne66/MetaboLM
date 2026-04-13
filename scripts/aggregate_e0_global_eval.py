#!/usr/bin/env python3
"""Aggregate per-checkpoint E0 global-eval worker outputs.

This script is intended to run locally on CPU after parallel Nebula workers
have written per-checkpoint subdirectories such as:

    outputs/E0_global_eval/T2D/metrics_ckpt_T2D.csv
    outputs/E0_global_eval/obesity/metrics_ckpt_obesity.csv

It reads only the diagonal rows (``ckpt_disease == eval_label``) to build the
same top-level summary files the old serial evaluators used to emit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.endpoints import get_disease_names


def infer_variant(eval_dir: Path, variant: str) -> str:
    if variant != "auto":
        return variant
    return "cohort" if "cohort" in eval_dir.name.lower() else "shared"


def output_filenames(variant: str) -> tuple[str, str]:
    if variant == "cohort":
        return "e0_global_val_metrics_own_preproc.csv", "summary.json"
    return "e0_global_val_metrics.csv", "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate E0 global-eval worker outputs.")
    parser.add_argument("--eval-dir", required=True, help="Root eval directory to aggregate.")
    parser.add_argument(
        "--variant",
        choices=["auto", "shared", "cohort"],
        default="auto",
        help="Select output naming convention. Default infers from eval-dir name.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    variant = infer_variant(eval_dir, args.variant)
    csv_name, summary_name = output_filenames(variant)
    diseases = get_disease_names()

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for disease in diseases:
        metrics_path = eval_dir / disease / f"metrics_ckpt_{disease}.csv"
        if not metrics_path.exists():
            missing.append(disease)
            continue

        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            diag_row = None
            for row in reader:
                if row.get("ckpt_disease") == disease and row.get("eval_label") == disease:
                    diag_row = row
                    break

        if diag_row is None:
            missing.append(disease)
            continue

        auc_raw = (diag_row.get("Val_AUC") or "").strip()
        auc = float("nan") if auc_raw == "" else float(auc_raw)
        rows.append({
            "Disease": disease,
            "Metric": auc,
            "N_Positive": int(diag_row["N_Positive"]),
            "N_Negative": int(diag_row["N_Negative"]),
            "Prevalence_pct": float(diag_row["Prevalence_pct"]),
        })

    metric_col = (
        "Global_Val_AUC_Own_Preproc" if variant == "cohort" else "Global_Val_AUC"
    )
    csv_path = eval_dir / csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Disease", metric_col, "N_Positive", "N_Negative", "Prevalence_pct"],
        )
        writer.writeheader()
        for row in rows:
            metric_value = row["Metric"]
            writer.writerow({
                "Disease": row["Disease"],
                metric_col: (
                    "" if isinstance(metric_value, float) and math.isnan(metric_value)
                    else f"{metric_value:.4f}"
                ),
                "N_Positive": row["N_Positive"],
                "N_Negative": row["N_Negative"],
                "Prevalence_pct": f"{row['Prevalence_pct']:.2f}",
            })

    valid = [
        float(row["Metric"]) for row in rows
        if not (isinstance(row["Metric"], float) and math.isnan(row["Metric"]))
    ]
    mean_auc = sum(valid) / len(valid) if valid else float("nan")

    summary = {
        "eval_set": "global val.csv",
        "preprocessing": (
            "per-disease cohort z-score replayed" if variant == "cohort"
            else "shared preprocess (global z-score only)"
        ),
        "n_completed_ckpts": len(rows),
        "missing_ckpt_diseases": missing,
        "mean_auc": mean_auc,
        "per_disease": {
            row["Disease"]: (
                None if isinstance(row["Metric"], float) and math.isnan(row["Metric"])
                else round(float(row["Metric"]), 4)
            )
            for row in rows
        },
    }
    summary_path = eval_dir / summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    if missing:
        print(f"Missing diagonal outputs for: {missing}")
    print(
        f"Aggregated {len(rows)}/{len(diseases)} checkpoint directories, "
        f"mean_auc={'NaN' if math.isnan(mean_auc) else f'{mean_auc:.4f}'}"
    )


if __name__ == "__main__":
    main()
