#!/usr/bin/env python3
"""Run hF1 evaluation on E0, E1(full_ft), E2, E3, E4 checkpoints and aggregate results."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_hf1_all_experiments")


def _resolve_output_path(path: str) -> Path:
    """Match eval_hierarchical_f1.py path resolution (METABOLM_OUTPUT_BASE)."""
    from src.config import _resolve_output_path as _cfg_resolve
    return Path(_cfg_resolve(path))


def run_e0_hf1(e0_dir: str) -> dict:
    """Run hF1 eval on E0 (per-disease checkpoints)."""
    cmd = [
        sys.executable,
        "scripts/eval_hierarchical_f1.py",
        "e0",
        "--e0-dir", e0_dir,
        "--threshold", "0.5",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("E0 hF1 eval failed:\nstdout: %s\nstderr: %s", result.stdout, result.stderr)
        raise RuntimeError("E0 hF1 eval failed")

    json_path = _resolve_output_path(e0_dir) / "hierarchical_f1.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("E0 → hF1=%.4f flat_F1=%.4f delta=%.4f", data["hF1"], data["flat_F1"], data["delta_hF1_flat_F1"])
    return data


def run_multitask_hf1(exp_name: str, config_path: str, output_dir: str) -> dict:
    """Run hF1 eval on a multitask experiment."""
    cmd = [
        sys.executable,
        "scripts/eval_hierarchical_f1.py",
        "multitask",
        "--config", config_path,
        "--output-dir", output_dir,
        "--threshold", "0.5",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("%s hF1 eval failed:\nstdout: %s\nstderr: %s", exp_name, result.stdout, result.stderr)
        raise RuntimeError(f"{exp_name} hF1 eval failed")

    json_path = _resolve_output_path(output_dir) / "hierarchical_f1.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("%s → hF1=%.4f flat_F1=%.4f delta=%.4f", exp_name, data["hF1"], data["flat_F1"], data["delta_hF1_flat_F1"])
    return data


# Each experiment maps to (config_yaml, output_dir)
EXPERIMENTS = {
    "E1_full_ft":   ("configs/sft_full_ft.yaml",   "outputs/E1_sft_full_ft"),
    "E2_head_only": ("configs/sft_head_only.yaml",  "outputs/E2_sft_head_only"),
    "E3_adapter":   ("configs/sft_adapter.yaml",    "outputs/E3_sft_adapter"),
    "E4_lora":      ("configs/sft_lora.yaml",       "outputs/E4_sft_lora"),
}


def main() -> int:
    results = {}
    failed = []

    # E0: per-disease checkpoints
    try:
        data = run_e0_hf1("outputs/E0_reproduction")
        results["E0"] = {
            "hF1": data["hF1"], "flat_F1": data["flat_F1"],
            "delta_hF1_flat_F1": data["delta_hF1_flat_F1"],
            "hP": data["hP"], "hR": data["hR"],
            "n_samples": data["n_samples"],
        }
    except RuntimeError as e:
        logger.error("E0 failed: %s", e)
        failed.append("E0")

    # E1-E4: multitask checkpoints with correct configs
    for exp_name, (config_path, output_dir) in EXPERIMENTS.items():
        try:
            data = run_multitask_hf1(exp_name, config_path, output_dir)
            results[exp_name] = {
                "hF1": data["hF1"], "flat_F1": data["flat_F1"],
                "delta_hF1_flat_F1": data["delta_hF1_flat_F1"],
                "hP": data["hP"], "hR": data["hR"],
                "n_samples": data["n_samples"],
            }
        except RuntimeError as e:
            logger.error("%s failed: %s", exp_name, e)
            failed.append(exp_name)

    if not results:
        logger.error("All experiments failed: %s", failed)
        return 1

    # Aggregate results
    exp_order = ["E0", "E1_full_ft", "E2_head_only", "E3_adapter", "E4_lora"]
    summary = {
        "experiments": [e for e in exp_order if e in results],
        "failed": failed,
        "results": results,
        "best_exp": max(results, key=lambda e: results[e]["hF1"]),
        "best_hF1": max(r["hF1"] for r in results.values()),
    }

    summary_path = _resolve_output_path("outputs/hf1_all_experiments.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", summary_path)
    logger.info("Best: %s with hF1=%.4f", summary["best_exp"], summary["best_hF1"])
    if failed:
        logger.warning("Failed experiments: %s", failed)

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Exp':>14} {'hF1':>8} {'flat_F1':>10} {'delta':>8} {'hP':>8} {'hR':>8}")
    print("-" * 70)
    for exp in exp_order:
        if exp in results:
            r = results[exp]
            print(f"{exp:>14} {r['hF1']:>8.4f} {r['flat_F1']:>10.4f} {r['delta_hF1_flat_F1']:>8.4f} {r['hP']:>8.4f} {r['hR']:>8.4f}")
        elif exp in failed:
            print(f"{exp:>14} {'FAILED':>8}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
