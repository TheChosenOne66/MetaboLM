#!/usr/bin/env python3
"""Run hF1 evaluation on all μ sweep checkpoints and aggregate results."""

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
logger = logging.getLogger("run_hf1_mu_sweep")


def _resolve_output_path(path: str) -> Path:
    """Match eval_hierarchical_f1.py path resolution (METABOLM_OUTPUT_BASE)."""
    from src.config import _resolve_output_path as _cfg_resolve
    return Path(_cfg_resolve(path))


def run_hf1_for_mu(mu: str, config_path: str, output_dir: str) -> dict:
    """Run eval_hierarchical_f1.py for a single μ value."""
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
        logger.error("hF1 eval failed for μ=%s:\nstdout: %s\nstderr: %s", mu, result.stdout, result.stderr)
        raise RuntimeError(f"hF1 eval failed for μ={mu}")

    # Read the generated JSON — use resolved path to match eval script
    json_path = _resolve_output_path(output_dir) / "hierarchical_f1.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("μ=%s → hF1=%.4f flat_F1=%.4f delta=%.4f", mu, data["hF1"], data["flat_F1"], data["delta_hF1_flat_F1"])
    return data


def main() -> int:
    mu_values = ["0.0", "0.1", "0.5", "1.0", "3.0", "10.0"]
    config_path = "configs/sft_full_ft.yaml"
    results = {}

    for mu in mu_values:
        output_dir = f"outputs/E1_mu_sweep/mu_{mu}"
        try:
            data = run_hf1_for_mu(mu, config_path, output_dir)
            results[mu] = {
                "hF1": data["hF1"],
                "flat_F1": data["flat_F1"],
                "delta_hF1_flat_F1": data["delta_hF1_flat_F1"],
                "hP": data["hP"],
                "hR": data["hR"],
                "n_samples": data["n_samples"],
            }
        except RuntimeError:
            return 1

    # Aggregate results
    summary = {
        "mu_values": mu_values,
        "results": results,
        "best_mu": max(results, key=lambda m: results[m]["hF1"]),
        "best_hF1": max(r["hF1"] for r in results.values()),
    }

    summary_path = _resolve_output_path("outputs/E1_mu_sweep/hf1_comparison.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", summary_path)
    logger.info("Best μ=%s with hF1=%.4f", summary["best_mu"], summary["best_hF1"])

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'μ':>6} {'hF1':>8} {'flat_F1':>10} {'delta':>8} {'hP':>8} {'hR':>8}")
    print("-" * 60)
    for mu in mu_values:
        r = results[mu]
        print(f"{mu:>6} {r['hF1']:>8.4f} {r['flat_F1']:>10.4f} {r['delta_hF1_flat_F1']:>8.4f} {r['hP']:>8.4f} {r['hR']:>8.4f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
