#!/usr/bin/env python3
"""Run the EXP-000 hierarchy-loss mu sweep.

This is the Nebula-friendly equivalent of the local shell sweep:
train each ``configs/sft_full_ft_mu_*.yaml`` variant, run the HVR sidecar,
then aggregate the sweep into summary files.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MUS = ["0.0", "0.1", "0.5", "1.0", "3.0", "10.0"]


def resolve_output_path(path: Path) -> Path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import _resolve_output_path

    return Path(_resolve_output_path(str(path)))


def detect_num_gpus() -> int:
    if os.environ.get("NUM_GPUS"):
        return max(1, int(os.environ["NUM_GPUS"]))

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 1

    return max(1, len([line for line in result.stdout.splitlines() if line.strip()]))


def stream_command(cmd: list[str], log_path: Path, append: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    print(f"[run_mu_sweep] cmd={' '.join(cmd)}")
    with log_path.open(mode, encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_one(mu: str, args: argparse.Namespace, num_gpus: int) -> None:
    cfg = args.config_dir / f"sft_full_ft_mu_{mu}.yaml"
    out_dir = args.sweep_dir / f"mu_{mu}"
    log_path = out_dir / "train.log"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config for mu={mu}: {cfg}")

    out_dir.mkdir(parents=True, exist_ok=True)
    header = (
        f"=== config={cfg} | output_dir={out_dir} | "
        f"NUM_GPUS={num_gpus} | log={log_path} ===\n"
    )
    print(header, end="")
    log_path.write_text(header, encoding="utf-8")

    if num_gpus > 1:
        train_cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "scripts/train_multitask.py",
            "--config",
            str(cfg),
        ]
    else:
        train_cmd = [
            sys.executable,
            "scripts/train_multitask.py",
            "--config",
            str(cfg),
        ]
    stream_command(train_cmd, log_path, append=True)

    hvr_cmd = [
        sys.executable,
        "scripts/eval_hierarchy_violation.py",
        "multitask",
        "--config",
        str(cfg),
        "--output-dir",
        str(out_dir),
    ]
    stream_command(hvr_cmd, log_path, append=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP-000 mu sweep.")
    parser.add_argument(
        "--mus",
        nargs="+",
        default=DEFAULT_MUS,
        help="Mu values to run. Default: full sweep.",
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("outputs/E1_mu_sweep"),
        help="Directory for mu_* outputs.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing sft_full_ft_mu_*.yaml.",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Train/evaluate only; do not regenerate sweep summaries.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "Number of GPUs for torch.distributed.run. Default: auto-detect "
            "visible GPUs via NUM_GPUS or nvidia-smi."
        ),
    )
    args = parser.parse_args()
    args.sweep_dir = resolve_output_path(args.sweep_dir)

    num_gpus = detect_num_gpus() if args.num_gpus is None else max(1, args.num_gpus)
    for mu in args.mus:
        run_one(mu, args, num_gpus)

    if not args.skip_analyze:
        analyze_cmd = [
            sys.executable,
            "scripts/analyze_mu_sweep.py",
            "--sweep-dir",
            str(args.sweep_dir),
            "--config-dir",
            str(args.config_dir),
        ]
        stream_command(analyze_cmd, args.sweep_dir / "analyze_mu_sweep.log")


if __name__ == "__main__":
    main()
