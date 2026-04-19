#!/usr/bin/env bash
# do.sh — task router for MetaboLM ablations (EXP-000+).
# Usage: bash do.sh <task_name>

set -euo pipefail
cd "$(dirname "$0")"

# Resolve python interpreter: explicit PY env overrides, then tjyprotenix, then PATH.
PY="${PY:-/SPXvePFS/share/miniconda3/envs/tjyprotenix/bin/python}"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python)"
fi

# Number of GPUs for DDP — auto-detect from nvidia-smi, but NUM_GPUS env can override.
if [[ -z "${NUM_GPUS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS="$(nvidia-smi -L | wc -l)"
    else
        NUM_GPUS=1
    fi
fi

# Shared sweep helper: single-GPU → python, multi-GPU → torchrun.
# Multi-GPU is launched via ``${PY} -m torch.distributed.run`` instead of bare
# ``torchrun`` on PATH. ``torchrun`` is a thin shim over
# torch.distributed.run, and in environments where ${PY} is an explicit conda
# env (e.g. tjyprotenix) a PATH-resolved torchrun may point at a different
# python interpreter that lacks project deps.
run_one() {
    local cfg="$1"
    local out_dir="$2"
    mkdir -p "${out_dir}"
    local log="${out_dir}/train.log"
    echo "=== $(date -Iseconds) | config=${cfg} | NUM_GPUS=${NUM_GPUS} | log=${log} ==="
    if (( NUM_GPUS > 1 )); then
        "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
            scripts/train_multitask.py --config "${cfg}" 2>&1 | tee "${log}"
    else
        "${PY}" scripts/train_multitask.py --config "${cfg}" 2>&1 | tee "${log}"
    fi
    # Post-train HVR sidecar (rank-0 operation, always single-process)
    "${PY}" scripts/eval_hierarchy_violation.py multitask \
        --config "${cfg}" \
        --output-dir "${out_dir}" 2>&1 | tee -a "${log}"
}

case "${1:-}" in

    e1_mu_sweep)
        # 6 μ configs serial; each owns all GPUs for its run.
        for mu in 0.0 0.1 0.5 1.0 3.0 10.0; do
            cfg="configs/sft_full_ft_mu_${mu}.yaml"
            out="outputs/E1_mu_sweep/mu_${mu}"
            run_one "${cfg}" "${out}"
        done
        "${PY}" scripts/analyze_mu_sweep.py --sweep-dir outputs/E1_mu_sweep
        ;;

    e1_mu_single)
        # Run a single μ, selected by $2. Example: bash do.sh e1_mu_single 1.0
        mu="${2:?usage: bash do.sh e1_mu_single <mu>}"
        cfg="configs/sft_full_ft_mu_${mu}.yaml"
        out="outputs/E1_mu_sweep/mu_${mu}"
        run_one "${cfg}" "${out}"
        ;;

    analyze_mu_sweep)
        # Re-run aggregation without re-training
        "${PY}" scripts/analyze_mu_sweep.py --sweep-dir outputs/E1_mu_sweep
        ;;

    *)
        echo "Unknown task: ${1:-<empty>}"
        echo "Available: e1_mu_sweep | e1_mu_single <mu> | analyze_mu_sweep"
        exit 2
        ;;
esac
