"""Distributed training utilities for MetaboLM.

Provides DDP setup following the pattern used in holon_2 on the Nebula
platform.  Supports both single-GPU (no-op) and multi-GPU via torchrun.

Usage::

    torchrun --nproc_per_node=4 scripts/train_multitask.py --config ...

Environment variables set by torchrun:
    RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise distributed process group (NCCL) if running under torchrun.

    Returns:
        Tuple of ``(rank, world_size, device)``.
        For single-GPU runs: ``(0, 1, cuda:0)``.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched via torchrun
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            logger.info(
                "Distributed: rank=%d, world_size=%d, local_rank=%d",
                rank, world_size, local_rank,
            )
    else:
        # Single-GPU fallback
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, device


def cleanup_distributed() -> None:
    """Destroy the process group if it was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True if this is the main process (rank 0) or single-GPU."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier() -> None:
    """Synchronise all processes (no-op for single-GPU)."""
    if dist.is_initialized():
        dist.barrier()
