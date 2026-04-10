#!/usr/bin/env python3
"""Generate LEADERBOARD.md from experiment outputs.

Reads configs/leaderboard_manifest.yaml to determine which experiments to
display, dispatches to type-specific parsers that read existing CSV/JSON
files under outputs/, and renders a two-section Markdown leaderboard.

This script is READ-ONLY w.r.t. training outputs. It never modifies any
file under outputs/, src/, or any trainer script. The only file it writes
is LEADERBOARD.md at the repo root.

Usage:
    python scripts/update_leaderboard.py
    python scripts/update_leaderboard.py --manifest configs/leaderboard_manifest.yaml
    python scripts/update_leaderboard.py --output LEADERBOARD.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


class ExperimentType(str, Enum):
    PER_DISEASE_SFT = "per_disease_sft"
    MULTITASK_SFT = "multitask_sft"
    MULTITASK_RL = "multitask_rl"


class ExperimentStatus(str, Enum):
    PLANNED = "planned"
    PARTIAL = "partial"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ExperimentRow:
    """Unified in-memory representation of one experiment for rendering."""

    # Fields from manifest (always present)
    id: str
    display_name: str
    phase: str
    exp_type: ExperimentType
    config: str
    output_dir: str
    description: str
    innovation: str | None

    # Fields from parser (None if parser returned without data)
    status: ExperimentStatus = ExperimentStatus.PLANNED
    mean_auroc: float | None = None
    mean_auprc: float | None = None
    hierarchy_violation_rate: float | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    freeze_strategy: str | None = None
    best_epoch: int | None = None
    per_disease_auroc: dict[str, float] = field(default_factory=dict)
    per_disease_auprc: dict[str, float] = field(default_factory=dict)

    # Metadata
    num_completed_diseases: int | None = None  # E0 only
    total_diseases: int = 16
    error_message: str | None = None


def main() -> int:
    """CLI entry point. Returns exit code."""
    raise NotImplementedError("main() implemented in Task 9")


if __name__ == "__main__":
    sys.exit(main())
