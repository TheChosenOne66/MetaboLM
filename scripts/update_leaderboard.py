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


# ── Manifest loading ──────────────────────────────────────────────────────

REQUIRED_EXPERIMENT_FIELDS = (
    "id", "display_name", "phase", "type", "config", "output_dir", "description",
)


@dataclass
class ManifestExperiment:
    """One entry from the manifest YAML."""
    id: str
    display_name: str
    phase: str
    exp_type: ExperimentType
    config: str
    output_dir: str
    description: str
    innovation: str | None


@dataclass
class Manifest:
    experiments: list[ManifestExperiment]
    diseases: list[str]


def load_manifest(path: Path) -> Manifest:
    """Load and validate manifest YAML. Exit 1 on any failure."""
    path = Path(path)
    if not path.exists():
        print(f"ERROR: manifest file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"ERROR: failed to parse manifest YAML: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(raw, dict):
        print(f"ERROR: manifest root must be a mapping, got {type(raw).__name__}", file=sys.stderr)
        sys.exit(1)

    if "experiments" not in raw or "diseases" not in raw:
        print(
            "ERROR: manifest must contain 'experiments' and 'diseases' keys",
            file=sys.stderr,
        )
        sys.exit(1)

    experiments: list[ManifestExperiment] = []
    for i, entry in enumerate(raw["experiments"]):
        if not isinstance(entry, dict):
            print(
                f"ERROR: experiment #{i} must be a mapping, got {type(entry).__name__}",
                file=sys.stderr,
            )
            sys.exit(1)

        missing = [f for f in REQUIRED_EXPERIMENT_FIELDS if f not in entry]
        if missing:
            print(
                f"ERROR: experiment #{i} (id={entry.get('id', '?')}) missing required fields: {missing}",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            exp_type = ExperimentType(entry["type"])
        except ValueError:
            print(
                f"ERROR: experiment {entry['id']} has invalid type '{entry['type']}'. "
                f"Must be one of: {[t.value for t in ExperimentType]}",
                file=sys.stderr,
            )
            sys.exit(1)

        experiments.append(
            ManifestExperiment(
                id=entry["id"],
                display_name=entry["display_name"],
                phase=entry["phase"],
                exp_type=exp_type,
                config=entry["config"],
                output_dir=entry["output_dir"],
                description=entry["description"],
                innovation=entry.get("innovation"),
            )
        )

    diseases = raw["diseases"]
    if not isinstance(diseases, list) or not all(isinstance(d, str) for d in diseases):
        print("ERROR: manifest 'diseases' must be a list of strings", file=sys.stderr)
        sys.exit(1)

    return Manifest(experiments=experiments, diseases=list(diseases))


# ── Parsers ───────────────────────────────────────────────────────────────

def parse_per_disease_sft(
    exp_dir: Path, canonical_diseases: list[str]
) -> dict[str, Any]:
    """Parse E0 output: read finetune_summary_metrics.csv.

    Returns a dict with keys that map onto ExperimentRow fields.
    Never raises — on any error returns status=ERROR with error_message.
    """
    exp_dir = Path(exp_dir)
    csv_path = exp_dir / "finetune_summary_metrics.csv"

    empty_result: dict[str, Any] = {
        "status": ExperimentStatus.PLANNED,
        "mean_auroc": None,
        "mean_auprc": None,
        "hierarchy_violation_rate": None,
        "per_disease_auroc": {},
        "per_disease_auprc": {},
        "trainable_params": None,
        "total_params": None,
        "freeze_strategy": None,
        "best_epoch": None,
        "num_completed_diseases": None,
        "error_message": None,
    }

    if not exp_dir.exists() or not csv_path.exists():
        return empty_result

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {**empty_result, "status": ExperimentStatus.ERROR,
                "error_message": f"CSV parse error: {e}"}

    if "Disease" not in df.columns or "Val_AUC" not in df.columns:
        return {**empty_result, "status": ExperimentStatus.ERROR,
                "error_message": "CSV missing required columns Disease/Val_AUC"}

    per_disease_auroc: dict[str, float] = {}
    canonical_set = set(canonical_diseases)
    for _, row in df.iterrows():
        disease = str(row["Disease"])
        if disease not in canonical_set:
            print(
                f"[WARN] parse_per_disease_sft({exp_dir.name}): "
                f"unknown disease '{disease}' in CSV, skipping",
                file=sys.stderr,
            )
            continue
        try:
            per_disease_auroc[disease] = float(row["Val_AUC"])
        except (ValueError, TypeError):
            print(
                f"[WARN] parse_per_disease_sft({exp_dir.name}): "
                f"non-numeric Val_AUC for '{disease}', skipping",
                file=sys.stderr,
            )
            continue

    n_done = len(per_disease_auroc)
    if n_done == 0:
        return empty_result

    mean_auroc = sum(per_disease_auroc.values()) / n_done

    # Best epoch: take max across completed rows (for display only)
    best_epoch: int | None = None
    if "Best_Epoch" in df.columns:
        try:
            valid_epochs = [
                int(e) for e, d in zip(df["Best_Epoch"], df["Disease"])
                if str(d) in canonical_set
            ]
            if valid_epochs:
                best_epoch = max(valid_epochs)
        except (ValueError, TypeError):
            pass

    status = (
        ExperimentStatus.COMPLETED if n_done == len(canonical_diseases)
        else ExperimentStatus.PARTIAL
    )

    return {
        "status": status,
        "mean_auroc": mean_auroc,
        "mean_auprc": None,  # E0 CSV has no AUPRC
        "hierarchy_violation_rate": None,
        "per_disease_auroc": per_disease_auroc,
        "per_disease_auprc": {},
        "trainable_params": None,  # E0 hardcoded size, not in CSV
        "total_params": None,
        "freeze_strategy": None,
        "best_epoch": best_epoch,
        "num_completed_diseases": n_done,
        "error_message": None,
    }


def parse_multitask(
    exp_dir: Path, canonical_diseases: list[str]
) -> dict[str, Any]:
    """Parse E1-E6 output: read summary.json + multitask_metrics.csv.

    Both files required for COMPLETED status. Missing file(s) → PLANNED.
    Parse error → ERROR with error_message. Used for both multitask_sft
    and multitask_rl experiment types (identical output schema).
    """
    exp_dir = Path(exp_dir)
    summary_path = exp_dir / "summary.json"
    csv_path = exp_dir / "multitask_metrics.csv"

    empty_result: dict[str, Any] = {
        "status": ExperimentStatus.PLANNED,
        "mean_auroc": None,
        "mean_auprc": None,
        "hierarchy_violation_rate": None,
        "per_disease_auroc": {},
        "per_disease_auprc": {},
        "trainable_params": None,
        "total_params": None,
        "freeze_strategy": None,
        "best_epoch": None,
        "num_completed_diseases": None,
        "error_message": None,
    }

    if not exp_dir.exists() or not summary_path.exists() or not csv_path.exists():
        return empty_result

    # Parse summary.json
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except json.JSONDecodeError as e:
        return {**empty_result, "status": ExperimentStatus.ERROR,
                "error_message": f"summary.json parse error: {e}"}
    except Exception as e:
        return {**empty_result, "status": ExperimentStatus.ERROR,
                "error_message": f"summary.json read error: {e}"}

    # Parse multitask_metrics.csv
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {**empty_result, "status": ExperimentStatus.ERROR,
                "error_message": f"multitask_metrics.csv parse error: {e}"}

    if "Disease" not in df.columns or "Val_AUC" not in df.columns:
        return {**empty_result, "status": ExperimentStatus.ERROR,
                "error_message": "CSV missing Disease/Val_AUC columns"}

    # Filter out the MEAN row explicitly; it's an aggregate, not a disease
    df_diseases = df[df["Disease"] != "MEAN"]

    per_disease_auroc: dict[str, float] = {}
    canonical_set = set(canonical_diseases)
    for _, row in df_diseases.iterrows():
        disease = str(row["Disease"])
        if disease not in canonical_set:
            print(
                f"[WARN] parse_multitask({exp_dir.name}): "
                f"unknown disease '{disease}' in CSV, skipping",
                file=sys.stderr,
            )
            continue
        try:
            per_disease_auroc[disease] = float(row["Val_AUC"])
        except (ValueError, TypeError):
            continue

    # Prefer summary.json.best_mean_auc; fall back to CSV MEAN row
    mean_auroc: float | None = None
    if "best_mean_auc" in summary:
        try:
            mean_auroc = float(summary["best_mean_auc"])
        except (ValueError, TypeError):
            pass

    csv_mean: float | None = None
    mean_rows = df[df["Disease"] == "MEAN"]
    if len(mean_rows) > 0:
        try:
            csv_mean = float(mean_rows.iloc[0]["Val_AUC"])
        except (ValueError, TypeError):
            pass

    if mean_auroc is not None and csv_mean is not None:
        if abs(mean_auroc - csv_mean) > 1e-3:
            print(
                f"[INFO] parse_multitask({exp_dir.name}): "
                f"mean AUROC mismatch: json={mean_auroc:.4f}, "
                f"csv_mean_row={csv_mean:.4f}, using json",
                file=sys.stderr,
            )
    elif mean_auroc is None and csv_mean is not None:
        mean_auroc = csv_mean

    # trainable_params / total_params from summary.json
    def _int_or_none(val: Any) -> int | None:
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    return {
        "status": ExperimentStatus.COMPLETED,
        "mean_auroc": mean_auroc,
        "mean_auprc": None,  # Not emitted by current trainers
        "hierarchy_violation_rate": summary.get("hierarchy_violation_rate"),
        "per_disease_auroc": per_disease_auroc,
        "per_disease_auprc": {},
        "trainable_params": _int_or_none(summary.get("trainable_params")),
        "total_params": _int_or_none(summary.get("total_params")),
        "freeze_strategy": summary.get("freeze_strategy"),
        "best_epoch": _int_or_none(summary.get("best_epoch")),
        "num_completed_diseases": len(per_disease_auroc),
        "error_message": None,
    }


# ── Collection / dispatch ─────────────────────────────────────────────────

# Map experiment type → parser function
_PARSERS = {
    ExperimentType.PER_DISEASE_SFT: parse_per_disease_sft,
    ExperimentType.MULTITASK_SFT: parse_multitask,
    ExperimentType.MULTITASK_RL: parse_multitask,
}


def collect_experiment_rows(
    manifest: Manifest, repo_root: Path
) -> list[ExperimentRow]:
    """Dispatch each manifest entry to its parser and build ExperimentRows."""
    repo_root = Path(repo_root)
    rows: list[ExperimentRow] = []

    for entry in manifest.experiments:
        # Resolve output_dir relative to repo_root if it's not absolute
        out_path = Path(entry.output_dir)
        if not out_path.is_absolute():
            out_path = repo_root / out_path

        parser = _PARSERS[entry.exp_type]
        try:
            parsed = parser(out_path, manifest.diseases)
        except Exception as e:
            # Defensive: parsers should NEVER raise, but belt-and-suspenders.
            print(
                f"[ERROR] {entry.id} parser raised unexpectedly: {e}",
                file=sys.stderr,
            )
            parsed = {
                "status": ExperimentStatus.ERROR,
                "mean_auroc": None,
                "mean_auprc": None,
                "hierarchy_violation_rate": None,
                "per_disease_auroc": {},
                "per_disease_auprc": {},
                "trainable_params": None,
                "total_params": None,
                "freeze_strategy": None,
                "best_epoch": None,
                "num_completed_diseases": None,
                "error_message": f"Unexpected exception: {e}",
            }

        row = ExperimentRow(
            id=entry.id,
            display_name=entry.display_name,
            phase=entry.phase,
            exp_type=entry.exp_type,
            config=entry.config,
            output_dir=entry.output_dir,
            description=entry.description,
            innovation=entry.innovation,
            status=parsed["status"],
            mean_auroc=parsed["mean_auroc"],
            mean_auprc=parsed["mean_auprc"],
            hierarchy_violation_rate=parsed["hierarchy_violation_rate"],
            trainable_params=parsed["trainable_params"],
            total_params=parsed["total_params"],
            freeze_strategy=parsed["freeze_strategy"],
            best_epoch=parsed["best_epoch"],
            per_disease_auroc=parsed["per_disease_auroc"],
            per_disease_auprc=parsed["per_disease_auprc"],
            num_completed_diseases=parsed["num_completed_diseases"],
            total_diseases=len(manifest.diseases),
            error_message=parsed["error_message"],
        )
        rows.append(row)

    return rows


def main() -> int:
    """CLI entry point. Returns exit code."""
    raise NotImplementedError("main() implemented in Task 9")


if __name__ == "__main__":
    sys.exit(main())
