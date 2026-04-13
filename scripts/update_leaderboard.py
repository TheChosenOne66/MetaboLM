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
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_output_dir(path_str: str, repo_root: Path) -> Path:
    """Resolve leaderboard output dirs under repo root or shared output base."""
    path = Path(path_str)
    if path.is_absolute():
        return path

    output_base = os.environ.get("METABOLM_OUTPUT_BASE", "").strip()
    if output_base:
        parts = path.parts
        if parts and parts[0] == "outputs":
            path = Path(*parts[1:])
        return Path(output_base) / path

    return repo_root / path


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

    # Optional global-eval variants (populated from manifest ``eval_dirs``).
    # Keys are eval-variant labels ("global_shared", "global_cohort", ...),
    # values are per-disease AUROC dicts / their mean. ``None`` means the
    # variant was not declared in the manifest or its output dir was absent.
    eval_variant_mean_auroc: dict[str, float | None] = field(default_factory=dict)
    eval_variant_per_disease_auroc: dict[str, dict[str, float]] = field(default_factory=dict)

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
    # Optional: additional evaluation output directories keyed by variant name
    # (e.g. ``global_shared``, ``global_cohort``). Each maps a label shown in
    # LEADERBOARD.md to the filesystem path produced by one of the eval scripts
    # under ``scripts/eval_e0_global*.py`` following the per-ckpt subdir
    # contract (``<dir>/<ckpt_disease>/metrics_ckpt_<ckpt_disease>.csv``).
    eval_dirs: dict[str, str] = field(default_factory=dict)


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

        # ``eval_dirs`` is optional; tolerate missing/empty/malformed gracefully.
        raw_eval_dirs = entry.get("eval_dirs") or {}
        if not isinstance(raw_eval_dirs, dict):
            print(
                f"ERROR: experiment {entry['id']} 'eval_dirs' must be a mapping, "
                f"got {type(raw_eval_dirs).__name__}",
                file=sys.stderr,
            )
            sys.exit(1)
        eval_dirs = {str(k): str(v) for k, v in raw_eval_dirs.items()}

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
                eval_dirs=eval_dirs,
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
    """Parse E0 output from summary CSV and per-disease metric files.

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

    if not exp_dir.exists():
        return empty_result

    per_disease_auroc: dict[str, float] = {}
    best_epoch_by_disease: dict[str, int] = {}
    canonical_set = set(canonical_diseases)

    def _merge_metrics(df: pd.DataFrame, source_label: str) -> str | None:
        if "Disease" not in df.columns or "Val_AUC" not in df.columns:
            return f"{source_label} missing required columns Disease/Val_AUC"

        for _, row in df.iterrows():
            disease = str(row["Disease"])
            if disease not in canonical_set:
                print(
                    f"[WARN] parse_per_disease_sft({exp_dir.name}): "
                    f"unknown disease '{disease}' in {source_label}, skipping",
                    file=sys.stderr,
                )
                continue
            try:
                per_disease_auroc[disease] = float(row["Val_AUC"])
            except (ValueError, TypeError):
                print(
                    f"[WARN] parse_per_disease_sft({exp_dir.name}): "
                    f"non-numeric Val_AUC for '{disease}' in {source_label}, skipping",
                    file=sys.stderr,
                )
                continue

            if "Best_Epoch" in df.columns:
                try:
                    best_epoch_by_disease[disease] = int(row["Best_Epoch"])
                except (ValueError, TypeError):
                    pass

        return None

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return {
                **empty_result,
                "status": ExperimentStatus.ERROR,
                "error_message": f"CSV parse error: {e}",
            }

        error_message = _merge_metrics(df, "finetune_summary_metrics.csv")
        if error_message is not None:
            return {
                **empty_result,
                "status": ExperimentStatus.ERROR,
                "error_message": error_message,
            }

    # Fall back to per-disease metric files when the summary CSV is missing
    # or incomplete. This keeps the leaderboard aligned with finished runs
    # even if the aggregate file was not refreshed.
    for disease in canonical_diseases:
        if disease in per_disease_auroc:
            continue

        metrics_path = exp_dir / disease / f"finetune_metrics_{disease}.csv"
        if not metrics_path.exists():
            continue

        try:
            disease_df = pd.read_csv(metrics_path)
        except Exception as e:
            return {
                **empty_result,
                "status": ExperimentStatus.ERROR,
                "error_message": f"{metrics_path.name} parse error: {e}",
            }

        error_message = _merge_metrics(disease_df, metrics_path.name)
        if error_message is not None:
            return {
                **empty_result,
                "status": ExperimentStatus.ERROR,
                "error_message": error_message,
            }

    n_done = len(per_disease_auroc)
    if n_done == 0:
        return empty_result

    mean_auroc = sum(per_disease_auroc.values()) / n_done

    # Best epoch: take max across completed rows (for display only)
    best_epoch = max(best_epoch_by_disease.values()) if best_epoch_by_disease else None

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


def parse_global_eval_dir(
    eval_dir: Path, canonical_diseases: list[str]
) -> tuple[dict[str, float], float | None]:
    """Parse per-ckpt metrics under an eval output dir.

    The eval scripts under ``scripts/eval_e0_global*.py`` write per-ckpt
    metrics to ``<eval_dir>/<ckpt_disease>/metrics_ckpt_<ckpt_disease>.csv``
    with the schema ``ckpt_disease, eval_label, Val_AUC, N_Positive,
    N_Negative, Prevalence_pct``. Only diagonal rows (``ckpt_disease ==
    eval_label``) feed the main leaderboard columns — off-diagonal cross-
    disease rows (if ever added for 16×16 analysis) are ignored here.

    Returns ``(per_disease_auroc, mean_auroc)``. Missing/empty dir returns
    ``({}, None)``.
    """
    per_disease: dict[str, float] = {}
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        return per_disease, None

    for disease in canonical_diseases:
        metrics_path = eval_dir / disease / f"metrics_ckpt_{disease}.csv"
        if not metrics_path.exists():
            continue
        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(
                f"[WARN] parse_global_eval_dir({eval_dir.name}/{disease}): "
                f"CSV parse error ({e}); skipping",
                file=sys.stderr,
            )
            continue

        required = {"ckpt_disease", "eval_label", "Val_AUC"}
        if not required.issubset(df.columns):
            print(
                f"[WARN] parse_global_eval_dir({eval_dir.name}/{disease}): "
                f"missing columns {required - set(df.columns)}; skipping",
                file=sys.stderr,
            )
            continue

        diag = df[(df["ckpt_disease"] == disease) & (df["eval_label"] == disease)]
        if diag.empty:
            continue
        val_auc = diag["Val_AUC"].iloc[0]
        # Eval scripts write an empty ``Val_AUC`` cell when ``roc_auc_score``
        # fails (e.g. single-class labels on the eval set). pandas loads the
        # empty cell as ``NaN``, and ``float(NaN)`` silently returns ``NaN``
        # instead of raising — so the ``except`` clause below is *not*
        # sufficient on its own. Guard with ``pd.isna`` first, otherwise a
        # NaN would slip into ``per_disease`` and poison the mean_auroc
        # computed from it.
        if pd.isna(val_auc):
            continue
        try:
            per_disease[disease] = float(val_auc)
        except (ValueError, TypeError):
            continue

    mean_auroc = (
        sum(per_disease.values()) / len(per_disease) if per_disease else None
    )
    return per_disease, mean_auroc


def collect_experiment_rows(
    manifest: Manifest, repo_root: Path
) -> list[ExperimentRow]:
    """Dispatch each manifest entry to its parser and build ExperimentRows."""
    repo_root = Path(repo_root)
    rows: list[ExperimentRow] = []

    for entry in manifest.experiments:
        out_path = resolve_output_dir(entry.output_dir, repo_root)
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

        # Parse any additional eval variants declared in the manifest (e.g.
        # E0's global-val shared/cohort evaluations). Missing dirs are OK —
        # the variant just shows up as a dash in the rendered diagnostics.
        eval_variant_mean: dict[str, float | None] = {}
        eval_variant_per_disease: dict[str, dict[str, float]] = {}
        for variant_label, variant_dir in entry.eval_dirs.items():
            resolved = resolve_output_dir(variant_dir, repo_root)
            per_disease, mean_auc = parse_global_eval_dir(resolved, manifest.diseases)
            eval_variant_mean[variant_label] = mean_auc
            eval_variant_per_disease[variant_label] = per_disease

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
            eval_variant_mean_auroc=eval_variant_mean,
            eval_variant_per_disease_auroc=eval_variant_per_disease,
            num_completed_diseases=parsed["num_completed_diseases"],
            total_diseases=len(manifest.diseases),
            error_message=parsed["error_message"],
        )
        rows.append(row)

    return rows


# ── Rendering ─────────────────────────────────────────────────────────────

def _format_float(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _format_params(trainable: int | None, total: int | None) -> str:
    if trainable is None:
        return "—"

    def _humanize(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.0f}K"
        return str(n)

    trainable_str = _humanize(trainable)
    if total and total > 0:
        pct = 100.0 * trainable / total
        return f"{trainable_str} ({pct:.2f}%)"
    return trainable_str


_PHASE_ABBREV = {"phase1": "P1", "phase2": "P2", "phase3": "P3"}


# Human-readable names for eval variants declared in the manifest's
# ``eval_dirs`` mapping. Unknown variant labels fall back to the raw key.
_EVAL_VARIANT_DISPLAY = {
    "global_shared": "Global val — shared preprocess (E1-E4 pipeline)",
    "global_cohort": "Global val — own pipeline (per-disease cohort z-score)",
}


def _rows_with_eval_variants(rows: list[ExperimentRow]) -> list[ExperimentRow]:
    """Filter to rows that have at least one non-empty eval variant."""
    result = []
    for r in rows:
        if any(r.eval_variant_per_disease_auroc.get(v)
               for v in r.eval_variant_mean_auroc):
            result.append(r)
    return result


def _render_global_eval_diagnostics(
    rows: list[ExperimentRow], diseases: list[str]
) -> list[str]:
    """Render the E0 Global-Val Diagnostics section.

    Renders one sub-section per experiment that declared ``eval_dirs`` in
    the manifest. Each shows a small Summary table (one row per variant +
    the primary Mean AUROC as context) and a per-disease table with the
    primary row and each variant's row.

    Returns ``[]`` when no experiment has any eval variant data — the
    caller simply omits the section.
    """
    relevant = _rows_with_eval_variants(rows)
    if not relevant:
        return []

    lines: list[str] = []
    lines.append("## Global-Val Diagnostics")
    lines.append("")
    lines.append(
        "Additional eval modes for experiments that declare ``eval_dirs`` in "
        "the manifest. The primary *Mean AUROC* above may use a different "
        "evaluation set (e.g. E0's balanced per-disease subset). The rows "
        "below re-evaluate the same checkpoints on the shared ``val.csv`` "
        "so they can be compared like-for-like with E1-E4."
    )
    lines.append("")

    for row in relevant:
        # ``display_name`` typically already contains the experiment id
        # (e.g. ``E0 — Per-Disease Baseline (Reproduction)``), so use it
        # directly rather than prefixing with ``row.id`` again.
        lines.append(f"### {row.display_name}")
        lines.append("")
        # Summary table
        lines.append("| Eval Mode | Mean AUROC |")
        lines.append("|---|:---:|")
        primary_label = (
            "Primary (paper-style balanced subset)"
            if row.exp_type == ExperimentType.PER_DISEASE_SFT
            else "Primary"
        )
        lines.append(
            f"| {primary_label} | {_format_float(row.mean_auroc)} |"
        )
        for variant_label, mean_auc in row.eval_variant_mean_auroc.items():
            display = _EVAL_VARIANT_DISPLAY.get(variant_label, variant_label)
            lines.append(f"| {display} | {_format_float(mean_auc)} |")
        lines.append("")

        # Per-disease table: one column for the primary, one per variant
        variant_labels = list(row.eval_variant_mean_auroc.keys())
        header_cells = ["Disease", "Primary"] + [
            _EVAL_VARIANT_DISPLAY.get(v, v) for v in variant_labels
        ]
        lines.append("| " + " | ".join(header_cells) + " |")
        align_cells = ["---"] + [":---:"] * (len(variant_labels) + 1)
        lines.append("| " + " | ".join(align_cells) + " |")

        for disease in diseases:
            cells = [disease, _format_float(row.per_disease_auroc.get(disease))]
            for variant_label in variant_labels:
                variant_per_disease = row.eval_variant_per_disease_auroc.get(
                    variant_label, {}
                )
                cells.append(_format_float(variant_per_disease.get(disease)))
            lines.append("| " + " | ".join(cells) + " |")

        # MEAN row for this sub-section
        mean_cells = ["**MEAN**", f"**{_format_float(row.mean_auroc)}**"]
        for variant_label in variant_labels:
            mean_cells.append(
                f"**{_format_float(row.eval_variant_mean_auroc.get(variant_label))}**"
            )
        lines.append("| " + " | ".join(mean_cells) + " |")
        lines.append("")

    return lines


def _format_status(row: ExperimentRow) -> str:
    if row.status == ExperimentStatus.COMPLETED:
        return "✅ Done"
    if row.status == ExperimentStatus.PARTIAL:
        total = row.total_diseases
        n = row.num_completed_diseases or 0
        return f"⚡ Partial ({n}/{total})"
    if row.status == ExperimentStatus.PLANNED:
        return "⬜ Planned"
    if row.status == ExperimentStatus.ERROR:
        return "❌ Error"
    return "?"


def _format_mean_with_partial_warning(row: ExperimentRow) -> str:
    base = _format_float(row.mean_auroc)
    if row.status == ExperimentStatus.PARTIAL and row.mean_auroc is not None:
        return f"{base} ⚠️"
    return base


def render_markdown(rows: list[ExperimentRow], diseases: list[str]) -> str:
    """Render the full LEADERBOARD.md content as a string."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []

    # Header
    lines.append("# MetaboLM Post-Training — Experiment Leaderboard")
    lines.append("")
    lines.append(
        "*Auto-generated by `scripts/update_leaderboard.py`. "
        "Do not edit manually.*"
    )
    lines.append(f"*Last updated: {now}*")
    lines.append(
        "*Manifest: "
        "[`configs/leaderboard_manifest.yaml`]"
        "(configs/leaderboard_manifest.yaml)*"
    )
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| ID | Experiment | Phase | Status | Mean AUROC | "
        "Mean AUPRC | Hier. Violation | Trainable Params | Best Epoch |"
    )
    lines.append(
        "|----|-----------|:-----:|:------:|:----------:|:----------:|"
        ":---------------:|:----------------:|:----------:|"
    )
    for row in rows:
        lines.append(
            f"| {row.id} | {row.display_name} | "
            f"{_PHASE_ABBREV.get(row.phase, row.phase)} | "
            f"{_format_status(row)} | "
            f"{_format_mean_with_partial_warning(row)} | "
            f"{_format_float(row.mean_auprc)} | "
            f"{_format_float(row.hierarchy_violation_rate)} | "
            f"{_format_params(row.trainable_params, row.total_params)} | "
            f"{row.best_epoch if row.best_epoch is not None else '—'} |"
        )
    lines.append("")
    lines.append(
        "**Legend**: ✅ Done · ⚡ Partial · ⬜ Planned · ❌ Error · "
        "⚠️ = partial data, metric computed over completed diseases only"
    )
    lines.append("")

    # Per-disease table
    lines.append("## Per-Disease AUROC")
    lines.append("")
    header_cells = ["Disease"] + [row.id for row in rows]
    lines.append("| " + " | ".join(header_cells) + " |")
    align_cells = ["---"] + [":---:" for _ in rows]
    lines.append("| " + " | ".join(align_cells) + " |")

    for disease in diseases:
        cells = [disease]
        for row in rows:
            value = row.per_disease_auroc.get(disease)
            cells.append(_format_float(value))
        lines.append("| " + " | ".join(cells) + " |")

    # MEAN row
    mean_cells = ["**MEAN**"]
    for row in rows:
        mean_cells.append(f"**{_format_float(row.mean_auroc)}**")
    lines.append("| " + " | ".join(mean_cells) + " |")
    lines.append("")

    # Optional diagnostics section for experiments with ``eval_dirs``.
    lines.extend(_render_global_eval_diagnostics(rows, diseases))

    # Experiment details
    lines.append("## Experiment Details")
    lines.append("")
    for row in rows:
        lines.append(f"### {row.display_name}")
        lines.append(f"- **ID**: {row.id}")
        lines.append(f"- **Phase**: {row.phase}")
        lines.append(f"- **Config**: `{row.config}`")
        lines.append(f"- **Output dir**: `{row.output_dir}`")
        lines.append(f"- **Description**: {row.description}")
        if row.innovation:
            lines.append(f"- **Innovation**: {row.innovation}")
        lines.append(f"- **Status**: {_format_status(row)}")
        if row.freeze_strategy:
            lines.append(f"- **Freeze strategy**: `{row.freeze_strategy}`")
        if row.status == ExperimentStatus.ERROR and row.error_message:
            lines.append(f"- **Error**: {row.error_message}")
        if (row.status == ExperimentStatus.PARTIAL
                and row.num_completed_diseases is not None):
            done = set(row.per_disease_auroc.keys())
            pending = [d for d in diseases if d not in done]
            lines.append(f"- **Completed diseases**: {', '.join(sorted(done))}")
            if pending:
                lines.append(f"- **Pending diseases**: {', '.join(pending)}")
        lines.append("")

    return "\n".join(lines)


# ── README injection ─────────────────────────────────────────────────────

README_START_MARKER = "<!-- LEADERBOARD:START -->"
README_END_MARKER = "<!-- LEADERBOARD:END -->"


def render_readme_section(
    rows: list[ExperimentRow], diseases: list[str]
) -> str:
    """Render the subset of the leaderboard suitable for embedding in README.

    Includes Summary table + Per-Disease AUROC table with ``###`` subheadings
    (so it nests cleanly under an existing ``##`` section in README). Does NOT
    include the ``# Experiment Leaderboard`` title or the Experiment Details
    section (those live in the full ``LEADERBOARD.md``).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append(
        f"*Auto-generated by `scripts/update_leaderboard.py` — "
        f"Last updated: {now}. "
        f"See [`LEADERBOARD.md`](LEADERBOARD.md) for full details.*"
    )
    lines.append("")

    # Summary table
    lines.append("### Summary")
    lines.append("")
    lines.append(
        "| ID | Experiment | Phase | Status | Mean AUROC | "
        "Mean AUPRC | Hier. Violation | Trainable Params | Best Epoch |"
    )
    lines.append(
        "|----|-----------|:-----:|:------:|:----------:|:----------:|"
        ":---------------:|:----------------:|:----------:|"
    )
    for row in rows:
        lines.append(
            f"| {row.id} | {row.display_name} | "
            f"{_PHASE_ABBREV.get(row.phase, row.phase)} | "
            f"{_format_status(row)} | "
            f"{_format_mean_with_partial_warning(row)} | "
            f"{_format_float(row.mean_auprc)} | "
            f"{_format_float(row.hierarchy_violation_rate)} | "
            f"{_format_params(row.trainable_params, row.total_params)} | "
            f"{row.best_epoch if row.best_epoch is not None else '—'} |"
        )
    lines.append("")
    lines.append(
        "**Legend**: ✅ Done · ⚡ Partial · ⬜ Planned · ❌ Error · "
        "⚠️ = partial data, metric computed over completed diseases only"
    )
    lines.append("")

    # Per-disease table
    lines.append("### Per-Disease AUROC")
    lines.append("")
    header_cells = ["Disease"] + [row.id for row in rows]
    lines.append("| " + " | ".join(header_cells) + " |")
    align_cells = ["---"] + [":---:" for _ in rows]
    lines.append("| " + " | ".join(align_cells) + " |")

    for disease in diseases:
        cells = [disease]
        for row in rows:
            value = row.per_disease_auroc.get(disease)
            cells.append(_format_float(value))
        lines.append("| " + " | ".join(cells) + " |")

    mean_cells = ["**MEAN**"]
    for row in rows:
        mean_cells.append(f"**{_format_float(row.mean_auroc)}**")
    lines.append("| " + " | ".join(mean_cells) + " |")

    # Mirror render_markdown: append the diagnostics section when any
    # experiment declared eval_dirs. In README the section uses ``###``
    # subheadings (one level deeper) since the README block lives under a
    # top-level ``## 实验进度`` heading.
    diag_lines = _render_global_eval_diagnostics(rows, diseases)
    if diag_lines:
        # Demote ``##`` / ``###`` one level so nesting under README ``##`` works.
        for line in diag_lines:
            if line.startswith("## "):
                lines.append("### " + line[3:])
            elif line.startswith("### "):
                lines.append("#### " + line[4:])
            else:
                lines.append(line)

    return "\n".join(lines)


def inject_into_readme(readme_path: Path, section: str) -> bool:
    """Replace content between leaderboard markers in ``readme_path``.

    Returns ``True`` if the README was updated, ``False`` if the file is
    missing or does not contain both markers. Existing prose outside the
    marker pair is preserved verbatim.
    """
    readme_path = Path(readme_path)
    if not readme_path.exists():
        return False

    content = readme_path.read_text()
    if README_START_MARKER not in content or README_END_MARKER not in content:
        return False

    start_idx = content.index(README_START_MARKER) + len(README_START_MARKER)
    end_idx = content.index(README_END_MARKER)
    if end_idx < start_idx:
        # End marker appears before start marker — malformed, skip
        return False

    new_content = (
        content[:start_idx]
        + "\n"
        + section
        + "\n"
        + content[end_idx:]
    )
    readme_path.write_text(new_content)
    return True


def main() -> int:
    """CLI entry point. Returns exit code.

    Exit codes:
        0 — success (includes all-planned and partial states).
        1 — manifest load failure (raised from load_manifest via sys.exit(1)).
        2 — at least one experiment parser produced an ERROR status.
            The LEADERBOARD.md file is still written in this case.
    """
    parser = argparse.ArgumentParser(
        description="Generate LEADERBOARD.md from experiment outputs."
    )
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "configs" / "leaderboard_manifest.yaml"),
        help="Path to manifest YAML (default: configs/leaderboard_manifest.yaml)",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "LEADERBOARD.md"),
        help="Path to output markdown file (default: LEADERBOARD.md)",
    )
    parser.add_argument(
        "--readme",
        default=str(REPO_ROOT / "README.md"),
        help=(
            "Path to README.md to inject summary section into. The README "
            "must contain <!-- LEADERBOARD:START --> and "
            "<!-- LEADERBOARD:END --> markers; without them, injection is "
            "silently skipped. (default: README.md)"
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repo root for resolving relative output_dir paths (default: auto-detect)",
    )
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    rows = collect_experiment_rows(manifest, repo_root=Path(args.repo_root))

    markdown = render_markdown(rows, manifest.diseases)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)

    # Also inject a subset into README.md if markers are present
    readme_section = render_readme_section(rows, manifest.diseases)
    readme_updated = inject_into_readme(Path(args.readme), readme_section)

    # Summary line to stdout
    counts: dict[ExperimentStatus, int] = {s: 0 for s in ExperimentStatus}
    for row in rows:
        counts[row.status] += 1

    parts = []
    if counts[ExperimentStatus.COMPLETED]:
        parts.append(f"{counts[ExperimentStatus.COMPLETED]} done")
    if counts[ExperimentStatus.PARTIAL]:
        parts.append(f"{counts[ExperimentStatus.PARTIAL]} partial")
    if counts[ExperimentStatus.PLANNED]:
        parts.append(f"{counts[ExperimentStatus.PLANNED]} planned")
    if counts[ExperimentStatus.ERROR]:
        parts.append(f"{counts[ExperimentStatus.ERROR]} error")
    summary_str = ", ".join(parts) if parts else "no experiments"
    readme_suffix = " + README" if readme_updated else ""
    print(f"Leaderboard updated: {len(rows)} experiments ({summary_str}){readme_suffix}")

    # Exit code 2 if any parser produced an error
    if counts[ExperimentStatus.ERROR] > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
