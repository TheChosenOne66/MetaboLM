"""Tests for scripts/update_leaderboard.py."""

import sys
from pathlib import Path

# Make repo root importable for `scripts` and `tests` paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# The script lives at scripts/update_leaderboard.py, imported as update_leaderboard
import update_leaderboard as ulb

FIXTURES = Path(__file__).parent / "fixtures" / "leaderboard"


def test_module_imports():
    """Smoke test: script can be imported."""
    assert hasattr(ulb, "ExperimentRow")
    assert hasattr(ulb, "ExperimentType")
    assert hasattr(ulb, "ExperimentStatus")


def test_experiment_type_values():
    assert ulb.ExperimentType.PER_DISEASE_SFT.value == "per_disease_sft"
    assert ulb.ExperimentType.MULTITASK_SFT.value == "multitask_sft"
    assert ulb.ExperimentType.MULTITASK_RL.value == "multitask_rl"


def test_experiment_status_values():
    assert ulb.ExperimentStatus.PLANNED.value == "planned"
    assert ulb.ExperimentStatus.PARTIAL.value == "partial"
    assert ulb.ExperimentStatus.COMPLETED.value == "completed"
    assert ulb.ExperimentStatus.ERROR.value == "error"


def test_experiment_row_defaults():
    row = ulb.ExperimentRow(
        id="E0",
        display_name="Test",
        phase="phase1",
        exp_type=ulb.ExperimentType.PER_DISEASE_SFT,
        config="configs/reproduce.yaml",
        output_dir="outputs/E0_reproduction",
        description="test",
        innovation=None,
    )
    assert row.status == ulb.ExperimentStatus.PLANNED
    assert row.mean_auroc is None
    assert row.per_disease_auroc == {}
    assert row.total_diseases == 16
