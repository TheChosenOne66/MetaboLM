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


# ── ManifestLoader tests ────────────────────────────────────────────────

def test_manifest_loader_valid():
    manifest = ulb.load_manifest(FIXTURES / "manifest_valid.yaml")
    assert len(manifest.experiments) == 2
    assert manifest.experiments[0].id == "E0"
    assert manifest.experiments[0].exp_type == ulb.ExperimentType.PER_DISEASE_SFT
    assert manifest.experiments[1].id == "E1"
    assert manifest.experiments[1].exp_type == ulb.ExperimentType.MULTITASK_SFT
    assert len(manifest.diseases) == 16
    assert manifest.diseases[0] == "T2D"
    assert manifest.diseases[-1] == "prostate_cancer"


def test_manifest_loader_malformed_exits(capsys):
    import pytest
    with pytest.raises(SystemExit) as exc:
        ulb.load_manifest(FIXTURES / "manifest_malformed.yaml")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "manifest" in err.lower()


def test_manifest_loader_missing_file_exits(capsys):
    import pytest
    with pytest.raises(SystemExit) as exc:
        ulb.load_manifest(FIXTURES / "nonexistent.yaml")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "no such" in err.lower()


def test_manifest_loader_rejects_missing_required_field(tmp_path):
    import pytest
    bad = tmp_path / "bad_manifest.yaml"
    bad.write_text("""
experiments:
  - id: E0
    # missing display_name, phase, type, etc.
diseases:
  - T2D
""")
    with pytest.raises(SystemExit) as exc:
        ulb.load_manifest(bad)
    assert exc.value.code == 1


# ── parse_per_disease_sft tests ─────────────────────────────────────────

CANONICAL_DISEASES = [
    "T2D", "obesity", "hypertension", "ischemic_heart", "atrial_fib",
    "heart_failure", "rheumatoid", "asthma", "dementia", "copd",
    "stroke", "parkinsons", "breast_cancer", "colon_cancer",
    "lung_cancer", "prostate_cancer",
]


def test_parse_e0_complete_returns_completed_status():
    result = ulb.parse_per_disease_sft(
        FIXTURES / "outputs" / "e0_complete", CANONICAL_DISEASES
    )
    assert result["status"] == ulb.ExperimentStatus.COMPLETED
    assert result["num_completed_diseases"] == 16
    assert len(result["per_disease_auroc"]) == 16
    assert result["per_disease_auroc"]["T2D"] == 0.866
    assert result["per_disease_auroc"]["dementia"] == 0.890


def test_parse_e0_mean_auroc_is_arithmetic_mean():
    result = ulb.parse_per_disease_sft(
        FIXTURES / "outputs" / "e0_complete", CANONICAL_DISEASES
    )
    # Arithmetic mean of the 16 Val_AUC values in e0_complete fixture
    expected = (0.866 + 0.812 + 0.795 + 0.840 + 0.820 + 0.855 + 0.760 + 0.730
                + 0.890 + 0.870 + 0.790 + 0.830 + 0.720 + 0.700 + 0.870 + 0.810) / 16
    assert abs(result["mean_auroc"] - expected) < 1e-6


def test_parse_e0_partial_returns_partial_status():
    result = ulb.parse_per_disease_sft(
        FIXTURES / "outputs" / "e0_partial", CANONICAL_DISEASES
    )
    assert result["status"] == ulb.ExperimentStatus.PARTIAL
    assert result["num_completed_diseases"] == 3
    assert len(result["per_disease_auroc"]) == 3
    assert "T2D" in result["per_disease_auroc"]
    assert "prostate_cancer" not in result["per_disease_auroc"]


def test_parse_e0_missing_dir_returns_planned():
    result = ulb.parse_per_disease_sft(
        FIXTURES / "outputs" / "e0_empty", CANONICAL_DISEASES
    )
    assert result["status"] == ulb.ExperimentStatus.PLANNED
    assert result["num_completed_diseases"] is None
    assert result["per_disease_auroc"] == {}


def test_parse_e0_nonexistent_dir_returns_planned():
    result = ulb.parse_per_disease_sft(
        FIXTURES / "outputs" / "does_not_exist", CANONICAL_DISEASES
    )
    assert result["status"] == ulb.ExperimentStatus.PLANNED


# ── parse_multitask tests ───────────────────────────────────────────────

def test_parse_multitask_complete():
    result = ulb.parse_multitask(
        FIXTURES / "outputs" / "e1_complete", CANONICAL_DISEASES
    )
    assert result["status"] == ulb.ExperimentStatus.COMPLETED
    assert result["mean_auroc"] == 0.8472  # from summary.json
    assert result["trainable_params"] == 85123456
    assert result["total_params"] == 85123456
    assert result["freeze_strategy"] == "none"
    assert result["best_epoch"] == 28
    assert len(result["per_disease_auroc"]) == 16
    assert result["per_disease_auroc"]["T2D"] == 0.871
    assert result["error_message"] is None


def test_parse_multitask_filters_mean_row():
    """MEAN row in CSV must NOT appear as a disease key."""
    result = ulb.parse_multitask(
        FIXTURES / "outputs" / "e1_complete", CANONICAL_DISEASES
    )
    assert "MEAN" not in result["per_disease_auroc"]


def test_parse_multitask_prefers_json_over_csv_mean():
    """When summary.json and csv MEAN row disagree, json wins."""
    result = ulb.parse_multitask(
        FIXTURES / "outputs" / "e1_json_csv_mismatch", CANONICAL_DISEASES
    )
    assert result["status"] == ulb.ExperimentStatus.COMPLETED
    # json says 0.85, csv MEAN row says 0.83 — expect json value
    assert result["mean_auroc"] == 0.85


def test_parse_multitask_missing_disease_in_csv():
    """15/16 diseases in CSV: status still completed, missing disease absent from dict."""
    result = ulb.parse_multitask(
        FIXTURES / "outputs" / "e1_missing_disease", CANONICAL_DISEASES
    )
    assert result["status"] == ulb.ExperimentStatus.COMPLETED
    assert len(result["per_disease_auroc"]) == 15
    assert "prostate_cancer" not in result["per_disease_auroc"]
    assert "T2D" in result["per_disease_auroc"]


def test_parse_multitask_malformed_json_returns_error():
    result = ulb.parse_multitask(
        FIXTURES / "outputs" / "e1_malformed_json", CANONICAL_DISEASES
    )
    assert result["status"] == ulb.ExperimentStatus.ERROR
    assert result["error_message"] is not None
    assert "json" in result["error_message"].lower()


def test_parse_multitask_missing_dir_returns_planned():
    result = ulb.parse_multitask(
        FIXTURES / "outputs" / "e0_empty", CANONICAL_DISEASES  # dir exists but no files
    )
    assert result["status"] == ulb.ExperimentStatus.PLANNED


def test_parse_multitask_nonexistent_dir_returns_planned():
    result = ulb.parse_multitask(
        FIXTURES / "outputs" / "does_not_exist", CANONICAL_DISEASES
    )
    assert result["status"] == ulb.ExperimentStatus.PLANNED
