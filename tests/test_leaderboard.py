"""Tests for scripts/update_leaderboard.py."""

import json
import sys
from pathlib import Path

import pytest

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


# ── collect_experiment_rows tests ────────────────────────────────────────

def test_collect_experiment_rows_mixed(tmp_path):
    # Build a manifest pointing at real fixtures
    manifest = ulb.Manifest(
        experiments=[
            ulb.ManifestExperiment(
                id="E0", display_name="E0 test", phase="phase1",
                exp_type=ulb.ExperimentType.PER_DISEASE_SFT,
                config="configs/reproduce.yaml",
                output_dir=str(FIXTURES / "outputs" / "e0_partial"),
                description="test", innovation=None,
            ),
            ulb.ManifestExperiment(
                id="E1", display_name="E1 test", phase="phase2",
                exp_type=ulb.ExperimentType.MULTITASK_SFT,
                config="configs/sft_full_ft.yaml",
                output_dir=str(FIXTURES / "outputs" / "e1_complete"),
                description="test", innovation="innov",
            ),
            ulb.ManifestExperiment(
                id="E2", display_name="E2 test", phase="phase2",
                exp_type=ulb.ExperimentType.MULTITASK_SFT,
                config="configs/sft_head_only.yaml",
                output_dir=str(FIXTURES / "outputs" / "does_not_exist"),
                description="test", innovation=None,
            ),
        ],
        diseases=CANONICAL_DISEASES,
    )
    rows = ulb.collect_experiment_rows(manifest, repo_root=tmp_path)
    assert len(rows) == 3
    assert rows[0].id == "E0"
    assert rows[0].status == ulb.ExperimentStatus.PARTIAL
    assert rows[1].id == "E1"
    assert rows[1].status == ulb.ExperimentStatus.COMPLETED
    assert rows[2].id == "E2"
    assert rows[2].status == ulb.ExperimentStatus.PLANNED


def test_collect_experiment_rows_error_is_isolated(tmp_path):
    """A parser error on one experiment must not affect others."""
    manifest = ulb.Manifest(
        experiments=[
            ulb.ManifestExperiment(
                id="E1", display_name="E1 bad", phase="phase2",
                exp_type=ulb.ExperimentType.MULTITASK_SFT,
                config="configs/sft_full_ft.yaml",
                output_dir=str(FIXTURES / "outputs" / "e1_malformed_json"),
                description="test", innovation=None,
            ),
            ulb.ManifestExperiment(
                id="E2", display_name="E2 good", phase="phase2",
                exp_type=ulb.ExperimentType.MULTITASK_SFT,
                config="configs/sft_head_only.yaml",
                output_dir=str(FIXTURES / "outputs" / "e1_complete"),
                description="test", innovation=None,
            ),
        ],
        diseases=CANONICAL_DISEASES,
    )
    rows = ulb.collect_experiment_rows(manifest, repo_root=tmp_path)
    assert len(rows) == 2
    assert rows[0].status == ulb.ExperimentStatus.ERROR
    assert rows[0].error_message is not None
    assert rows[1].status == ulb.ExperimentStatus.COMPLETED


def test_collect_experiment_rows_resolves_relative_paths(tmp_path):
    """Output_dir relative paths are resolved against repo_root."""
    # Copy a fixture to a relative location under tmp_path
    import shutil
    relative_target = tmp_path / "outputs" / "E0_test"
    relative_target.parent.mkdir(parents=True)
    shutil.copytree(FIXTURES / "outputs" / "e0_complete", relative_target)

    manifest = ulb.Manifest(
        experiments=[
            ulb.ManifestExperiment(
                id="E0", display_name="E0 test", phase="phase1",
                exp_type=ulb.ExperimentType.PER_DISEASE_SFT,
                config="configs/reproduce.yaml",
                output_dir="outputs/E0_test",  # RELATIVE path
                description="test", innovation=None,
            ),
        ],
        diseases=CANONICAL_DISEASES,
    )
    rows = ulb.collect_experiment_rows(manifest, repo_root=tmp_path)
    assert rows[0].status == ulb.ExperimentStatus.COMPLETED


# ── render_markdown tests ────────────────────────────────────────────────

def _make_completed_multitask_row(id="E1"):
    return ulb.ExperimentRow(
        id=id,
        display_name=f"{id} — Test",
        phase="phase2",
        exp_type=ulb.ExperimentType.MULTITASK_SFT,
        config=f"configs/{id.lower()}.yaml",
        output_dir=f"outputs/{id.lower()}",
        description="Test experiment",
        innovation="Test innovation",
        status=ulb.ExperimentStatus.COMPLETED,
        mean_auroc=0.8472,
        mean_auprc=0.3215,
        hierarchy_violation_rate_mean=0.150,
        hierarchy_violation_rate_head=0.028,
        trainable_params=85123456,
        total_params=85123456,
        freeze_strategy="none",
        best_epoch=28,
        per_disease_auroc={d: 0.80 + i * 0.005 for i, d in enumerate(CANONICAL_DISEASES)},
        num_completed_diseases=16,
    )


def _make_planned_row(id="E5"):
    return ulb.ExperimentRow(
        id=id,
        display_name=f"{id} — Planned",
        phase="phase3",
        exp_type=ulb.ExperimentType.MULTITASK_RL,
        config=f"configs/{id.lower()}.yaml",
        output_dir=f"outputs/{id.lower()}",
        description="Not yet run",
        innovation=None,
        status=ulb.ExperimentStatus.PLANNED,
    )


def test_render_summary_table_has_all_experiments():
    rows = [_make_completed_multitask_row("E1"), _make_planned_row("E5")]
    md = ulb.render_markdown(rows, CANONICAL_DISEASES)
    assert "E1 — Test" in md
    assert "E5 — Planned" in md


def test_render_per_disease_table_dimensions():
    rows = [_make_completed_multitask_row("E1"), _make_planned_row("E5")]
    md = ulb.render_markdown(rows, CANONICAL_DISEASES)
    # Per-disease section header
    assert "## Per-Disease AUROC" in md
    # All 16 diseases in the left column
    for disease in CANONICAL_DISEASES:
        assert disease in md
    # MEAN row
    assert "**MEAN**" in md or "| MEAN " in md


def test_render_status_icons():
    rows = [
        _make_completed_multitask_row("E1"),
        _make_planned_row("E5"),
    ]
    md = ulb.render_markdown(rows, CANONICAL_DISEASES)
    assert "✅" in md
    assert "⬜" in md


def test_render_partial_shows_fraction():
    row = ulb.ExperimentRow(
        id="E0", display_name="E0 — Partial", phase="phase1",
        exp_type=ulb.ExperimentType.PER_DISEASE_SFT,
        config="configs/reproduce.yaml", output_dir="outputs/E0",
        description="test", innovation=None,
        status=ulb.ExperimentStatus.PARTIAL,
        mean_auroc=0.866,
        num_completed_diseases=1,
        per_disease_auroc={"T2D": 0.866},
    )
    md = ulb.render_markdown([row], CANONICAL_DISEASES)
    assert "⚡" in md
    assert "1/16" in md
    assert "⚠️" in md  # partial mean warning


def test_render_formats_params_human_readable():
    row = _make_completed_multitask_row("E1")
    md = ulb.render_markdown([row], CANONICAL_DISEASES)
    # 85123456 should render as ~85.1M
    assert "85.1M" in md or "85M" in md


def test_render_placeholder_for_none():
    row = _make_planned_row("E5")
    md = ulb.render_markdown([row], CANONICAL_DISEASES)
    # Planned row has all-None metrics; should show dashes
    assert "—" in md


def test_render_error_row():
    row = ulb.ExperimentRow(
        id="E3", display_name="E3 — Broken", phase="phase2",
        exp_type=ulb.ExperimentType.MULTITASK_SFT,
        config="configs/sft_adapter.yaml", output_dir="outputs/E3",
        description="test", innovation=None,
        status=ulb.ExperimentStatus.ERROR,
        error_message="JSON parse error: ...",
    )
    md = ulb.render_markdown([row], CANONICAL_DISEASES)
    assert "❌" in md


# ── main() end-to-end tests ──────────────────────────────────────────────

_SIXTEEN_DISEASES_YAML = """
  - T2D
  - obesity
  - hypertension
  - ischemic_heart
  - atrial_fib
  - heart_failure
  - rheumatoid
  - asthma
  - dementia
  - copd
  - stroke
  - parkinsons
  - breast_cancer
  - colon_cancer
  - lung_cancer
  - prostate_cancer
"""


def test_main_writes_leaderboard_file(tmp_path, monkeypatch, capsys):
    """Smoke test: given a valid manifest + empty outputs dir, main writes a file."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(f"""
experiments:
  - id: E0
    display_name: "E0 Test"
    phase: phase1
    type: per_disease_sft
    config: configs/reproduce.yaml
    output_dir: outputs/does_not_exist
    description: "Test"
    innovation: null
diseases:{_SIXTEEN_DISEASES_YAML}
""")
    output_path = tmp_path / "LEADERBOARD.md"
    readme_path = tmp_path / "README.md"  # isolate from real README
    monkeypatch.setattr(
        sys, "argv",
        [
            "update_leaderboard.py",
            "--manifest", str(manifest_path),
            "--output", str(output_path),
            "--readme", str(readme_path),
            "--repo-root", str(tmp_path),
        ],
    )
    exit_code = ulb.main()
    assert exit_code == 0
    assert output_path.exists()
    content = output_path.read_text()
    assert "E0 Test" in content
    assert "⬜" in content  # planned icon


def test_main_exit_code_0_on_success(tmp_path, monkeypatch):
    """Purely PLANNED state should exit 0, not 2."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("""
experiments: []
diseases:
  - T2D
""")
    output_path = tmp_path / "LEADERBOARD.md"
    readme_path = tmp_path / "README.md"  # isolate from real README
    monkeypatch.setattr(
        sys, "argv",
        [
            "update_leaderboard.py",
            "--manifest", str(manifest_path),
            "--output", str(output_path),
            "--readme", str(readme_path),
            "--repo-root", str(tmp_path),
        ],
    )
    exit_code = ulb.main()
    assert exit_code == 0


def test_main_exit_code_2_on_parser_error(tmp_path, monkeypatch):
    """A parser error should cause exit code 2 but still write the file."""
    import shutil
    shutil.copytree(
        FIXTURES / "outputs" / "e1_malformed_json",
        tmp_path / "outputs" / "e1_bad",
    )

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(f"""
experiments:
  - id: E1
    display_name: "E1 Bad"
    phase: phase2
    type: multitask_sft
    config: configs/sft_full_ft.yaml
    output_dir: outputs/e1_bad
    description: "Test"
    innovation: null
diseases:{_SIXTEEN_DISEASES_YAML}
""")
    output_path = tmp_path / "LEADERBOARD.md"
    readme_path = tmp_path / "README.md"  # isolate from real README
    monkeypatch.setattr(
        sys, "argv",
        [
            "update_leaderboard.py",
            "--manifest", str(manifest_path),
            "--output", str(output_path),
            "--readme", str(readme_path),
            "--repo-root", str(tmp_path),
        ],
    )
    exit_code = ulb.main()
    assert exit_code == 2
    assert output_path.exists()  # file still written
    assert "❌" in output_path.read_text()


# ── render_readme_section tests ─────────────────────────────────────────

def test_render_readme_section_has_summary_table():
    """Readme section must contain a summary row for every experiment."""
    rows = [_make_planned_row("E0"), _make_planned_row("E1")]
    section = ulb.render_readme_section(rows, CANONICAL_DISEASES)
    assert "Summary" in section
    assert "| E0 |" in section
    assert "| E1 |" in section


def test_render_readme_section_has_per_disease_table():
    """Readme section must contain the per-disease AUROC table."""
    rows = [_make_planned_row("E0")]
    section = ulb.render_readme_section(rows, CANONICAL_DISEASES)
    assert "Per-Disease AUROC" in section
    # every disease must appear as a row label
    for disease in CANONICAL_DISEASES:
        assert f"| {disease} |" in section
    # MEAN row must be present
    assert "**MEAN**" in section


def test_render_readme_section_omits_experiment_details():
    """Experiment Details section belongs in LEADERBOARD.md, not README."""
    rows = [_make_planned_row("E0")]
    section = ulb.render_readme_section(rows, CANONICAL_DISEASES)
    assert "Experiment Details" not in section


def test_render_readme_section_uses_h3_subheadings():
    """README already uses ## for top-level sections; keep leaderboard at ###."""
    rows = [_make_planned_row("E0")]
    section = ulb.render_readme_section(rows, CANONICAL_DISEASES)
    # Top-level title (#) must not appear — that's the README's job
    assert "# MetaboLM Post-Training — Experiment Leaderboard" not in section
    # Must use ### for subsections, not ## — check line-by-line to avoid
    # the substring false-positive where "## Summary" ⊂ "### Summary".
    section_lines = section.splitlines()
    assert "### Summary" in section_lines
    assert "### Per-Disease AUROC" in section_lines
    assert "## Summary" not in section_lines
    assert "## Per-Disease AUROC" not in section_lines


def test_render_readme_section_has_timestamp():
    """Helps user trust the data is fresh."""
    rows = [_make_planned_row("E0")]
    section = ulb.render_readme_section(rows, CANONICAL_DISEASES)
    assert "Last updated" in section


# ── inject_into_readme tests ────────────────────────────────────────────

def test_inject_into_readme_replaces_content_between_markers(tmp_path):
    """Marker-delimited region is replaced; surrounding text is preserved."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# My Project\n"
        "\n"
        "Some prose.\n"
        "\n"
        "<!-- LEADERBOARD:START -->\n"
        "OLD CONTENT\n"
        "<!-- LEADERBOARD:END -->\n"
        "\n"
        "More prose.\n"
    )
    result = ulb.inject_into_readme(readme, "NEW CONTENT")
    assert result is True
    content = readme.read_text()
    assert "NEW CONTENT" in content
    assert "OLD CONTENT" not in content
    # Surrounding text preserved
    assert "# My Project" in content
    assert "Some prose." in content
    assert "More prose." in content
    # Markers themselves preserved
    assert "<!-- LEADERBOARD:START -->" in content
    assert "<!-- LEADERBOARD:END -->" in content


def test_inject_into_readme_returns_false_without_markers(tmp_path):
    """README without markers is left untouched and function returns False."""
    readme = tmp_path / "README.md"
    original = "# My Project\n\nNo markers here.\n"
    readme.write_text(original)
    result = ulb.inject_into_readme(readme, "NEW CONTENT")
    assert result is False
    assert readme.read_text() == original


def test_inject_into_readme_returns_false_if_file_missing(tmp_path):
    """Missing README is not an error."""
    readme = tmp_path / "does_not_exist.md"
    result = ulb.inject_into_readme(readme, "NEW CONTENT")
    assert result is False
    assert not readme.exists()


def test_inject_into_readme_is_idempotent(tmp_path):
    """Running twice with same content must not duplicate or corrupt the file."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n"
        "<!-- LEADERBOARD:START -->\n"
        "old\n"
        "<!-- LEADERBOARD:END -->\n"
        "after\n"
    )
    ulb.inject_into_readme(readme, "injected")
    first = readme.read_text()
    ulb.inject_into_readme(readme, "injected")
    second = readme.read_text()
    assert first == second
    # Only one occurrence of the markers
    assert first.count("<!-- LEADERBOARD:START -->") == 1
    assert first.count("<!-- LEADERBOARD:END -->") == 1


# ── main() + README injection ───────────────────────────────────────────

def test_main_injects_into_readme(tmp_path, monkeypatch):
    """End-to-end: main() with --readme flag updates the README in place."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(f"""
experiments:
  - id: E0
    display_name: "E0 Test"
    phase: phase1
    type: per_disease_sft
    config: configs/reproduce.yaml
    output_dir: outputs/does_not_exist
    description: "Test"
    innovation: null
diseases:{_SIXTEEN_DISEASES_YAML}
""")
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "# Project\n"
        "\n"
        "Intro text.\n"
        "\n"
        "<!-- LEADERBOARD:START -->\n"
        "stale\n"
        "<!-- LEADERBOARD:END -->\n"
        "\n"
        "Outro text.\n"
    )
    output_path = tmp_path / "LEADERBOARD.md"
    monkeypatch.setattr(
        sys, "argv",
        [
            "update_leaderboard.py",
            "--manifest", str(manifest_path),
            "--output", str(output_path),
            "--readme", str(readme_path),
            "--repo-root", str(tmp_path),
        ],
    )
    exit_code = ulb.main()
    assert exit_code == 0
    readme_content = readme_path.read_text()
    # Stale content is gone
    assert "stale" not in readme_content
    # Summary table injected
    assert "E0 Test" in readme_content
    assert "### Summary" in readme_content
    # Surrounding README text untouched
    assert "# Project" in readme_content
    assert "Intro text." in readme_content
    assert "Outro text." in readme_content


def test_main_readme_missing_markers_is_not_an_error(tmp_path, monkeypatch):
    """If README lacks markers, main() succeeds (exit 0) without modifying it."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("""
experiments: []
diseases:
  - T2D
""")
    readme_path = tmp_path / "README.md"
    original = "# Project\n\nNo markers.\n"
    readme_path.write_text(original)
    output_path = tmp_path / "LEADERBOARD.md"
    monkeypatch.setattr(
        sys, "argv",
        [
            "update_leaderboard.py",
            "--manifest", str(manifest_path),
            "--output", str(output_path),
            "--readme", str(readme_path),
            "--repo-root", str(tmp_path),
        ],
    )
    exit_code = ulb.main()
    assert exit_code == 0
    assert readme_path.read_text() == original


# ── HVR sidecar reads + two-column rendering (option B) ──────────────────

def _minimal_multitask_files(exp_dir: Path, extras_in_summary: dict | None = None):
    """Write the minimum files parse_multitask needs so it returns COMPLETED
    instead of PLANNED, plus any extra keys in summary.json."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "multitask_metrics.csv").write_text(
        "Disease,Val_AUC,Val_F1\nT2D,0.75,0.5\nMEAN,0.75,0.5\n"
    )
    summary = {
        "experiment": "E1", "freeze_strategy": "none",
        "best_epoch": 10, "best_mean_auc": 0.75,
        "trainable_params": 100, "total_params": 100,
    }
    if extras_in_summary:
        summary.update(extras_in_summary)
    (exp_dir / "summary.json").write_text(json.dumps(summary))


def _hvr_payload(rate: float, method: str) -> str:
    return json.dumps({
        "hierarchy_violation_rate": rate, "method": method,
        "n_samples": 1, "n_leaf_chapter_pairs": 16, "method_details": {},
    })


def test_parse_multitask_reads_both_hvr_jsons(tmp_path):
    """Both ``_mean`` and ``_head`` sidecars populate their respective fields."""
    exp_dir = tmp_path / "E1_test"
    _minimal_multitask_files(exp_dir)
    (exp_dir / "hierarchy_violation_mean.json").write_text(
        _hvr_payload(0.15, "mean_aggregate_multitask")
    )
    (exp_dir / "hierarchy_violation_head.json").write_text(
        _hvr_payload(0.028, "explicit_chapter_head")
    )
    r = ulb.parse_multitask(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] == pytest.approx(0.15)
    assert r["hierarchy_violation_rate_head"] == pytest.approx(0.028)


def test_parse_multitask_falls_back_to_summary_for_head(tmp_path):
    """Pre-sidecar runs dumped HVR in summary.json; treat as head metric."""
    exp_dir = tmp_path / "E1_legacy"
    _minimal_multitask_files(
        exp_dir, extras_in_summary={"hierarchy_violation_rate": 0.042},
    )
    r = ulb.parse_multitask(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] is None
    assert r["hierarchy_violation_rate_head"] == pytest.approx(0.042)


def test_parse_multitask_no_hvr_sources_gives_none(tmp_path):
    """Absent sidecars + no legacy summary key → both HVR fields are None."""
    exp_dir = tmp_path / "E1_no_hvr"
    _minimal_multitask_files(exp_dir)
    r = ulb.parse_multitask(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] is None
    assert r["hierarchy_violation_rate_head"] is None


def test_parse_multitask_sidecar_wins_over_summary(tmp_path):
    """Sidecar JSON takes priority over summary.json's legacy field."""
    exp_dir = tmp_path / "E1_both"
    _minimal_multitask_files(
        exp_dir, extras_in_summary={"hierarchy_violation_rate": 0.999},
    )
    (exp_dir / "hierarchy_violation_head.json").write_text(
        _hvr_payload(0.028, "explicit_chapter_head")
    )
    r = ulb.parse_multitask(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_head"] == pytest.approx(0.028)


def test_parse_multitask_handles_malformed_sidecar(tmp_path):
    """A corrupt sidecar is warned about, not fatal; HVR just stays None."""
    exp_dir = tmp_path / "E1_bad"
    _minimal_multitask_files(exp_dir)
    (exp_dir / "hierarchy_violation_mean.json").write_text("{not valid json")
    r = ulb.parse_multitask(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] is None
    # Still returns COMPLETED — the main metrics are fine.
    assert r["status"] == ulb.ExperimentStatus.COMPLETED


def test_parse_per_disease_sft_reads_hvr_mean_json(tmp_path):
    """E0 parser reads only ``_mean`` sidecar; ``_head`` stays None (no chapter head)."""
    exp_dir = tmp_path / "E0_test"
    exp_dir.mkdir()
    (exp_dir / "finetune_summary_metrics.csv").write_text(
        "Disease,Val_AUC,Best_Epoch\nT2D,0.85,5\n"
    )
    (exp_dir / "hierarchy_violation_mean.json").write_text(
        _hvr_payload(0.37, "mean_aggregate_baseline")
    )
    r = ulb.parse_per_disease_sft(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] == pytest.approx(0.37)
    assert r["hierarchy_violation_rate_head"] is None


def test_parse_per_disease_sft_without_hvr_sidecar(tmp_path):
    exp_dir = tmp_path / "E0_no_hvr"
    exp_dir.mkdir()
    (exp_dir / "finetune_summary_metrics.csv").write_text(
        "Disease,Val_AUC,Best_Epoch\nT2D,0.85,5\n"
    )
    r = ulb.parse_per_disease_sft(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] is None
    assert r["hierarchy_violation_rate_head"] is None


def test_render_markdown_shows_two_hvr_columns():
    rows = [_make_completed_multitask_row("E1")]
    md = ulb.render_markdown(rows, CANONICAL_DISEASES)
    # New column headers present
    assert "HVR (mean)" in md
    assert "HVR (head)" in md
    # Old single header is gone
    assert "Hier. Violation" not in md
    # Both values render (0.150 and 0.028 from _make_completed_multitask_row)
    assert "0.150" in md
    assert "0.028" in md


def test_render_readme_section_shows_two_hvr_columns():
    rows = [_make_completed_multitask_row("E1")]
    md = ulb.render_readme_section(rows, CANONICAL_DISEASES)
    assert "HVR (mean)" in md
    assert "HVR (head)" in md
    assert "Hier. Violation" not in md


# ── HVR coverage annotation (codex P2 round 3) ───────────────────────────

def test_parse_multitask_surfaces_n_pairs_from_sidecar(tmp_path):
    """Sidecar ``n_leaf_chapter_pairs`` is preserved into the parsed dict."""
    exp_dir = tmp_path / "E1_coverage"
    _minimal_multitask_files(exp_dir)
    (exp_dir / "hierarchy_violation_mean.json").write_text(json.dumps({
        "hierarchy_violation_rate": 0.15, "method": "mean_aggregate_multitask",
        "n_samples": 1, "n_leaf_chapter_pairs": 16, "method_details": {},
    }))
    (exp_dir / "hierarchy_violation_head.json").write_text(json.dumps({
        "hierarchy_violation_rate": 0.028, "method": "explicit_chapter_head",
        "n_samples": 1, "n_leaf_chapter_pairs": 12, "method_details": {},
    }))
    r = ulb.parse_multitask(exp_dir, ["T2D"])
    assert r["hierarchy_violation_n_pairs_mean"] == 16
    assert r["hierarchy_violation_n_pairs_head"] == 12


def test_parse_per_disease_sft_surfaces_partial_n_pairs(tmp_path):
    """E0 baseline often runs partial (locally only T2D); n_pairs reflects that."""
    exp_dir = tmp_path / "E0_partial"
    exp_dir.mkdir()
    (exp_dir / "finetune_summary_metrics.csv").write_text(
        "Disease,Val_AUC,Best_Epoch\nT2D,0.85,5\n"
    )
    (exp_dir / "hierarchy_violation_mean.json").write_text(json.dumps({
        "hierarchy_violation_rate": 0.0, "method": "mean_aggregate_baseline",
        "n_samples": 84611, "n_leaf_chapter_pairs": 1, "method_details": {},
    }))
    r = ulb.parse_per_disease_sft(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_mean"] == pytest.approx(0.0)
    assert r["hierarchy_violation_n_pairs_mean"] == 1
    assert r["hierarchy_violation_n_pairs_head"] is None


def test_parse_multitask_no_n_pairs_in_legacy_summary(tmp_path):
    """Legacy summary.json HVR fallback has no coverage info; n_pairs stays None."""
    exp_dir = tmp_path / "E1_legacy_cov"
    _minimal_multitask_files(
        exp_dir, extras_in_summary={"hierarchy_violation_rate": 0.042},
    )
    r = ulb.parse_multitask(exp_dir, ["T2D"])
    assert r["hierarchy_violation_rate_head"] == pytest.approx(0.042)
    # No way to know coverage of the legacy training-time HVR.
    assert r["hierarchy_violation_n_pairs_head"] is None


def test_format_hvr_full_coverage_no_annotation():
    assert ulb._format_hvr(0.150, n_pairs=16, total=16) == "0.150"


def test_format_hvr_partial_coverage_annotated():
    assert ulb._format_hvr(0.000, n_pairs=1, total=16) == "0.000 (1/16)"


def test_format_hvr_unknown_coverage_treated_as_full():
    # Legacy fallback (summary.json) has no n_pairs — we presume full so as
    # not to noisily annotate historical data that was almost certainly
    # computed on the full eval set.
    assert ulb._format_hvr(0.042, n_pairs=None, total=16) == "0.042"


def test_format_hvr_none_value_renders_dash():
    assert ulb._format_hvr(None, n_pairs=None, total=16) == "—"
    assert ulb._format_hvr(None, n_pairs=16, total=16) == "—"


def test_render_markdown_annotates_partial_hvr_coverage():
    """E0 locally (T2D only → 1/16) must show ``(1/16)`` next to the rate."""
    row = ulb.ExperimentRow(
        id="E0", display_name="E0 — Partial HVR", phase="phase1",
        exp_type=ulb.ExperimentType.PER_DISEASE_SFT,
        config="configs/reproduce.yaml", output_dir="outputs/E0",
        description="test", innovation=None,
        status=ulb.ExperimentStatus.PARTIAL,
        mean_auroc=0.866,
        hierarchy_violation_rate_mean=0.0,
        hierarchy_violation_n_pairs_mean=1,
        num_completed_diseases=1,
        per_disease_auroc={"T2D": 0.866},
    )
    md = ulb.render_markdown([row], CANONICAL_DISEASES)
    assert "(1/16)" in md


def test_render_markdown_full_coverage_no_annotation():
    """Full coverage rows render plain rates with no (n/total) suffix."""
    row = _make_completed_multitask_row("E1")
    # Default _make_completed_multitask_row has n_pairs=None → presumed full.
    # Explicitly set n_pairs=16 to exercise the equal-to-total branch.
    row.hierarchy_violation_n_pairs_mean = 16
    row.hierarchy_violation_n_pairs_head = 16
    md = ulb.render_markdown([row], CANONICAL_DISEASES)
    assert "(16/16)" not in md
    assert "/16)" not in md
