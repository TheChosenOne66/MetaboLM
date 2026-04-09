"""Tests for NMR biomarker selection (168 fields)."""

import pandas as pd
import pytest

from src.data.biomarkers import (
    FIELD_TO_NAME,
    METABOLITE_NAMES,
    NAME_TO_FIELD,
    get_field_ids,
    get_metabolite_names,
    select_biomarkers,
)


# ──────────────────────────────────────────────────────────────────────
# Basic invariant tests (no external data needed)
# ──────────────────────────────────────────────────────────────────────

def test_metabolite_count():
    """There must be exactly 168 metabolite names."""
    names = get_metabolite_names()
    assert len(names) == 168


def test_field_id_count():
    """get_field_ids must return exactly 168 field IDs."""
    field_ids = get_field_ids()
    assert len(field_ids) == 168


def test_names_are_unique():
    """All 168 metabolite names must be distinct."""
    assert len(set(METABOLITE_NAMES)) == 168


def test_field_ids_are_unique():
    """All 168 UKB field IDs must be distinct."""
    assert len(set(FIELD_TO_NAME.keys())) == 168


def test_mapping_round_trip():
    """NAME_TO_FIELD and FIELD_TO_NAME must be inverses of each other."""
    for field, name in FIELD_TO_NAME.items():
        assert NAME_TO_FIELD[name] == field


def test_names_match_expected_order():
    """METABOLITE_NAMES order must match the correlation-matrix column order.

    The correlation matrix uses sorted() on the original metabolite names,
    which is case-sensitive (uppercase before lowercase). The ordering is
    validated against the actual CSV header in test_names_match_correlation_matrix.
    """
    # Verify the list is not accidentally shuffled by checking a few known positions
    assert METABOLITE_NAMES[0] == "3-Hydroxybutyrate"
    assert METABOLITE_NAMES[-1] == "VLDL Cholesterol"
    assert len(METABOLITE_NAMES) == 168


def test_field_ids_in_valid_range():
    """All field IDs must be in the p23400-p23648 range."""
    for field in FIELD_TO_NAME:
        num = int(field[1:])
        assert 23400 <= num <= 23648, f"Field {field} outside valid range"


def test_get_metabolite_names_returns_copy():
    """get_metabolite_names must return a copy, not the module-level list."""
    names = get_metabolite_names()
    names.append("BOGUS")
    assert len(get_metabolite_names()) == 168


def test_field_ids_match_model_order():
    """Field IDs from get_field_ids must correspond to METABOLITE_NAMES in order."""
    field_ids = get_field_ids()
    for fid, name in zip(field_ids, METABOLITE_NAMES):
        assert FIELD_TO_NAME[fid] == name


# ──────────────────────────────────────────────────────────────────────
# Consistency with the reference correlation matrix
# ──────────────────────────────────────────────────────────────────────

def test_names_match_correlation_matrix():
    """Metabolite names must exactly match the correlation-matrix columns."""
    import csv

    csv_path = "/SPXvePFS/users/jytang/metabolm_posttrain/_reference_correlation_matrix.csv"
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
    csv_names = [h for h in header if h.strip()]
    assert csv_names == list(METABOLITE_NAMES), (
        "METABOLITE_NAMES does not match the correlation-matrix header"
    )


# ──────────────────────────────────────────────────────────────────────
# select_biomarkers on real UKB metabolomics data
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_metabolomics_df() -> pd.DataFrame:
    """Load first 10 rows of the raw UKB metabolomics CSV."""
    return pd.read_csv(
        "/SPXvePFS/users/jytang/storage_tmp/metabolomics.csv",
        nrows=10,
    )


def test_select_biomarkers_shape(raw_metabolomics_df: pd.DataFrame):
    """Output must have eid + 168 columns = 169 columns."""
    result = select_biomarkers(raw_metabolomics_df, instance=0)
    assert "eid" in result.columns
    assert len(result.columns) == 169  # eid + 168 metabolites


def test_select_biomarkers_column_order(raw_metabolomics_df: pd.DataFrame):
    """Metabolite columns must follow METABOLITE_NAMES order."""
    result = select_biomarkers(raw_metabolomics_df, instance=0)
    result_metabolite_cols = [c for c in result.columns if c != "eid"]
    assert result_metabolite_cols == list(METABOLITE_NAMES)


def test_select_biomarkers_eid_preserved(raw_metabolomics_df: pd.DataFrame):
    """Participant eids must be carried through."""
    result = select_biomarkers(raw_metabolomics_df, instance=0)
    expected_eids = raw_metabolomics_df["participant.eid"].tolist()
    assert result["eid"].tolist() == expected_eids


def test_select_biomarkers_values_match(raw_metabolomics_df: pd.DataFrame):
    """Spot-check that values are correctly transferred from the raw data."""
    result = select_biomarkers(raw_metabolomics_df, instance=0)
    # Check a few known mappings
    spot_checks = {
        "p23400": "Total Cholesterol",
        "p23474": "3-Hydroxybutyrate",
        "p23470": "Glucose",
    }
    for field_id, name in spot_checks.items():
        raw_col = f"participant.{field_id}_i0"
        pd.testing.assert_series_equal(
            result[name].reset_index(drop=True),
            raw_metabolomics_df[raw_col].reset_index(drop=True),
            check_names=False,
        )


def test_select_biomarkers_row_count(raw_metabolomics_df: pd.DataFrame):
    """Output must have the same number of rows as input."""
    result = select_biomarkers(raw_metabolomics_df, instance=0)
    assert len(result) == len(raw_metabolomics_df)


def test_select_biomarkers_missing_column_raises():
    """Should raise KeyError when expected columns are absent."""
    dummy_df = pd.DataFrame({"participant.eid": [1, 2], "bogus": [0.1, 0.2]})
    with pytest.raises(KeyError, match="expected UKB columns not found"):
        select_biomarkers(dummy_df, instance=0)


def test_select_biomarkers_missing_eid_raises():
    """Should raise KeyError when neither eid column is present."""
    # Build a DataFrame that has the 168 biomarker columns but no eid
    cols = {f"participant.{NAME_TO_FIELD[n]}_i0": [0.0] for n in METABOLITE_NAMES}
    dummy_df = pd.DataFrame(cols)
    with pytest.raises(KeyError, match="eid"):
        select_biomarkers(dummy_df, instance=0)
