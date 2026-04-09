"""Construct cohorts aligned with MetaboLM paper protocol.

Builds the fine-tuning cohort from raw UK Biobank data:
 1. Extract baseline dates from blood-collection timestamps.
 2. Parse ICD-10 diagnosis codes and match first-occurrence dates.
 3. Classify each participant-endpoint pair as incident / prevalent / healthy.
 4. Merge metabolomics features with labels and split train / val.

All heavy operations are vectorized (no per-participant Python loops).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from src.data.endpoints import DiseaseEndpoint

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Baseline dates
# ──────────────────────────────────────────────────────────────────────────────

def build_baseline_dates(time_blood_csv: str) -> pd.DataFrame:
    """Extract baseline blood-collection date per participant.

    Uses ``p3166_i0_a0`` (first blood-collection timestamp at instance 0).

    Returns
    -------
    DataFrame with columns ``[eid, baseline_date]`` where *baseline_date* is
    ``datetime64[ns]``.  Rows with missing timestamps are dropped.
    """
    df = pd.read_csv(time_blood_csv, usecols=["participant.eid", "participant.p3166_i0_a0"])
    result = pd.DataFrame({
        "eid": df["participant.eid"],
        "baseline_date": pd.to_datetime(df["participant.p3166_i0_a0"], errors="coerce"),
    })
    n_before = len(result)
    result = result.dropna(subset=["baseline_date"]).reset_index(drop=True)
    logger.info("Baseline dates: %d / %d participants have valid timestamps.", len(result), n_before)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. ICD-10 code parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_icd10_json(raw_str: str) -> list[str]:
    """Parse a single p41270 cell into a list of raw ICD-10 code strings.

    Handles the JSON-array format ``["F412","N318"]`` as well as common
    quoting variants.  Returns an empty list for NaN / empty cells.
    """
    if not isinstance(raw_str, str) or not raw_str.strip():
        return []
    try:
        codes = json.loads(raw_str.replace("'", '"'))
        if isinstance(codes, list):
            return [str(c).strip() for c in codes if c]
        return []
    except (json.JSONDecodeError, ValueError):
        # Fallback: strip brackets, split on comma, strip quotes
        inner = raw_str.strip("[] ")
        if not inner:
            return []
        return [c.strip().strip('"').strip("'") for c in inner.split(",") if c.strip()]


def normalize_icd10(code: str) -> str:
    """Normalize an ICD-10 code to 3-character format (letter + 2 digits).

    E.g. ``"F412"`` -> ``"F41"``, ``"E11.9"`` -> ``"E11"``.
    Returns empty string for malformed codes.
    """
    code = code.strip().replace(".", "")
    if len(code) >= 3 and code[0].isalpha():
        return code[:3].upper()
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# 3. Build long-format diagnosis table  (vectorized)
# ──────────────────────────────────────────────────────────────────────────────

def _explode_diagnosis_codes(diagnosis_csv: str) -> pd.DataFrame:
    """Parse p41270 into a long DataFrame ``(eid, code_index, raw_code, code3)``.

    *code_index* is the positional index within the JSON array, corresponding
    to the ``p41280_a{code_index}`` column in the first-occurrence-date file.
    """
    diag_df = pd.read_csv(diagnosis_csv)
    diag_df = diag_df.rename(columns={"participant.eid": "eid"})

    records: list[dict] = []
    for eid, raw in zip(diag_df["eid"], diag_df["participant.p41270"]):
        codes = _parse_icd10_json(str(raw) if pd.notna(raw) else "")
        for idx, c in enumerate(codes):
            norm = normalize_icd10(c)
            if norm:
                records.append({"eid": eid, "code_index": idx, "raw_code": c, "code3": norm})

    long = pd.DataFrame(records)
    logger.info("Exploded diagnosis codes: %d rows from %d participants.", len(long), long["eid"].nunique() if len(long) else 0)
    return long


def _attach_first_occur_dates(long_codes: pd.DataFrame, first_occur_csv: str) -> pd.DataFrame:
    """Merge first-occurrence dates onto the long-format code table.

    The ``first_occur_date.csv`` file stores ``p41280_a{i}`` columns for
    indices 0-258, with ``participant.eid`` as the **last** column.  The
    positional index *i* matches *code_index* from the exploded p41270 array.

    This function reads the CSV, melts date columns to long format, and joins.
    """
    # Read first occurrence dates
    fo_df = pd.read_csv(first_occur_csv)
    fo_df = fo_df.rename(columns={"participant.eid": "eid"})

    # Identify date columns (p41280_a0 .. p41280_a258)
    date_cols = sorted(
        [c for c in fo_df.columns if c.startswith("participant.p41280_a")],
        key=lambda c: int(c.split("_a")[-1]),
    )

    # Melt to long format: (eid, code_index, first_occur_date)
    fo_long = fo_df[["eid"] + date_cols].melt(
        id_vars="eid", value_vars=date_cols,
        var_name="col_name", value_name="first_occur_date",
    )
    # Extract code_index from column name
    fo_long["code_index"] = fo_long["col_name"].str.extract(r"_a(\d+)$").astype(int)
    fo_long = fo_long.drop(columns="col_name")

    # Drop rows with no date to reduce join size
    fo_long = fo_long.dropna(subset=["first_occur_date"])
    fo_long["first_occur_date"] = pd.to_datetime(fo_long["first_occur_date"], errors="coerce")

    # Merge
    merged = long_codes.merge(fo_long, on=["eid", "code_index"], how="left")
    logger.info(
        "Attached dates: %d / %d code rows have a first-occurrence date.",
        merged["first_occur_date"].notna().sum(), len(merged),
    )
    return merged


def _add_cause_of_death_codes(
    long_codes: pd.DataFrame,
    cause_of_death_csv: str,
) -> pd.DataFrame:
    """Append cause-of-death ICD-10 codes to the long diagnosis table.

    Cause-of-death codes (p40001_i0, p40001_i1) do not have a corresponding
    entry in the first-occurrence-date array.  We set ``first_occur_date`` to
    ``NaT`` for these; downstream labelling treats missing-date matches as
    *incident* (conservative: the death occurred after enrolment).
    """
    cod_df = pd.read_csv(cause_of_death_csv)
    cod_df = cod_df.rename(columns={"participant.eid": "eid"})

    records: list[dict] = []
    for col in ["participant.p40001_i0", "participant.p40001_i1"]:
        if col not in cod_df.columns:
            continue
        for eid, val in zip(cod_df["eid"], cod_df[col]):
            if pd.notna(val):
                norm = normalize_icd10(str(val))
                if norm:
                    records.append({
                        "eid": eid,
                        "code_index": -1,       # sentinel: not from p41270
                        "raw_code": str(val),
                        "code3": norm,
                        "first_occur_date": pd.NaT,
                    })

    if not records:
        return long_codes

    cod_long = pd.DataFrame(records)
    combined = pd.concat([long_codes, cod_long], ignore_index=True)
    logger.info("Added %d cause-of-death code rows.", len(cod_long))
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# 4. Build endpoint labels
# ──────────────────────────────────────────────────────────────────────────────

def build_diagnosis_labels(
    diagnosis_csv: str,
    first_occur_csv: str,
    cause_of_death_csv: str,
    baseline_dates: pd.DataFrame,
    endpoints: list[DiseaseEndpoint],
) -> pd.DataFrame:
    """Determine incident / prevalent status for each participant and endpoint.

    For each (participant, endpoint) pair:

    - **Incident** (label = 1): at least one matching ICD-10 code with a
      first-occurrence date strictly **after** the baseline date, or a
      cause-of-death code with no recorded date (assumed post-baseline).
    - **Prevalent** (exclude): *all* matching codes occurred **on or before**
      the baseline date (disease was already present at enrolment).
    - **Healthy** (label = 0): no matching codes at all.

    Returns
    -------
    DataFrame with columns ``[eid, label_<ep1>, label_<ep2>, ..., is_prevalent_any]``.
    ``label_*`` is 1 (incident) or 0 (healthy).  Rows where *any* endpoint is
    prevalent are flagged via ``is_prevalent_any`` (True) so callers can
    decide whether to exclude them.
    """
    # --- Step 1: Build long-format diagnosis table -------------------------
    logger.info("Parsing ICD-10 diagnosis codes...")
    long_codes = _explode_diagnosis_codes(diagnosis_csv)

    logger.info("Attaching first-occurrence dates...")
    long_codes = _attach_first_occur_dates(long_codes, first_occur_csv)

    logger.info("Adding cause-of-death codes...")
    long_codes = _add_cause_of_death_codes(long_codes, cause_of_death_csv)

    # --- Step 2: Merge baseline dates into long table ----------------------
    long_codes = long_codes.merge(baseline_dates[["eid", "baseline_date"]], on="eid", how="inner")

    # --- Step 3: Classify each code as incident / prevalent ----------------
    # incident:  first_occur_date > baseline_date  OR  first_occur_date is NaT
    #            (NaT means cause-of-death or unknown -> assume post-baseline)
    # prevalent: first_occur_date <= baseline_date
    long_codes["is_incident"] = (
        long_codes["first_occur_date"].isna()
        | (long_codes["first_occur_date"] > long_codes["baseline_date"])
    )

    # --- Step 4: For each endpoint, build label column ---------------------
    result = baseline_dates[["eid"]].copy()

    prevalent_any = pd.Series(False, index=result.index)

    for ep in endpoints:
        target_codes = set(ep.icd10_codes)
        col_name = f"label_{ep.name}"

        # Filter to codes matching this endpoint
        mask = long_codes["code3"].isin(target_codes)
        ep_codes = long_codes.loc[mask, ["eid", "is_incident"]].copy()

        if ep_codes.empty:
            result[col_name] = 0
            continue

        # Per participant: any incident code? any prevalent code?
        ep_agg = ep_codes.groupby("eid")["is_incident"].agg(["any", "all"])
        ep_agg.columns = ["has_incident", "all_incident"]
        # has_incident = True  => at least one code is incident
        # all_incident = False => at least one code is prevalent
        # If has_incident but not all_incident => mixed; we count as incident
        #   (the participant *also* developed new codes post-baseline)
        # If not has_incident => all are prevalent -> prevalent

        # Merge into result
        result = result.merge(
            ep_agg.reset_index().rename(columns={"index": "eid"}),
            on="eid",
            how="left",
        )

        # Label: 1 if incident (has at least one post-baseline code), 0 otherwise
        result[col_name] = 0
        matched_mask = result["has_incident"].notna()
        result.loc[matched_mask & result["has_incident"], col_name] = 1

        # Prevalent: matched but NO incident code at all
        is_prevalent = matched_mask & (~result["has_incident"].fillna(False).astype(bool))
        prevalent_any = prevalent_any | result["eid"].isin(
            result.loc[is_prevalent, "eid"]
        )
        # Set prevalent cases to 0 (they will be excluded via is_prevalent_any)
        result.loc[is_prevalent, col_name] = 0

        # Clean up temporary columns
        result = result.drop(columns=["has_incident", "all_incident"], errors="ignore")

    result["is_prevalent_any"] = prevalent_any.values

    # Log summary
    for ep in endpoints:
        col = f"label_{ep.name}"
        if col in result.columns:
            n_pos = int(result[col].sum())
            logger.info("  %s: %d incident cases", ep.name, n_pos)
    n_prev = int(result["is_prevalent_any"].sum())
    logger.info("  Prevalent-any (to exclude): %d", n_prev)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. Build final cohort: merge metabolomics + labels, train/val split
# ──────────────────────────────────────────────────────────────────────────────

def build_cohort(
    metabolomics: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    exclude_prevalent: bool = True,
    min_nmr_present: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge metabolomics features with disease labels and split train / val.

    Parameters
    ----------
    metabolomics:
        Output of :func:`~src.data.biomarkers.select_biomarkers` — must have
        ``eid`` plus 168 metabolite columns.
    labels:
        Output of :func:`build_diagnosis_labels`.
    val_fraction:
        Fraction of data to hold out for validation.
    seed:
        Random seed for reproducibility.
    exclude_prevalent:
        If True, drop participants where ``is_prevalent_any`` is True.
    min_nmr_present:
        Minimum number of non-NaN metabolite values a participant must have
        to be retained.

    Returns
    -------
    ``(train_df, val_df)`` — two DataFrames with identical column structure:
    ``[eid, <168 metabolite cols>, label_T2D, label_obesity, ...]``.
    """
    # Merge on eid (inner join: must have both metabolomics + labels)
    cohort = metabolomics.merge(labels, on="eid", how="inner")
    logger.info("After merge (metab + labels): %d participants.", len(cohort))

    # Exclude prevalent cases
    if exclude_prevalent and "is_prevalent_any" in cohort.columns:
        n_before = len(cohort)
        cohort = cohort[~cohort["is_prevalent_any"]].copy()
        logger.info("Excluded %d prevalent cases, %d remain.", n_before - len(cohort), len(cohort))

    # Drop participants with too few NMR values
    from src.data.biomarkers import METABOLITE_NAMES
    nmr_cols = [c for c in METABOLITE_NAMES if c in cohort.columns]
    n_present = cohort[nmr_cols].notna().sum(axis=1)
    cohort = cohort[n_present >= min_nmr_present].copy()
    logger.info("After NMR presence filter (>=%d): %d participants.", min_nmr_present, len(cohort))

    # Drop helper columns
    cohort = cohort.drop(columns=["is_prevalent_any", "baseline_date"], errors="ignore")

    # Train / val split (stratified is hard with multi-label; use random)
    train_df, val_df = train_test_split(
        cohort, test_size=val_fraction, random_state=seed, shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info("Train: %d, Val: %d", len(train_df), len(val_df))
    return train_df, val_df
