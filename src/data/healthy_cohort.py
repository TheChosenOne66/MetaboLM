"""Construct the "relatively healthy" cohort for MetaboLM pretraining evaluation.

Replicates the paper's definition as closely as possible using available UKB data:

    "The healthy population was defined as participants who did not have any of
     these 16 common chronic diseases OR ANY CANCERS either at baseline or
     during follow-up."

Data sources used:
  - diagnosis_icd10.csv  (p41270 — hospital inpatient ICD-10 codes)
  - cause_of_death.csv   (p40001 — ICD-10 death cause codes)

Data sources NOT available (causes slight overcount of "healthy"):
  - Self-reported diagnoses (p20002)
  - Primary care data (Field 3000)
  - Cancer registry (p40005/p40006)
  - Algorithmically-defined outcomes (asthma, dementia, COPD, stroke, Parkinson's)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from src.data.cohort import _parse_icd10_json, normalize_icd10

logger = logging.getLogger(__name__)

# ── ICD-10 codes to exclude ─────────────────────────────────────────────────
# 16 chronic diseases from MetaboLM paper
EXCLUDE_DISEASE_CODES_3CHAR: set[str] = {
    "E11",        # Type 2 Diabetes
    "E66",        # Obesity
    "I10",        # Essential Hypertension
    "I25",        # Chronic Ischaemic Heart Disease
    "I48",        # Atrial Fibrillation and Flutter
    "I50",        # Heart Failure
    "M05", "M06", # Rheumatoid Arthritis
    "J45",        # Asthma
    "F00", "F01", "F02", "F03", "G30",  # Dementia
    "J44",        # COPD
    "I60", "I61", "I62", "I63", "I64",  # Stroke
    "G20",        # Parkinson's Disease
    "C50",        # Breast Cancer
    "C18",        # Colon Cancer
    "C34",        # Lung Cancer
    "C61",        # Prostate Cancer
}


def _is_any_cancer(code3: str) -> bool:
    """Check if a 3-char ICD-10 code is any cancer (C00-C97)."""
    return code3.startswith("C") and len(code3) == 3


def _has_excluded_code(icd10_codes: list[str]) -> bool:
    """Check if any code in the list matches an exclusion criterion."""
    for raw_code in icd10_codes:
        code3 = normalize_icd10(raw_code)
        if not code3:
            continue
        if code3 in EXCLUDE_DISEASE_CODES_3CHAR:
            return True
        if _is_any_cancer(code3):
            return True
    return False


def build_healthy_eids(
    diagnosis_csv: str,
    cause_of_death_csv: str,
    metabolomics_eids: Optional[set[int]] = None,
) -> set[int]:
    """Identify "relatively healthy" participants by exclusion.

    Parameters
    ----------
    diagnosis_csv:
        Path to diagnosis_icd10.csv (p41270).
    cause_of_death_csv:
        Path to cause_of_death.csv (p40001).
    metabolomics_eids:
        If provided, only consider participants in this set
        (i.e., those with valid NMR metabolomics data).

    Returns
    -------
    Set of eids that pass all exclusion criteria.
    """
    # ── Step 1: Find eids to exclude from hospital diagnoses ─────────────
    logger.info("Parsing hospital diagnosis codes (p41270)...")
    diag_df = pd.read_csv(diagnosis_csv)
    diag_df = diag_df.rename(columns={"participant.eid": "eid"})

    excluded_eids: set[int] = set()
    n_total = len(diag_df)
    for eid, raw in zip(diag_df["eid"], diag_df["participant.p41270"]):
        codes = _parse_icd10_json(str(raw) if pd.notna(raw) else "")
        if _has_excluded_code(codes):
            excluded_eids.add(int(eid))

    logger.info(
        "Hospital diagnoses: %d / %d participants excluded (have 16-disease or cancer codes).",
        len(excluded_eids), n_total,
    )

    # ── Step 2: Find eids to exclude from cause of death ─────────────────
    logger.info("Parsing cause-of-death codes (p40001)...")
    cod_df = pd.read_csv(cause_of_death_csv)
    cod_df = cod_df.rename(columns={"participant.eid": "eid"})

    n_cod_excluded = 0
    for col in ["participant.p40001_i0", "participant.p40001_i1"]:
        if col not in cod_df.columns:
            continue
        for eid, val in zip(cod_df["eid"], cod_df[col]):
            if pd.notna(val):
                code3 = normalize_icd10(str(val))
                if code3 and (code3 in EXCLUDE_DISEASE_CODES_3CHAR or _is_any_cancer(code3)):
                    if int(eid) not in excluded_eids:
                        n_cod_excluded += 1
                    excluded_eids.add(int(eid))

    logger.info(
        "Cause-of-death: %d additional participants excluded. Total excluded: %d.",
        n_cod_excluded, len(excluded_eids),
    )

    # ── Step 3: Build healthy set ────────────────────────────────────────
    if metabolomics_eids is not None:
        all_eids = metabolomics_eids
    else:
        # Use all eids from the diagnosis file as the universe
        all_eids = set(diag_df["eid"].astype(int).tolist())

    healthy = all_eids - excluded_eids
    logger.info(
        "Healthy cohort: %d / %d participants (%.1f%% excluded).",
        len(healthy), len(all_eids),
        100 * len(excluded_eids & all_eids) / len(all_eids) if all_eids else 0,
    )
    return healthy


def split_healthy_cohort(
    healthy_eids: set[int],
    pretrain_fraction: float = 0.9,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split healthy eids 9:1 into pretraining and fine-tuning control sets.

    Parameters
    ----------
    healthy_eids:
        Set of healthy participant eids.
    pretrain_fraction:
        Fraction for pretraining (default 0.9, matching paper).
    seed:
        Random seed.

    Returns
    -------
    ``(pretrain_eids, finetune_eids)`` — sorted lists.
    """
    import random
    eids_sorted = sorted(healthy_eids)
    rng = random.Random(seed)
    rng.shuffle(eids_sorted)

    split_idx = int(len(eids_sorted) * pretrain_fraction)
    pretrain = sorted(eids_sorted[:split_idx])
    finetune = sorted(eids_sorted[split_idx:])

    logger.info(
        "Healthy split: pretrain=%d (%.0f%%), finetune_ctrl=%d (%.0f%%).",
        len(pretrain), 100 * pretrain_fraction,
        len(finetune), 100 * (1 - pretrain_fraction),
    )
    return pretrain, finetune
