"""16 disease endpoint definitions from MetaboLM paper."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiseaseEndpoint:
    name: str              # e.g. "T2D"
    icd10_codes: tuple[str, ...]  # e.g. ("E11",) — use tuple for frozen dataclass
    chapter: str           # ICD-10 chapter key, e.g. "chapter_04"
    description: str       # Human-readable name


# 16 endpoints from MetaboLM paper (Nature Communications 2025)
# ICD-10 codes normalized to 3-character format (letter + 2 digits)
DISEASE_ENDPOINTS: list[DiseaseEndpoint] = [
    DiseaseEndpoint("T2D", ("E11",), "chapter_04", "Type 2 Diabetes"),
    DiseaseEndpoint("obesity", ("E66",), "chapter_04", "Obesity"),
    DiseaseEndpoint("hypertension", ("I10",), "chapter_09", "Essential Hypertension"),
    DiseaseEndpoint("ischemic_heart", ("I25",), "chapter_09", "Chronic Ischemic Heart Disease"),
    DiseaseEndpoint("atrial_fib", ("I48",), "chapter_09", "Atrial Fibrillation and Flutter"),
    DiseaseEndpoint("heart_failure", ("I50",), "chapter_09", "Heart Failure"),
    DiseaseEndpoint("rheumatoid", ("M05", "M06"), "chapter_13", "Rheumatoid Arthritis"),
    DiseaseEndpoint("asthma", ("J45",), "chapter_10", "Asthma"),
    DiseaseEndpoint("dementia", ("F00", "F01", "F02", "F03", "G30"), "chapter_06", "Dementia"),
    DiseaseEndpoint("copd", ("J44",), "chapter_10", "COPD"),
    DiseaseEndpoint("stroke", ("I60", "I61", "I62", "I63", "I64"), "chapter_09", "Stroke"),
    DiseaseEndpoint("parkinsons", ("G20",), "chapter_06", "Parkinson's Disease"),
    DiseaseEndpoint("breast_cancer", ("C50",), "chapter_02", "Breast Cancer"),
    DiseaseEndpoint("colon_cancer", ("C18",), "chapter_02", "Colon Cancer"),
    DiseaseEndpoint("lung_cancer", ("C34",), "chapter_02", "Lung Cancer"),
    DiseaseEndpoint("prostate_cancer", ("C61",), "chapter_02", "Prostate Cancer"),
]


def get_disease_names() -> list[str]:
    """Return ordered list of 16 disease names."""
    return [ep.name for ep in DISEASE_ENDPOINTS]


def get_unique_chapters() -> list[str]:
    """Return sorted unique chapter keys across all 16 endpoints."""
    return sorted(set(ep.chapter for ep in DISEASE_ENDPOINTS))


def get_chapter_to_diseases() -> dict[str, list[str]]:
    """Return mapping from chapter key to list of disease names."""
    mapping: dict[str, list[str]] = {}
    for ep in DISEASE_ENDPOINTS:
        mapping.setdefault(ep.chapter, []).append(ep.name)
    return mapping


def get_disease_to_chapter_idx() -> dict[int, int]:
    """Return mapping from disease index (0-15) to chapter index (0-5).
    Used by HierarchicalLoss to enforce P(leaf) <= P(chapter).
    """
    chapters = get_unique_chapters()
    chapter_to_idx = {ch: i for i, ch in enumerate(chapters)}
    return {i: chapter_to_idx[ep.chapter] for i, ep in enumerate(DISEASE_ENDPOINTS)}


def get_all_icd10_codes() -> set[str]:
    """Return set of all ICD-10 codes across all endpoints."""
    codes: set[str] = set()
    for ep in DISEASE_ENDPOINTS:
        codes.update(ep.icd10_codes)
    return codes
