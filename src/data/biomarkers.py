"""168 original NMR biomarker field definitions for UK Biobank.

Maps between MetaboLM's 168 metabolite names (from the pretrained model's
correlation matrix) and UK Biobank Nightingale NMR metabolomics field IDs
(p23400-p23648).  Only the 168 "original" biomarkers are included; derived
ratios and percentage compositions are excluded.

The canonical ordering of metabolites is alphabetical, matching the column
order of the pretrained correlation matrix.
"""

from __future__ import annotations

import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Ordered list of 168 metabolite names, matching the pretrained model's
# correlation-matrix column order (alphabetical).
# ──────────────────────────────────────────────────────────────────────
METABOLITE_NAMES: list[str] = [
    "3-Hydroxybutyrate",
    "Acetate",
    "Acetoacetate",
    "Acetone",
    "Alanine",
    "Albumin",
    "Apolipoprotein A1",
    "Apolipoprotein B",
    "Average Diameter for HDL Particles",
    "Average Diameter for LDL Particles",
    "Average Diameter for VLDL Particles",
    "Cholesterol in Chylomicrons and Extremely Large VLDL",
    "Cholesterol in IDL",
    "Cholesterol in Large HDL",
    "Cholesterol in Large LDL",
    "Cholesterol in Large VLDL",
    "Cholesterol in Medium HDL",
    "Cholesterol in Medium LDL",
    "Cholesterol in Medium VLDL",
    "Cholesterol in Small HDL",
    "Cholesterol in Small LDL",
    "Cholesterol in Small VLDL",
    "Cholesterol in Very Large HDL",
    "Cholesterol in Very Large VLDL",
    "Cholesterol in Very Small VLDL",
    "Cholesteryl Esters in Chylomicrons and Extremely Large VLDL",
    "Cholesteryl Esters in HDL",
    "Cholesteryl Esters in IDL",
    "Cholesteryl Esters in Large HDL",
    "Cholesteryl Esters in Large LDL",
    "Cholesteryl Esters in Large VLDL",
    "Cholesteryl Esters in LDL",
    "Cholesteryl Esters in Medium HDL",
    "Cholesteryl Esters in Medium LDL",
    "Cholesteryl Esters in Medium VLDL",
    "Cholesteryl Esters in Small HDL",
    "Cholesteryl Esters in Small LDL",
    "Cholesteryl Esters in Small VLDL",
    "Cholesteryl Esters in Very Large HDL",
    "Cholesteryl Esters in Very Large VLDL",
    "Cholesteryl Esters in Very Small VLDL",
    "Cholesteryl Esters in VLDL",
    "Citrate",
    "Clinical LDL Cholesterol",
    "Concentration of Chylomicrons and Extremely Large VLDL Particles",
    "Concentration of HDL Particles",
    "Concentration of IDL Particles",
    "Concentration of Large HDL Particles",
    "Concentration of Large LDL Particles",
    "Concentration of Large VLDL Particles",
    "Concentration of LDL Particles",
    "Concentration of Medium HDL Particles",
    "Concentration of Medium LDL Particles",
    "Concentration of Medium VLDL Particles",
    "Concentration of Small HDL Particles",
    "Concentration of Small LDL Particles",
    "Concentration of Small VLDL Particles",
    "Concentration of Very Large HDL Particles",
    "Concentration of Very Large VLDL Particles",
    "Concentration of Very Small VLDL Particles",
    "Concentration of VLDL Particles",
    "Creatinine",
    "Degree of Unsaturation",
    "Docosahexaenoic Acid",
    "Free Cholesterol in Chylomicrons and Extremely Large VLDL",
    "Free Cholesterol in HDL",
    "Free Cholesterol in IDL",
    "Free Cholesterol in Large HDL",
    "Free Cholesterol in Large LDL",
    "Free Cholesterol in Large VLDL",
    "Free Cholesterol in LDL",
    "Free Cholesterol in Medium HDL",
    "Free Cholesterol in Medium LDL",
    "Free Cholesterol in Medium VLDL",
    "Free Cholesterol in Small HDL",
    "Free Cholesterol in Small LDL",
    "Free Cholesterol in Small VLDL",
    "Free Cholesterol in Very Large HDL",
    "Free Cholesterol in Very Large VLDL",
    "Free Cholesterol in Very Small VLDL",
    "Free Cholesterol in VLDL",
    "Glucose",
    "Glutamine",
    "Glycine",
    "Glycoprotein Acetyls",
    "HDL Cholesterol",
    "Histidine",
    "Isoleucine",
    "Lactate",
    "LDL Cholesterol",
    "Leucine",
    "Linoleic Acid",
    "Monounsaturated Fatty Acids",
    "Omega-3 Fatty Acids",
    "Omega-6 Fatty Acids",
    "Phenylalanine",
    "Phosphatidylcholines",
    "Phosphoglycerides",
    "Phospholipids in Chylomicrons and Extremely Large VLDL",
    "Phospholipids in HDL",
    "Phospholipids in IDL",
    "Phospholipids in Large HDL",
    "Phospholipids in Large LDL",
    "Phospholipids in Large VLDL",
    "Phospholipids in LDL",
    "Phospholipids in Medium HDL",
    "Phospholipids in Medium LDL",
    "Phospholipids in Medium VLDL",
    "Phospholipids in Small HDL",
    "Phospholipids in Small LDL",
    "Phospholipids in Small VLDL",
    "Phospholipids in Very Large HDL",
    "Phospholipids in Very Large VLDL",
    "Phospholipids in Very Small VLDL",
    "Phospholipids in VLDL",
    "Polyunsaturated Fatty Acids",
    "Pyruvate",
    "Remnant Cholesterol (Non-HDL, Non-LDL -Cholesterol)",
    "Saturated Fatty Acids",
    "Sphingomyelins",
    "Total Cholesterol",
    "Total Cholesterol Minus HDL-C",
    "Total Cholines",
    "Total Concentration of Branched-Chain Amino Acids (Leucine + Isoleucine + Valine)",
    "Total Concentration of Lipoprotein Particles",
    "Total Esterified Cholesterol",
    "Total Fatty Acids",
    "Total Free Cholesterol",
    "Total Lipids in Chylomicrons and Extremely Large VLDL",
    "Total Lipids in HDL",
    "Total Lipids in IDL",
    "Total Lipids in Large HDL",
    "Total Lipids in Large LDL",
    "Total Lipids in Large VLDL",
    "Total Lipids in LDL",
    "Total Lipids in Lipoprotein Particles",
    "Total Lipids in Medium HDL",
    "Total Lipids in Medium LDL",
    "Total Lipids in Medium VLDL",
    "Total Lipids in Small HDL",
    "Total Lipids in Small LDL",
    "Total Lipids in Small VLDL",
    "Total Lipids in Very Large HDL",
    "Total Lipids in Very Large VLDL",
    "Total Lipids in Very Small VLDL",
    "Total Lipids in VLDL",
    "Total Phospholipids in Lipoprotein Particles",
    "Total Triglycerides",
    "Triglycerides in Chylomicrons and Extremely Large VLDL",
    "Triglycerides in HDL",
    "Triglycerides in IDL",
    "Triglycerides in Large HDL",
    "Triglycerides in Large LDL",
    "Triglycerides in Large VLDL",
    "Triglycerides in LDL",
    "Triglycerides in Medium HDL",
    "Triglycerides in Medium LDL",
    "Triglycerides in Medium VLDL",
    "Triglycerides in Small HDL",
    "Triglycerides in Small LDL",
    "Triglycerides in Small VLDL",
    "Triglycerides in Very Large HDL",
    "Triglycerides in Very Large VLDL",
    "Triglycerides in Very Small VLDL",
    "Triglycerides in VLDL",
    "Tyrosine",
    "Valine",
    "VLDL Cholesterol",
]

# ──────────────────────────────────────────────────────────────────────
# UKB field ID -> metabolite name mapping  (168 original biomarkers)
# Derived from UK Biobank Data_description.rtf field table.
# Fields p23400-p23648 cover 249 Nightingale NMR biomarkers; only the
# 168 non-ratio, non-percentage fields are included here.
# ──────────────────────────────────────────────────────────────────────
FIELD_TO_NAME: dict[str, str] = {
    "p23474": "3-Hydroxybutyrate",
    "p23475": "Acetate",
    "p23476": "Acetoacetate",
    "p23477": "Acetone",
    "p23460": "Alanine",
    "p23479": "Albumin",
    "p23440": "Apolipoprotein A1",
    "p23439": "Apolipoprotein B",
    "p23433": "Average Diameter for HDL Particles",
    "p23432": "Average Diameter for LDL Particles",
    "p23431": "Average Diameter for VLDL Particles",
    "p23484": "Cholesterol in Chylomicrons and Extremely Large VLDL",
    "p23526": "Cholesterol in IDL",
    "p23561": "Cholesterol in Large HDL",
    "p23533": "Cholesterol in Large LDL",
    "p23498": "Cholesterol in Large VLDL",
    "p23568": "Cholesterol in Medium HDL",
    "p23540": "Cholesterol in Medium LDL",
    "p23505": "Cholesterol in Medium VLDL",
    "p23575": "Cholesterol in Small HDL",
    "p23547": "Cholesterol in Small LDL",
    "p23512": "Cholesterol in Small VLDL",
    "p23554": "Cholesterol in Very Large HDL",
    "p23491": "Cholesterol in Very Large VLDL",
    "p23519": "Cholesterol in Very Small VLDL",
    "p23485": "Cholesteryl Esters in Chylomicrons and Extremely Large VLDL",
    "p23418": "Cholesteryl Esters in HDL",
    "p23527": "Cholesteryl Esters in IDL",
    "p23562": "Cholesteryl Esters in Large HDL",
    "p23534": "Cholesteryl Esters in Large LDL",
    "p23499": "Cholesteryl Esters in Large VLDL",
    "p23417": "Cholesteryl Esters in LDL",
    "p23569": "Cholesteryl Esters in Medium HDL",
    "p23541": "Cholesteryl Esters in Medium LDL",
    "p23506": "Cholesteryl Esters in Medium VLDL",
    "p23576": "Cholesteryl Esters in Small HDL",
    "p23548": "Cholesteryl Esters in Small LDL",
    "p23513": "Cholesteryl Esters in Small VLDL",
    "p23555": "Cholesteryl Esters in Very Large HDL",
    "p23492": "Cholesteryl Esters in Very Large VLDL",
    "p23520": "Cholesteryl Esters in Very Small VLDL",
    "p23416": "Cholesteryl Esters in VLDL",
    "p23473": "Citrate",
    "p23404": "Clinical LDL Cholesterol",
    "p23481": "Concentration of Chylomicrons and Extremely Large VLDL Particles",
    "p23430": "Concentration of HDL Particles",
    "p23523": "Concentration of IDL Particles",
    "p23558": "Concentration of Large HDL Particles",
    "p23530": "Concentration of Large LDL Particles",
    "p23495": "Concentration of Large VLDL Particles",
    "p23429": "Concentration of LDL Particles",
    "p23565": "Concentration of Medium HDL Particles",
    "p23537": "Concentration of Medium LDL Particles",
    "p23502": "Concentration of Medium VLDL Particles",
    "p23572": "Concentration of Small HDL Particles",
    "p23544": "Concentration of Small LDL Particles",
    "p23509": "Concentration of Small VLDL Particles",
    "p23551": "Concentration of Very Large HDL Particles",
    "p23488": "Concentration of Very Large VLDL Particles",
    "p23516": "Concentration of Very Small VLDL Particles",
    "p23428": "Concentration of VLDL Particles",
    "p23478": "Creatinine",
    "p23443": "Degree of Unsaturation",
    "p23450": "Docosahexaenoic Acid",
    "p23486": "Free Cholesterol in Chylomicrons and Extremely Large VLDL",
    "p23422": "Free Cholesterol in HDL",
    "p23528": "Free Cholesterol in IDL",
    "p23563": "Free Cholesterol in Large HDL",
    "p23535": "Free Cholesterol in Large LDL",
    "p23500": "Free Cholesterol in Large VLDL",
    "p23421": "Free Cholesterol in LDL",
    "p23570": "Free Cholesterol in Medium HDL",
    "p23542": "Free Cholesterol in Medium LDL",
    "p23507": "Free Cholesterol in Medium VLDL",
    "p23577": "Free Cholesterol in Small HDL",
    "p23549": "Free Cholesterol in Small LDL",
    "p23514": "Free Cholesterol in Small VLDL",
    "p23556": "Free Cholesterol in Very Large HDL",
    "p23493": "Free Cholesterol in Very Large VLDL",
    "p23521": "Free Cholesterol in Very Small VLDL",
    "p23420": "Free Cholesterol in VLDL",
    "p23470": "Glucose",
    "p23461": "Glutamine",
    "p23462": "Glycine",
    "p23480": "Glycoprotein Acetyls",
    "p23406": "HDL Cholesterol",
    "p23463": "Histidine",
    "p23465": "Isoleucine",
    "p23471": "Lactate",
    "p23405": "LDL Cholesterol",
    "p23466": "Leucine",
    "p23449": "Linoleic Acid",
    "p23447": "Monounsaturated Fatty Acids",
    "p23444": "Omega-3 Fatty Acids",
    "p23445": "Omega-6 Fatty Acids",
    "p23468": "Phenylalanine",
    "p23437": "Phosphatidylcholines",
    "p23434": "Phosphoglycerides",
    "p23483": "Phospholipids in Chylomicrons and Extremely Large VLDL",
    "p23414": "Phospholipids in HDL",
    "p23525": "Phospholipids in IDL",
    "p23560": "Phospholipids in Large HDL",
    "p23532": "Phospholipids in Large LDL",
    "p23497": "Phospholipids in Large VLDL",
    "p23413": "Phospholipids in LDL",
    "p23567": "Phospholipids in Medium HDL",
    "p23539": "Phospholipids in Medium LDL",
    "p23504": "Phospholipids in Medium VLDL",
    "p23574": "Phospholipids in Small HDL",
    "p23546": "Phospholipids in Small LDL",
    "p23511": "Phospholipids in Small VLDL",
    "p23553": "Phospholipids in Very Large HDL",
    "p23490": "Phospholipids in Very Large VLDL",
    "p23518": "Phospholipids in Very Small VLDL",
    "p23412": "Phospholipids in VLDL",
    "p23446": "Polyunsaturated Fatty Acids",
    "p23472": "Pyruvate",
    "p23402": "Remnant Cholesterol (Non-HDL, Non-LDL -Cholesterol)",
    "p23448": "Saturated Fatty Acids",
    "p23438": "Sphingomyelins",
    "p23400": "Total Cholesterol",
    "p23401": "Total Cholesterol Minus HDL-C",
    "p23436": "Total Cholines",
    "p23464": "Total Concentration of Branched-Chain Amino Acids (Leucine + Isoleucine + Valine)",
    "p23427": "Total Concentration of Lipoprotein Particles",
    "p23415": "Total Esterified Cholesterol",
    "p23442": "Total Fatty Acids",
    "p23419": "Total Free Cholesterol",
    "p23482": "Total Lipids in Chylomicrons and Extremely Large VLDL",
    "p23426": "Total Lipids in HDL",
    "p23524": "Total Lipids in IDL",
    "p23559": "Total Lipids in Large HDL",
    "p23531": "Total Lipids in Large LDL",
    "p23496": "Total Lipids in Large VLDL",
    "p23425": "Total Lipids in LDL",
    "p23423": "Total Lipids in Lipoprotein Particles",
    "p23566": "Total Lipids in Medium HDL",
    "p23538": "Total Lipids in Medium LDL",
    "p23503": "Total Lipids in Medium VLDL",
    "p23573": "Total Lipids in Small HDL",
    "p23545": "Total Lipids in Small LDL",
    "p23510": "Total Lipids in Small VLDL",
    "p23552": "Total Lipids in Very Large HDL",
    "p23489": "Total Lipids in Very Large VLDL",
    "p23517": "Total Lipids in Very Small VLDL",
    "p23424": "Total Lipids in VLDL",
    "p23411": "Total Phospholipids in Lipoprotein Particles",
    "p23407": "Total Triglycerides",
    "p23487": "Triglycerides in Chylomicrons and Extremely Large VLDL",
    "p23410": "Triglycerides in HDL",
    "p23529": "Triglycerides in IDL",
    "p23564": "Triglycerides in Large HDL",
    "p23536": "Triglycerides in Large LDL",
    "p23501": "Triglycerides in Large VLDL",
    "p23409": "Triglycerides in LDL",
    "p23571": "Triglycerides in Medium HDL",
    "p23543": "Triglycerides in Medium LDL",
    "p23508": "Triglycerides in Medium VLDL",
    "p23578": "Triglycerides in Small HDL",
    "p23550": "Triglycerides in Small LDL",
    "p23515": "Triglycerides in Small VLDL",
    "p23557": "Triglycerides in Very Large HDL",
    "p23494": "Triglycerides in Very Large VLDL",
    "p23522": "Triglycerides in Very Small VLDL",
    "p23408": "Triglycerides in VLDL",
    "p23469": "Tyrosine",
    "p23467": "Valine",
    "p23403": "VLDL Cholesterol",
}

# Reverse mapping: metabolite name -> UKB field ID
NAME_TO_FIELD: dict[str, str] = {v: k for k, v in FIELD_TO_NAME.items()}


# ──────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────

def get_metabolite_names() -> list[str]:
    """Return ordered list of 168 metabolite names matching pretrained model."""
    return list(METABOLITE_NAMES)


def get_field_ids() -> list[str]:
    """Return 168 UKB field IDs in model order (alphabetical by name)."""
    return [NAME_TO_FIELD[name] for name in METABOLITE_NAMES]


def select_biomarkers(
    metabolomics_df: pd.DataFrame,
    instance: int = 0,
) -> pd.DataFrame:
    """Select and rename 168 NMR biomarker columns from raw UKB metabolomics.

    Args:
        metabolomics_df: Raw UKB CSV with ``participant.p23XXX_iN`` columns
            (and ``participant.eid`` for the participant identifier).
        instance: UKB assessment instance (0 = baseline). Default 0.

    Returns:
        DataFrame with ``eid`` (int) + 168 metabolite-name columns, ordered to
        match :data:`METABOLITE_NAMES` (i.e. the pretrained model order).

    Raises:
        KeyError: If expected UKB columns are missing from *metabolomics_df*.
    """
    # Build the column rename map: "participant.p23XXX_i0" -> metabolite name
    rename_map: dict[str, str] = {}
    missing_cols: list[str] = []
    for field_id, name in FIELD_TO_NAME.items():
        col = f"participant.{field_id}_i{instance}"
        if col in metabolomics_df.columns:
            rename_map[col] = name
        else:
            missing_cols.append(col)

    if missing_cols:
        raise KeyError(
            f"{len(missing_cols)} expected UKB columns not found in DataFrame. "
            f"First 5 missing: {missing_cols[:5]}"
        )

    # Determine eid column
    if "participant.eid" in metabolomics_df.columns:
        eid_col = "participant.eid"
    elif "eid" in metabolomics_df.columns:
        eid_col = "eid"
    else:
        raise KeyError("Neither 'participant.eid' nor 'eid' found in DataFrame.")

    # Select eid + 168 biomarker columns
    cols_to_select = [eid_col] + list(rename_map.keys())
    result = metabolomics_df[cols_to_select].copy()

    # Rename
    result = result.rename(columns={eid_col: "eid", **rename_map})

    # Reorder metabolite columns to match METABOLITE_NAMES (model order)
    result = result[["eid"] + list(METABOLITE_NAMES)]

    return result
