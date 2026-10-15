"""Vocabulary shared by every layer: code families, verdict tiers, evidence types."""
from __future__ import annotations

FAMILIES = ("achi", "mbs", "prosthesis", "dx")
# What runs unless a caller says otherwise. The diagnosis family answers a different question (which OTHER diagnoses make a target
# plausible, rather than which codes show it was treated / investigated / examined), costs extra, and is opt-in.
DEFAULT_FAMILIES = ("achi", "mbs", "prosthesis")
FAMILY_LABELS = {
    "achi": "ACHI procedure codes",
    "mbs": "MBS billing items",
    "prosthesis": "prosthesis device groups (each stands for every billing code under it)",
    "dx": "ICD-10-AM diagnosis categories that could be coded in the same episode",
}

SUPPORT_TIERS = ("core", "broad", "unclear", "not_supporting")
EVIDENCE_TYPES = ("treated", "investigated", "examined", "none")
# For the diagnosis family the "evidence type" field holds the RELATION between the candidate and the target category.
EVIDENCE_TYPES_BY_FAMILY = {"dx": ("underlying_condition", "complication_or_manifestation", "associated_comorbidity", "none")}
AT_LEAST_BROAD = ("core", "broad")
STRICTLY_CORE = ("core",)
DEFAULT_MERGE_TIERS = AT_LEAST_BROAD

THINKING_PRESETS = ("disabled", "low", "medium", "high", "adaptive")
DEFAULT_THINKING = "low"


def evidence_types(family: str) -> tuple:
    return EVIDENCE_TYPES_BY_FAMILY.get(family, EVIDENCE_TYPES)


def parse_families(value: str) -> tuple[str, ...]:
    """'achi,mbs' -> ('achi', 'mbs'); 'all' -> every family. Raises ValueError on an unknown name or an empty list."""
    v = (value or "").strip().lower()
    fams = FAMILIES if v == "all" else tuple(dict.fromkeys(f.strip() for f in v.split(",") if f.strip()))
    bad = [f for f in fams if f not in FAMILIES]
    if bad or not fams:
        raise ValueError(f"families must be a comma-separated subset of {FAMILIES} (or 'all'); got {value!r}")
    return fams
