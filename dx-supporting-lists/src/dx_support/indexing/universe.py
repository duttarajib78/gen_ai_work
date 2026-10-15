"""Loads the four reference files from the data directory and builds the categories and the four family indices."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dx_support.indexing.families import (build_achi_index, build_dx_index, build_mbs_index, build_prosthesis_index, check_index_integrity)
from dx_support.indexing.models import FamilyIndex
from dx_support.reference.categories import CategoryInfo, build_categories
from dx_support.reference.loaders import file_sha256, load_achi_codes, load_diagnosis_codes, load_mbs_items, load_prostheses

logger = logging.getLogger(__name__)

DIAGNOSIS_FILE = "diagnosis_codes.tsv"      # headerless TSV, ICD-10-AM leaf codes
ACHI_FILE = "achi_codes.tsv"                # headerless TSV, ACHI procedure codes
PROSTHESES_FILE = "all_pros.csv"            # prostheses billing codes (stacked extracts)
MBS_DEFAULT_NAME = "MBS-XML-20260801.XML"   # the schedule's date changes each release


def find_mbs_xml(data_dir: Path) -> Path:
    """The MBS schedule file. Its name carries the schedule date (MBS-XML-YYYYMMDD.XML), so pick the newest file in the data directory that
    matches the pattern; fall back to the default name (whose absence the loader reports clearly)."""
    data_dir = Path(data_dir)
    candidates = sorted(p for p in data_dir.glob("*") if p.is_file() and p.name.upper().startswith("MBS-XML-") and p.suffix.upper() == ".XML") \
        if data_dir.exists() else []
    return candidates[-1] if candidates else data_dir / MBS_DEFAULT_NAME


def data_files(data_dir: Path) -> dict[str, Path]:
    data_dir = Path(data_dir)
    return {"diagnosis": data_dir / DIAGNOSIS_FILE, "achi": data_dir / ACHI_FILE, "prostheses": data_dir / PROSTHESES_FILE, "mbs": find_mbs_xml(data_dir)}


def data_digest(data_dir: Path) -> str:
    """SHA-256 over the four input files (name + content digest). Part of the cache fingerprint: new reference data means new results."""
    h = hashlib.sha256()
    for key, path in sorted(data_files(data_dir).items()):
        h.update(key.encode())
        h.update(path.name.encode())
        h.update(file_sha256(path).encode())
    return h.hexdigest()


@dataclass
class Universe:
    categories: dict[str, CategoryInfo]
    indices: dict[str, FamilyIndex]
    report: dict = field(default_factory=dict)
    digest: str = ""
    include_coarse_prostheses: bool = False

    def leaf_to_category(self) -> dict[str, str]:
        return {code: cat for cat, info in self.categories.items() for code in info.member_codes}


def load_universe(data_dir: Path, include_coarse_prostheses: bool = False, warm_indexes: bool = True) -> Universe:
    """Loads all four reference files and builds the categories and the four family indices (ACHI, MBS, prostheses, diagnoses)."""
    files = data_files(data_dir)
    categories = build_categories(load_diagnosis_codes(files["diagnosis"]))
    pros_records, pros_report = load_prostheses(files["prostheses"])
    indices = {
        "achi": build_achi_index(load_achi_codes(files["achi"])),
        "mbs": build_mbs_index(load_mbs_items(files["mbs"])),
        "prosthesis": build_prosthesis_index(pros_records, include_coarse=include_coarse_prostheses),
        "dx": build_dx_index(categories),
    }
    problems = [p for idx in indices.values() for p in check_index_integrity(idx)]
    if problems:
        raise RuntimeError("Index integrity check failed: " + "; ".join(problems))
    if warm_indexes:                      # build the lexical indexes now: a first request must not pay for them (and threads must not race to)
        for idx in indices.values():
            idx.bm25()
    return Universe(categories=categories, indices=indices, report={"prostheses": pros_report}, digest=data_digest(data_dir),
                    include_coarse_prostheses=include_coarse_prostheses)


def select_categories(categories: dict[str, CategoryInfo], requested: Optional[list[str]] = None, chapter=None,
                      max_categories: Optional[int] = None) -> tuple[list[str], list[str]]:
    """(selected category ids in code order, requested ids that do not exist). `requested` overrides chapter / max_categories; otherwise the
    chapter filter applies first, then the first-N cut."""
    if requested:
        want = [c.strip().upper() for c in requested if c.strip()]
        missing = [c for c in want if c not in categories]
        return [c for c in categories if c in set(want)], missing
    chapters = None if not chapter else ({chapter} if isinstance(chapter, str) else set(chapter))
    ids = [c for c, i in categories.items() if chapters is None or i.chapter in chapters]
    if max_categories:
        ids = ids[:max_categories]
    return ids, []
