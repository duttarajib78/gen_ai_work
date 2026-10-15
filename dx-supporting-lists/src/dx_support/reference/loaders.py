"""Loads and cleans the four reference files. Nothing here calls an API and nothing here writes to the data directory.

Structure of each input (confirmed by inspecting the real files)
-----------------------------------------------------------------
diagnosis_codes.tsv / achi_codes.tsv -- NO header row, 8 tab-separated positional columns:
    0: constant "ALL"; 1: constant "ICD10_DIAGNOSIS" (diagnosis) / "ICD10_PROCEDURE" (ACHI); 2: the code (diagnosis "A000", "M4052";
    ACHI 7 digits); 3: constant "13" (not interpreted); 4, 5: blank; 6, 7: description, identical -- only 6 is used.
  The diagnosis file lists LEAF codes only (3-, 4- and 5-character): no category rows for categories that have subdivisions, no chapter or
  block titles. A "category" is therefore DERIVED: the first three characters of a leaf code.

MBS-XML-*.XML -- one <Data> element per item (ItemNum, Category, Group, SubGroup, SubHeading, Description, ItemEndDate, ...).
  Category/Group/SubGroup are CODES only; the file carries no titles for them.

all_pros.csv -- header row, 7 columns (Product Category, Sub Category, Product Group, Product Sub Group, Suffix, Billing Code, Benefit). SEVERAL
  EXTRACTS STACKED TOGETHER with no version column: a code appears 1-4 times; benefit values differ (irrelevant, benefits are never loaded);
  some rows are column-shifted (the real code sits in "Suffix", a dollar amount in "Billing Code") and are recovered; some codes carry only a
  coarse category and are excluded by default. The last stacked block is the one whose benefits are "$"-formatted (heuristic `in_latest_extract`).

Security posture: all four files are EXTERNAL INPUT and are treated as untrusted -- every code is validated against a strict pattern, the MBS XML
is parsed with defusedxml (DTD and entity declarations refused), and file sizes are bounded.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dx_support.text import display_text

logger = logging.getLogger(__name__)

_CODE_TYPE_COL, _CODE_COL, _DESC_COL = 1, 2, 6
_EXPECTED_TSV_COLUMNS = 8
_ICD_CODE_RE = re.compile(r"^[A-Z]\d{2}[A-Z0-9]{0,2}$")
_ACHI_CODE_RE = re.compile(r"^\d{7}$")
MAX_FILE_BYTES = 200 * 1024 * 1024


def _check_size(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- see data/README.txt for the four reference files.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{path} is larger than {MAX_FILE_BYTES:,} bytes -- refusing to load an unexpectedly large file.")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_positional_tsv(path: Path, expected_code_type: str, code_re: re.Pattern) -> dict[str, str]:
    """Shared loader for the two headerless TSVs. Sanity-checks the shape: a wrong column count or an unexpected code-type constant raises,
    rather than silently mis-reading a different file."""
    _check_size(path)
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if df.shape[1] != _EXPECTED_TSV_COLUMNS:
        raise ValueError(f"{path.name} has {df.shape[1]} columns, expected {_EXPECTED_TSV_COLUMNS}.")
    seen_types = set(df[_CODE_TYPE_COL].unique())
    if seen_types != {expected_code_type}:
        raise ValueError(f"{path.name} column {_CODE_TYPE_COL} contains {sorted(seen_types)}, expected only {expected_code_type!r}.")
    out: dict[str, str] = {}
    n_bad = n_dup = 0
    for code, desc in zip(df[_CODE_COL].tolist(), df[_DESC_COL].tolist()):
        code = str(code).strip()
        if not code_re.match(code):
            n_bad += 1
            continue
        if code in out:
            n_dup += 1
            continue
        out[code] = display_text(desc)
    if n_bad or n_dup:
        logger.warning("%s: skipped %d row(s) with an invalid code and %d duplicate code(s).", path.name, n_bad, n_dup)
    if not out:
        raise ValueError(f"{path.name}: no valid rows found.")
    return out


def load_diagnosis_codes(path: Path) -> dict[str, str]:
    """ICD-10-AM leaf code -> description (3-, 4- and 5-character codes)."""
    return _load_positional_tsv(Path(path), "ICD10_DIAGNOSIS", _ICD_CODE_RE)


def load_achi_codes(path: Path) -> dict[str, str]:
    """ACHI 7-digit code -> description."""
    return _load_positional_tsv(Path(path), "ICD10_PROCEDURE", _ACHI_CODE_RE)


# ---------------------------------------------------------------------------
# MBS (XML)
# ---------------------------------------------------------------------------
try:  # preferred: defusedxml blocks entity expansion / external entities
    from defusedxml.ElementTree import iterparse as _iterparse  # type: ignore
    _DEFUSED = True
except ImportError:  # pragma: no cover - exercised only where defusedxml is absent
    from xml.etree.ElementTree import iterparse as _iterparse  # noqa: S405
    _DEFUSED = False

_MBS_CODE_RE = re.compile(r"^\d{1,6}$")
_MBS_SUB_RE = re.compile(r"^[A-Za-z0-9]{1,4}$")


def canonical_mbs_code(code: object) -> str:
    """Strips leading zeros so "00023" and "23" compare equal."""
    stripped = str(code).strip().lstrip("0")
    return stripped or "0"


@dataclass(frozen=True)
class MbsItem:
    code: str
    category: str
    group: str
    subgroup: str
    subheading: str
    description: str
    ended: bool


def _assert_no_dtd(path: Path) -> None:
    """Fallback when defusedxml is not installed: a DOCTYPE (and so any ENTITY declaration) must appear before the root element, so refusing
    a document with one in its first 64 KB is sufficient."""
    with open(path, "rb") as fh:
        head = fh.read(64 * 1024).upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
        raise ValueError(f"{path.name} declares a DOCTYPE/ENTITY -- refusing to parse it (install defusedxml to override safely).")


def load_mbs_items(path: Path) -> list[MbsItem]:
    """All items in an MBS XML schedule. Streams the file (iterparse). Items with a non-blank ItemEndDate are flagged `ended`."""
    path = Path(path)
    _check_size(path)
    if not _DEFUSED:
        _assert_no_dtd(path)
    items: list[MbsItem] = []
    n_bad = 0
    kwargs = {"forbid_dtd": True} if _DEFUSED else {}
    root_checked = False
    for _event, elem in _iterparse(str(path), events=("end",), **kwargs):
        if elem.tag != "Data":
            continue
        root_checked = True
        raw_code = (elem.findtext("ItemNum") or "").strip()
        sub = (elem.findtext("SubItemNum") or "").strip()
        desc = display_text(elem.findtext("Description") or "")
        if not _MBS_CODE_RE.match(raw_code) or not desc or (sub and not _MBS_SUB_RE.match(sub)):
            n_bad += 1
            elem.clear()
            continue
        code = canonical_mbs_code(raw_code) + (f"-{sub}" if sub else "")
        items.append(MbsItem(
            code=code, category=(elem.findtext("Category") or "").strip(), group=(elem.findtext("Group") or "").strip(),
            subgroup=(elem.findtext("SubGroup") or "").strip(), subheading=(elem.findtext("SubHeading") or "").strip(),
            description=desc, ended=bool((elem.findtext("ItemEndDate") or "").strip()),
        ))
        elem.clear()
    if not root_checked or not items:
        raise ValueError(f"{path.name}: no <Data> items found -- is this an MBS XML schedule?")
    if n_bad:
        logger.warning("%s: skipped %d item(s) with an invalid item number or empty description.", path.name, n_bad)
    seen: set[str] = set()
    unique: list[MbsItem] = []
    for it in items:
        if it.code in seen:
            continue
        seen.add(it.code)
        unique.append(it)
    if len(unique) != len(items):
        logger.warning("%s: %d duplicate item number(s) dropped.", path.name, len(items) - len(unique))
    return unique


# ---------------------------------------------------------------------------
# Prostheses (stacked, partly column-shifted CSV)
# ---------------------------------------------------------------------------
_PROS_COLUMNS = ["Product Category", "Sub Category", "Product Group", "Product Sub Group", "Suffix", "Billing Code", "Benefit"]
# Two real formats exist side by side: 2 letters + 3 digits (the main scheme) and 3 letters + 2 digits (the coarse-labelled scheme).
_PROS_CODE_RE = re.compile(r"^[A-Z]{2,3}\d{2,3}$")


@dataclass(frozen=True)
class ProsRecord:
    code: str
    path: tuple[str, str, str, str]    # (category, sub category, product group, product sub group); "" when absent
    suffixes: tuple[str, ...]
    labelled: bool                      # has a sub category, i.e. a real device-type hierarchy
    in_latest_extract: bool
    n_label_variants: int               # >1 means the extracts disagree on this code's labels


def load_prostheses(path: Path) -> tuple[dict[str, ProsRecord], dict]:
    """Cleans the stacked prostheses CSV into one record per billing code.

    Billing Code valid -> normal row; Billing Code starts with "$" and Suffix is a valid code -> COLUMN-SHIFTED row (real code recovered from
    Suffix); anything else -> dropped and counted. Per code, the primary label path is the most frequent across extracts (ties: the most
    complete, then the latest in file order); the number of distinct paths is kept as `n_label_variants`. Benefit is never read.
    Returns (records, report)."""
    path = Path(path)
    _check_size(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    missing = set(_PROS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing column(s) {sorted(missing)}; got {list(df.columns)}.")
    sub = df[_PROS_COLUMNS]
    rows = sub.map(lambda v: str(v).strip()) if hasattr(sub, "map") else sub.applymap(lambda v: str(v).strip())

    per_code: dict[str, dict] = {}
    n_shifted = n_unrecoverable = 0
    any_dollar = False
    for idx, (cat, sub_cat, grp, ssg, sfx, bc, ben) in enumerate(rows.itertuples(index=False, name=None)):
        if _PROS_CODE_RE.match(bc):
            code, suffix = bc, sfx
        elif bc.startswith("$") and _PROS_CODE_RE.match(sfx):
            code, suffix = sfx, ""
            n_shifted += 1
        else:
            n_unrecoverable += 1
            continue
        is_dollar = ben.startswith("$") or bc.startswith("$")
        any_dollar = any_dollar or is_dollar
        rec = per_code.setdefault(code, {"paths": Counter(), "last": {}, "dollar": False, "suffixes": set()})
        p = (cat, sub_cat, grp, ssg)
        rec["paths"][p] += 1
        rec["last"][p] = idx
        rec["dollar"] = rec["dollar"] or is_dollar
        if suffix and suffix.lower() != "no suffix":
            rec["suffixes"].add(suffix)

    records: dict[str, ProsRecord] = {}
    n_conflict = 0
    for code in sorted(per_code):
        rec = per_code[code]
        best = max(rec["paths"], key=lambda p: (rec["paths"][p], sum(1 for x in p if x), rec["last"][p]))
        if len(rec["paths"]) > 1:
            n_conflict += 1
        records[code] = ProsRecord(
            code=code, path=best, suffixes=tuple(sorted(rec["suffixes"])), labelled=bool(best[1]),
            in_latest_extract=rec["dollar"] if any_dollar else True, n_label_variants=len(rec["paths"]),
        )
    report = {
        "rows": len(df), "rows_column_shifted_recovered": n_shifted, "rows_unrecoverable": n_unrecoverable, "unique_codes": len(records),
        "labelled_codes": sum(1 for r in records.values() if r.labelled), "coarse_only_codes": sum(1 for r in records.values() if not r.labelled),
        "codes_with_conflicting_labels": n_conflict, "latest_extract_detected": any_dollar,
        "codes_in_latest_extract": sum(1 for r in records.values() if r.in_latest_extract),
    }
    if not records:
        raise ValueError(f"{path.name}: no valid billing codes found.")
    return records, report
