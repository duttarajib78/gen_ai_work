"""Code -> description lookups for ICD-10-AM diagnoses, ACHI procedures, and MBS items.

Layout inside the SageMaker model artifact (``/opt/ml/model``)::

    manifest.json                       sha256 + size of every file below
    reference/diagnosis_codes.tsv       8 columns, no header, ICD10_DIAGNOSIS
    reference/achi_codes.tsv            8 columns, no header, ICD10_PROCEDURE
    reference/MBS-XML-YYYYMMDD.XML      latest file by name wins
    drg/...                             DRG cost-weight parquet (used by the grouper)

Hardening applied here because descriptions are placed into an LLM prompt:

* the artifact is verified against ``manifest.json`` at startup (tamper/corruption
  detection) and the service refuses to start if it does not match;
* the MBS XML is parsed with ``defusedxml`` (no entity expansion, no external
  entities) and streamed to bound memory;
* every description is stripped of control and bidi characters, collapsed to one
  line, and length-capped before it can reach a prompt.

Lookups never raise for an unknown code: they return a visible placeholder so a
missing description is never mistaken for "the code does not exist".
"""
from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Optional

import pandas as pd
from defusedxml import ElementTree as SafeET

from .errors import ReferenceDataError
from .textutil import clean_line

NOT_FOUND = "(description not found)"
DESCRIPTION_MAX_LEN = 300
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1

DIAGNOSIS_FILE = "diagnosis_codes.tsv"
ACHI_FILE = "achi_codes.tsv"
MBS_GLOB = "MBS-XML-*.XML"

_CODE_TYPE_COL, _CODE_COL, _DESC_COL = 1, 2, 6
_EXPECTED_TSV_COLUMNS = 8


# ---------------------------------------------------------------------------
# Integrity manifest
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    files: Mapping[str, Mapping]
    sha256: str  # hash of the manifest file itself; recorded in every response's provenance


def load_manifest(model_dir: Path) -> Optional[Manifest]:
    path = Path(model_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
        files = doc["files"]
        if doc.get("schema_version") != MANIFEST_SCHEMA_VERSION or not isinstance(files, dict) or not files:
            raise ValueError("unexpected manifest structure")
        for rel, meta in files.items():
            if not isinstance(meta, dict) or not isinstance(meta.get("sha256"), str):
                raise ValueError("bad file entry")
    except (ValueError, KeyError, TypeError) as exc:
        raise ReferenceDataError("manifest.json is not a valid manifest") from exc
    return Manifest(files=MappingProxyType(files), sha256=hashlib.sha256(raw).hexdigest())


def _safe_relative(rel: str) -> PurePosixPath:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ReferenceDataError("manifest contains an unsafe path")
    return p


def verify_manifest(model_dir: Path, manifest: Manifest) -> None:
    model_dir = Path(model_dir)
    for rel, meta in manifest.files.items():
        target = model_dir.joinpath(*_safe_relative(rel).parts)
        if not target.is_file():
            raise ReferenceDataError(f"artifact file listed in manifest is missing: {rel}")
        if sha256_file(target) != meta["sha256"]:
            raise ReferenceDataError(f"artifact file failed integrity check: {rel}")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_positional_tsv(path: Path, expected_code_type: str) -> dict[str, str]:
    """Shared loader for diagnosis_codes.tsv / achi_codes.tsv. Fails loudly on a
    shape or code_type mismatch rather than silently mis-reading the wrong file."""
    if not path.is_file():
        raise ReferenceDataError(f"reference file not found: {path.name}")
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] != _EXPECTED_TSV_COLUMNS:
        raise ReferenceDataError(
            f"{path.name} has {df.shape[1]} columns, expected {_EXPECTED_TSV_COLUMNS}; "
            "the file structure may have changed"
        )
    seen = set(df[_CODE_TYPE_COL].dropna().unique())
    if seen != {expected_code_type}:
        raise ReferenceDataError(f"{path.name} does not contain only {expected_code_type!r} rows; wrong file?")
    df = df.dropna(subset=[_CODE_COL])
    codes = df[_CODE_COL].astype(str).str.strip()
    descriptions = df[_DESC_COL].fillna("").astype(str)
    return {code: clean_line(desc, DESCRIPTION_MAX_LEN) for code, desc in zip(codes, descriptions)}


@dataclass(frozen=True)
class MbsItem:
    item_num: str
    description: str
    category: str
    group: str
    sub_group: str
    sub_heading: str
    item_end_date: str  # non-empty: superseded as of this extract
    derived_fee: str    # non-empty: fee depends on another item


def find_latest_mbs_xml(directory: Path) -> Path:
    matches = sorted(glob.glob(str(Path(directory) / MBS_GLOB)))
    if not matches:
        raise ReferenceDataError(f"no file matching {MBS_GLOB} in the reference directory")
    return Path(matches[-1])  # YYYYMMDD in the name sorts correctly as text


def _mbs_key(code: object) -> str:
    return str(code).strip().lstrip("0") or "0"


def load_mbs_items(path: Path) -> dict[str, MbsItem]:
    """item number (leading zeros stripped) -> MbsItem."""
    if not Path(path).is_file():
        raise ReferenceDataError("MBS XML file not found")
    items: dict[str, MbsItem] = {}
    try:
        for _event, elem in SafeET.iterparse(str(path), events=("end",)):
            if elem.tag != "Data":
                continue

            def text(tag: str, _elem=elem) -> str:
                child = _elem.find(tag)
                return (child.text or "").strip() if child is not None and child.text else ""

            raw_num = text("ItemNum")
            if raw_num:
                items[_mbs_key(raw_num)] = MbsItem(
                    item_num=raw_num,
                    description=clean_line(text("Description"), DESCRIPTION_MAX_LEN),
                    category=text("Category"), group=text("Group"), sub_group=text("SubGroup"),
                    sub_heading=text("SubHeading"), item_end_date=text("ItemEndDate"),
                    derived_fee=text("DerivedFee"),
                )
            elem.clear()
    except SafeET.ParseError as exc:
        raise ReferenceDataError("MBS XML could not be parsed") from exc
    except Exception as exc:  # defusedxml raises its own exception types for forbidden constructs
        if type(exc).__module__.startswith("defusedxml"):
            raise ReferenceDataError("MBS XML contains forbidden constructs (DTD/entities)") from exc
        raise
    return items


class ReferenceData:
    def __init__(
        self,
        diagnosis: Mapping[str, str],
        achi: Mapping[str, str],
        mbs: Mapping[str, MbsItem],
        *,
        mbs_source_file: str = "",
        manifest_sha256: Optional[str] = None,
    ) -> None:
        self.diagnosis = MappingProxyType(dict(diagnosis))
        self.achi = MappingProxyType(dict(achi))
        self.mbs = MappingProxyType(dict(mbs))
        self.mbs_source_file = mbs_source_file
        self.manifest_sha256 = manifest_sha256

    @classmethod
    def from_directory(cls, directory: Path, *, manifest_sha256: Optional[str] = None) -> "ReferenceData":
        """Load from a directory that directly contains the three reference files."""
        directory = Path(directory)
        mbs_path = find_latest_mbs_xml(directory)
        return cls(
            _load_positional_tsv(directory / DIAGNOSIS_FILE, "ICD10_DIAGNOSIS"),
            _load_positional_tsv(directory / ACHI_FILE, "ICD10_PROCEDURE"),
            load_mbs_items(mbs_path),
            mbs_source_file=mbs_path.name,
            manifest_sha256=manifest_sha256,
        )

    @classmethod
    def from_model_dir(cls, model_dir: Path, *, require_manifest: bool = True) -> "ReferenceData":
        """Production path: verify the artifact, then load ``reference/`` from it."""
        model_dir = Path(model_dir)
        manifest = load_manifest(model_dir)
        if manifest is None:
            if require_manifest:
                raise ReferenceDataError(
                    "manifest.json not found in the model artifact; build it with "
                    "tools/deploy/package_model_artifact.py (or set AUDIT_REQUIRE_MANIFEST=false for local use only)"
                )
        else:
            verify_manifest(model_dir, manifest)
            required = {f"reference/{DIAGNOSIS_FILE}", f"reference/{ACHI_FILE}"}
            if not required.issubset(manifest.files) or not any(
                PurePosixPath(rel).match(f"reference/{MBS_GLOB}") for rel in manifest.files
            ):
                raise ReferenceDataError("manifest does not list all required reference files")
        reference_dir = model_dir / "reference"
        data = cls.from_directory(reference_dir, manifest_sha256=manifest.sha256 if manifest else None)
        if manifest is not None and f"reference/{data.mbs_source_file}" not in manifest.files:
            raise ReferenceDataError("the MBS file selected for use is not covered by the manifest")
        return data

    # -- lookups ---------------------------------------------------------------
    def resolve_diagnosis(self, code: str) -> str:
        return self.diagnosis.get(str(code).strip(), NOT_FOUND)

    def resolve_achi(self, code: str) -> str:
        return self.achi.get(str(code).strip(), NOT_FOUND)

    def resolve_mbs(self, code: str) -> str:
        item = self.mbs.get(_mbs_key(code))
        if item is None:
            return NOT_FOUND
        suffix = " [SUPERSEDED per current MBS extract]" if item.item_end_date else ""
        return item.description + suffix

    def counts(self) -> dict:
        return {"diagnosis": len(self.diagnosis), "achi": len(self.achi), "mbs": len(self.mbs)}
