"""The licensed source documents (IHACPA/AIHW 13th Edition set + MBS XML). Filenames are overridable via
``sources:`` in the YAML config when your editions differ."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SourceDocument:
    key: str
    filename: str
    label: str
    edition: str
    kind: str   # tabular | alphabetic | standards | xml

    def path(self, source_dir: Path) -> Path:
        return source_dir / self.filename


DEFAULT_SOURCES: dict[str, SourceDocument] = {
    "icd10am_tabular": SourceDocument("icd10am_tabular", "ICD-10-AM_Tabular_List_Thirteenth_Edition__10_Mar_2025_.pdf",
                                      "ICD-10-AM Tabular List", "13th Edition (1 Jul 2025)", "tabular"),
    "icd10am_alphabetic": SourceDocument("icd10am_alphabetic", "ICD-10-AM_Alphabetic_Index_Thirteenth_Edition__11_Mar_2025_.pdf",
                                         "ICD-10-AM Alphabetic Index", "13th Edition (1 Jul 2025)", "alphabetic"),
    "achi_tabular": SourceDocument("achi_tabular", "ACHI_Tabular_List_Thirteenth_Edition__10_Mar_2025_.pdf",
                                   "ACHI Tabular List", "13th Edition (1 Jul 2025)", "tabular"),
    "achi_alphabetic": SourceDocument("achi_alphabetic", "ACHI_Alphabetic_Index_Thirteenth_Edition__10_Mar_2025_.pdf",
                                      "ACHI Alphabetic Index", "13th Edition (1 Jul 2025)", "alphabetic"),
    "acs": SourceDocument("acs", "Australian_Coding_Standards_Thirteenth_Edition__11_Mar_2025_.pdf",
                          "Australian Coding Standards", "13th Edition (1 Jul 2025)", "standards"),
    "mbs_xml": SourceDocument("mbs_xml", "MBS-XML-20260801.XML", "MBS Schedule", "20260801", "xml"),
}
PDF_DOC_KEYS = ("icd10am_tabular", "achi_tabular", "icd10am_alphabetic", "achi_alphabetic", "acs")


def resolve_sources(overrides: Mapping[str, Any] | None = None) -> dict[str, SourceDocument]:
    """Apply ``sources: {key: {filename|label|edition}}`` overrides from configuration."""
    result = dict(DEFAULT_SOURCES)
    for key, override in (overrides or {}).items():
        if key not in result or not isinstance(override, Mapping):
            raise ValueError(f"unknown source document key in config: {key!r}")
        base = result[key]
        result[key] = SourceDocument(
            key=key,
            filename=str(override.get("filename", base.filename)),
            label=str(override.get("label", base.label)),
            edition=str(override.get("edition", base.edition)),
            kind=base.kind,
        )
    for doc in result.values():
        if "/" in doc.filename or "\\" in doc.filename or doc.filename.startswith("."):
            raise ValueError(f"source filename must be a plain file name: {doc.filename!r}")
    return result
