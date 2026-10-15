"""Corpus builder and reader.

Layout (this is what the agent's Read/Grep/Glob tools see; it is read-only at serving time):

  corpus/
    manifest.json                 documents, sha256 of sources, page counts, builder version
    acs_toc.json                  ACS standard anchors + chunk page ranges
    known_codes.json              canonical ICD-10-AM / ACHI / MBS codes found in the sources
    <doc_key>/pNNNN.txt           one file per page; first line is a header with pdf and printed page numbers
    mbs_items.txt                 one line per MBS item
    reference/mbs_billing_facts_reference.jsonl
"""
from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import __version__
from ..errors import IngestionError
from ..settings import Settings
from ..util import atomic_write_text, canonical_json, sha256_bytes, sha256_file, utc_now_iso
from .chunking import StandardChunk, standard_chunks
from .mbs import billing_facts, lookup_lines, parse_mbs_xml
from .pdf import PageText, extract_pages
from .sources import PDF_DOC_KEYS, SourceDocument, resolve_sources
from .toc import find_toc_pages, locate_anchors, parse_toc

FORMAT_VERSION = 1
HEADER_RE = re.compile(r"^=== (\S+) \| pdf_page (\d+) \| printed_page (.*) ===$")
_ICD_LINE = re.compile(r"^\s*([A-Z]\d{2}(?:\.\d{1,2})?)\s+\S")
_ACHI_LINE = re.compile(r"^\s*(\d{5}-\d{2})\s+\S")
_CANON = re.compile(r"[.\-\s]")


def canon_code(code: str) -> str:
    return _CANON.sub("", code).upper()


@dataclass(frozen=True)
class CodeRegistry:
    icd10am: frozenset[str]
    achi: frozenset[str]
    mbs: frozenset[str]

    @classmethod
    def empty(cls) -> "CodeRegistry":
        return cls(frozenset(), frozenset(), frozenset())

    @property
    def available(self) -> bool:
        return bool(self.icd10am or self.achi or self.mbs)


def page_header(doc_key: str, page: PageText) -> str:
    return f"=== {doc_key} | pdf_page {page.index} | printed_page {page.printed_label or '?'} ==="


# ------------------------------------------------------------------------------------------ build
def _extract_doc(args: tuple[str, str, str]) -> tuple[str, int, str]:
    """Worker (process-safe): PDF -> corpus/<key>/pNNNN.txt via a temp dir swap. Returns (key, pages, sha256)."""
    key, pdf_path, corpus_dir = args
    root = Path(corpus_dir)
    tmp = root / f".tmp-{key}"
    final = root / key
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    count = 0
    for page in extract_pages(pdf_path):
        (tmp / f"p{page.index:04d}.txt").write_text(page_header(key, page) + "\n" + page.text + "\n", encoding="utf-8")
        count += 1
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return key, count, sha256_file(pdf_path)


def build_corpus(settings: Settings, *, force: bool = False, workers: int = 1, only: Sequence[str] | None = None,
                 sources: Mapping[str, SourceDocument] | None = None) -> dict[str, Any]:
    sources = dict(sources or resolve_sources(settings.extras.get("sources")))
    source_dir, corpus_dir = settings.paths.source_docs_dir, settings.paths.corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    previous = _read_json(corpus_dir / "manifest.json") or {}
    prev_docs = previous.get("documents", {})
    wanted = [k for k in PDF_DOC_KEYS if only is None or k in only]

    todo: list[tuple[str, str, str]] = []
    docs: dict[str, dict[str, Any]] = dict(prev_docs)
    for key in wanted:
        pdf = sources[key].path(source_dir)
        if not pdf.is_file():
            raise IngestionError(f"missing source document {sources[key].filename} in {source_dir}")
        digest = sha256_file(pdf)
        old = prev_docs.get(key, {})
        intact = (corpus_dir / key).is_dir() and old.get("sha256") == digest and previous.get("format_version") == FORMAT_VERSION
        if intact and not force:
            continue
        todo.append((key, str(pdf), str(corpus_dir)))
    if todo:
        if workers > 1 and len(todo) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as pool:
                results = list(pool.map(_extract_doc, todo))
        else:
            results = [_extract_doc(t) for t in todo]
        for key, pages, digest in results:
            src = sources[key]
            docs[key] = {"label": src.label, "edition": src.edition, "kind": src.kind, "filename": src.filename,
                         "sha256": digest, "pages": pages}

    if "acs" in wanted or "acs" in docs:
        toc = _build_acs_toc(corpus_dir, docs["acs"]["pages"], settings.chunking.min_acs_standards)
        atomic_write_text(corpus_dir / "acs_toc.json", json.dumps(toc, indent=2))
    registry = _scan_known_codes(corpus_dir, docs)

    mbs_stats: dict[str, int] = {}
    mbs_src = sources["mbs_xml"]
    mbs_path = mbs_src.path(source_dir)
    if mbs_path.is_file() and (only is None or "mbs_xml" in only):
        items = parse_mbs_xml(mbs_path)
        atomic_write_text(corpus_dir / "mbs_items.txt", "\n".join(lookup_lines(items)) + "\n")
        facts = list(billing_facts(items))
        (corpus_dir / "reference").mkdir(exist_ok=True)
        atomic_write_text(corpus_dir / "reference" / "mbs_billing_facts_reference.jsonl",
                          "".join(json.dumps(f, ensure_ascii=False, sort_keys=True) + "\n" for f in facts))
        registry = CodeRegistry(registry.icd10am, registry.achi, frozenset(canon_code(i.item_num) for i in items))
        docs["mbs_xml"] = {"label": mbs_src.label, "edition": mbs_src.edition, "kind": "xml", "filename": mbs_src.filename,
                           "sha256": sha256_file(mbs_path), "pages": 0}
        mbs_stats = {"items": len(items), "facts": len(facts)}
    elif "mbs_xml" in prev_docs:
        mbs_stats = previous.get("mbs", {})
        prior = _read_json(corpus_dir / "known_codes.json") or {}
        registry = CodeRegistry(registry.icd10am, registry.achi, frozenset(prior.get("mbs", [])))

    atomic_write_text(corpus_dir / "known_codes.json", json.dumps({
        "icd10am": sorted(registry.icd10am), "achi": sorted(registry.achi), "mbs": sorted(registry.mbs)}))
    manifest = {
        "format_version": FORMAT_VERSION, "builder_version": __version__, "built_at": utc_now_iso(),
        "documents": docs, "mbs": mbs_stats,
        "known_codes": {"icd10am": len(registry.icd10am), "achi": len(registry.achi), "mbs": len(registry.mbs)},
    }
    atomic_write_text(corpus_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _read_page_texts(corpus_dir: Path, key: str, pages: int) -> list[PageText]:
    out = []
    for i in range(1, pages + 1):
        raw = (corpus_dir / key / f"p{i:04d}.txt").read_text(encoding="utf-8")
        head, _, body = raw.partition("\n")
        m = HEADER_RE.match(head)
        out.append(PageText(index=i, text=body.rstrip("\n"), printed_label="" if not m or m.group(3) == "?" else m.group(3)))
    return out


def _build_acs_toc(corpus_dir: Path, pages: int, min_standards: int) -> dict[str, Any]:
    texts = _read_page_texts(corpus_dir, "acs", pages)
    toc_pages = find_toc_pages(texts)
    if not toc_pages:
        raise IngestionError("could not find the ACS table of contents (no page with >=5 'NNNN Title ... page' lines)")
    entries = []
    for page in toc_pages:
        entries.extend(parse_toc(page.text))
    seen: set[str] = set()
    entries = [e for e in entries if not (e.code in seen or seen.add(e.code))]
    anchors = locate_anchors(entries, texts, toc_end_page=max(p.index for p in toc_pages))
    resolved = [a for a in anchors if a["start_page"]]
    if len(resolved) < min_standards:
        raise IngestionError(
            f"parsed {len(entries)} ACS standards and located {len(resolved)}, below the required minimum of {min_standards}. "
            "Inspect corpus/acs/p*.txt and adjust ingestion/toc.py or RULE_EXTRACTOR_MIN_ACS_STANDARDS.")
    chunks = standard_chunks(anchors, pages)
    return {"toc_pages": [p.index for p in toc_pages], "anchors": anchors,
            "chunks": [c.__dict__ for c in chunks]}


def _scan_known_codes(corpus_dir: Path, docs: Mapping[str, Any]) -> CodeRegistry:
    icd: set[str] = set()
    achi: set[str] = set()
    for key, pattern, bucket in (("icd10am_tabular", _ICD_LINE, icd), ("achi_tabular", _ACHI_LINE, achi)):
        if key not in docs:
            continue
        for path in (corpus_dir / key).glob("p*.txt"):
            for line in path.read_text(encoding="utf-8").splitlines():
                m = pattern.match(line)
                if m:
                    bucket.add(canon_code(m.group(1)))
    return CodeRegistry(frozenset(icd), frozenset(achi), frozenset())


# ----------------------------------------------------------------------------------------- reader
def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class CorpusIndex:
    """Read-only access to a built corpus (used by the planner, prompt builder and finalizer)."""

    def __init__(self, corpus_dir: Path | str) -> None:
        self.root = Path(corpus_dir)
        self._manifest: dict[str, Any] | None = None
        self._toc: dict[str, Any] | None = None
        self._codes: CodeRegistry | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            data = _read_json(self.root / "manifest.json")
            if data is None:
                raise IngestionError(f"no corpus manifest at {self.root}; run `rule-extractor corpus build` first")
            self._manifest = data
        return self._manifest

    def digest(self) -> str:
        return sha256_bytes(canonical_json(self.manifest).encode("utf-8"))

    def page_count(self, doc_key: str) -> int:
        try:
            return int(self.manifest["documents"][doc_key]["pages"])
        except KeyError as exc:
            raise IngestionError(f"document {doc_key!r} is not in the corpus") from exc

    def doc_label(self, doc_key: str) -> str:
        return str(self.manifest["documents"][doc_key].get("label", doc_key))

    def page_text(self, doc_key: str, page: int) -> str:
        if not isinstance(page, int) or not 1 <= page <= self.page_count(doc_key):
            raise IngestionError(f"page {page!r} out of range for {doc_key}")
        return (self.root / doc_key / f"p{page:04d}.txt").read_text(encoding="utf-8")

    def window_text(self, doc_key: str, start: int, end: int) -> str:
        return "\n\n".join(self.page_text(doc_key, p) for p in range(start, end + 1))

    def standards(self) -> list[StandardChunk]:
        if self._toc is None:
            self._toc = _read_json(self.root / "acs_toc.json") or {}
        return [StandardChunk(**c) for c in self._toc.get("chunks", [])]

    def known_codes(self) -> CodeRegistry:
        if self._codes is None:
            data = _read_json(self.root / "known_codes.json")
            self._codes = CodeRegistry.empty() if not data else CodeRegistry(
                frozenset(data.get("icd10am", [])), frozenset(data.get("achi", [])), frozenset(data.get("mbs", [])))
        return self._codes

    def mbs_facts_path(self) -> Path:
        return self.root / "reference" / "mbs_billing_facts_reference.jsonl"


def verify_corpus(corpus_dir: Path | str, *, min_standards: int = 100) -> list[str]:
    """Return a list of problems (empty = healthy)."""
    root = Path(corpus_dir)
    problems: list[str] = []
    manifest = _read_json(root / "manifest.json")
    if not manifest:
        return [f"manifest.json missing in {root}"]
    if manifest.get("format_version") != FORMAT_VERSION:
        problems.append(f"unsupported corpus format {manifest.get('format_version')!r}")
    for key in PDF_DOC_KEYS:
        info = manifest.get("documents", {}).get(key)
        if not info:
            problems.append(f"document {key} missing from manifest")
            continue
        missing = [i for i in range(1, int(info["pages"]) + 1) if not (root / key / f"p{i:04d}.txt").is_file()]
        if missing:
            problems.append(f"{key}: {len(missing)} page file(s) missing (first: p{missing[0]:04d})")
    toc = _read_json(root / "acs_toc.json") or {}
    resolved = [a for a in toc.get("anchors", []) if a.get("start_page")]
    if len(resolved) < min_standards:
        problems.append(f"only {len(resolved)} ACS standards resolved (minimum {min_standards})")
    codes = manifest.get("known_codes", {})
    if not codes.get("icd10am") or not codes.get("achi"):
        problems.append("code registry is empty for ICD-10-AM or ACHI (check the Tabular List text extraction)")
    return problems
