"""PDF -> per-page text. PDFs are treated as untrusted input: bounded size/pages, text sanitised."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..errors import IngestionError
from ..util import clean_text

MAX_PDF_BYTES = 300 * 1024 * 1024
MAX_PAGES = 5000
MAX_PAGE_CHARS = 200_000

_NUMBER_LINE = re.compile(r"^(?:page\s+)?(\d{1,4}|[ivxlc]{1,7})$", re.I)
_PIPE_LINE = re.compile(r"^(?:.{0,60}\|\s*(\d{1,4}|[ivxlc]{1,7})|(\d{1,4}|[ivxlc]{1,7})\s*\|.{0,60})$", re.I)


@dataclass(frozen=True)
class PageText:
    index: int          # 1-based physical page number
    text: str
    printed_label: str  # printed page number when detected, else ""


def detect_printed_page(text: str) -> str:
    """Heuristic: a stand-alone number / roman numeral in the first or last 3 lines is the printed page number."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[-3:][::-1] + lines[:3]:
        if len(line) > 80:
            continue
        m = _NUMBER_LINE.match(line) or _PIPE_LINE.match(line)
        if m:
            return next(g for g in m.groups() if g) if m.groups() else m.group(0)
    return ""


def _explicit_labels(reader) -> list[str]:
    try:
        labels = [str(label) for label in reader.page_labels]
    except Exception:  # malformed /PageLabels - fall back to the heuristic
        return []
    if labels == [str(i) for i in range(1, len(labels) + 1)]:
        return []  # the PDF defines no real labels (reader synthesises 1..N)
    return labels


def extract_pages(pdf_path: Path | str) -> Iterator[PageText]:
    path = Path(pdf_path)
    if not path.is_file():
        raise IngestionError(f"source PDF not found: {path.name}")
    if path.stat().st_size > MAX_PDF_BYTES:
        raise IngestionError(f"source PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB: {path.name}")
    try:
        import pdfplumber
        import pypdf
    except ImportError as exc:  # pragma: no cover - declared dependencies
        raise IngestionError("pdfplumber and pypdf are required for corpus building") from exc

    try:
        labels = _explicit_labels(pypdf.PdfReader(str(path)))
    except Exception as exc:
        raise IngestionError(f"cannot read PDF structure of {path.name}: {type(exc).__name__}") from exc
    with pdfplumber.open(str(path)) as pdf:
        total = len(pdf.pages)
        if total == 0 or total > MAX_PAGES:
            raise IngestionError(f"{path.name} has {total} pages (allowed 1..{MAX_PAGES})")
        for i, page in enumerate(pdf.pages, 1):
            raw = page.extract_text() or ""
            text = clean_text(raw[:MAX_PAGE_CHARS])
            label = labels[i - 1] if i - 1 < len(labels) else detect_printed_page(text)
            yield PageText(index=i, text=text, printed_label=label)
            page.flush_cache()
