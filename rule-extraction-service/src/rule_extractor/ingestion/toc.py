"""Parse the ACS table of contents into (standard code, title, printed page) entries and locate each
standard's heading page in the extracted text."""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Sequence

from .pdf import PageText

_TOC_LINE = re.compile(r"^\s*(?:ACS\s+)?(\d{4})\s+(.+?)\s*(?:\.{2,}|\s)\s*(\d{1,4})\s*$")
_TOC_START = re.compile(r"^\s*(?:ACS\s+)?(\d{4})\s+(\S.*)$")
_TRAILING_PAGE = re.compile(r"^(.*?)\s*(?:\.{2,}|\s)\s*(\d{1,4})\s*$")
_WS = re.compile(r"\s+")
MAX_CONTINUATION_LINES = 3


@dataclass(frozen=True)
class TocEntry:
    code: str
    title: str
    printed_page: int


def _norm(text: str) -> str:
    return _WS.sub(" ", text.replace("\u2013", "-").replace("\u2014", "-")).strip().lower()


def parse_toc(text: str) -> list[TocEntry]:
    """Parse TOC text. Handles dot leaders, optional 'ACS' prefixes, and titles wrapped over several lines."""
    entries: list[TocEntry] = []
    pending: tuple[str, str, int] | None = None   # code, title so far, continuation lines used
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _TOC_LINE.match(line)
        if m:
            entries.append(TocEntry(m.group(1), _clean_title(m.group(2)), int(m.group(3))))
            pending = None
            continue
        start = _TOC_START.match(line)
        if start:
            pending = (start.group(1), start.group(2), 0)
            continue
        if pending:
            code, title, used = pending
            joined = f"{title} {line}"
            tail = _TRAILING_PAGE.match(joined)
            if tail and tail.group(1).strip():
                entries.append(TocEntry(code, _clean_title(tail.group(1)), int(tail.group(2))))
                pending = None
            elif used + 1 >= MAX_CONTINUATION_LINES:
                pending = None
            else:
                pending = (code, joined, used + 1)
    seen: set[str] = set()
    unique = []
    for entry in entries:
        if entry.code not in seen:
            seen.add(entry.code)
            unique.append(entry)
    return unique


def _clean_title(title: str) -> str:
    return _WS.sub(" ", title.strip(" .\t"))


def find_toc_pages(pages: Sequence[PageText], *, search_first: int = 40, min_lines: int = 5) -> list[PageText]:
    out = []
    for page in pages[:search_first]:
        hits = sum(1 for ln in page.text.splitlines() if _TOC_LINE.match(ln.strip()))
        if hits >= min_lines:
            out.append(page)
    return out


def locate_anchors(entries: Sequence[TocEntry], pages: Sequence[PageText], toc_end_page: int) -> list[dict]:
    """Return ``[{code, title, start_page, method}]`` where ``start_page`` is the physical page of the heading.

    Strategy: (1) scan forward for a line that starts with the standard's code and contains the start of its
    title; (2) fall back to printed page + the median offset measured from the standards found by (1).
    """
    located: dict[str, int] = {}
    cursor = toc_end_page + 1
    by_index = {p.index: p for p in pages}
    for entry in entries:
        needle = _norm(entry.title)[:14]
        code_re = re.compile(rf"^\s*(?:acs\s+)?{entry.code}\b", re.I)
        for idx in range(cursor, len(pages) + 1):
            page = by_index.get(idx)
            if page is None:
                continue
            if any(code_re.match(ln) and needle in _norm(ln) for ln in page.text.splitlines()):
                located[entry.code] = idx
                cursor = idx
                break
    offsets = [located[e.code] - e.printed_page for e in entries if e.code in located]
    offset = int(statistics.median(offsets)) if offsets else None
    anchors = []
    for entry in entries:
        if entry.code in located:
            anchors.append({"code": entry.code, "title": entry.title, "start_page": located[entry.code], "method": "heading"})
        elif offset is not None and 1 <= entry.printed_page + offset <= len(pages):
            anchors.append({"code": entry.code, "title": entry.title, "start_page": entry.printed_page + offset, "method": "offset"})
        else:
            anchors.append({"code": entry.code, "title": entry.title, "start_page": None, "method": "unresolved"})
    return anchors
