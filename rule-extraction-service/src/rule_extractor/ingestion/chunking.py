"""Chunk planning: sliding page windows for the sweep pass and one chunk per ACS standard."""
from __future__ import annotations

from dataclasses import dataclass


def sliding_windows(total_pages: int, pages_per_chunk: int, overlap: int) -> list[tuple[int, int]]:
    """Inclusive 1-based (start, end) windows covering every page with ``overlap`` pages re-shown at each boundary.

    Reproduces the measured plan sizes for the 13th Edition set: 614pp -> 123, 418pp -> 84 (6/1) and
    598pp -> 67, 286pp -> 32 (10/1).
    """
    if total_pages < 1:
        return []
    if pages_per_chunk < 1 or not 0 <= overlap < pages_per_chunk:
        raise ValueError("need pages_per_chunk >= 1 and 0 <= overlap < pages_per_chunk")
    stride = pages_per_chunk - overlap
    windows: list[tuple[int, int]] = []
    start = 1
    while True:
        end = min(start + pages_per_chunk - 1, total_pages)
        windows.append((start, end))
        if end >= total_pages:
            break
        start += stride
    return windows


@dataclass(frozen=True)
class StandardChunk:
    code: str
    title: str
    start_page: int
    end_page: int


def standard_chunks(anchors: list[dict], total_pages: int, *, max_pages: int = 40) -> list[StandardChunk]:
    """One chunk per ACS standard: from its heading page to the page its successor starts on (shared page included)."""
    ordered = sorted((a for a in anchors if a.get("start_page")), key=lambda a: (a["start_page"], a["code"]))
    chunks: list[StandardChunk] = []
    for i, anchor in enumerate(ordered):
        start = int(anchor["start_page"])
        end = int(ordered[i + 1]["start_page"]) if i + 1 < len(ordered) else total_pages
        if end < start:
            end = start
        if end - start + 1 > max_pages:
            end = start + max_pages - 1
        chunks.append(StandardChunk(code=anchor["code"], title=anchor["title"], start_page=start, end_page=min(end, total_pages)))
    return chunks
