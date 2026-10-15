"""Sanitisation helpers for untrusted or externally sourced text.

Used for (a) reference-data descriptions before they enter a prompt,
(b) LLM-produced reasoning before it is returned or embedded in an alert, and
(c) error text before it is logged.
"""
from __future__ import annotations

import html
import re

# C0/C1 controls (except tab/newline), DEL, bidi overrides/isolates (the
# "Trojan Source" family), zero-width characters, BOM, line/paragraph separators.
_UNSAFE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    "\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]"
)
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_WHITESPACE = re.compile(r"\s+")

TRUNCATION_MARK = " [truncated]"


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    keep = max(0, max_len - len(TRUNCATION_MARK))
    return text[:keep].rstrip() + TRUNCATION_MARK


def clean_text(value: object, max_len: int = 2000) -> str:
    """Multi-line text: strip unsafe characters, normalise newlines, cap length."""
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _UNSAFE.sub("", text)
    text = _MULTI_NEWLINE.sub("\n\n", text).strip()
    return _truncate(text, max_len)


def clean_line(value: object, max_len: int = 300) -> str:
    """Single-line text: also collapse every run of whitespace to one space."""
    text = "" if value is None else str(value)
    text = _UNSAFE.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return _truncate(text, max_len)


def escape_html(value: object) -> str:
    """Output-encode text that will be rendered as HTML by a downstream viewer."""
    return html.escape("" if value is None else str(value), quote=True)
