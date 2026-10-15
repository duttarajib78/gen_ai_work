"""Text hygiene. Reference-file text and model output are UNTRUSTED: nothing here trusts either."""
from __future__ import annotations

import re
import unicodedata

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_WS_RE = re.compile(r"\s+")
# Strings this project itself uses as prompt delimiters -- stripped from data so a description can never spoof the end of a data block.
_DELIM_RE = re.compile(r"(<<<|>>>|```|</?\s*(?:data|system|instructions?|candidates?)\s*>)", re.I)


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "\u2026"


def display_text(value: object) -> str:
    """Original text with only whitespace normalised -- what is stored in outputs. (Not NFKC-normalised: outputs show the source wording.)"""
    return _WS_RE.sub(" ", _CONTROL_RE.sub(" ", str(value if value is not None else ""))).strip()


def sanitize_prompt_text(value: object, max_chars: int = 300) -> str:
    """Single-line, delimiter-free, length-bounded text for inclusion in a prompt. Removing control characters and our own
    delimiter strings means a description cannot even *look* like structure."""
    s = unicodedata.normalize("NFKC", str(value if value is not None else ""))
    s = _CONTROL_RE.sub(" ", s)
    s = _DELIM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return truncate(s, max_chars)


_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value: object) -> object:
    """Spreadsheet formula-injection guard for free-text cells: a cell starting with = + - @ (or tab/CR) is a FORMULA to Excel and
    LibreOffice. Prefix such strings with an apostrophe so they stay inert. Non-strings pass through."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


_SECRET_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{6,}|sk-[A-Za-z0-9_\-]{20,}|Bearer\s+[A-Za-z0-9._\-]{12,}"
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|(?i:aws_secret_access_key|aws_session_token|x-amz-security-token)\s*[=:]\s*\S+)"
)


def redact(text: str) -> str:
    """Remove anything that looks like a credential from text that is about to be logged or returned."""
    return _SECRET_RE.sub("[REDACTED]", text or "")
