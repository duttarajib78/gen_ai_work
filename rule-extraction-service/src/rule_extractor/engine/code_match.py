"""Exact / prefix / wildcard code matching shared by the condition and action interpreters.

Codes are canonicalised (upper-case, dots/hyphens/spaces removed) so ``E11.9``, ``E119`` and ``e11.9``
are equal, and ACHI ``30571-00`` equals ``3057100``. Patterns may end in ``*`` (prefix) or contain
``*``/``?`` anywhere (wildcard).
"""
from __future__ import annotations

import fnmatch
import re
from typing import Iterable, Sequence

_STRIP = re.compile(r"[.\-\s]")
_PATTERN_CHARS = re.compile(r"^[A-Za-z0-9.\-*?]{1,32}$")
MAX_PATTERNS = 20


def canonical(code: object) -> str:
    return _STRIP.sub("", str(code)).upper()


def validate_pattern(pattern: object) -> list[str]:
    if not isinstance(pattern, str):
        return ["code pattern must be a string"]
    if not _PATTERN_CHARS.match(pattern):
        return [f"invalid code pattern {pattern[:40]!r}: use letters, digits, '.', '-', '*' and '?' (max 32 chars)"]
    literal = re.sub(r"[.\-*?]", "", pattern)
    if not literal:
        return [f"code pattern {pattern!r} has no literal characters and would match everything"]
    return []


def normalize_patterns(value: object) -> list[str]:
    """Accept one pattern or a list of patterns and return a list (raises ValueError when invalid)."""
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)) or not items or len(items) > MAX_PATTERNS:
        raise ValueError(f"expected one pattern or a list of 1-{MAX_PATTERNS} patterns")
    problems: list[str] = []
    for item in items:
        problems.extend(validate_pattern(item))
    if problems:
        raise ValueError("; ".join(problems))
    return [str(i) for i in items]


def matches(code: object, pattern: str) -> bool:
    c = canonical(code)
    p = canonical(pattern)
    if "*" in p or "?" in p:
        return fnmatch.fnmatchcase(c, p)
    return c == p


def any_match(code: object, patterns: Iterable[str]) -> bool:
    return any(matches(code, p) for p in patterns)


def matching_indices(codes: Sequence[object], patterns: Sequence[str]) -> list[int]:
    return [i for i, code in enumerate(codes) if any_match(code, patterns)]
