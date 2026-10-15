"""Small dependency-free helpers shared across the package."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# C0/C1 control characters except TAB / LF / CR, plus bidi override characters
# (used for text-spoofing attacks).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_BIDI_CHARS = re.compile("[\u202a-\u202e\u2066-\u2069\u200e\u200f]")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch_now() -> float:
    return time.time()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def clean_text(value: Any) -> str:
    """Normalise untrusted text: NFC, unified newlines, no control / bidi-override characters."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _BIDI_CHARS.sub("", text)
    return text.strip()


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same directory + rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def validate_object_key(key: str) -> str:
    """Reject keys that could escape a storage root (absolute paths, '..', empty segments, NULs)."""
    if not key or key.startswith("/") or "\\" in key or "\x00" in key:
        raise ValueError(f"unsafe object key: {key!r}")
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError(f"unsafe object key: {key!r}")
    return key


def validate_object_prefix(prefix: str) -> str:
    """Like :func:`validate_object_key` but allows ''/'' and a single trailing slash (listing prefixes)."""
    if prefix == "":
        return prefix
    validate_object_key(prefix[:-1] if prefix.endswith("/") else prefix)
    return prefix


def is_within(root: Path | str, candidate: Path | str) -> bool:
    """True when ``candidate`` (after resolving symlinks) is inside ``root``."""
    root_real = os.path.realpath(root)
    cand_real = os.path.realpath(candidate)
    try:
        return os.path.commonpath([root_real, cand_real]) == root_real
    except ValueError:
        return False
