"""Request validation. Everything a caller sends is untrusted.

The API never puts caller text into a prompt: a request names categories (validated against a strict pattern AND against the loaded universe)
and families (an allowlist); everything the model sees comes from the reference files. Unknown fields are rejected rather than ignored, so a
typo cannot silently change what a request means.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from dx_support.constants import DEFAULT_FAMILIES, DEFAULT_MERGE_TIERS, FAMILIES, SUPPORT_TIERS
from dx_support.indexing.universe import Universe
from dx_support.reference.chapters import CONTEXT_ONLY_CHAPTERS
from dx_support.settings import Settings

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")
CATEGORY_RE = re.compile(r"^[A-Z][0-9]{2}$")
LEAF_RE = re.compile(r"^[A-Z][0-9]{2}[A-Z0-9]{0,2}$")
ACTIONS = ("supporting_lists", "describe")
SUPPORT_KEYS = {"action", "request_id", "categories", "codes", "families", "merge_tiers", "include_audit", "prosthesis_extract", "cache_only", "stream"}
DESCRIBE_KEYS = {"action", "request_id"}


class RequestError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass(frozen=True)
class SupportRequest:
    request_id: str
    categories: tuple[str, ...]
    families: tuple[str, ...]
    merge_tiers: tuple[str, ...]
    include_audit: bool
    prosthesis_extract: str
    cache_only: bool
    stream: bool


def parse_body(raw: bytes) -> dict:
    try:
        body: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RequestError(400, "invalid_json", "The request body must be a UTF-8 JSON object.") from None
    if not isinstance(body, dict):
        raise RequestError(400, "invalid_json", "The request body must be a JSON object.")
    return body


def _str_list(body: Mapping, key: str, max_items: int, max_len: int = 16) -> list[str]:
    v = body.get(key)
    if v is None:
        return []
    if not isinstance(v, list) or len(v) > max_items or not all(isinstance(x, str) and 0 < len(x.strip()) <= max_len for x in v):
        raise RequestError(422, f"invalid_{key}", f"`{key}` must be a list of at most {max_items} short strings.")
    return [x.strip().upper() if key in ("categories", "codes") else x.strip().lower() for x in v]


def _bool(body: Mapping, key: str, default: bool) -> bool:
    v = body.get(key, default)
    if not isinstance(v, bool):
        raise RequestError(422, f"invalid_{key}", f"`{key}` must be true or false.")
    return v


def request_id_of(body: Mapping) -> str:
    rid = body.get("request_id")
    if rid is None:
        return uuid.uuid4().hex
    if not isinstance(rid, str) or not REQUEST_ID_RE.match(rid):
        raise RequestError(422, "invalid_request_id", "`request_id` must be 1-64 characters: letters, digits, '.', '_' or '-'.")
    return rid


def check_keys(body: Mapping, allowed: set[str]) -> None:
    unknown = sorted(str(k)[:40] for k in body if k not in allowed)
    if unknown:
        raise RequestError(400, "unknown_field", f"Unknown field(s): {unknown}. Allowed: {sorted(allowed)}.")


def parse_support_request(body: Mapping, settings: Settings, universe: Universe, leaf_map: Mapping[str, str]) -> SupportRequest:
    check_keys(body, SUPPORT_KEYS)
    rid = request_id_of(body)
    cats = _str_list(body, "categories", 50)
    codes = _str_list(body, "codes", 50)
    cat_set: dict[str, None] = {}
    for c in cats:
        if not CATEGORY_RE.match(c):
            raise RequestError(422, "invalid_category", "A category is a 3-character ICD-10-AM category such as H25.")
        cat_set.setdefault(c, None)
    for code in codes:
        if not LEAF_RE.match(code) or code not in leaf_map:
            raise RequestError(422, "unknown_code", "`codes` must be ICD-10-AM codes present in the loaded diagnosis file (they are mapped to their category).")
        cat_set.setdefault(leaf_map[code], None)
    if not cat_set:
        raise RequestError(422, "missing_categories", "Provide `categories` (for example [\"H25\"]) or `codes`.")
    if len(cat_set) > settings.max_categories_per_request:
        raise RequestError(422, "too_many_categories", f"At most {settings.max_categories_per_request} categories per request.")
    for c in cat_set:
        info = universe.categories.get(c)
        if info is None:
            raise RequestError(422, "unknown_category", f"Category {c} is not in the loaded reference data.")
        if info.chapter in CONTEXT_ONLY_CHAPTERS and not settings.include_context_chapters:
            raise RequestError(422, "context_only_category", f"Category {c} is in a context-only chapter ({info.chapter}) and is not built.")
    fams = tuple(dict.fromkeys(_str_list(body, "families", len(FAMILIES)))) or DEFAULT_FAMILIES
    bad = [f for f in fams if f not in FAMILIES]
    if bad:
        raise RequestError(422, "invalid_families", f"`families` must be a subset of {list(FAMILIES)}.")
    tiers = tuple(dict.fromkeys(_str_list(body, "merge_tiers", len(SUPPORT_TIERS)))) or DEFAULT_MERGE_TIERS
    if any(t not in SUPPORT_TIERS for t in tiers):
        raise RequestError(422, "invalid_merge_tiers", f"`merge_tiers` must be a subset of {list(SUPPORT_TIERS)}.")
    extract = body.get("prosthesis_extract", "all")
    if extract not in ("all", "latest"):
        raise RequestError(422, "invalid_prosthesis_extract", "`prosthesis_extract` must be \"all\" or \"latest\".")
    return SupportRequest(request_id=rid, categories=tuple(cat_set), families=fams, merge_tiers=tiers, include_audit=_bool(body, "include_audit", False),
                          prosthesis_extract=extract, cache_only=_bool(body, "cache_only", False), stream=_bool(body, "stream", False))
