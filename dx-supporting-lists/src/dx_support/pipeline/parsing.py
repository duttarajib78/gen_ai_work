"""Response parsing: every id must be one we sent, and everything is bounded. Model output is untrusted."""
from __future__ import annotations

from typing import Any, Optional

from dx_support.constants import SUPPORT_TIERS, evidence_types
from dx_support.pipeline.constants import MAX_RATIONALE_CHARS
from dx_support.text import sanitize_prompt_text


def _clean_list(value: Any, max_items: int, width: int) -> list[str]:
    out: list[str] = []
    if isinstance(value, list):
        for x in value[:max_items]:
            s = sanitize_prompt_text(x, width)
            if s:
                out.append(s)
    return out


def parse_brief(structured: Any) -> Optional[dict]:
    if not isinstance(structured, dict):
        return None
    brief = {
        "clinical_summary": sanitize_prompt_text(structured.get("clinical_summary", ""), 400),
        "treatment": _clean_list(structured.get("treatment"), 10, 100),
        "investigation": _clean_list(structured.get("investigation"), 10, 100),
        "examination": _clean_list(structured.get("examination"), 10, 100),
        "devices": _clean_list(structured.get("devices"), 10, 100),
        "search_terms": _clean_list(structured.get("search_terms"), 30, 40),
        "typically_no_procedural_evidence": structured.get("typically_no_procedural_evidence") is True,
    }
    if not brief["clinical_summary"] and not any(brief[k] for k in ("treatment", "investigation", "examination", "devices", "search_terms")):
        return None
    return brief


def parse_route(structured: Any, valid_ids: set[str]) -> Optional[list[str]]:
    """None = unusable response (retry later). [] = a valid 'none of these'."""
    if not isinstance(structured, dict) or not isinstance(structured.get("selected"), list):
        return None
    seen: dict[str, None] = {}
    for x in structured["selected"]:
        sid = str(x).strip()
        if sid in valid_ids:
            seen.setdefault(sid, None)
    return list(seen)


def parse_judge(structured: Any, sent_ids: set[str], member_codes: set[str], family: str = "achi") -> dict[str, dict]:
    """{unit_id: verdict} for the ids that were sent and carry a valid tier. Unknown ids and invalid tiers are dropped (those units stay pending).
    The evidence type must be valid FOR THE FAMILY (a diagnosis relation is not a procedure type); anything else is stored as "unspecified"."""
    out: dict[str, dict] = {}
    if not isinstance(structured, dict) or not isinstance(structured.get("verdicts"), list):
        return out
    allowed = evidence_types(family)
    for v in structured["verdicts"]:
        if not isinstance(v, dict):
            continue
        uid = str(v.get("id", "")).strip()
        tier = v.get("support_tier")
        if uid not in sent_ids or tier not in SUPPORT_TIERS:
            continue
        evidence = v.get("evidence_type")
        if evidence not in allowed:
            evidence = "unspecified"
        if tier in ("not_supporting", "unclear"):
            evidence = "none"
        elif evidence == "none":
            evidence = "unspecified"
        scope = v.get("member_scope")
        members = sorted({str(m).strip() for m in scope if str(m).strip() in member_codes}) if isinstance(scope, list) else []
        out[uid] = {"support_tier": tier, "evidence_type": evidence, "rationale": sanitize_prompt_text(v.get("rationale", ""), MAX_RATIONALE_CHARS),
                    "member_scope": members or "all"}
    return out
