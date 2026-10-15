"""Builds the LLM-facing ``EPISODE`` text block from a validated episode.

Format (unchanged from the validated prototype)::

    EPISODE

    PDx:
      J189 — Pneumonia, unspecified

    ADx:
      I509 — Heart failure

    ACHI:
      3237500 — <procedure description> (day 1)

    MBS / hospital services:
      23 — <item description> (day unknown)

Data minimisation is part of the design: the model sees diagnosis, procedure,
and service *descriptions* and day offsets only. It never sees identifiers,
dates, age, sex, length of stay, DRG, or any financial field, so the prompt is
de-identified by construction and the judgement is independent of the billed
outcome.

Duplicate diagnosis codes are collapsed (earliest occurrence wins, so a repeat of
the PDx code stays PDx and is never also an ADx). ``diagnosis_order`` keeps the
full raw array because paired-array corrections must stay positionally aligned.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .reference_data import ReferenceData
from .schemas import Episode


@dataclass(frozen=True)
class FormattedEpisode:
    text: str
    pdx_code: str
    adx_codes: tuple[str, ...]
    achi_codes: tuple[str, ...]
    mbs_codes: tuple[str, ...]
    diagnosis_order: tuple[str, ...]


def _day_offset(value: Optional[datetime], admitted: Optional[datetime]) -> str:
    """Calendar-day offset from admission (day 0 = admission day), never a calendar date."""
    if value is None or admitted is None:
        return "day unknown"
    return f"day {(value.date() - admitted.date()).days}"


def build_episode_text(
    episode: Episode, ref: ReferenceData, mbs_array_field: str = "medical_claims_array"
) -> FormattedEpisode:
    seen: set[str] = set()
    unique: list[str] = []
    for code in episode.diagnosis_codes:
        if code not in seen:
            seen.add(code)
            unique.append(code)
    pdx_code, adx_codes = unique[0], tuple(unique[1:])

    achi_codes = episode.procedure_codes
    mbs_codes = episode.mbs_codes(mbs_array_field)

    lines = ["EPISODE"]

    def diagnosis_block(label: str, codes) -> None:
        lines.append("")
        lines.append(f"{label}:")
        if not codes:
            lines.append("  (none)")
        for code in codes:
            lines.append(f"  {code} \u2014 {ref.resolve_diagnosis(code)}")

    diagnosis_block("PDx", [pdx_code])
    diagnosis_block("ADx", adx_codes)

    lines.append("")
    lines.append("ACHI:")
    if not achi_codes:
        lines.append("  (none)")
    for i, code in enumerate(achi_codes):
        when = episode.procedure_dates[i] if i < len(episode.procedure_dates) else None
        lines.append(f"  {code} \u2014 {ref.resolve_achi(code)} ({_day_offset(when, episode.admit_dt)})")

    lines.append("")
    lines.append("MBS / hospital services:")
    if not mbs_codes:
        lines.append("  (none)")
    for code in mbs_codes:
        lines.append(f"  {code} \u2014 {ref.resolve_mbs(code)} (day unknown)")

    return FormattedEpisode(
        text="\n".join(lines),
        pdx_code=pdx_code,
        adx_codes=adx_codes,
        achi_codes=achi_codes,
        mbs_codes=mbs_codes,
        diagnosis_order=episode.diagnosis_codes,
    )
