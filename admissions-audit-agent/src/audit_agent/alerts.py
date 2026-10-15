"""Reviewer-facing alert text and the alert record.

``build_alert_info`` produces the free-text finding a human coder reads. Every
piece of model-written text is output-encoded with ``escape_html`` because the
downstream viewer renders this field as HTML (it intentionally contains one
``<br>``). Wording is carried over from the validated prototype, minus
references to internal file names.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .grouper.interface import RegroupResult
from .llm.verdicts import CodeVerdict
from .schemas import Episode
from .textutil import escape_html

TARGET_SERVICE_CLASS_CODES = ("DRG_CASE_PAYMENT", "EPM", "CWO")


def get_claim_seqno(episode: Episode) -> Optional[str]:
    """Sequence number of the first hospital claim whose service class is a DRG payment class."""
    seqnos = episode.extras.get("hospital_claims_seqno_array") or ()
    classes = episode.extras.get("hospital_claims_service_class_code_array") or ()
    for i, service_class in enumerate(classes):
        if service_class in TARGET_SERVICE_CLASS_CODES and i < len(seqnos) and seqnos[i] is not None:
            return str(int(seqnos[i]))
    return None


def _support_headline(verdict: CodeVerdict, threshold: float) -> tuple[str, str]:
    """(headline, closing note) for one failing diagnosis."""
    if verdict.support_level == "none":
        headline = (
            f"{verdict.code}: NOT SUPPORTED -- NO ACHI or MBS code on this admission gives it even "
            "weak or indirect clinical support, let alone clear support. A code that was merely "
            "CONSISTENT WITH this diagnosis (e.g. a general test or imaging study that doesn't "
            "confirm it, or a specimen collection of unrecorded type) would already have been "
            "enough to avoid this finding -- only a fully checked absence of any plausible "
            "connection reaches it."
        )
        closing = (
            f"Every ACHI/MBS code on the claim was checked individually against {verdict.code}, "
            "and the model found none of them -- not even weakly or indirectly -- to support it."
        )
        return headline, closing
    if verdict.support_level == "weak":
        headline = (
            f"{verdict.code}: LOW-CONFIDENCE SUPPORT -- the AI found at least one ACHI or MBS code "
            "plausibly, though only weakly or indirectly, connected to this diagnosis, but was not "
            f"confident in that finding (confidence {verdict.confidence:.2f}, below the "
            f"{threshold:.2f} threshold used to flag this). This is a DIFFERENT, weaker finding than "
            "a checked zero-relation case -- it means the AI itself is unsure whether the evidence it "
            "found amounts to real support, not that it found nothing at all."
        )
        closing = (
            f"At least one ACHI/MBS code on the claim was found plausibly connected to {verdict.code}, "
            "but the AI's own confidence in that connection was low."
        )
        return headline, closing
    raise ValueError("only 'none' or low-confidence 'weak' verdicts produce a finding")


def build_alert_info(
    *,
    pdx_verdict: CodeVerdict,
    adx_verdicts: Sequence[CodeVerdict],
    regroup: RegroupResult,
    replacement_reasoning: Optional[str],
    mechanical_fallback: bool,
    threshold: float,
) -> str:
    """Three shapes reach here: PDx failed only (resequence), ADx failed only
    (removal), or both together (one alert, one regroup)."""
    pdx_failed = regroup.pdx_resequence is not None
    old_code = new_code = None
    if pdx_failed:
        old_code, new_code = regroup.pdx_resequence

    sections: list[str] = []

    if pdx_failed:
        headline, closing = _support_headline(pdx_verdict, threshold)
        headline = headline.replace(f"{pdx_verdict.code}:", f"PDx {pdx_verdict.code}:", 1)
        headline += " This diagnosis is currently coded as the PRINCIPAL reason for admission."
        reasoning = escape_html(replacement_reasoning or "")
        if mechanical_fallback:
            note = (
                f"{old_code} was NOT deleted -- it has been kept as an additional diagnosis. The "
                f"model did NOT find {new_code}, or any other surviving diagnosis already "
                "coded on this episode, to be a clinically well-supported replacement principal "
                f"diagnosis. Its own assessment of why none fit was:\n{reasoning}\n"
                f"{new_code} was selected as the replacement principal diagnosis MECHANICALLY "
                "-- it is simply the first surviving additional diagnosis in the original coded order, "
                "not a clinical judgment that it is the right one. The DRG/cost-weight figures below "
                "reflect that mechanical choice only. This episode specifically needs a clinical coder "
                "to determine the correct principal diagnosis directly -- more so than a typical "
                "resequencing case, since no clinical basis for this particular replacement exists."
            )
        else:
            note = (
                f"{old_code} was NOT deleted -- it has been kept as an additional diagnosis, since "
                "the evidence not supporting it as principal doesn't mean it isn't a real condition. "
                f"{new_code} was identified as the best-supported replacement principal "
                f"diagnosis, for this reason:\n{reasoning}\n"
                "This resequencing is model-reasoned, not a mechanical fallback -- but it is still an "
                "AI-assisted read of the ACHI/MBS evidence only, not a clinical determination. Please "
                "confirm the correct principal diagnosis by clinical review before treating the figures "
                "below as a confirmed outcome."
            )
        sections.append(
            f"{headline}\n{escape_html(pdx_verdict.reasoning)}\n\n"
            f"IMPORTANT -- this includes the PRINCIPAL diagnosis. {note}\n\n{closing}"
        )

    removed = set(regroup.codes_removed)
    failed_adx = [v for v in adx_verdicts if v.code in removed]
    if failed_adx:
        blocks = []
        for v in failed_adx:
            headline, closing = _support_headline(v, threshold)
            headline = headline.replace(f"{v.code}:", f"ADx {v.code}:", 1)
            headline += (
                " This diagnosis is being REMOVED from the claim entirely, not resequenced -- unlike the "
                "PDx, an unsupported ADx means the evidence says it shouldn't be coded on this episode at all."
            )
            blocks.append(f"{headline}\n{escape_html(v.reasoning)}\n{closing}")
        one = len(failed_adx) == 1
        intro = (
            "The following additional diagnos" + ("is" if one else "es")
            + " on this episode " + ("was" if one else "were")
            + (" also" if pdx_failed else "")
            + " found unsupported by the ACHI/MBS evidence and "
            + ("has" if one else "have") + " been removed:"
        )
        sections.append(intro + "\n\n" + "\n\n".join(blocks))

    if regroup.drg_changed:
        removal_text = f", and removing {', '.join(regroup.codes_removed)}," if failed_adx else ""
        if pdx_failed:
            change = (
                f"Resequencing {old_code} to secondary and promoting {new_code} to principal{removal_text} "
                f"regroups to DRG {escape_html(regroup.regrouped_drg)} - {escape_html(regroup.regrouped_drg_desc)}"
            )
        else:
            change = (
                f"Removing {', '.join(regroup.codes_removed)} regroups to DRG "
                f"{escape_html(regroup.regrouped_drg)} - {escape_html(regroup.regrouped_drg_desc)}"
            )
        desc = f" ({escape_html(regroup.original_drg_desc)})" if regroup.original_drg_desc else ""
        numbers = (
            f"\nThis admission has DRG {escape_html(regroup.original_drg)}{desc} "
            f"with cost weight {regroup.original_cost_weight}.\n"
            f"{change} with cost weight {regroup.regrouped_cost_weight} in DRG version "
            f"{escape_html(regroup.drg_ver_grouper)} (cost weight drop: {regroup.cost_weight_drop})."
        )
    else:
        numbers = (
            "\nApplying the correction(s) above did not change the assigned DRG in this instance -- "
            "flagged for coding-accuracy review regardless of cost-weight impact."
        )

    return (
        "The admission was flagged by an AI-assisted coding-support review for the reason(s) below; "
        "please review the coding and add a suitable comment for any confirmed coding error.<br>\n\n"
        + "\n\n---\n\n".join(sections) + f"\n{numbers}\n\n"
        "This is an AI-assisted screening signal based ONLY on the ACHI procedures and MBS "
        "services billed for this admission -- no administrative/demographic context or other "
        "evidence was available to the model. This is NOT a clinical determination. The model "
        "was not shown the assigned DRG or any financial information, so this reflects an "
        "independent read of the procedural/billing evidence, not agreement or disagreement "
        "with the billed outcome."
    )


def build_alert_record(
    *, alert_info: str, claim_seqno: Optional[str], possible_recovery: Optional[float], rule_id: int
) -> dict:
    """Same schema the downstream alert consumers already read."""
    return {
        "claim_seqno": claim_seqno,
        "alert_info": alert_info,
        "recovery_amount": round(possible_recovery) if possible_recovery is not None else None,
        "alert_instance_id": int(rule_id),
    }
