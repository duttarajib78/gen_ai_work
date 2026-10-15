"""The audit pipeline for one episode.

  Phase 1  per-diagnosis verdicts (PDx and every ADx judged independently in one
           model call); the episode is a potential alert if the PDx or any ADx fails
           the decision rule (``rules.diagnosis_fails``).
  Phase 2  decide the correction. Every failed ADx is removed. If the PDx failed,
           ask the model for a replacement among the *surviving* ADx; if none fits,
           fall back mechanically to the first surviving ADx in coded order; if no
           ADx survives, the episode is dropped (no alert). A PDx that passed with
           only ADx failures skips the replacement search.
  Phase 3  regroup the corrected episode with the AR-DRG grouper and assemble the
           alert text and record.

``mode="verdicts_only"`` stops after phase 1 and needs no grouper. With the grouper
disabled in configuration, ``mode="full"`` stops after phase 2 and returns the
proposed correction without regroup figures or an alert record.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .alerts import build_alert_info, build_alert_record, get_claim_seqno
from .config import Settings
from .deadline import Deadline
from .episode_text import FormattedEpisode, build_episode_text
from .errors import AuditError
from .grouper.corrections import apply_diagnosis_corrections, rows_to_full_diagnosis_fields
from .grouper.interface import Grouper, RegroupResult
from .llm.base import LlmClient, UsageMeter
from .llm.verdicts import CodeVerdict, LlmTaskConfig, get_diagnosis_verdicts, get_replacement_pdx
from .reference_data import ReferenceData
from .rules import diagnosis_fails
from .schemas import Episode

logger = logging.getLogger(__name__)

DISPOSITIONS = ("alert", "no_alert", "dropped_no_surviving_adx", "verdicts_only", "correction_proposed")


@dataclass
class AuditOutcome:
    formatted: FormattedEpisode
    pdx_verdict: CodeVerdict
    adx_verdicts: list[CodeVerdict]
    failed_adx_codes: tuple[str, ...]
    potential_alert: bool
    removed_adx_codes: tuple[str, ...] = ()
    pdx_resequence: Optional[tuple[str, str]] = None
    replacement_reasoning: Optional[str] = None
    mechanical_fallback: bool = False
    regroup: Optional[RegroupResult] = None
    corrected_diagnosis_fields: Optional[dict] = None
    alert: Optional[dict] = None
    flagged: bool = False
    disposition: str = "no_alert"
    warnings: list[str] = field(default_factory=list)


class AuditEngine:
    def __init__(self, *, reference: ReferenceData, llm: LlmClient, grouper: Optional[Grouper], settings: Settings) -> None:
        self._ref = reference
        self._llm = llm
        self._grouper = grouper
        self._settings = settings
        self._task_cfg = LlmTaskConfig.from_settings(settings)

    # -- phase 1 ---------------------------------------------------------------
    def evaluate(
        self, episode: Episode, *, mbs_array_field: str, deadline: Deadline, meter: UsageMeter
    ) -> AuditOutcome:
        formatted = build_episode_text(episode, self._ref, mbs_array_field)
        verdicts = get_diagnosis_verdicts(
            self._llm, episode_text=formatted.text, pdx_code=formatted.pdx_code, adx_codes=formatted.adx_codes,
            deadline=deadline, meter=meter, cfg=self._task_cfg,
        )
        by_code = {v.code: v for v in verdicts}
        pdx = by_code[formatted.pdx_code]
        adx = [by_code[c] for c in formatted.adx_codes]
        threshold = self._settings.low_confidence_weak_threshold
        failed = tuple(v.code for v in adx if diagnosis_fails(v.support_level, v.confidence, threshold))
        potential = diagnosis_fails(pdx.support_level, pdx.confidence, threshold) or bool(failed)
        return AuditOutcome(
            formatted=formatted, pdx_verdict=pdx, adx_verdicts=adx, failed_adx_codes=failed, potential_alert=potential,
        )

    # -- phase 2 ---------------------------------------------------------------
    def plan_correction(self, outcome: AuditOutcome, *, deadline: Deadline, meter: UsageMeter) -> bool:
        """Fill in the correction on ``outcome``. Returns False when the episode is dropped."""
        threshold = self._settings.low_confidence_weak_threshold
        pdx = outcome.pdx_verdict
        outcome.removed_adx_codes = outcome.failed_adx_codes
        if not diagnosis_fails(pdx.support_level, pdx.confidence, threshold):
            return True  # ADx-only alert: removal only, no replacement search

        surviving = [c for c in outcome.formatted.adx_codes if c not in outcome.failed_adx_codes]
        if not surviving:
            return False  # nothing to promote; checked before spending a model call

        choice = get_replacement_pdx(
            self._llm, episode_text=outcome.formatted.text, old_pdx=pdx, candidate_codes=surviving,
            deadline=deadline, meter=meter, cfg=self._task_cfg,
        )
        if choice.new_principal_code is not None:
            outcome.pdx_resequence = (pdx.code, choice.new_principal_code)
        else:
            outcome.pdx_resequence = (pdx.code, surviving[0])
            outcome.mechanical_fallback = True
        outcome.replacement_reasoning = choice.reasoning
        return True

    # -- phase 3 ---------------------------------------------------------------
    def finalize(self, outcome: AuditOutcome, episode: Episode, *, deadline: Deadline) -> None:
        removed = frozenset(outcome.removed_adx_codes)
        rows = apply_diagnosis_corrections(episode, removed, outcome.pdx_resequence)
        outcome.corrected_diagnosis_fields = rows_to_full_diagnosis_fields(rows)

        if self._grouper is None:
            outcome.disposition = "correction_proposed"
            outcome.warnings.append("grouper_disabled: no regroup figures or alert record were produced")
            return

        regroup = self._grouper.regroup(
            episode, removed_codes=removed, pdx_resequence=outcome.pdx_resequence, deadline=deadline
        )
        outcome.regroup = regroup
        alert_info = build_alert_info(
            pdx_verdict=outcome.pdx_verdict, adx_verdicts=outcome.adx_verdicts, regroup=regroup,
            replacement_reasoning=outcome.replacement_reasoning, mechanical_fallback=outcome.mechanical_fallback,
            threshold=self._settings.low_confidence_weak_threshold,
        )
        outcome.alert = build_alert_record(
            alert_info=alert_info, claim_seqno=get_claim_seqno(episode),
            possible_recovery=regroup.possible_recovery, rule_id=self._settings.rule_id,
        )
        outcome.flagged = True
        outcome.disposition = "alert"

    # -- all phases ------------------------------------------------------------
    def audit(
        self, episode: Episode, *, mode: str, mbs_array_field: str, deadline: Deadline, meter: UsageMeter
    ) -> AuditOutcome:
        outcome = self.evaluate(episode, mbs_array_field=mbs_array_field, deadline=deadline, meter=meter)
        try:
            if mode == "verdicts_only":
                outcome.disposition = "verdicts_only"
                return outcome
            if not outcome.potential_alert:
                outcome.disposition = "no_alert"
                return outcome
            if not self.plan_correction(outcome, deadline=deadline, meter=meter):
                outcome.disposition = "dropped_no_surviving_adx"
                return outcome
            self.finalize(outcome, episode, deadline=deadline)
            return outcome
        except AuditError as exc:
            exc.partial = outcome  # lets the service return the verdicts that were already paid for
            raise
