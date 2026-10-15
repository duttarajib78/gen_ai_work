"""Transport-agnostic service layer: payload in, (status code, JSON body) out.

The HTTP app (``serving/app.py``), the offline benchmark, and the diagnostics all
call this same class, so a benchmark run exercises exactly the code that serves
production traffic.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from .config import Settings
from .deadline import Deadline
from .engine import AuditEngine, AuditOutcome
from .errors import AuditError, DeadlineExceededError
from .llm.base import UsageMeter
from .llm.verdicts import CodeVerdict
from .logging_utils import exception_fields, log_event
from .prompts import PROMPT_VERSION
from .reference_data import ReferenceData
from .rules import RULE_VERSION, diagnosis_fails, flag_reason
from .schemas import AuditRequest, parse_request

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceResult:
    status_code: int
    body: dict


class AuditService:
    def __init__(self, *, settings: Settings, engine: AuditEngine, reference: ReferenceData) -> None:
        self.settings = settings
        self._engine = engine
        self._reference = reference

    # -- public ----------------------------------------------------------------
    def validate(self, payload: Any) -> AuditRequest:
        return parse_request(payload, self.settings)

    def provenance(self) -> dict:
        s = self.settings
        return {
            "service_version": s.service_version,
            "model_id": s.bedrock_model_id,
            "routing_scope": s.routing_scope,
            "reasoning_effort": s.llm_effort,
            "thinking_mode": s.llm_thinking_mode,
            "prompt_version": PROMPT_VERSION,
            "rule_version": RULE_VERSION,
            "low_confidence_weak_threshold": s.low_confidence_weak_threshold,
            "reference_data_manifest_sha256": self._reference.manifest_sha256,
            "mbs_source_file": self._reference.mbs_source_file,
        }

    def audit(self, request: AuditRequest, deadline: Deadline) -> ServiceResult:
        """Never raises for expected failures; every outcome is a ServiceResult."""
        meter = UsageMeter()
        started = time.monotonic()
        try:
            outcome = self._engine.audit(
                request.episode, mode=request.mode, mbs_array_field=request.mbs_array_field,
                deadline=deadline, meter=meter,
            )
            body = self._success_body(request, outcome, meter, started)
            self._log_done(request, "completed", 200, started, meter, outcome=outcome)
            return ServiceResult(200, body)
        except AuditError as exc:
            body = self._failure_body(request, exc, meter, started, getattr(exc, "partial", None))
            self._log_done(request, "failed", exc.http_status, started, meter, error=exc)
            return ServiceResult(exc.http_status, body)
        except Exception as exc:  # noqa: BLE001 - last line of defence; details go to logs only
            log_event(
                logger, "unexpected_error", level=logging.ERROR, request_id=request.request_id,
                phase="audit", **exception_fields(exc),
            )
            error = AuditError("an unexpected internal error occurred", code="internal_error", http_status=500)
            body = self._failure_body(request, error, meter, started, None)
            self._log_done(request, "failed", 500, started, meter, error=error)
            return ServiceResult(500, body)

    def deadline_exceeded_result(self, request: AuditRequest, deadline: Deadline) -> ServiceResult:
        error = DeadlineExceededError(
            f"the {deadline.budget_s:.0f}s time budget ran out before the audit finished; retry with "
            "options.stream=true (Accept: application/x-ndjson) for a longer budget, or lower the reasoning effort",
            details={"budget_s": deadline.budget_s},
        )
        meter = UsageMeter()
        started = time.monotonic() - deadline.elapsed()
        body = self._failure_body(request, error, meter, started, None)
        body["usage"]["note"] = "usage unknown: the audit was still running when the response was sent"
        self._log_done(request, "failed", error.http_status, started, meter, error=error)
        return ServiceResult(error.http_status, body)

    # -- bodies ----------------------------------------------------------------
    def _verdict_json(self, verdict: CodeVerdict) -> dict:
        t = self.settings.low_confidence_weak_threshold
        data = verdict.to_dict()
        data["failed"] = diagnosis_fails(verdict.support_level, verdict.confidence, t)
        data["flag_reason"] = flag_reason(verdict.support_level, verdict.confidence, t)
        return data

    def _success_body(self, request: AuditRequest, outcome: AuditOutcome, meter: UsageMeter, started: float) -> dict:
        dropped = outcome.disposition == "dropped_no_surviving_adx"
        correction = None
        if (outcome.removed_adx_codes or outcome.pdx_resequence) and not dropped:
            resequence = None
            if outcome.pdx_resequence:
                resequence = {
                    "old_pdx_code": outcome.pdx_resequence[0],
                    "new_principal_code": outcome.pdx_resequence[1],
                    "mechanical_fallback": outcome.mechanical_fallback,
                    "reasoning": outcome.replacement_reasoning,
                }
            correction = {
                "removed_adx_codes": list(outcome.removed_adx_codes),
                "pdx_resequence": resequence,
                "corrected_diagnosis_fields": outcome.corrected_diagnosis_fields,
            }
        warnings = list(request.warnings) + list(outcome.warnings)
        if dropped:
            warnings.append(
                "no_surviving_adx: the principal diagnosis failed but no additional diagnosis survives to "
                "promote, so no alert was produced; route to a human coder if this episode matters"
            )
        body: dict[str, Any] = {
            "request_id": request.request_id,
            "status": "completed",
            "admission_no": request.episode.admission_no,
            "mode": request.mode,
            "potential_alert": outcome.potential_alert,
            "flagged": outcome.flagged,
            "disposition": outcome.disposition,
            "verdicts": [self._verdict_json(outcome.pdx_verdict)] + [self._verdict_json(v) for v in outcome.adx_verdicts],
            "failed_adx_codes": list(outcome.failed_adx_codes),
            "correction": correction,
            "regroup": outcome.regroup.to_dict() if outcome.regroup else None,
            "alert": outcome.alert,
            "warnings": warnings,
            "provenance": self.provenance(),
            "usage": {**meter.to_dict(), "elapsed_ms": int((time.monotonic() - started) * 1000)},
        }
        if request.include_episode_text:
            body["episode_text"] = outcome.formatted.text
        return body

    def _failure_body(
        self, request: AuditRequest, error: AuditError, meter: UsageMeter, started: float,
        partial: Optional[AuditOutcome],
    ) -> dict:
        body: dict[str, Any] = {
            "request_id": request.request_id,
            "status": "failed",
            "admission_no": request.episode.admission_no,
            "mode": request.mode,
            "error": error.to_error_body(),
            "partial": None,
            "provenance": self.provenance(),
            "usage": {**meter.to_dict(), "elapsed_ms": int((time.monotonic() - started) * 1000)},
        }
        if partial is not None:
            body["partial"] = {
                "potential_alert": partial.potential_alert,
                "verdicts": [self._verdict_json(partial.pdx_verdict)] + [self._verdict_json(v) for v in partial.adx_verdicts],
            }
        return body

    def _log_done(
        self, request: AuditRequest, status: str, http_status: int, started: float, meter: UsageMeter,
        *, outcome: Optional[AuditOutcome] = None, error: Optional[AuditError] = None,
    ) -> None:
        failed = status == "failed"
        log_event(
            logger,
            "episode_failed" if failed else "episode_completed",
            level=(logging.ERROR if (failed and http_status >= 500) else logging.INFO),
            log_raw_identifiers=self.settings.log_raw_identifiers,
            request_id=request.request_id, admission_no=request.episode.admission_no, status=status,
            http_status=http_status, mode=request.mode,
            n_diagnoses=len(set(request.episode.diagnosis_codes)),
            flagged=outcome.flagged if outcome else None,
            potential_alert=outcome.potential_alert if outcome else None,
            disposition=outcome.disposition if outcome else None,
            n_failed_adx=len(outcome.failed_adx_codes) if outcome else None,
            error_code=error.code if error else None, retryable=error.retryable if error else None,
            llm_calls=meter.calls, input_tokens=meter.input_tokens, output_tokens=meter.output_tokens,
            cache_read_tokens=meter.cache_read_tokens, cache_write_tokens=meter.cache_write_tokens,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )


def build_service(settings: Settings) -> AuditService:
    """Wire the production dependencies. Everything that can fail does so here, at
    start-up, with a clear message; nothing calls Bedrock until a request arrives."""
    from .grouper.au_drg import AuDrgGrouper
    from .llm.bedrock import BedrockClaudeClient

    reference = ReferenceData.from_model_dir(settings.model_dir, require_manifest=settings.require_manifest)
    llm = BedrockClaudeClient.from_settings(settings)
    grouper = AuDrgGrouper.from_settings(settings) if settings.grouper_backend == "au_drg" else None
    engine = AuditEngine(reference=reference, llm=llm, grouper=grouper, settings=settings)
    return AuditService(settings=settings, engine=engine, reference=reference)
