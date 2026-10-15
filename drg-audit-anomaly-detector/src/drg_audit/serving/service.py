"""Service layer behind the SageMaker handlers. Framework-agnostic and unit-testable with a stub engine."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from drg_audit.logging_utils import get_logger, log_event
from drg_audit.schemas import AuditResult, InpatientRecord
from drg_audit.serving.contract import RejectedRecord, ScoreRequest, parse_request
from drg_audit.settings import RequestLimits

logger = get_logger("serving")


class ScoringEngine(Protocol):
    model_version: str
    default_threshold: float

    def predict(
        self, records: Sequence[InpatientRecord], confidence_threshold: float | None = None
    ) -> list[AuditResult]: ...


class InferenceService:
    def __init__(self, engine: ScoringEngine, limits: RequestLimits) -> None:
        self.engine = engine
        self.limits = limits

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, env: Mapping[str, str] | None = None) -> InferenceService:
        """Load, integrity-check and warm up the model bundle. Imports torch lazily."""
        from drg_audit.artifacts import load_bundle
        from drg_audit.settings import ServingConfig

        # Verify integrity first (pure python, fails fast), then pay for the heavy torch import.
        bundle = load_bundle(model_dir, verify=True)
        config = ServingConfig.from_mapping(bundle.inference_config).with_env(env)
        from drg_audit.inference.engine import AuditEngine

        engine = AuditEngine(
            bundle, settings=config.runtime, default_threshold=config.decision.confidence_threshold
        )
        log_event(
            logger, "model_loaded", model_version=bundle.model_version, device=str(engine.device),
            tokenization=bundle.tokenization.name,
        )
        return cls(engine, config.limits)

    def parse(self, body: bytes | bytearray | str, content_type: str | None) -> ScoreRequest:
        return parse_request(body, content_type, self.limits)

    def score(self, request: ScoreRequest) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        threshold = (
            request.confidence_threshold
            if request.confidence_threshold is not None
            else self.engine.default_threshold
        )

        valid = request.valid
        scored = self.engine.predict([p.record for p in valid], threshold) if valid else []
        if len(scored) != len(valid):
            raise RuntimeError("engine returned an unexpected number of results")
        scored_by_index = {p.index: (result, p.warnings) for p, result in zip(valid, scored)}

        results: list[dict[str, Any]] = []
        for item in request.items:
            if isinstance(item, RejectedRecord):
                results.append(AuditResult.rejected(item.admission_no, item.reason_text).to_dict())
                continue
            result, warnings = scored_by_index[item.index]
            result.warnings = [*result.warnings, *warnings]
            results.append(result.to_dict())

        summary = {
            "n_records": len(results),
            "n_scored": sum(r["status"] == "ok" for r in results),
            "n_anomalies": sum(bool(r["is_billing_anomaly"]) for r in results),
            "n_skipped": sum(r["status"] == "skipped" for r in results),
            "n_rejected": sum(r["status"] == "rejected" for r in results),
        }
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        log_event(logger, "request_scored", request_id=request_id, latency_ms=latency_ms,
                  confidence_threshold=threshold, model_version=self.engine.model_version, **summary)
        return {
            "request_id": request_id,
            "model_version": self.engine.model_version,
            "confidence_threshold": threshold,
            "summary": summary,
            "results": results,
        }
