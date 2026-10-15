"""Finalise a job: dedupe -> append to the catalogue template -> remap executable rules -> summary.

All outputs are written under ``jobs/<job_id>/outputs/`` with deterministic keys, so re-running finalisation after a
crash is idempotent. The raw (pre-dedupe) ledger is written FIRST so nothing paid for is lost if a later step fails.
"""
from __future__ import annotations

import dataclasses
import logging
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.jobs import JobRecord, JobStatus, UnitRecord, UnitStatus
from ..domain.schema import Rule, rule_from_ledger_record
from ..engine.rules_store import ExecutableRule, load_all, remap_ids, save_all
from ..errors import OutputError
from ..ingestion.corpus import CorpusIndex
from ..output.excel_writer import append_rules_to_template
from ..output.jsonl import iter_jsonl, to_jsonl_bytes
from ..settings import Settings
from ..storage.interfaces import ArtifactStore
from ..util import truncate, utc_now_iso
from .dedupe import dedupe_rules

log = logging.getLogger(__name__)
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
NDJSON = "application/x-ndjson"


@dataclass
class FinalizeResult:
    status: str
    artifacts: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Finalizer:
    def __init__(self, settings: Settings, artifacts: ArtifactStore, corpus: CorpusIndex | None = None) -> None:
        self._s, self._artifacts, self._corpus = settings, artifacts, corpus

    # ---------------------------------------------------------------- resolution
    def _template_path(self) -> Path | None:
        p = self._s.paths
        for cand in (p.template_path, p.model_dir / "templates" / "rule_catalogue.xlsx", p.source_docs_dir / "rule_catalogue.xlsx"):
            if cand and Path(cand).is_file():
                return Path(cand)
        return None

    def _base_rules(self) -> list[ExecutableRule]:
        p = self._s.paths
        for cand in (p.base_rules_path, p.model_dir / "base" / "executable_rules.jsonl"):
            if cand and Path(cand).is_file():
                return load_all(cand)
        return []

    @staticmethod
    def _status(job: JobRecord, counts: Counter, total: int) -> str:
        succeeded = counts[UnitStatus.SUCCEEDED.value]
        if job.cancel_requested:
            return JobStatus.CANCELLED.value
        if succeeded == total:
            return JobStatus.SUCCEEDED.value
        return JobStatus.PARTIAL.value if succeeded else JobStatus.FAILED.value

    # ------------------------------------------------------------------ finalize
    def finalize(self, job: JobRecord, units: list[UnitRecord]) -> FinalizeResult:
        counts = Counter(u.status for u in units)
        base = f"jobs/{job.job_id}/outputs"
        artifacts: dict[str, str] = {}
        status = self._status(job, counts, len(units))
        summary: dict[str, Any] = {
            "job_id": job.job_id, "kind": job.kind, "finalized_at": utc_now_iso(), "units_total": len(units),
            "units": {k.value: counts.get(k.value, 0) for k in UnitStatus},
            "cost_usd": round(sum(u.cost_usd for u in units), 4), "turns": sum(u.turns for u in units),
            "failures": [{"unit_id": u.unit_id, "kind": u.kind, "error": truncate(u.error or "", 300)}
                         for u in units if u.status == UnitStatus.FAILED.value][:50],
            "cost_note": "cost_usd is the SDK-reported estimate; reconcile with AWS Cost Explorer / Bedrock invocation logs",
            "versions": job.version_info,
        }
        error: str | None = None
        if job.kind == "contract_check":
            key = f"{base}/contract_report.json"
            if self._artifacts.exists(key):
                artifacts["contract_report"] = self._artifacts.uri(key)
        else:
            try:
                self._extraction_outputs(job, units, base, artifacts, summary)
            except Exception as exc:
                log.exception("finalization failed")
                status, error = JobStatus.FAILED.value, f"finalization failed: {type(exc).__name__}: {truncate(str(exc), 300)}"
        table = [{"unit_id": u.unit_id, "kind": u.kind, "status": u.status, "attempts": u.attempts, "rules": u.rules_count,
                  "rejected": u.rejected_count, "cost_usd": round(u.cost_usd, 4), "turns": u.turns,
                  "duration_s": round(u.duration_s, 1), "model": u.model, "error": truncate(u.error or "", 300)} for u in units]
        artifacts["units"] = self._artifacts.put_bytes(f"{base}/units.jsonl", to_jsonl_bytes(table), content_type=NDJSON)
        summary["status"] = status
        summary["artifacts"] = dict(artifacts)
        artifacts["summary"] = self._artifacts.put_bytes(f"{base}/summary.json", _json_bytes(summary), content_type="application/json")
        return FinalizeResult(status=status, artifacts=artifacts, summary=summary, error=error)

    def _extraction_outputs(self, job: JobRecord, units: list[UnitRecord], base: str, artifacts: dict[str, str],
                            summary: dict[str, Any]) -> None:
        loaded: list[tuple[UnitRecord, Rule, ExecutableRule | None]] = []
        for unit in sorted((u for u in units if u.status == UnitStatus.SUCCEEDED.value and u.rules_key), key=lambda u: u.order):
            for record in iter_jsonl(self._artifacts.get_bytes(unit.rules_key or "")):
                exe = ExecutableRule.from_dict(record["executable"]) if record.get("executable") else None
                loaded.append((unit, rule_from_ledger_record(record["rule"]), exe))
        artifacts["rules_raw"] = self._artifacts.put_bytes(
            f"{base}/rules_raw.jsonl", to_jsonl_bytes([r.as_ledger_record() for _, r, _ in loaded]), content_type=NDJSON)

        kept, dropped = dedupe_rules([(r, e) for _, r, e in loaded])
        summary.update(rules_raw=len(loaded), rules_after_dedupe=len(kept), duplicates_dropped=dropped)
        id_map: dict[str, str] = {}
        work_dir = self._s.paths.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="finalize-", dir=str(work_dir)) as tmp:
            if kept:
                template = self._template_path()
                if template is None:
                    raise OutputError("catalogue template not found (package templates/rule_catalogue.xlsx into the model artifact)")
                out = Path(tmp) / "rule_catalogue.xlsx"
                footer = self._s.output.footer_lines if self._s.output.footer_enabled else ()
                id_map = append_rules_to_template(template, out, [r for r, _ in kept], sheet_name=self._s.output.excel_sheet_name,
                                                  footer_lines=footer)
                artifacts["catalogue"] = self._artifacts.put_file(f"{base}/rule_catalogue.xlsx", out, content_type=XLSX_MIME)

            final = [(dataclasses.replace(r, rule_id=id_map.get(r.rule_id, r.rule_id)), e) for r, e in kept]
            artifacts["rules_final"] = self._artifacts.put_bytes(
                f"{base}/rules_final.jsonl",
                to_jsonl_bytes([{"rule": r.as_ledger_record(), "executable": remapped.to_dict() if remapped else None}
                                for (r, _), remapped in zip(final, _remap_each(kept, id_map))]), content_type=NDJSON)

            new_exec = remap_ids([e for _, e in kept if e is not None], id_map)
            exec_path = Path(tmp) / "executable_rules.jsonl"
            save_all(exec_path, self._base_rules() + new_exec)
            artifacts["executable_rules"] = self._artifacts.put_file(f"{base}/executable_rules.jsonl", exec_path, content_type=NDJSON)
        summary.update(executable_rules=len(new_exec), automatable_rules=sum(1 for e in new_exec if e.automatable),
                       narrative_only_rules=len(kept) - len(new_exec))
        if self._corpus is not None and self._corpus.mbs_facts_path().is_file():
            artifacts["mbs_facts"] = self._artifacts.put_file(f"{base}/mbs_billing_facts_reference.jsonl",
                                                              self._corpus.mbs_facts_path(), content_type=NDJSON)


def _remap_each(kept: list[tuple[Rule, ExecutableRule | None]], id_map: dict[str, str]) -> list[ExecutableRule | None]:
    return [remap_ids([e], id_map)[0] if e is not None else None for _, e in kept]


def _json_bytes(obj: Any) -> bytes:
    import json
    return json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
