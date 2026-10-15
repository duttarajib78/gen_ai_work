"""``emit_rule``: the validation boundary between untrusted model output and everything downstream.

A rule is recorded only if it passes: schema + length limits, citation check, rule-shape check, DSL validation
(``build_executable``), duplicate and per-unit caps. Rejections return an explanatory error so the model can fix
and retry within the same session. ``confidence`` is always ``needs_review``; the model cannot influence it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ..domain.schema import (MATERIALITY_LEVELS, RECOMMENDED_ACTIONS, RULE_SHAPES, Rule, RuleValidationError,
                             validate_rule_dict)
from ..engine.rules_store import ExecutableRule, build_executable
from ..ingestion.corpus import CodeRegistry, canon_code
from .types import ToolResult, ToolSpec

MODEL_FIELDS = (
    "rule_name", "rule_logic_plain_english", "icd10am_codes", "achi_codes", "mbs_codes", "upcoding_mechanism",
    "recommended_action", "source_reference", "materiality", "rule_shape",
)
_CITATION = re.compile(r"(?i)(\bp{1,2}\.?\s*\d|\bpages?\s+\d|https?://)")
_URL = re.compile(r"https?://[^\s;,)\]]+")
_TOKEN_SPLIT = re.compile(r"[,;|/\s]+")
_ICD_PLAIN = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,2})?$")
_ACHI_PLAIN = re.compile(r"^\d{5}-\d{2}$")

EMIT_RULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rule_name": {"type": "string", "description": "Short, specific rule name (max 200 characters)."},
        "rule_logic_plain_english": {"type": "string", "description": "The detection logic in plain English, using the claims-schema field names."},
        "icd10am_codes": {"type": "string", "description": "Comma-separated ICD-10-AM codes the rule involves, or N/A."},
        "achi_codes": {"type": "string", "description": "Comma-separated ACHI codes, or N/A."},
        "mbs_codes": {"type": "string", "description": "Comma-separated MBS item numbers, or N/A."},
        "upcoding_mechanism": {"type": "string", "description": "How the pattern could inflate the AR-DRG cost weight, in plain English. Do not invent numeric weights."},
        "recommended_action": {"type": "string", "enum": RECOMMENDED_ACTIONS},
        "source_reference": {"type": "string", "description": "Document, printed page and standard/block, e.g. 'icd10am_tabular printed p.27; acs printed p.101 (ACS 0110 Example 4)'."},
        "materiality": {"type": "string", "enum": MATERIALITY_LEVELS},
        "rule_shape": {"type": "string", "enum": RULE_SHAPES},
        "flag_condition": {"type": "object", "description": "Optional executable condition (see the grammar in the system prompt)."},
        "correction_action": {"type": "object", "description": "Optional executable action (see the grammar). Required when recommended_action is not flag_for_clinical_review and flag_condition is given."},
    },
    "required": ["rule_name", "rule_logic_plain_english", "upcoding_mechanism", "recommended_action", "source_reference", "rule_shape"],
}


@dataclass
class EmittedRule:
    rule: Rule
    executable: ExecutableRule | None


@dataclass
class RuleCollector:
    job_id: str
    unit_id: str
    topic: str
    source_model: str
    prompt_version: str
    extraction_method: str = "LLM-Synthesis"
    max_rules: int = 40
    registry: CodeRegistry | None = None
    required_url_hosts: tuple[str, ...] = ()
    emitted: list[EmittedRule] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    # ---------------------------------------------------------------- tool surface
    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name="emit_rule",
            description=("Record ONE validated upcoding/compliance rule. Call once per rule. If the response is an error, "
                         "fix exactly the listed problems and call again. Never call it for a rule you could not verify "
                         "against the corpus."),
            input_schema=EMIT_RULE_SCHEMA,
            handler=self.handle,
        )

    def handle(self, args: Any) -> ToolResult:
        try:
            return self._handle(args)
        except Exception as exc:  # never let a bug here surface as a model-visible stack trace
            self._reject("internal_error", "")
            return ToolResult(f"emit_rule failed internally ({type(exc).__name__}); the rule was NOT recorded.", is_error=True)

    # ------------------------------------------------------------------ internals
    def _reject(self, reason: str, name: Any) -> None:
        self.rejected.append({"reason": reason[:300], "rule_name": str(name)[:120]})

    def _handle(self, args: Any) -> ToolResult:
        if not isinstance(args, dict):
            self._reject("arguments were not an object", "")
            return ToolResult("emit_rule expects a JSON object.", is_error=True)
        name = args.get("rule_name", "")
        if len(self.emitted) >= self.max_rules:
            self._reject("per-unit rule cap reached", name)
            return ToolResult(f"Limit reached: at most {self.max_rules} rules per unit. Stop emitting and finish.", is_error=True)

        problems: list[str] = []
        fields = {k: args[k] for k in MODEL_FIELDS if args.get(k) not in (None, "")}
        fields.update(job_id=self.job_id, unit_id=self.unit_id, topic=self.topic, source_model=self.source_model,
                      prompt_version=self.prompt_version, extraction_method=self.extraction_method)
        rule: Rule | None = None
        try:
            rule = validate_rule_dict(fields)
        except RuleValidationError as exc:
            problems.extend(exc.errors)
        executable: ExecutableRule | None = None
        if rule is not None:
            problems.extend(self._semantic_checks(rule, args))
            # Run DSL validation in the same pass so the model sees every problem in one round trip.
            executable, dsl_errors = build_executable(rule.rule_id, rule.recommended_action,
                                                      args.get("flag_condition"), args.get("correction_action"))
            problems.extend(dsl_errors)
        if rule is not None and not problems:
            key = rule.dedupe_key()
            if key in self._seen:
                problems.append("duplicate of a rule you already emitted in this session")
        if problems or rule is None:
            self._reject("; ".join(problems), name)
            return ToolResult("Rule NOT recorded. Fix these problems and call emit_rule again:\n- " + "\n- ".join(problems), is_error=True)

        self._seen.add(rule.dedupe_key())
        rule.validation_warnings = list(rule.validation_warnings) + self._unknown_code_warnings(rule)
        self.emitted.append(EmittedRule(rule=rule, executable=executable))
        note = "executable" if executable else "narrative-only"
        warn = f" warnings: {rule.validation_warnings}" if rule.validation_warnings else ""
        return ToolResult(f"Recorded rule #{len(self.emitted)} '{rule.rule_name}' ({note}).{warn}")

    def _semantic_checks(self, rule: Rule, args: dict[str, Any]) -> list[str]:
        problems = []
        if args.get("rule_shape") not in RULE_SHAPES:
            problems.append(f"rule_shape must be one of {RULE_SHAPES}")
        if not _CITATION.search(rule.source_reference):
            problems.append("source_reference must cite the document and page (for example 'acs printed p.101')")
        if self.required_url_hosts:
            hosts = {(urlsplit(u).hostname or "").lower() for u in _URL.findall(rule.source_reference)}
            if not hosts or not hosts <= {h.lower() for h in self.required_url_hosts}:
                problems.append(f"source_reference must include a URL on an allowed host {list(self.required_url_hosts)}")
        return problems

    def _unknown_code_warnings(self, rule: Rule) -> list[str]:
        reg = self.registry
        if reg is None or not reg.available:
            return []
        warnings: list[str] = []
        for label, text, known, plain in (("icd10am", rule.icd10am_codes, reg.icd10am, _ICD_PLAIN),
                                          ("achi", rule.achi_codes, reg.achi, _ACHI_PLAIN)):
            for token in _TOKEN_SPLIT.split(text.upper()):
                if token and token != "N/A" and plain.match(token) and known and canon_code(token) not in known:
                    warnings.append(f"unknown_{label}_code:{token}")
        if reg.mbs:
            for token in _TOKEN_SPLIT.split(rule.mbs_codes):
                if token.isdigit() and canon_code(token) not in reg.mbs:
                    warnings.append(f"unknown_mbs_item:{token}")
        return warnings[:20]
