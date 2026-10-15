"""Test doubles and fixtures shared by the suite (also used by tools/dev_server.py). Not shipped in the image."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from audit_agent.config import Settings
from audit_agent.deadline import Deadline
from audit_agent.grouper.interface import RegroupResult
from audit_agent.llm.base import LlmUsage, ToolCallResult, UsageMeter

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BASE_ENV = {
    "AUDIT_AWS_REGION": "ap-southeast-2",
    "AUDIT_BEDROCK_MODEL_ID": "global.anthropic.claude-sonnet-5-5",
    "AUDIT_ALLOW_GLOBAL_INFERENCE": "true",
    "AUDIT_MODEL_DIR": "/nonexistent",
    "AUDIT_GROUPER_BACKEND": "disabled",
}


def make_settings(**overrides: str) -> Settings:
    return Settings.from_mapping({**BASE_ENV, **overrides})


# ---------------------------------------------------------------------------
# Reference data fixture
# ---------------------------------------------------------------------------
DIAGNOSES = {
    "J189": "Pneumonia, unspecified",
    "I509": "Heart failure, unspecified",
    "N179": "Acute kidney failure, unspecified",
    "E119": "Type 2 diabetes mellitus without complications",
    "Y8313": "Surgical operation with anastomosis as cause of abnormal reaction",
}
ACHI = {"3237500": "Intravenous administration of pharmacological agent", "9200900": "Chest physiotherapy"}
MBS = {"3": "Professional attendance by a general practitioner", "23": "Consultation (superseded)"}


def _tsv(path: Path, code_type: str, rows: dict[str, str]) -> None:
    lines = [f"{i}\t{code_type}\t{code}\t12\tx\ty\t{desc}\tz" for i, (code, desc) in enumerate(rows.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mbs_xml(path: Path) -> None:
    items = []
    for num, desc in MBS.items():
        end = "<ItemEndDate>01.07.2020</ItemEndDate>" if num == "23" else "<ItemEndDate></ItemEndDate>"
        items.append(
            f"<Data><ItemNum>{num}</ItemNum><Description>{desc}</Description><Category>1</Category>"
            f"<Group>A1</Group><SubGroup>1</SubGroup><SubHeading>x</SubHeading>{end}<DerivedFee></DerivedFee></Data>"
        )
    path.write_text('<?xml version="1.0" encoding="utf-8"?><MBS_XML>' + "".join(items) + "</MBS_XML>", encoding="utf-8")


def make_reference_dir(root: Path) -> Path:
    """Writes the three reference files directly into ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    _tsv(root / "diagnosis_codes.tsv", "ICD10_DIAGNOSIS", DIAGNOSES)
    _tsv(root / "achi_codes.tsv", "ICD10_PROCEDURE", ACHI)
    _mbs_xml(root / "MBS-XML-20260801.XML")
    return root


def write_manifest(model_dir: Path) -> str:
    files = {}
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rel = path.relative_to(model_dir).as_posix()
            files[rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    (model_dir / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}), encoding="utf-8")
    return "ok"


def make_model_dir(root: Path) -> Path:
    make_reference_dir(root / "reference")
    write_manifest(root)
    return root


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
SAMPLE_EPISODE: dict[str, Any] = {
    "admission_no": 555123,
    "fund_id": 101,
    "provider_id": "0000123X",
    "drg": "E62B",
    "drg_vers": "100",
    "los": 5,
    "hos_benefit": 9200.0,
    "hospital_claims_seqno_array": [1001, 1002],
    "hospital_claims_service_class_code_array": ["DRG_CASE_PAYMENT", "OTHER"],
    "hospital_claims_array": ["E62B", "X"],
    "diagnosis_code_array": ["J189", "I509", "N179", "E119"],
    "diagnosis_poa_array": [1, 1, 0, 1],
    "condition_onset_array": ["ADMIT", "ADMIT", "DURING", "ADMIT"],
    "dx_code_idx_array": [0, 1, 2, 3],
    "procedure_code_array": ["3237500", "9200900"],
    "procedure_date_array": ["2026-08-02", "2026-08-03"],
    "medical_claims_array": ["3", "23"],
    "dte_admitted": "2026-08-01",
    "dte_separated": "2026-08-06",
    "urgency_of_admission": "Emergency",
    "mode_of_separation": 1,
    "hours_of_mechanical_ventilation": 0,
    "minutes_operating_theatre": 0,
    "infant_weight": None,
    "source_of_referral": 2,
    "member_age": 74,
    "member_sex": "F",
    "member_dob": "1952-01-01",
}


def sample_episode(**overrides: Any) -> dict:
    episode = copy.deepcopy(SAMPLE_EPISODE)
    episode.update(overrides)
    return episode


def sample_payload(options: Optional[dict] = None, **episode_overrides: Any) -> dict:
    payload: dict[str, Any] = {"request_id": "req-1", "episode": sample_episode(**episode_overrides)}
    if options is not None:
        payload["options"] = options
    return payload


# ---------------------------------------------------------------------------
# LLM fakes
# ---------------------------------------------------------------------------
_CODE_LINE = re.compile(r'^\s+"([A-Z0-9.\-]+)" \((PDx|ADx)\)$', re.M)
_CANDIDATE_LINE = re.compile(r"^  ([A-Z0-9.\-]+)$", re.M)


class PolicyLlm:
    """Answers verdict/replacement requests from a per-code policy.

    ``levels``: code -> (support_level, confidence); codes not listed get
    ``default``. ``replacement``: code to choose in the replacement search (None = none fit).
    """

    def __init__(
        self,
        levels: Optional[dict[str, tuple[str, float]]] = None,
        *,
        default: tuple[str, float] = ("clear", 0.9),
        replacement: Optional[str] = None,
        usage: LlmUsage = LlmUsage(input_tokens=1200, output_tokens=800, latency_ms=2500),
    ) -> None:
        self.levels = levels or {}
        self.default = default
        self.replacement = replacement
        self.usage = usage
        self.calls: list[dict] = []

    def invoke_tool(self, *, system_prompt, user_text, tool, deadline, max_output_tokens=None, meter=None):
        self.calls.append({"tool": tool.name, "user_text": user_text, "max_output_tokens": max_output_tokens})
        if tool.name == "submit_diagnosis_verdicts":
            verdicts = []
            for code, role in _CODE_LINE.findall(user_text):
                level, conf = self.levels.get(code, self.default)
                verdicts.append({
                    "code": code, "role": role, "support_level": level, "confidence": conf,
                    "reasoning": f"Weighed the ACHI/MBS codes for {code}; judged {level}.",
                })
            args: dict[str, Any] = {"verdicts": verdicts}
        else:
            args = {"new_principal_code": self.replacement, "reasoning": "Considered each candidate against the evidence."}
        if meter is not None:
            meter.add(self.usage)
        return ToolCallResult(tool_name=tool.name, arguments=args, usage=self.usage)


class ScriptedLlm:
    """Returns/raises queued items, one per call (callables receive the user text)."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def invoke_tool(self, *, system_prompt, user_text, tool, deadline, max_output_tokens=None, meter=None):
        self.calls.append({"tool": tool.name, "user_text": user_text, "max_output_tokens": max_output_tokens})
        item = self.script.pop(0)
        if callable(item):
            item = item(user_text)
        if isinstance(item, BaseException):
            raise item
        usage = LlmUsage(input_tokens=100, output_tokens=50, latency_ms=100)
        if meter is not None:
            meter.add(usage)
        return ToolCallResult(tool_name=tool.name, arguments=item, usage=usage)


class FakeClientError(Exception):
    """Mimics botocore.exceptions.ClientError's ``response`` attribute."""

    def __init__(self, code: str, status: int = 400, message: str = "boom", request_id: str = "req-aws-1") -> None:
        super().__init__(f"{code}: {message}")
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status, "RequestId": request_id},
        }


class FakeBedrockRuntime:
    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    def converse(self, **kwargs: Any) -> dict:
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def converse_response(arguments: Any, *, tool_name: str = "submit_diagnosis_verdicts", stop: str = "tool_use",
                      with_reasoning: bool = True, usage: Optional[dict] = None) -> dict:
    content: list[dict] = []
    if with_reasoning:
        content.append({"reasoningContent": {"reasoningText": {"text": "", "signature": "sig"}}})
    content.append({"toolUse": {"toolUseId": "tooluse_1", "name": tool_name, "input": arguments}})
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop,
        "usage": usage or {"inputTokens": 1000, "outputTokens": 500, "totalTokens": 1500},
        "metrics": {"latencyMs": 1234},
    }


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------------------
# Grouper stub
# ---------------------------------------------------------------------------
class StubGrouper:
    """Deterministic regroup: any correction moves E62B (0.70) to E62A (0.55)."""

    def __init__(self, fail_with: Optional[BaseException] = None) -> None:
        self.calls: list[dict] = []
        self.fail_with = fail_with

    def regroup(self, episode, *, removed_codes, pdx_resequence, deadline):
        self.calls.append({"removed": set(removed_codes), "resequence": pdx_resequence})
        if self.fail_with:
            raise self.fail_with
        return RegroupResult(
            codes_removed=tuple(sorted(removed_codes)), pdx_resequence=pdx_resequence,
            original_drg=episode.extras.get("drg"), original_drg_desc="Respiratory infections W CC",
            original_cost_weight=0.70, regrouped_drg="E62A", regrouped_drg_desc="Respiratory infections W/O CC",
            regrouped_cost_weight=0.55, drg_changed=True, cost_weight_drop=0.15,
            possible_recovery=1971.43, drg_ver_grouper="100",
        )


def new_deadline(budget: float = 50.0) -> Deadline:
    return Deadline(budget)


def new_meter() -> UsageMeter:
    return UsageMeter()
