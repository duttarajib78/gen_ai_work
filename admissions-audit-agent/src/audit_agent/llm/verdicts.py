"""Model tasks: per-diagnosis verdicts and the replacement-principal search.

Each task asks the model, validates the answer against what was actually asked,
and re-asks (up to ``max_attempts``) with a precise rejection reason when the
answer is unusable. Retries are fresh requests with the reason appended, not a
continued conversation, so no reasoning blocks ever have to be replayed.

Validation is strict because nothing upstream guarantees schema conformance on
this model: every expected code exactly once, no strays, roles consistent, enum
values exact, confidence a finite number (clamped to 0-1), reasoning non-empty.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from ..config import Settings
from ..deadline import Deadline
from ..errors import LlmOutputError
from ..logging_utils import log_event
from ..prompts import (
    REPLACEMENT_SYSTEM_PROMPT, REPLACEMENT_TOOL, SYSTEM_PROMPT, VERDICTS_TOOL,
    replacement_user_message, verdicts_user_message,
)
from ..textutil import clean_text
from .base import LlmClient, UsageMeter

logger = logging.getLogger(__name__)

REASONING_MAX_LEN = 8000
_REQUIRED_VERDICT_FIELDS = ("code", "role", "reasoning", "support_level", "confidence")
_LEVELS = ("none", "weak", "clear")


@dataclass(frozen=True)
class CodeVerdict:
    code: str
    role: str  # "PDx" | "ADx"
    support_level: str  # "none" | "weak" | "clear"
    supported: bool  # derived: anything other than "none"
    reasoning: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "code": self.code, "role": self.role, "support_level": self.support_level,
            "supported": self.supported, "reasoning": self.reasoning, "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ReplacementChoice:
    new_principal_code: Optional[str]
    reasoning: str


@dataclass(frozen=True)
class LlmTaskConfig:
    max_output_tokens: int = 16000
    max_output_tokens_ceiling: int = 64000
    max_attempts: int = 3

    @classmethod
    def from_settings(cls, settings: Settings) -> "LlmTaskConfig":
        return cls(
            max_output_tokens=settings.llm_max_output_tokens,
            max_output_tokens_ceiling=settings.llm_max_output_tokens_ceiling,
            max_attempts=settings.llm_max_output_attempts,
        )


class InvalidAnswer(ValueError):
    """The model answered, but not with something usable. ``category`` is a fixed
    label safe to log and return; ``str(exc)`` is detailed feedback for the model."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


# ---------------------------------------------------------------------------
def _confidence(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise InvalidAnswer("bad_confidence", f"{where} \"confidence\" must be a number from 0.0 to 1.0")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InvalidAnswer("bad_confidence", f"{where} \"confidence\" must be a number from 0.0 to 1.0") from None
    if not math.isfinite(number):
        raise InvalidAnswer("bad_confidence", f"{where} \"confidence\" must be a finite number")
    return max(0.0, min(1.0, number))


def verdicts_validator(pdx_code: str, adx_codes: Sequence[str]) -> Callable[[Mapping[str, Any]], list[CodeVerdict]]:
    expected = {pdx_code: "PDx", **{c: "ADx" for c in adx_codes}}

    def validate(args: Mapping[str, Any]) -> list[CodeVerdict]:
        entries = args.get("verdicts")
        if not isinstance(entries, list):
            raise InvalidAnswer("not_a_list", '"verdicts" must be an array')
        by_code: dict[str, CodeVerdict] = {}
        for i, entry in enumerate(entries):
            where = f"verdicts[{i}]"
            if not isinstance(entry, Mapping):
                raise InvalidAnswer("bad_entry", f"{where} must be an object")
            missing = [k for k in _REQUIRED_VERDICT_FIELDS if k not in entry]
            if missing:
                raise InvalidAnswer("missing_field", f"{where} is missing required field(s): {missing}")
            code = str(entry["code"]).strip().upper()
            if code not in expected:
                raise InvalidAnswer("stray_code", f"{where} answered about a code that is not on this episode")
            if code in by_code:
                raise InvalidAnswer("duplicate_code", f"{where} duplicates an earlier entry for code {code}")
            role = str(entry["role"]).strip()
            if role != expected[code]:
                raise InvalidAnswer("wrong_role", f"{where} gave code {code} role {role!r}; expected {expected[code]!r}")
            level = str(entry["support_level"]).strip().lower()
            if level not in _LEVELS:
                raise InvalidAnswer("bad_support_level", f'{where} "support_level" must be exactly "none", "weak", or "clear"')
            reasoning = clean_text(entry["reasoning"], REASONING_MAX_LEN)
            if not reasoning:
                raise InvalidAnswer("empty_reasoning", f'{where} "reasoning" must not be empty')
            by_code[code] = CodeVerdict(
                code=code, role=role, support_level=level, supported=(level != "none"),
                reasoning=reasoning, confidence=_confidence(entry["confidence"], where),
            )
        missing_codes = [c for c in expected if c not in by_code]
        if missing_codes:
            raise InvalidAnswer("missing_code", f"no entry was given for code(s): {missing_codes}")
        return [by_code[c] for c in expected]  # PDx first, then ADx in coded order

    return validate


def replacement_validator(candidates: Sequence[str]) -> Callable[[Mapping[str, Any]], ReplacementChoice]:
    allowed = set(candidates)

    def validate(args: Mapping[str, Any]) -> ReplacementChoice:
        missing = [k for k in ("new_principal_code", "reasoning") if k not in args]
        if missing:
            raise InvalidAnswer("missing_field", f"the answer is missing required field(s): {missing}")
        raw = args["new_principal_code"]
        code: Optional[str]
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "null", "none")):
            code = None
        else:
            code = str(raw).strip().upper()
            if code not in allowed:
                raise InvalidAnswer("not_a_candidate", f"{code} is not in the candidate list; choose from {sorted(allowed)} or null")
        reasoning = clean_text(args["reasoning"], 4000)
        if not reasoning:
            raise InvalidAnswer("empty_reasoning", '"reasoning" must not be empty')
        return ReplacementChoice(new_principal_code=code, reasoning=reasoning)

    return validate


# ---------------------------------------------------------------------------
def _ask(
    llm: LlmClient,
    *,
    system_prompt: str,
    user_builder: Callable[[Optional[str]], str],
    tool,
    validate: Callable[[Mapping[str, Any]], Any],
    deadline: Deadline,
    meter: UsageMeter,
    cfg: LlmTaskConfig,
    task: str,
) -> Any:
    feedback: Optional[str] = None
    tokens = cfg.max_output_tokens
    last_category = "none"
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            result = llm.invoke_tool(
                system_prompt=system_prompt, user_text=user_builder(feedback), tool=tool,
                deadline=deadline, max_output_tokens=tokens, meter=meter,
            )
            return validate(result.arguments)
        except LlmOutputError as exc:
            last_category = exc.code
            if exc.code == "llm_output_truncated":
                tokens = min(tokens * 2, cfg.max_output_tokens_ceiling)
                feedback = "the response was cut off; keep each reasoning field concise."
            else:
                feedback = str(exc)
        except InvalidAnswer as exc:
            last_category = exc.category
            feedback = str(exc)
        log_event(
            logger, "llm_answer_rejected", level=logging.WARNING, phase=task, attempt=attempt,
            reason=last_category, max_tokens=tokens,
        )
    raise LlmOutputError(
        f"the model did not return a usable answer after {cfg.max_attempts} attempt(s)",
        details={"task": task, "last_reason": last_category, "attempts": cfg.max_attempts},
    )


def get_diagnosis_verdicts(
    llm: LlmClient, *, episode_text: str, pdx_code: str, adx_codes: Sequence[str],
    deadline: Deadline, meter: UsageMeter, cfg: LlmTaskConfig,
) -> list[CodeVerdict]:
    return _ask(
        llm,
        system_prompt=SYSTEM_PROMPT,
        user_builder=lambda fb: verdicts_user_message(episode_text, pdx_code, adx_codes, fb),
        tool=VERDICTS_TOOL,
        validate=verdicts_validator(pdx_code, adx_codes),
        deadline=deadline, meter=meter, cfg=cfg, task="verdicts",
    )


def get_replacement_pdx(
    llm: LlmClient, *, episode_text: str, old_pdx: CodeVerdict, candidate_codes: Sequence[str],
    deadline: Deadline, meter: UsageMeter, cfg: LlmTaskConfig,
) -> ReplacementChoice:
    return _ask(
        llm,
        system_prompt=REPLACEMENT_SYSTEM_PROMPT,
        user_builder=lambda fb: replacement_user_message(episode_text, old_pdx.code, old_pdx.reasoning, candidate_codes, fb),
        tool=REPLACEMENT_TOOL,
        validate=replacement_validator(candidate_codes),
        deadline=deadline, meter=meter, cfg=cfg, task="replacement",
    )
