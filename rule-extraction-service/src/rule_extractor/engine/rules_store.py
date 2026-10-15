"""Executable-rule store (JSONL), joined to ``rule_catalogue.xlsx`` rows by ``rule_id``."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..util import atomic_write_text
from .actions import ACTION_FOR_RECOMMENDATION, validate_action
from .conditions import validate_condition

FLAG_ONLY = {"action": "flag_only"}


@dataclass
class ExecutableRule:
    rule_id: str
    flag_condition: dict[str, Any]
    correction_action: dict[str, Any]
    automatable: bool
    recommended_action: str = "flag_for_clinical_review"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExecutableRule":
        problems = validate_condition(raw.get("flag_condition")) + validate_action(raw.get("correction_action"))
        if problems:
            raise ValueError(f"invalid executable rule {raw.get('rule_id')!r}: " + "; ".join(problems))
        action = raw["correction_action"]
        return cls(
            rule_id=str(raw["rule_id"]),
            flag_condition=raw["flag_condition"],
            correction_action=action,
            automatable=bool(raw.get("automatable", False)) and action["action"] != "flag_only",
            recommended_action=str(raw.get("recommended_action", "flag_for_clinical_review")),
        )


def build_executable(rule_id: str, recommended_action: str, flag_condition: Any, correction_action: Any
                     ) -> tuple[ExecutableRule | None, list[str]]:
    """Validate a model-supplied condition/action pair.

    Returns ``(None, [])`` for a narrative-only rule, ``(None, errors)`` when the pair is invalid (the
    caller asks the model to retry), or ``(rule, [])``. ``flag_for_clinical_review`` always yields a
    flag-only action regardless of what the model sent - enforced here, not just requested in the prompt.
    """
    if flag_condition in (None, {}):
        return None, []
    errors = validate_condition(flag_condition)
    if recommended_action == "flag_for_clinical_review":
        action: Any = dict(FLAG_ONLY)
    else:
        if correction_action in (None, {}):
            errors.append(
                f"correction_action is required when recommended_action is '{recommended_action}'; "
                "provide it or use recommended_action 'flag_for_clinical_review'"
            )
            action = None
        else:
            action = correction_action
            errors.extend(validate_action(action))
            expected = ACTION_FOR_RECOMMENDATION.get(recommended_action)
            if not errors and expected and action["action"] != expected:
                errors.append(
                    f"correction_action.action '{action['action']}' does not implement "
                    f"recommended_action '{recommended_action}' (expected '{expected}')"
                )
    if errors:
        return None, errors
    return ExecutableRule(rule_id, flag_condition, action, action["action"] != "flag_only", recommended_action), []


def load_all(path: str | Path) -> list[ExecutableRule]:
    p = Path(path)
    if not p.is_file():
        return []
    rules: list[ExecutableRule] = []
    with open(p, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rules.append(ExecutableRule.from_dict(json.loads(line)))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{p}:{line_no}: {exc}") from exc
    return rules


def save_all(path: str | Path, rules: Iterable[ExecutableRule]) -> int:
    lines = [json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) for r in rules]
    atomic_write_text(path, "".join(line + "\n" for line in lines))
    return len(lines)


def remap_ids(rules: Iterable[ExecutableRule], id_map: Mapping[str, str]) -> list[ExecutableRule]:
    """Re-key rules after the Excel writer assigns sequential ``RULE_NNN`` ids (keeps the join correct)."""
    out = []
    for rule in rules:
        data = rule.to_dict()
        data["rule_id"] = id_map.get(rule.rule_id, rule.rule_id)
        out.append(ExecutableRule(**data))
    return out
