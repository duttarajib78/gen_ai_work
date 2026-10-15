"""Safe ``flag_condition`` grammar and interpreter.

Every node is an object with exactly ONE operator key:

  logical : {"all": [node, ...]}  {"any": [node, ...]}  {"not": node}
  codes   : {"has_diagnosis": P}  {"has_procedure": P}  {"has_mbs_item": P}
            {"principal_is": P}   {"secondary_has": P}          (P = pattern or list of patterns)
  count   : {"count": {"in": "diagnosis"|"procedure"|"mbs", "match": P, "op": CMP, "value": int}}
  scalar  : {"field": {"name": FIELD, "op": OP, "value": v}}
  onset   : {"onset": {"match": P, "flag": "1"|"2"}}
  order   : {"sequence": {"first": P, "then": P}}    (a code matching `first` is sequenced before one matching `then`)

Missing data never produces a flag: comparisons against a null scalar evaluate to False (except is_null).
"""
from __future__ import annotations

import operator
from typing import Any, Callable

from .code_match import any_match, matching_indices, normalize_patterns
from .episode import SCALAR_FIELDS, Episode

MAX_DEPTH = 8
MAX_NODES = 64
_CMP: dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}
COUNT_TARGETS = {"diagnosis": "diagnosis_code_array", "procedure": "procedure_code_array", "mbs": "medical_claims_array"}
CODE_OPERATORS = ("has_diagnosis", "has_procedure", "has_mbs_item", "principal_is", "secondary_has")
OPERATORS = ("all", "any", "not") + CODE_OPERATORS + ("count", "field", "onset", "sequence")


class ConditionError(ValueError):
    pass


# --------------------------------------------------------------------------- validation
def validate_condition(node: Any, path: str = "$") -> list[str]:
    budget = [0]
    return _validate(node, path, 1, budget)


def _validate(node: Any, path: str, depth: int, budget: list[int]) -> list[str]:
    budget[0] += 1
    if budget[0] > MAX_NODES:
        return [f"{path}: condition has more than {MAX_NODES} nodes"]
    if depth > MAX_DEPTH:
        return [f"{path}: condition nesting deeper than {MAX_DEPTH}"]
    if not isinstance(node, dict) or len(node) != 1:
        return [f"{path}: each condition node must be an object with exactly one operator key (one of {list(OPERATORS)})"]
    (op, value), = node.items()
    here = f"{path}.{op}"
    if op in ("all", "any"):
        if not isinstance(value, list) or not value or len(value) > 16:
            return [f"{here}: must be a non-empty list of at most 16 conditions"]
        errors: list[str] = []
        for i, child in enumerate(value):
            errors.extend(_validate(child, f"{here}[{i}]", depth + 1, budget))
        return errors
    if op == "not":
        return _validate(value, here, depth + 1, budget)
    if op in CODE_OPERATORS:
        return _validate_patterns(value, here)
    if op == "count":
        return _validate_count(value, here)
    if op == "field":
        return _validate_field(value, here)
    if op == "onset":
        if not isinstance(value, dict) or set(value) != {"match", "flag"}:
            return [f"{here}: must be an object with keys 'match' and 'flag'"]
        errors = _validate_patterns(value["match"], f"{here}.match")
        if value["flag"] not in ("1", "2"):
            errors.append(f"{here}.flag: must be the string '1' or '2'")
        return errors
    if op == "sequence":
        if not isinstance(value, dict) or set(value) != {"first", "then"}:
            return [f"{here}: must be an object with keys 'first' and 'then'"]
        return _validate_patterns(value["first"], f"{here}.first") + _validate_patterns(value["then"], f"{here}.then")
    return [f"{path}: unknown operator {op!r}; allowed: {list(OPERATORS)}"]


def _validate_patterns(value: Any, path: str) -> list[str]:
    try:
        normalize_patterns(value)
    except ValueError as exc:
        return [f"{path}: {exc}"]
    return []


def _validate_count(value: Any, path: str) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"in", "match", "op", "value"}:
        return [f"{path}: must be an object with keys 'in', 'match', 'op', 'value'"]
    errors = _validate_patterns(value["match"], f"{path}.match")
    if value["in"] not in COUNT_TARGETS:
        errors.append(f"{path}.in: must be one of {sorted(COUNT_TARGETS)}")
    if value["op"] not in _CMP:
        errors.append(f"{path}.op: must be one of {sorted(_CMP)}")
    if isinstance(value["value"], bool) or not isinstance(value["value"], int) or not 0 <= value["value"] <= 1000:
        errors.append(f"{path}.value: must be an integer between 0 and 1000")
    return errors


def _validate_field(value: Any, path: str) -> list[str]:
    if not isinstance(value, dict) or not {"name", "op"} <= set(value) or not set(value) <= {"name", "op", "value"}:
        return [f"{path}: must be an object with keys 'name', 'op' and (except for is_null/not_null) 'value'"]
    name, op = value["name"], value["op"]
    if name not in SCALAR_FIELDS:
        return [f"{path}.name: must be one of {sorted(SCALAR_FIELDS)}"]
    kind = SCALAR_FIELDS[name]
    allowed_ops = sorted(_CMP) + ["in", "is_null", "not_null"]
    if op not in allowed_ops:
        return [f"{path}.op: must be one of {allowed_ops}"]
    if op in ("is_null", "not_null"):
        return [] if "value" not in value else [f"{path}: 'value' is not allowed with {op}"]
    if "value" not in value:
        return [f"{path}: 'value' is required with {op}"]
    sample = value["value"]
    items = sample if op == "in" else [sample]
    if op == "in" and (not isinstance(sample, list) or not sample or len(sample) > 20):
        return [f"{path}.value: 'in' needs a list of 1-20 values"]
    for item in items:
        if kind == "number" and (isinstance(item, bool) or not isinstance(item, (int, float))):
            return [f"{path}.value: {name} is numeric"]
        if kind == "string" and not isinstance(item, str):
            return [f"{path}.value: {name} is a string"]
    if kind == "string" and op not in ("==", "!=", "in"):
        return [f"{path}.op: string field {name} only supports ==, != and in"]
    return []


# ---------------------------------------------------------------------------- evaluation
def evaluate(node: Any, episode: Episode) -> bool:
    problems = validate_condition(node)
    if problems:
        raise ConditionError("; ".join(problems))
    return _eval(node, episode)


def _eval(node: dict[str, Any], ep: Episode) -> bool:
    (op, value), = node.items()
    if op == "all":
        return all(_eval(c, ep) for c in value)
    if op == "any":
        return any(_eval(c, ep) for c in value)
    if op == "not":
        return not _eval(value, ep)
    if op == "has_diagnosis":
        pats = normalize_patterns(value)
        return any(any_match(c, pats) for c in ep.diagnosis_code_array)
    if op == "has_procedure":
        pats = normalize_patterns(value)
        return any(any_match(c, pats) for c in ep.procedure_code_array)
    if op == "has_mbs_item":
        pats = normalize_patterns(value)
        return any(any_match(c, pats) for c in ep.medical_claims_array)
    if op == "principal_is":
        pats = normalize_patterns(value)
        return bool(ep.principal_diagnosis_code) and any_match(ep.principal_diagnosis_code, pats)
    if op == "secondary_has":
        pats = normalize_patterns(value)
        return any(any_match(c, pats) for c in ep.diagnosis_code_array[1:])
    if op == "count":
        codes = getattr(ep, COUNT_TARGETS[value["in"]])
        n = len(matching_indices(codes, normalize_patterns(value["match"])))
        return _CMP[value["op"]](n, value["value"])
    if op == "field":
        return _eval_field(value, ep)
    if op == "onset":
        pats = normalize_patterns(value["match"])
        onset = ep.condition_onset_array
        return any(i < len(onset) and onset[i] == value["flag"] for i in matching_indices(ep.diagnosis_code_array, pats))
    if op == "sequence":
        first = matching_indices(ep.diagnosis_code_array, normalize_patterns(value["first"]))
        then = matching_indices(ep.diagnosis_code_array, normalize_patterns(value["then"]))
        return bool(first) and bool(then) and min(first) < max(then)
    raise ConditionError(f"unknown operator {op!r}")


def _eval_field(spec: dict[str, Any], ep: Episode) -> bool:
    actual = getattr(ep, spec["name"])
    op = spec["op"]
    if op == "is_null":
        return actual is None
    if op == "not_null":
        return actual is not None
    if actual is None:
        return False
    expected = spec["value"]
    if op == "in":
        return actual in expected
    if SCALAR_FIELDS[spec["name"]] == "string":
        return _CMP[op](str(actual).upper(), str(expected).upper())
    return _CMP[op](float(actual), float(expected))
