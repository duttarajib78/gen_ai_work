"""Safe ``correction_action`` grammar and interpreter.

  {"action": "remove_code", "target": "diagnosis"|"procedure"|"mbs", "match": P,
   "which": "all"|"first"|"last"|"all_but_first"}                       (``which`` defaults to "all")
  {"action": "resequence_principal", "new_principal": P}
  {"action": "replace_code", "target": ..., "match": P, "with": "CODE", "which": "all"|"first"}
  {"action": "flag_only"}

``automatable=False`` is a hard invariant: the interpreter then never changes the episode, whatever the
action says (matches the template's "route to a clinical coding reviewer" instruction).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .code_match import matching_indices, normalize_patterns
from .episode import Episode

ACTIONS = ("remove_code", "resequence_principal", "replace_code", "flag_only")
TARGET_ARRAYS = {"diagnosis": "diagnosis_code_array", "procedure": "procedure_code_array", "mbs": "medical_claims_array"}
REMOVE_WHICH = ("all", "first", "last", "all_but_first")
REPLACE_WHICH = ("all", "first")
_CODE_RE = re.compile(r"^[A-Za-z0-9.\-]{1,16}$")

# recommended_action token -> the interpreter action that implements it
ACTION_FOR_RECOMMENDATION = {
    "remove_code": "remove_code",
    "resequence_principal_diagnosis": "resequence_principal",
    "correct_code": "replace_code",
    "flag_for_clinical_review": "flag_only",
}
_KEYS = {
    "remove_code": ({"action", "target", "match"}, {"which"}),
    "resequence_principal": ({"action", "new_principal"}, set()),
    "replace_code": ({"action", "target", "match", "with"}, {"which"}),
    "flag_only": ({"action"}, set()),
}


@dataclass
class ActionResult:
    episode: Episode
    changed: bool = False
    notes: list[str] = field(default_factory=list)


def validate_action(action: Any, path: str = "$") -> list[str]:
    if not isinstance(action, dict) or action.get("action") not in ACTIONS:
        return [f"{path}: 'action' must be one of {list(ACTIONS)}"]
    kind = action["action"]
    required, optional = _KEYS[kind]
    missing, extra = required - set(action), set(action) - required - optional
    errors: list[str] = []
    if missing:
        errors.append(f"{path}: missing key(s) {sorted(missing)} for {kind}")
    if extra:
        errors.append(f"{path}: unexpected key(s) {sorted(extra)} for {kind}")
    if errors:
        return errors
    if kind in ("remove_code", "replace_code"):
        if action["target"] not in TARGET_ARRAYS:
            errors.append(f"{path}.target: must be one of {sorted(TARGET_ARRAYS)}")
        errors.extend(_pattern_errors(action["match"], f"{path}.match"))
        allowed = REMOVE_WHICH if kind == "remove_code" else REPLACE_WHICH
        if action.get("which", "all") not in allowed:
            errors.append(f"{path}.which: must be one of {list(allowed)}")
    if kind == "replace_code" and (not isinstance(action["with"], str) or not _CODE_RE.match(action["with"]) or "*" in action["with"]):
        errors.append(f"{path}.with: must be one exact code (no wildcards)")
    if kind == "resequence_principal":
        errors.extend(_pattern_errors(action["new_principal"], f"{path}.new_principal"))
    return errors


def _pattern_errors(value: Any, path: str) -> list[str]:
    try:
        normalize_patterns(value)
    except ValueError as exc:
        return [f"{path}: {exc}"]
    return []


def apply_action(action: dict[str, Any], episode: Episode, *, automatable: bool) -> ActionResult:
    """Return a corrected COPY of ``episode``; the input is never mutated."""
    problems = validate_action(action)
    if problems:
        raise ValueError("; ".join(problems))
    result = ActionResult(episode=episode.copy())
    kind = action["action"]
    if kind == "flag_only":
        result.notes.append("flag only: route to a clinical coding reviewer")
        return result
    if not automatable:
        result.notes.append(f"rule is not automatable: {kind} not applied")
        return result
    ep = result.episode
    if kind == "remove_code":
        arr_name = TARGET_ARRAYS[action["target"]]
        arr = getattr(ep, arr_name)
        idxs = matching_indices(arr, normalize_patterns(action["match"]))
        which = action.get("which", "all")
        chosen = {"all": idxs, "first": idxs[:1], "last": idxs[-1:], "all_but_first": idxs[1:]}[which]
        _delete_indices(ep, arr_name, set(chosen))
        result.changed = bool(chosen)
        result.notes.append(f"removed {len(chosen)} {action['target']} code(s)")
    elif kind == "replace_code":
        arr_name = TARGET_ARRAYS[action["target"]]
        arr = getattr(ep, arr_name)
        idxs = matching_indices(arr, normalize_patterns(action["match"]))
        if action.get("which", "all") == "first":
            idxs = idxs[:1]
        for i in idxs:
            arr[i] = action["with"]
        result.changed = bool(idxs)
        result.notes.append(f"replaced {len(idxs)} {action['target']} code(s) with {action['with']}")
    elif kind == "resequence_principal":
        pats = normalize_patterns(action["new_principal"])
        idxs = [i for i in matching_indices(ep.diagnosis_code_array, pats) if i >= 1]
        if idxs:
            i = idxs[0]
            ep.diagnosis_code_array.insert(0, ep.diagnosis_code_array.pop(i))
            if len(ep.condition_onset_array) == len(ep.diagnosis_code_array):
                ep.condition_onset_array.insert(0, ep.condition_onset_array.pop(i))
            result.changed = True
            result.notes.append("resequenced principal diagnosis")
        else:
            result.notes.append("no secondary diagnosis matched the new principal pattern")
    ep.principal_diagnosis_code = ep.diagnosis_code_array[0] if ep.diagnosis_code_array else None
    return result


def _delete_indices(ep: Episode, arr_name: str, indices: set[int]) -> None:
    arr = getattr(ep, arr_name)
    aligned = arr_name == "diagnosis_code_array" and len(ep.condition_onset_array) == len(arr)
    keep = [i for i in range(len(arr)) if i not in indices]
    setattr(ep, arr_name, [arr[i] for i in keep])
    if aligned:
        ep.condition_onset_array = [ep.condition_onset_array[i] for i in keep]
