"""Single source of truth for the vocabulary shown to the model: claims schema, rule shapes, DSL reference,
and worked examples. Prompts are rendered from here and the unit tests validate every example against the
real validators, so documentation and interpreter cannot drift apart.
"""
from __future__ import annotations

import json

from .actions import ACTIONS, REMOVE_WHICH, REPLACE_WHICH
from .conditions import OPERATORS
from .episode import CLAIMS_SCHEMA_FIELDS, SCALAR_FIELDS

RULE_SHAPE_DESCRIPTIONS: dict[str, str] = {
    "redundancy": "A code is assigned redundantly alongside another code that already includes or implies it "
                  "(Excludes notes, 'organism as cause' codes next to a code naming the organism). The extra code adds "
                  "complexity without independent clinical meaning.",
    "inherent_component": "A code describes something that is an inherent component of another code on the same "
                          "episode (unbundling), typically signalled by an Includes/Excludes note.",
    "sequencing": "A sequencing directive says a code must not be (or must be) the principal diagnosis, or must follow "
                  "another code; wrong sequencing can move the episode into a higher-weighted DRG.",
    "evidence_requirement": "A code is only supportable when other claim evidence exists (procedure codes, ventilation "
                            "hours, theatre minutes, prosthesis benefit); absent evidence suggests upcoding.",
    "must_never_assign": "An ACS directive states a code must never be assigned in a context (or together with another "
                         "code), ideally corroborated by a Tabular List flag.",
    "once_only": "A code may be assigned once per episode but appears more than once.",
    "demographic_implausibility": "A code is implausible for the member's sex or age.",
}
SWEEP_SHAPES = ("redundancy", "inherent_component")

EXAMPLES: list[dict] = [
    {
        "shape": "redundancy",
        "title": "Organism coded redundantly alongside a Chapter 1 code that already names it (illustrative codes)",
        "recommended_action": "remove_code",
        "flag_condition": {"all": [{"has_diagnosis": "A41.0"}, {"has_diagnosis": "B95.6"}]},
        "correction_action": {"action": "remove_code", "target": "diagnosis", "match": "B95.6"},
    },
    {
        "shape": "inherent_component",
        "title": "Procedure coded together with its own biopsy variant (illustrative codes)",
        "recommended_action": "flag_for_clinical_review",
        "flag_condition": {"all": [{"has_procedure": "40903-00"}, {"has_procedure": "40903-01"}]},
        "correction_action": {"action": "flag_only"},
    },
    {
        "shape": "sequencing",
        "title": "Asterisk (manifestation) code sequenced as principal diagnosis (illustrative codes)",
        "recommended_action": "resequence_principal_diagnosis",
        "flag_condition": {"all": [{"principal_is": "G01*"}, {"secondary_has": "A39.0"}]},
        "correction_action": {"action": "resequence_principal", "new_principal": "A39.0"},
    },
    {
        "shape": "evidence_requirement",
        "title": "Ventilation-duration code not supported by the recorded hours (illustrative codes)",
        "recommended_action": "correct_code",
        "flag_condition": {"all": [
            {"has_procedure": "13882-02"},
            {"field": {"name": "hours_of_mechanical_ventilation", "op": ">=", "value": 24}},
            {"field": {"name": "hours_of_mechanical_ventilation", "op": "<", "value": 96}},
        ]},
        "correction_action": {"action": "replace_code", "target": "procedure", "match": "13882-02", "with": "13882-01"},
    },
    {
        "shape": "must_never_assign",
        "title": "Two codes an ACS directive says are never assigned together (illustrative codes)",
        "recommended_action": "remove_code",
        "flag_condition": {"all": [{"has_diagnosis": "E10.*"}, {"has_diagnosis": "E11.*"}]},
        "correction_action": {"action": "remove_code", "target": "diagnosis", "match": "E11.*"},
    },
    {
        "shape": "once_only",
        "title": "Code assignable once per episode appears more than once (illustrative codes)",
        "recommended_action": "remove_code",
        "flag_condition": {"count": {"in": "diagnosis", "match": "Z37.*", "op": ">", "value": 1}},
        "correction_action": {"action": "remove_code", "target": "diagnosis", "match": "Z37.*", "which": "all_but_first"},
    },
    {
        "shape": "demographic_implausibility",
        "title": "Obstetric code on a male member (illustrative codes)",
        "recommended_action": "flag_for_clinical_review",
        "flag_condition": {"all": [{"has_diagnosis": "O80*"}, {"field": {"name": "member_sex", "op": "==", "value": "M"}}]},
        "correction_action": {"action": "flag_only"},
    },
]


def render_claims_schema() -> str:
    lines = [f"- `{name}` ({kind}): {desc}" for name, (kind, desc) in CLAIMS_SCHEMA_FIELDS.items()]
    return "\n".join(lines)


def render_rule_shapes(only: tuple[str, ...] | None = None) -> str:
    names = [n for n in RULE_SHAPE_DESCRIPTIONS if only is None or n in only]
    return "\n".join(f"- `{n}`: {RULE_SHAPE_DESCRIPTIONS[n]}" for n in names)


def render_dsl_reference() -> str:
    scalar = ", ".join(f"`{k}` ({v})" for k, v in SCALAR_FIELDS.items())
    return "\n".join([
        "Conditions (`flag_condition`) - each node is an object with exactly ONE operator key.",
        f"Operators: {', '.join('`%s`' % o for o in OPERATORS)}.",
        "- Logical: `{\"all\": [..]}`, `{\"any\": [..]}`, `{\"not\": node}`.",
        "- Code tests (pattern = exact code, prefix with trailing `*`, or wildcard with `*`/`?`; one pattern or a list):",
        "  `has_diagnosis`, `has_procedure`, `has_mbs_item`, `principal_is`, `secondary_has`.",
        "- Count: `{\"count\": {\"in\": \"diagnosis|procedure|mbs\", \"match\": P, \"op\": \"==|!=|<|<=|>|>=\", \"value\": N}}`.",
        f"- Scalar: `{{\"field\": {{\"name\": F, \"op\": \"==|!=|<|<=|>|>=|in|is_null|not_null\", \"value\": v}}}}` where F is one of {scalar}.",
        "- Onset: `{\"onset\": {\"match\": P, \"flag\": \"1\"|\"2\"}}`.",
        "- Order: `{\"sequence\": {\"first\": P, \"then\": P}}` (a `first` code is sequenced before a `then` code).",
        "",
        "Actions (`correction_action`):",
        f"- Allowed `action` values: {', '.join('`%s`' % a for a in ACTIONS)}.",
        f"- `remove_code`: `target` (diagnosis|procedure|mbs), `match` P, optional `which` in {list(REMOVE_WHICH)}.",
        "- `resequence_principal`: `new_principal` P (a secondary diagnosis is moved to the principal position).",
        f"- `replace_code`: `target`, `match` P, `with` (one exact code), optional `which` in {list(REPLACE_WHICH)}.",
        "- `flag_only`: no change to the episode.",
        "",
        "Mapping: recommended_action `remove_code` -> action `remove_code`; `resequence_principal_diagnosis` -> "
        "`resequence_principal`; `correct_code` -> `replace_code`; `flag_for_clinical_review` -> `flag_only` "
        "(enforced by the system regardless of what you send).",
        "Limits: at most 64 condition nodes, nesting depth 8.",
    ])


def render_examples(only_shapes: tuple[str, ...] | None = None) -> str:
    blocks = []
    for ex in EXAMPLES:
        if only_shapes is not None and ex["shape"] not in only_shapes:
            continue
        blocks.append(
            f"### {ex['shape']}: {ex['title']}\n"
            f"recommended_action: {ex['recommended_action']}\n"
            f"flag_condition: {json.dumps(ex['flag_condition'])}\n"
            f"correction_action: {json.dumps(ex['correction_action'])}"
        )
    return "\n\n".join(blocks)
