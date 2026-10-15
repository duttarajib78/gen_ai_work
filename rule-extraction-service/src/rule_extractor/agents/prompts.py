"""Prompt library: versioned Markdown templates rendered in a single pass.

* Templates live in ``agent/prompt_templates/*.md`` so they can be reviewed/diffed (and replaced with your own
  originals) without touching code. ``PROMPT_VERSION`` and a content digest are recorded with every rule.
* Rendering is one regex pass over the template only: untrusted text inserted into a placeholder is never
  re-scanned, so a document containing ``{{CODE}}`` cannot inject into later substitutions.
* Untrusted document text only ever appears in the USER prompt, between random-nonce markers.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from importlib import resources

from ..engine import grammar
from ..util import sha256_bytes

PROMPT_VERSION = "2026.10.1"
_PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")
_TEMPLATE_NAMES = (
    "_honesty.md", "_injection.md", "acs_topic.system.md", "acs_topic.user.md", "sweep_tabular.system.md",
    "sweep_alphabetic.system.md", "sweep.user.md", "web_research.system.md", "web_research.user.md",
)


def load_template(name: str) -> str:
    return (resources.files("rule_extractor.agents") / "prompt_templates" / name).read_text(encoding="utf-8")


def render(template: str, values: dict[str, str]) -> str:
    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"template placeholder {{{{{key}}}}} has no value")
        return values[key]

    return _PLACEHOLDER.sub(sub, template)


def prompt_digest() -> str:
    blob = "\n".join(load_template(n) for n in _TEMPLATE_NAMES).encode("utf-8")
    return sha256_bytes(blob)[:12]


def _strip_markers(text: str) -> str:
    return text.replace("UNTRUSTED_DOCUMENT", "UNTRUSTED-DOC")


@dataclass(frozen=True)
class PromptPair:
    system: str
    user: str


class PromptLibrary:
    def __init__(self) -> None:
        self._shared = {
            "HONESTY_RULES": load_template("_honesty.md").strip(),
            "INJECTION_DEFENCE": load_template("_injection.md").strip(),
            "CLAIMS_SCHEMA": grammar.render_claims_schema(),
            "DSL_REFERENCE": grammar.render_dsl_reference(),
        }

    def acs_topic(self, *, code: str, title: str, start_page: int, end_page: int, standard_text: str) -> PromptPair:
        system = render(load_template("acs_topic.system.md"), {
            **self._shared,
            "RULE_SHAPES": grammar.render_rule_shapes(),
            "EXAMPLES": grammar.render_examples(),
        })
        user = render(load_template("acs_topic.user.md"), {
            "CODE": code, "TITLE": title, "START_PAGE": f"{start_page:04d}", "END_PAGE": f"{end_page:04d}",
            "START_NUMBER": str(start_page), "END_NUMBER": str(end_page),
            "NONCE": secrets.token_hex(8), "DOCUMENT_TEXT": _strip_markers(standard_text),
        })
        return PromptPair(system, user)

    def sweep(self, *, doc_kind: str, doc_key: str, doc_label: str, start_page: int, end_page: int, window_text: str) -> PromptPair:
        template = "sweep_tabular.system.md" if doc_kind == "tabular" else "sweep_alphabetic.system.md"
        system = render(load_template(template), {
            **self._shared,
            "DOC_LABEL": doc_label,
            "RULE_SHAPES": grammar.render_rule_shapes(grammar.SWEEP_SHAPES),
            "EXAMPLES": grammar.render_examples(grammar.SWEEP_SHAPES),
        })
        user = render(load_template("sweep.user.md"), {
            "DOC_LABEL": doc_label, "DOC_KEY": doc_key, "START_NUMBER": str(start_page), "END_NUMBER": str(end_page),
            "NONCE": secrets.token_hex(8), "DOCUMENT_TEXT": _strip_markers(window_text),
        })
        return PromptPair(system, user)

    def web_research(self, *, topic: str, allowed_hosts: tuple[str, ...]) -> PromptPair:
        hosts = ", ".join(allowed_hosts)
        system = render(load_template("web_research.system.md"), {
            **self._shared, "ALLOWED_HOSTS": hosts, "RULE_SHAPES": grammar.render_rule_shapes(),
        })
        user = render(load_template("web_research.user.md"), {"TOPIC": topic, "ALLOWED_HOSTS": hosts})
        return PromptPair(system, user)
