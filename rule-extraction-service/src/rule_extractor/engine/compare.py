"""Evaluate -> correct -> regroup -> report, for one rule/episode or a whole rule set.

Results are a PREVIEW against an unreviewed correction, never a confirmed outcome, and nothing here writes
back to a real claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .actions import apply_action
from .conditions import evaluate
from .episode import Episode
from .grouper import GrouperAdapter, GroupResult
from .rules_store import ExecutableRule


@dataclass
class RuleRunResult:
    rule_id: str
    was_flagged: bool
    corrected: bool = False
    automatable: bool = False
    before: GroupResult | None = None
    after: GroupResult | None = None
    cost_weight_delta: float | None = None     # negative = the correction lowers the cost weight
    preview_only: bool = True
    notes: list[str] = field(default_factory=list)


def run_episode_against_rule(episode: Episode, rule: ExecutableRule, grouper: GrouperAdapter) -> RuleRunResult:
    if not evaluate(rule.flag_condition, episode):
        return RuleRunResult(rule_id=rule.rule_id, was_flagged=False, automatable=rule.automatable)
    outcome = apply_action(rule.correction_action, episode, automatable=rule.automatable)
    result = RuleRunResult(rule_id=rule.rule_id, was_flagged=True, corrected=outcome.changed,
                           automatable=rule.automatable, notes=list(outcome.notes))
    if outcome.changed:
        result.before = grouper.group(episode)
        result.after = grouper.group(outcome.episode)
        if result.before.cost_weight is not None and result.after.cost_weight is not None:
            result.cost_weight_delta = round(result.after.cost_weight - result.before.cost_weight, 6)
        if result.before.is_mock or result.after.is_mock:
            result.notes.append("cost weights come from the MOCK grouper (proxy count), not AR-DRG")
    return result


def run_episode_against_ruleset(episode: Episode, rules: Iterable[ExecutableRule], grouper: GrouperAdapter
                                ) -> Iterator[RuleRunResult]:
    for rule in rules:
        yield run_episode_against_rule(episode, rule, grouper)
