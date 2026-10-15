"""Grouper port: the interface rule evaluation uses to compare AR-DRG cost weights before/after a correction.

AR-DRG grouping is licensed classification software and is NOT reimplemented here. ``MockGrouperAdapter``
is a clearly-labelled stand-in (``is_mock=True``) so rule logic can be exercised end to end. To use the real
grouper, point ``RULE_EXTRACTOR_GROUPER_FACTORY`` at a ``module:callable`` that returns any object with a
``group(episode) -> GroupResult`` method (for example a factory wrapping your existing ``AuGrouperAdapter``);
no code in this repository changes.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .code_match import canonical
from .episode import Episode

_FACTORY_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


@dataclass
class GroupResult:
    drg_code: str | None
    cost_weight: float | None
    is_mock: bool
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GrouperAdapter(Protocol):
    def group(self, episode: Episode) -> GroupResult: ...


class MockGrouperAdapter:
    """NOT an AR-DRG grouper. The 'cost weight' is a proxy: the number of distinct diagnosis + procedure codes."""

    def group(self, episode: Episode) -> GroupResult:
        diags = {canonical(c) for c in episode.diagnosis_code_array}
        procs = {canonical(c) for c in episode.procedure_code_array}
        return GroupResult(drg_code="MOCK", cost_weight=float(len(diags) + len(procs)), is_mock=True,
                           details={"proxy": "distinct diagnosis + procedure code count"})


def load_grouper(factory: str | None) -> GrouperAdapter:
    """Resolve the configured grouper. ``factory`` comes from deployer-controlled configuration, never from requests."""
    if not factory:
        return MockGrouperAdapter()
    if not _FACTORY_RE.match(factory):
        raise ValueError("grouper factory must look like 'package.module:callable'")
    module_name, _, attr = factory.partition(":")
    adapter = getattr(importlib.import_module(module_name), attr)()
    if not isinstance(adapter, GrouperAdapter):
        raise TypeError(f"{factory} did not return an object with a group(episode) method")
    return adapter
