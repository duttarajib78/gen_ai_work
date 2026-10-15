"""Shared test helpers."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from dx_support.constants import DEFAULT_FAMILIES
from dx_support.llm.fake import FakeBackend
from dx_support.llm.gate import LLMGate
from dx_support.pipeline.context import RunContext
from dx_support.sample_data import synthetic_universe
from dx_support.storage.checkpoint import Checkpoint, latest_by_key
from dx_support.storage.objectstore import LocalObjectStore


def make_ctx(root, universe=None, backend=None, *, mode="routed", cap=1e6, max_retries=2, families=DEFAULT_FAMILIES, include_context=False,
             brief_skip=False, concurrency=8):
    universe = universe or synthetic_universe()
    store = LocalObjectStore(Path(root))
    ckpt = Checkpoint(store)
    gate = LLMGate(backend or FakeBackend(), concurrency, cap, run_log=ckpt.log_run, backoff_base_s=0.0, max_retries=max_retries)
    ctx = RunContext(categories=universe.categories, indices=universe.indices, gate=gate, ckpt=ckpt, families=tuple(families), mode=mode,
                     include_context_chapters=include_context, brief_skip=brief_skip)
    return ctx, store


def snapshot(store) -> dict:
    ck = Checkpoint(store)
    try:
        return {k: (v["support_tier"], v["evidence_type"]) for k, v in latest_by_key(ck.iter_verdicts()).items()}
    finally:
        ck.close()


def n_verdict_lines(store) -> int:
    ck = Checkpoint(store)
    try:
        return sum(1 for _ in ck.iter_verdicts())
    finally:
        ck.close()


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="dx_test_"))


def run(coro):
    return asyncio.run(coro)
