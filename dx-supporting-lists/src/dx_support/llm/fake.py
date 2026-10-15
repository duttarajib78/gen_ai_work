"""An offline, deterministic, NON-CLINICAL stand-in for a model.

Used by `--dry-run`, `dx-support serve-local --backend fake`, and the tests, so the whole pipeline (routing, checkpoints, resume, compile, the
HTTP service) can be exercised with no credentials and no spend. It decides purely from word overlap between the category text and each
node / unit text (the same tokenizer as the lexical index). Its verdicts carry NO clinical meaning, and production refuses to start with it.
"""
from __future__ import annotations

import asyncio
import random
import re

from dx_support.indexing.bm25 import tokenize
from dx_support.llm.types import LLMCallError, LLMRequest, LLMResult


class FakeBackend:
    name = "dry-run-stub"

    def __init__(self, fail_rate: float = 0.0, seed: int = 0, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.fail_rate = fail_rate
        self._rng = random.Random(seed)
        self.n_calls = 0

    async def complete(self, req: LLMRequest) -> LLMResult:
        await asyncio.sleep(self.delay_s)
        self.n_calls += 1
        if self.fail_rate and self._rng.random() < self.fail_rate:
            raise LLMCallError("simulated transient failure", cost_usd=0.0005, kind="transient")
        meta = req.meta
        q = set(tokenize(meta.get("category_text", "")))
        q.update(tokenize(" ".join(meta.get("search_terms", []))))
        if req.stage == "brief":
            toks = [t for t in dict.fromkeys(tokenize(meta.get("category_text", ""))) if len(t) > 3][:12]
            structured = {"clinical_summary": "(dry-run stub brief -- not clinical content)", "treatment": [], "investigation": [],
                          "examination": [], "devices": [], "search_terms": toks, "typically_no_procedural_evidence": False}
        elif req.stage == "route":
            ids = meta["valid_ids"]
            texts = meta["node_texts"]
            need = 2 if len(ids) > 10 else 1          # be pickier on big frontiers so dry runs stay small
            hits = [i for i in ids if len(q & set(tokenize(texts[i]))) >= need]
            structured = {"selected": hits[: max(1, len(ids) // 2)] if hits else []}
        elif req.stage == "judge":
            verdicts = []
            for uid, text in meta["unit_texts"].items():
                n = len(q & set(tokenize(text)))
                tier = "core" if n >= 3 else "broad" if n == 2 else "unclear" if n == 1 else "not_supporting"
                low = text.lower()
                ev = "none" if tier in ("not_supporting", "unclear") else (
                    "investigated" if re.search(r"ultrasound|radiograph|tomograph|imaging|angiograph|biops|test|stud(y|ies)|scan|monitor", low)
                    else "examined" if re.search(r"examination|assessment", low) else "treated")
                if req.family == "dx" and ev != "none":
                    ev = "associated_comorbidity"
                v = {"id": uid, "support_tier": tier, "evidence_type": ev, "rationale": f"stub: {n} shared term(s)"}
                if meta.get("include_member_scope") and tier == "core" and meta.get("member_codes"):
                    v["member_scope"] = list(meta["member_codes"][:2])
                verdicts.append(v)
            structured = {"category": req.category, "verdicts": verdicts}
        else:
            raise LLMCallError(f"unknown stage {req.stage!r}", kind="fatal")
        n_in = len(req.prompt) // 4
        cost = 0.001 + n_in * 2e-6
        return LLMResult(structured=structured, cost_usd=cost, diag={"input_tokens": n_in, "output_tokens": 50, "thinking_tokens": 0, "num_turns": 1})
