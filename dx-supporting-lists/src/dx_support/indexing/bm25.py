"""Dependency-free lexical index (Okapi BM25). Deterministic: ties break by document id."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Optional

_STOP = frozenset(
    "a an and are as at be by for from in is it of on or the to with not elsewhere classified other "
    "unspecified specified nec each per than that this which using use used".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _forms(token: str) -> list[str]:
    """The token plus its plural-stripped forms, so "lens" matches "lenses" without a stemming library. Documents and queries are expanded
    the same way, so the match is symmetric."""
    forms = [token]
    if len(token) > 3 and token.endswith("ies"):
        forms.append(token[:-3] + "y")
    if len(token) > 3 and token.endswith("es"):
        forms.append(token[:-2])
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        forms.append(token[:-1])
    return forms


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for t in _TOKEN_RE.findall((text or "").lower()):
        if t in _STOP or (len(t) < 2 and not t.isdigit()):
            continue
        out.extend(_forms(t))
    return out


class BM25:
    """Okapi BM25 over {doc_id: text}."""

    def __init__(self, docs: dict[str, str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        for i, doc_id in enumerate(self.ids):
            toks = tokenize(docs[doc_id])
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        n = len(self.ids)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, k: int, allowed: Optional[set[str]] = None, max_terms: int = 80) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(tokenize(query)))[:max_terms]
        scores: dict[int, float] = defaultdict(float)
        for t in terms:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                dl = self.doc_len[i]
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0)))
        ranked = sorted(((self.ids[i], s) for i, s in scores.items() if allowed is None or self.ids[i] in allowed), key=lambda x: (-x[1], x[0]))
        return ranked[:k]
