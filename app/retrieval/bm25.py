"""Self-contained BM25 ranking (no external dependency).

BM25 is a classic *sparse lexical* ranking function: it scores documents by term frequency
saturated by document length, weighted by inverse document frequency. Unlike dense
embeddings it matches exact terms — which is why **hybrid** search (dense + sparse) beats
either alone for enterprise jargon, codes, and acronyms. See ``docs/RAG_DEEP_DIVE.md``.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.doc_freqs: list[Counter] = [Counter(d) for d in corpus]
        df: Counter = Counter()
        for d in corpus:
            for term in set(d):
                df[term] += 1
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def score(self, query: list[str], index: int) -> float:
        freqs = self.doc_freqs[index]
        dl = self.doc_len[index]
        score = 0.0
        for term in query:
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0.0)
            tf = freqs[term]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (tf * (self.k1 + 1)) / (denom or 1)
        return score

    def rank(self, query: list[str], top_k: int) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.n)]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
