"""Lexical baseline: bm25s with German Snowball stemming (PyStemmer)."""

from __future__ import annotations

import re

import bm25s
import Stemmer
from bm25s.stopwords import STOPWORDS_GERMAN

from pruefstand.retrieval.ranking import tenant_mask, top_k

TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset(STOPWORDS_GERMAN)
_STEMMER = Stemmer.Stemmer("german")


def analyze(text: str) -> list[str]:
    """Lowercase, split on word characters, drop German stopwords and one-letter tokens, stem."""
    tokens = [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())
    ]
    return _STEMMER.stemWords(tokens)


class BM25Index:
    def __init__(self, texts: list[str], tenants: list[str] | None = None) -> None:
        self.n = len(texts)
        self.tenants = tenants or ["K1"] * self.n
        self._retriever = bm25s.BM25()
        self._retriever.index([analyze(text) for text in texts], show_progress=False)

    def search(self, query: str, k: int, tenant: str | None = None) -> list[tuple[int, float]]:
        tokens = analyze(query)
        if not tokens:
            return []
        scores = self._retriever.get_scores(tokens)
        return top_k(scores, k, allowed=tenant_mask(self.tenants, tenant), positive_only=True)
