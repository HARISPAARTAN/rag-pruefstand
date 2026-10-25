"""Reciprocal rank fusion (RRF): no fusion weight to tune on the test questions."""

from __future__ import annotations


def rrf(rankings: list[list[int]], k: int = 60, depth: int | None = None) -> list[tuple[int, float]]:
    """Fuse ranked lists of document indices; score(d) = sum over lists of 1 / (k + rank), rank starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    fused = sorted(
        ((doc, round(score, 9)) for doc, score in scores.items()), key=lambda item: (-item[1], item[0])
    )
    return fused[:depth] if depth is not None else fused
