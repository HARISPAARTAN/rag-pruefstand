"""Span-based relevance and per-question retrieval metrics.

A retrieved chunk is relevant to a gold unit when it covers at least 50% of the unit's characters or at least
200 of them. Metrics count gold units, not chunks, so overlapping C1 windows earn no extra credit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

MIN_OVERLAP_CHARS = 200
MIN_OVERLAP_SHARE = 0.5
K_VALUES = (1, 3, 5, 10)


def gold_units_covered(
    covers: Sequence[Mapping], gold_unit_ids: Sequence[str], unit_lengths: Mapping[str, int]
) -> list[str]:
    """Gold unit ids this chunk is relevant to, in the order of gold_unit_ids."""
    overlap: dict[str, int] = {}
    for cover in covers:
        unit_id = cover["unit_id"]
        overlap[unit_id] = overlap.get(unit_id, 0) + int(cover["end"]) - int(cover["start"])
    covered = []
    for unit_id in gold_unit_ids:
        chars = overlap.get(unit_id, 0)
        if chars <= 0:
            continue
        if chars >= MIN_OVERLAP_CHARS or chars >= MIN_OVERLAP_SHARE * unit_lengths[unit_id]:
            covered.append(unit_id)
    return covered


def recall_at_k(covered_per_rank: Sequence[Sequence[str]], n_gold: int, k: int) -> float:
    found: set[str] = set()
    for covered in covered_per_rank[:k]:
        found.update(covered)
    return len(found) / n_gold


def mrr_at_k(covered_per_rank: Sequence[Sequence[str]], k: int = 10) -> float:
    for rank, covered in enumerate(covered_per_rank[:k], start=1):
        if covered:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(covered_per_rank: Sequence[Sequence[str]], n_gold: int, k: int = 10) -> float:
    """Gain of a rank = number of gold units it covers for the first time; capped at 1.0."""
    seen: set[str] = set()
    dcg = 0.0
    for rank, covered in enumerate(covered_per_rank[:k], start=1):
        gain = len(set(covered) - seen)
        seen.update(covered)
        dcg += gain / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_gold, k) + 1))
    return min(1.0, dcg / ideal) if ideal else 0.0


def question_metrics(covered_per_rank: Sequence[Sequence[str]], n_gold: int) -> dict[str, float]:
    values = {f"recall@{k}": recall_at_k(covered_per_rank, n_gold, k) for k in K_VALUES}
    values["mrr@10"] = mrr_at_k(covered_per_rank, 10)
    values["ndcg@10"] = ndcg_at_k(covered_per_rank, n_gold, 10)
    return values


METRIC_NAMES = tuple([f"recall@{k}" for k in K_VALUES] + ["mrr@10", "ndcg@10"])
