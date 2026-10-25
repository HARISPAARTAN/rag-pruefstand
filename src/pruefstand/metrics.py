"""Span-based relevance and per-question retrieval metrics.

A retrieved chunk is relevant to a gold unit when it covers at least 50% of the unit's characters or at least
200 of them. Metrics count gold units, not chunks, so overlapping C1 windows earn no extra credit.

Which metric compares what:

- recall@k and MRR@10 are the cross-chunking comparison metrics. They count gold units found (recall) or the
  rank of the first hit (MRR) and do not depend on how wide a chunk is, so C1 and C2 can be compared with them.
- nDCG@10 compares retrievers within one chunking. The gain of a retrieved chunk is the share of the gold units
  it covers for the first time; the ideal DCG comes from the same chunking (the minimal greedy cover of the gold
  units by the chunks that overlap them, its gains placed at ranks 1, 2, ...). Both sides are normalised per
  chunking, so a value of 1.0 means "as good as this chunking allows", not "as good as another chunking allows".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from pruefstand.errors import PruefstandError

MIN_OVERLAP_CHARS = 200
MIN_OVERLAP_SHARE = 0.5
K_VALUES = (1, 3, 5, 10)


def validate_gold(gold_unit_ids: Sequence[str], unit_lengths: Mapping[str, int]) -> None:
    """Reject gold ids that are not corpus units or that repeat: both would silently distort recall."""
    unknown = [unit_id for unit_id in gold_unit_ids if unit_id not in unit_lengths]
    if unknown:
        raise PruefstandError(f"gold unit ids not in the corpus: {unknown}")
    if len(set(gold_unit_ids)) != len(gold_unit_ids):
        raise PruefstandError(f"duplicate gold unit ids: {list(gold_unit_ids)}")


def gold_units_covered(
    covers: Sequence[Mapping], gold_unit_ids: Sequence[str], unit_lengths: Mapping[str, int]
) -> list[str]:
    """Gold unit ids this chunk is relevant to, in the order of gold_unit_ids."""
    validate_gold(gold_unit_ids, unit_lengths)
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


def _require_gold(n_gold: int) -> None:
    if n_gold <= 0:
        raise PruefstandError("an answerable question needs at least one gold unit")


def recall_at_k(covered_per_rank: Sequence[Sequence[str]], n_gold: int, k: int) -> float:
    _require_gold(n_gold)
    found: set[str] = set()
    for covered in covered_per_rank[:k]:
        found.update(covered)
    return len(found) / n_gold


def mrr_at_k(covered_per_rank: Sequence[Sequence[str]], k: int = 10) -> float:
    for rank, covered in enumerate(covered_per_rank[:k], start=1):
        if covered:
            return 1.0 / rank
    return 0.0


def ideal_gains(chunk_covers: Iterable[Iterable[str]]) -> list[int]:
    """Gains of the minimal greedy cover of the gold units, best first.

    chunk_covers holds, for every chunk of one chunking, the gold units it is relevant to. Each step takes the
    chunk that covers the most still uncovered gold units (the first one on ties) and records that number.
    """
    remaining = [set(covered) for covered in chunk_covers if covered]
    uncovered = set().union(*remaining) if remaining else set()
    gains: list[int] = []
    while uncovered:
        best = max(remaining, key=lambda covered: len(covered & uncovered))
        gain = len(best & uncovered)
        gains.append(gain)
        uncovered -= best
    return gains


def _dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(
    covered_per_rank: Sequence[Sequence[str]], n_gold: int, ideal: Sequence[int], k: int = 10
) -> float:
    """nDCG@k within one chunking.

    Gain of a rank = share of the n_gold gold units that chunk covers for the first time. The ideal DCG places the
    gains of the minimal greedy cover of the same chunking (see ideal_gains) at ranks 1, 2, .... A wide chunk that
    covers two gold units therefore earns two gain units and is compared with an ideal that also needs that wide
    chunk, so it gets no advantage over two exact chunks. The result is guarded to at most 1.0; the guard only
    matters in contrived cases where the greedy cover is not the DCG optimum.
    """
    _require_gold(n_gold)
    seen: set[str] = set()
    gains = []
    for covered in covered_per_rank[:k]:
        new = set(covered) - seen
        seen.update(covered)
        gains.append(len(new) / n_gold)
    ideal_dcg = _dcg([gain / n_gold for gain in ideal[:k]])
    return min(1.0, _dcg(gains) / ideal_dcg) if ideal_dcg else 0.0


def question_metrics(
    covered_per_rank: Sequence[Sequence[str]], n_gold: int, ideal: Sequence[int]
) -> dict[str, float]:
    values = {f"recall@{k}": recall_at_k(covered_per_rank, n_gold, k) for k in K_VALUES}
    values["mrr@10"] = mrr_at_k(covered_per_rank, 10)
    values["ndcg@10"] = ndcg_at_k(covered_per_rank, n_gold, ideal, 10)
    return values


METRIC_NAMES = tuple([f"recall@{k}" for k in K_VALUES] + ["mrr@10", "ndcg@10"])
CROSS_CHUNKING_METRICS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10")
WITHIN_CHUNKING_METRICS = ("ndcg@10",)
