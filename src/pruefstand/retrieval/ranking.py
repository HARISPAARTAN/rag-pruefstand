"""Deterministic top-k selection shared by all retrievers."""

from __future__ import annotations

import numpy as np

SCORE_DECIMALS = 6


def top_k(
    scores: np.ndarray, k: int, allowed: np.ndarray | None = None, positive_only: bool = False
) -> list[tuple[int, float]]:
    """Return (index, score) pairs, best first.

    Scores are rounded to 6 decimals before ranking and ties are broken by the lower index, so the order is the
    same on every platform even when BLAS summation differs in the last bits.
    """
    rounded = np.round(scores.astype(np.float64), SCORE_DECIMALS)
    if allowed is not None:
        rounded = np.where(allowed, rounded, -np.inf)
    order = np.lexsort((np.arange(len(rounded)), -rounded))
    result: list[tuple[int, float]] = []
    for index in order:
        value = rounded[index]
        if not np.isfinite(value) or (positive_only and value <= 0):
            break
        result.append((int(index), float(value)))
        if len(result) == k:
            break
    return result


def tenant_mask(tenants: list[str], tenant: str | None) -> np.ndarray | None:
    if tenant is None:
        return None
    return np.array([value == tenant for value in tenants], dtype=bool)
