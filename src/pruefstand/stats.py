"""Percentile bootstrap intervals, paired bootstrap of differences, percentiles and weighted kappa."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DECIMALS = 6


def _r(value: float) -> float:
    return round(float(value), DECIMALS)


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """Mean with a percentile bootstrap interval over questions; empty input gives None values."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, data.size, size=(n_boot, data.size))
    means = data[samples].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": _r(data.mean()), "lo": _r(lo), "hi": _r(hi), "n": int(data.size)}


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_boot: int, seed: int, alpha: float = 0.05
) -> dict:
    """Mean difference a - b over the same questions with a percentile interval and a verdict."""
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("paired bootstrap needs two arrays over the same questions")
    result = bootstrap_ci(first - second, n_boot, seed, alpha)
    if result["n"] == 0:
        verdict = "no_data"
    elif result["lo"] > 0:
        verdict = "better"
    elif result["hi"] < 0:
        verdict = "worse"
    else:
        verdict = "no_measurable_difference"
    return {
        "diff": result["mean"],
        "lo": result["lo"],
        "hi": result["hi"],
        "n": result["n"],
        "verdict": verdict,
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    data = np.asarray(values, dtype=np.float64)
    return _r(np.percentile(data, q)) if data.size else None


def weighted_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int]) -> float | None:
    """Cohen's kappa with quadratic weights for ordinal labels (judge vs human)."""
    index = {category: i for i, category in enumerate(categories)}
    k = len(categories)
    observed = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(a, b, strict=True):
        observed[index[x], index[y]] += 1
    total = observed.sum()
    if total == 0:
        return None
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    grid = np.arange(k)
    weights = (grid[:, None] - grid[None, :]) ** 2 / (k - 1) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return 1.0
    return _r(1.0 - (weights * observed).sum() / denominator)
