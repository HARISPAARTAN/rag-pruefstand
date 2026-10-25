"""Compare a reproduction run with the published run (hits and metrics), ignoring where each was written."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pruefstand.errors import PruefstandError
from pruefstand.io import read_json, read_jsonl

IGNORED_KEYS = ("run_id",)
MAX_REPORTED = 20


def diff_json(left: Any, right: Any, path: str = "") -> list[str]:
    """Paths at which two JSON values differ (dicts by key, lists by index, scalars by value)."""
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(left.keys() | right.keys()):
            if key in IGNORED_KEYS and not path:
                continue
            where = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                found.append(f"{where}: only in {'reproduction' if key not in left else 'published'}")
            else:
                found.extend(diff_json(left[key], right[key], where))
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            d
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for d in diff_json(a, b, f"{path}[{i}]")
        ]
    return [] if left == right else [f"{path}: published {left!r} vs reproduction {right!r}"]


def compare_runs(published: Path, reproduction: Path) -> list[str]:
    """Differences between two run directories; an empty list means hits and metrics are identical.

    hits.jsonl.gz is compared by content (the gzip bytes depend on the zlib build), metrics.json by value.
    """
    if published.resolve() == reproduction.resolve():
        raise PruefstandError("a run cannot be compared with itself")
    for run in (published, reproduction):
        for name in ("hits.jsonl.gz", "metrics.json"):
            if not (run / name).is_file():
                raise PruefstandError(f"{run / name} not found")
    differences = []
    published_hits = read_jsonl(published / "hits.jsonl.gz")
    reproduced_hits = read_jsonl(reproduction / "hits.jsonl.gz")
    if published_hits != reproduced_hits:
        rows = sum(1 for a, b in zip(published_hits, reproduced_hits, strict=False) if a != b)
        differences.append(
            f"hits.jsonl.gz: {rows} of {len(published_hits)} rows differ"
            f"{'' if len(published_hits) == len(reproduced_hits) else ' (row counts differ)'}"
        )
    differences.extend(
        f"metrics.json {d}"
        for d in diff_json(read_json(published / "metrics.json"), read_json(reproduction / "metrics.json"))
    )
    return differences
