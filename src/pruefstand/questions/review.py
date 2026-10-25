"""Human check: export a stratified review sheet (CSV) and import the owner's verdicts."""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping
from pathlib import Path

from pruefstand.corpus.gii import Unit
from pruefstand.metrics import locate_excerpts
from pruefstand.questions.schema import TYPES

REVIEW_COLUMNS = (
    "qid",
    "type",
    "law",
    "question_de",
    "question_en",
    "gold_answer_de",
    "gold_answer_en",
    "gold_passage_de",
    "gold_unit_ids",
    "gold_unit_text_de",
    "source_url",
    "verdict",
    "edited_question_de",
    "edited_gold_answer_de",
    "note",
)
# Every unanswerable question is reviewed (abstention quality is only as good as its negatives); the other types
# share the remaining rows in proportion to their size in the set (largest remainder, ties by type order).
REVIEWED_IN_FULL = ("unanswerable",)
CELL_LIMIT = 30000  # spreadsheet cells hold at most 32,767 characters
CONTEXT_WINDOW = 1500  # characters of a long unit kept around each gold excerpt
VERDICTS = ("accept", "edit", "reject")


def review_quotas(questions: list[dict], n: int) -> dict[str, int]:
    """Rows per type: all questions of REVIEWED_IN_FULL types, the rest proportional to the type sizes."""
    sizes = {kind: sum(1 for q in questions if q["type"] == kind) for kind in TYPES}
    quotas = {kind: min(sizes[kind], n) for kind in REVIEWED_IN_FULL}
    rest = n - sum(quotas.values())
    others = [kind for kind in TYPES if kind not in REVIEWED_IN_FULL and sizes[kind]]
    total = sum(sizes[kind] for kind in others)
    exact = {kind: rest * sizes[kind] / total for kind in others} if total else {}
    quotas.update({kind: int(value) for kind, value in exact.items()})
    leftover = rest - sum(quotas[kind] for kind in others)
    by_remainder = sorted(others, key=lambda kind: (-(exact[kind] - int(exact[kind])), TYPES.index(kind)))
    for kind in by_remainder[: max(leftover, 0)]:
        quotas[kind] += 1
    return {kind: min(quotas.get(kind, 0), sizes[kind]) for kind in TYPES}


def sample_for_review(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified by type (see review_quotas); deterministic for a given seed."""
    rng = random.Random(seed)
    quotas = review_quotas(questions, n)
    chosen: list[dict] = []
    for kind in TYPES:
        pool = sorted((q for q in questions if q["type"] == kind), key=lambda q: q["qid"])
        chosen.extend(rng.sample(pool, quotas[kind]) if quotas[kind] < len(pool) else pool)
    return sorted(chosen, key=lambda q: q["qid"])


def unit_context(text: str, passage: str) -> str:
    """The unit text for the review sheet; very long units are cut to the neighbourhood of the gold excerpts."""
    if len(text) <= CELL_LIMIT // 4:
        return text
    spans = locate_excerpts(text, passage) or [(0, 0)]
    windows: list[list[int]] = []
    for start, end in sorted(spans):
        lo, hi = max(0, start - CONTEXT_WINDOW), min(len(text), end + CONTEXT_WINDOW)
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], hi)
        else:
            windows.append([lo, hi])
    parts = [
        ("[…] " if lo > 0 else "") + text[lo:hi] + (" […]" if hi < len(text) else "") for lo, hi in windows
    ]
    return " … ".join(parts)[:CELL_LIMIT]


def export_review(
    questions: list[dict], units_by_id: Mapping[str, Unit], out_path: Path, n: int, seed: int
) -> int:
    rows = []
    for question in sample_for_review(questions, n, seed):
        unit_ids = question["gold_unit_ids"] or question["source_unit_ids"]
        rows.append(
            {
                "qid": question["qid"],
                "type": question["type"],
                "law": question["law"],
                "question_de": question["question_de"],
                "question_en": question["question_en"],
                "gold_answer_de": question["gold_answer_de"],
                "gold_answer_en": question["gold_answer_en"],
                "gold_passage_de": question["gold_passage_de"],
                "gold_unit_ids": " ".join(question["gold_unit_ids"]),
                "gold_unit_text_de": "\n\n".join(
                    f"[{uid}] {unit_context(units_by_id[uid].text, question['gold_passage_de'])}"
                    for uid in unit_ids
                ),
                "source_url": " ".join(dict.fromkeys(units_by_id[uid].source_url for uid in unit_ids)),
                "verdict": "",
                "edited_question_de": "",
                "edited_gold_answer_de": "",
                "note": "",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def import_review(
    csv_path: Path, questions: list[dict], reviewed_at: str
) -> tuple[list[dict], int, list[str]]:
    """Apply filled-in verdicts; rows with an empty verdict are skipped. Returns (questions, updated, errors)."""
    by_qid = {question["qid"]: question for question in questions}
    errors: list[str] = []
    updates: list[tuple[dict, dict]] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            verdict = (row.get("verdict") or "").strip().lower()
            if not verdict:
                continue
            if verdict not in VERDICTS:
                errors.append(f"line {line_no}: verdict {verdict!r} not in {VERDICTS}")
                continue
            if row["qid"] not in by_qid:
                errors.append(f"line {line_no}: unknown qid {row['qid']}")
                continue
            if verdict == "edit" and not (
                row["edited_question_de"].strip() or row["edited_gold_answer_de"].strip()
            ):
                errors.append(f"line {line_no}: verdict 'edit' needs an edited question or answer")
                continue
            updates.append((by_qid[row["qid"]], {**row, "verdict": verdict}))
    if errors:
        return questions, 0, errors
    for question, row in updates:
        review = {"status": row["verdict"], "reviewed_at": reviewed_at, "note": row["note"].strip()}
        if row["verdict"] == "edit":
            review["original"] = {
                "question_de": question["question_de"],
                "gold_answer_de": question["gold_answer_de"],
            }
            if row["edited_question_de"].strip():
                question["question_de"] = row["edited_question_de"].strip()
            if row["edited_gold_answer_de"].strip():
                question["gold_answer_de"] = row["edited_gold_answer_de"].strip()
        question["review"] = review
    return questions, len(updates), []
