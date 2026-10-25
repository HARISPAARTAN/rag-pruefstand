"""Human check: export a stratified review sheet (CSV) and import the owner's verdicts."""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping
from pathlib import Path

from pruefstand.corpus.gii import Unit

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
TYPE_SHARE = {
    "factoid": 25,
    "definition": 20,
    "colloquial": 20,
    "crossref": 15,
    "number": 10,
    "unanswerable": 10,
}
VERDICTS = ("accept", "edit", "reject")


def sample_for_review(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified by type with the set's type shares; deterministic for a given seed."""
    rng = random.Random(seed)
    chosen: list[dict] = []
    for kind, share in TYPE_SHARE.items():
        pool = sorted((q for q in questions if q["type"] == kind), key=lambda q: q["qid"])
        quota = min(len(pool), round(n * share / 100))
        chosen.extend(rng.sample(pool, quota))
    return sorted(chosen, key=lambda q: q["qid"])


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
                "gold_unit_text_de": "\n\n".join(f"[{uid}] {units_by_id[uid].text}" for uid in unit_ids),
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
