"""Human check: export a stratified review sheet (CSV) and import the owner's verdicts."""

from __future__ import annotations

import copy
import csv
import random
import re
from collections.abc import Mapping
from pathlib import Path

from pruefstand.corpus.gii import Unit
from pruefstand.errors import PruefstandError
from pruefstand.metrics import locate_excerpts
from pruefstand.questions.filters import run_filters
from pruefstand.questions.schema import TYPES, validate_authored

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
# k1_v1 carries created_at 2026-10-26 although the questions were written on 2026-10-25 (a clock assumption of the
# authoring agent). The frozen set keeps the value (its SHA-256 is pinned by the published run); the successor
# set corrects it and records the old value.
CREATED_AT_CORRECTIONS = {"2026-10-26": "2026-10-25"}


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


def parse_verdicts(csv_path: Path, questions: list[dict], reviewed_at: str) -> tuple[list[dict], list[str]]:
    """Read the filled-in review sheet; rows with an empty verdict are skipped. Returns (verdicts, errors).

    A verdict record holds qid, status, reviewed_at, note and, for an edit, the edited texts plus the original
    texts as they stand in the frozen question set. Nothing is applied to the questions here.
    """
    by_qid = {question["qid"]: question for question in questions}
    errors: list[str] = []
    verdicts: list[dict] = []
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
            record = {
                "qid": row["qid"],
                "status": verdict,
                "reviewed_at": reviewed_at,
                "note": (row.get("note") or "").strip(),
            }
            if verdict == "edit":
                edited_question = (row.get("edited_question_de") or "").strip()
                edited_answer = (row.get("edited_gold_answer_de") or "").strip()
                if not (edited_question or edited_answer):
                    errors.append(f"line {line_no}: verdict 'edit' needs an edited question or answer")
                    continue
                original = by_qid[row["qid"]]
                record["original"] = {
                    "question_de": original["question_de"],
                    "gold_answer_de": original["gold_answer_de"],
                }
                if edited_question:
                    record["edited_question_de"] = edited_question
                if edited_answer:
                    record["edited_gold_answer_de"] = edited_answer
            verdicts.append(record)
    return verdicts, errors


def verdicts_path(questions_path: Path, version: str) -> Path:
    """Where the verdicts for a question set live: next to the review sheet, never inside the set itself."""
    return questions_path.parent / "review" / f"{version}_verdicts.jsonl"


def merge_verdicts(existing: list[dict], new: list[dict]) -> list[dict]:
    """The verdict file after an import: a later verdict for the same qid replaces the earlier one."""
    merged = {record["qid"]: record for record in existing}
    merged.update({record["qid"]: record for record in new})
    return [merged[qid] for qid in sorted(merged)]


def successor_version(version: str) -> str:
    """k1_v1 -> k1_v2: the id of the set that carries the edits."""
    match = re.fullmatch(r"(.+_v)(\d+)", version)
    if not match:
        raise PruefstandError(f"cannot derive the next set id from {version!r} (expected e.g. k1_v1)")
    return f"{match.group(1)}{int(match.group(2)) + 1}"


def review_block(verdict: dict) -> dict:
    block = {"status": verdict["status"], "reviewed_at": verdict["reviewed_at"], "note": verdict["note"]}
    if verdict["status"] == "edit":
        block["original"] = verdict["original"]
    return block


def build_successor(
    questions: list[dict], verdicts: list[dict], units_by_id: Mapping[str, Unit], source_version: str
) -> tuple[list[dict], list[str]]:
    """The next question set: the frozen set plus every verdict, with the edits applied and re-validated.

    Edited records must pass the authoring rules again (schema and grounding, then the automatic filters over the
    whole set). Records keep their qid. created_at values dated after the actual authoring day are corrected
    (see CREATED_AT_CORRECTIONS) and each record names the set it derives from.
    """
    by_qid = {verdict["qid"]: verdict for verdict in verdicts}
    successor = []
    for question in questions:
        record = copy.deepcopy(question)
        verdict = by_qid.get(record["qid"])
        if verdict is not None:
            record["review"] = review_block(verdict)
            if verdict["status"] == "edit":
                record["question_de"] = verdict.get("edited_question_de", record["question_de"])
                record["gold_answer_de"] = verdict.get("edited_gold_answer_de", record["gold_answer_de"])
        provenance = record["provenance"]
        created = provenance.get("created_at")
        if created in CREATED_AT_CORRECTIONS:
            provenance["created_at"] = CREATED_AT_CORRECTIONS[created]
            provenance["created_at_corrected_from"] = created
        provenance["derived_from"] = source_version
        successor.append(record)
    errors = []
    edited = {verdict["qid"] for verdict in verdicts if verdict["status"] == "edit"}
    for record in successor:
        if record["qid"] in edited:
            errors.extend(f"{record['qid']}: {problem}" for problem in validate_authored(record, units_by_id))
    if errors:
        return successor, errors
    results, filter_errors = run_filters(successor, units_by_id)
    errors.extend(f"{successor[index]['qid']}: {problem}" for index, problem in filter_errors)
    for record, result in zip(successor, results, strict=True):
        if record["qid"] in edited:
            record["provenance"]["filters"] = result
    return successor, errors


def apply_checked_verdicts(questions: list[dict], verdicts: list[dict]) -> list[dict]:
    """Overlay accept and reject verdicts of the verdict file onto the questions of the same set.

    An edit verdict is not applied here: the frozen set still holds the unedited text, so the question counts as
    checked only in the successor set, which carries the edited text.
    """
    by_qid = {verdict["qid"]: verdict for verdict in verdicts if verdict["status"] != "edit"}
    for question in questions:
        verdict = by_qid.get(question["qid"])
        if verdict is not None:
            question["review"] = review_block(verdict)
    return questions
