"""Question record rules for k1_v1: fields, types, per-law targets and gold-passage grounding."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

from pruefstand.corpus.gii import Unit

TYPES = ("factoid", "definition", "colloquial", "crossref", "number", "unanswerable")
ABSTAIN_ANSWER = "Die Frage lässt sich mit den vorliegenden Quellen nicht beantworten."
PASSAGE_SEPARATOR = " … "
AUTHORED_FIELDS = (
    "type",
    "question_de",
    "question_en",
    "gold_unit_ids",
    "gold_passage_de",
    "gold_answer_de",
    "gold_answer_en",
    "source_unit_ids",
)
# Exact counts per law and type (240 questions); columns sum to 60/48/48/36/24/24.
TARGETS: dict[str, dict[str, int]] = {
    "enwg_2005": {
        "factoid": 24,
        "definition": 19,
        "colloquial": 19,
        "crossref": 15,
        "number": 10,
        "unanswerable": 9,
    },
    "eeg_2014": {
        "factoid": 15,
        "definition": 12,
        "colloquial": 12,
        "crossref": 9,
        "number": 6,
        "unanswerable": 6,
    },
    "messbg": {"factoid": 8, "definition": 6, "colloquial": 6, "crossref": 4, "number": 3, "unanswerable": 3},
    "enfg": {"factoid": 7, "definition": 6, "colloquial": 5, "crossref": 4, "number": 2, "unanswerable": 3},
    "kwkg_2016": {
        "factoid": 6,
        "definition": 5,
        "colloquial": 6,
        "crossref": 4,
        "number": 3,
        "unanswerable": 3,
    },
}


def normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def law_of(record: Mapping) -> str:
    ids = record["gold_unit_ids"] or record["source_unit_ids"]
    return ids[0].split(":", 1)[0]


def validate_authored(record: Mapping, units_by_id: Mapping[str, Unit]) -> list[str]:
    errors = [f"missing field {name}" for name in AUTHORED_FIELDS if name not in record]
    if errors:
        return errors
    kind = record["type"]
    if kind not in TYPES:
        errors.append(f"unknown type {kind!r}")
    for name in ("question_de", "question_en", "gold_answer_de", "gold_answer_en"):
        if not isinstance(record[name], str) or not record[name].strip():
            errors.append(f"{name} must be a non-empty string")
    if not record["question_de"].strip().endswith("?"):
        errors.append("question_de must end with '?'")
    for unit_id in list(record["gold_unit_ids"]) + list(record["source_unit_ids"]):
        if unit_id not in units_by_id:
            errors.append(f"unknown unit id {unit_id}")
    if not record["source_unit_ids"]:
        errors.append("source_unit_ids must name at least one unit")
    if errors:
        return errors
    if kind == "unanswerable":
        if record["gold_unit_ids"]:
            errors.append("unanswerable questions have no gold units")
        if record["gold_answer_de"] != ABSTAIN_ANSWER:
            errors.append("unanswerable gold_answer_de must be the abstention sentence")
        if record["gold_passage_de"] != "":
            errors.append("unanswerable gold_passage_de must be empty")
        return errors
    if not record["gold_unit_ids"]:
        errors.append("answerable questions need gold_unit_ids")
        return errors
    if kind == "crossref" and len(set(record["gold_unit_ids"])) < 2:
        errors.append("crossref questions need at least two different gold units")
    gold_texts = [normalise_ws(units_by_id[unit_id].text) for unit_id in record["gold_unit_ids"]]
    for part in normalise_ws(record["gold_passage_de"]).split(PASSAGE_SEPARATOR.strip()):
        excerpt = part.strip()
        if not excerpt:
            errors.append("gold_passage_de has an empty excerpt")
        elif not any(excerpt in text for text in gold_texts):
            errors.append(f"gold passage excerpt not found verbatim in the gold units: {excerpt[:60]!r}")
    return errors


def check_distribution(records: list[Mapping]) -> list[str]:
    counts = Counter((law_of(record), record["type"]) for record in records)
    errors = []
    for law, per_type in TARGETS.items():
        for kind, expected in per_type.items():
            actual = counts.get((law, kind), 0)
            if actual != expected:
                errors.append(f"{law}/{kind}: {actual} questions, target {expected}")
    for (law, kind), actual in counts.items():
        if law not in TARGETS:
            errors.append(f"{law}/{kind}: law not in targets ({actual} questions)")
    return errors
