"""The frozen question set k1_v1 as committed: grounding, quotas, provenance and the honest review status."""

import csv
from collections import Counter
from pathlib import Path

import pytest

from pruefstand.corpus.gii import load_units
from pruefstand.io import read_jsonl
from pruefstand.prompts import load_prompt
from pruefstand.questions.review import CELL_LIMIT, review_quotas
from pruefstand.questions.schema import TARGETS, check_distribution, validate_authored

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data" / "questions"


@pytest.fixture(scope="module")
def questions():
    return read_jsonl(QUESTIONS / "k1_v1.jsonl")


@pytest.fixture(scope="module")
def units_by_id():
    return {unit.unit_id: unit for unit in load_units(ROOT / "data" / "corpus" / "k1_units.jsonl.gz")}


def test_set_has_240_numbered_questions_that_meet_every_quota(questions):
    assert [q["qid"] for q in questions] == [f"k1v1-{i:04d}" for i in range(1, 241)]
    assert check_distribution(questions) == []
    per_law = Counter(q["law"] for q in questions)
    assert per_law == {law: sum(per_type.values()) for law, per_type in TARGETS.items()}
    assert len({q["question_de"] for q in questions}) == 240


def test_every_record_is_grounded_in_the_committed_corpus(questions, units_by_id):
    for question in questions:
        assert validate_authored(question, units_by_id) == [], question["qid"]
        assert question["answerable"] == (question["type"] != "unanswerable")
        assert question["provenance"]["filters"]["passed"] is True


def test_provenance_points_at_the_frozen_protocol_and_nothing_claims_a_human_check(questions):
    protocol = load_prompt(ROOT / "docs" / "question-authoring.md")
    for question in questions:
        provenance = question["provenance"]
        assert provenance["method"] == "agent-authored" and provenance["prompt_id"] == protocol.prompt_id
        assert provenance["prompt_sha256"] == protocol.sha256
        assert question["review"] == {"status": "pending", "reviewed_at": None, "note": ""}
    assert len({q["provenance"]["model"] for q in questions}) == 1
    datasheet = (QUESTIONS / "DATASHEET.md").read_text(encoding="utf-8")
    assert "AUTHOR_MODEL_ID" not in datasheet
    assert questions[0]["provenance"]["model"] in datasheet and "**Pending.**" in datasheet


def test_review_sheet_lists_100_questions_with_the_set_type_shares(questions):
    with open(QUESTIONS / "review" / "k1_v1_review.csv", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_qid = {q["qid"]: q for q in questions}
    assert len(rows) == 100 and all(row["qid"] in by_qid for row in rows)
    assert Counter(row["type"] for row in rows) == {
        k: v for k, v in review_quotas(questions, 100).items() if v
    }
    assert (
        Counter(row["type"] for row in rows)["unanswerable"] == 24
    )  # every unanswerable question is reviewed
    assert all(row["verdict"] == "" for row in rows)
    assert all(len(row["gold_unit_text_de"]) <= CELL_LIMIT * 2 for row in rows)  # fits a spreadsheet cell
    for row in rows:  # the sheet shows the committed question text, not a stale copy
        assert row["question_de"] == by_qid[row["qid"]]["question_de"]
        assert row["gold_answer_de"] == by_qid[row["qid"]]["gold_answer_de"]


def test_every_long_gold_unit_has_a_located_passage_for_the_passage_based_relevance(questions, units_by_id):
    from pruefstand.metrics import LONG_UNIT_CHARS, passage_spans

    texts = {uid: unit.text for uid, unit in units_by_id.items()}
    long_units = 0
    for question in questions:
        spans = passage_spans(question["gold_unit_ids"], question["gold_passage_de"], texts, question["qid"])
        assert set(spans) == {u for u in question["gold_unit_ids"] if len(texts[u]) > LONG_UNIT_CHARS}
        long_units += len(spans)
    assert long_units >= 48  # at least the definition questions


def test_datasheet_claims_match_the_committed_set(questions, units_by_id):
    import re

    from pruefstand.metrics import passage_spans
    from pruefstand.questions.schema import section_of

    datasheet = (QUESTIONS / "DATASHEET.md").read_text(encoding="utf-8")
    named = re.compile(
        r"Energiewirtschaftsgesetz|Erneuerbare-Energien-Gesetz|Messstellenbetriebsgesetz|Energiefinanzierungsgesetz"
        r"|Kraft-Wärme-Kopplungsgesetz|\bEEG\b|\bEnWG\b|\bKWKG\b|\bMsbG\b|\bEnFG\b"
    )
    count = sum(1 for q in questions if named.search(q["question_de"]))
    assert f"{count} of the 240 questions name a law" in datasheet
    texts = {uid: unit.text for uid, unit in units_by_id.items()}
    passage_based = sum(
        1 for q in questions if passage_spans(q["gold_unit_ids"], q["gold_passage_de"], texts, q["qid"])
    )
    assert "62 questions use at least one" in datasheet and passage_based == 62
    crossref = [q for q in questions if q["type"] == "crossref"]
    cross_law = [q["qid"][-4:] for q in crossref if len({u.split(":")[0] for u in q["gold_unit_ids"]}) > 1]
    assert len(cross_law) == 6 and all(qid in datasheet for qid in cross_law)
    assert all(len({section_of(u) for u in q["gold_unit_ids"]}) >= 2 for q in crossref)
    unanswerable = [q for q in questions if q["type"] == "unanswerable"]
    cues = re.compile(
        r"derzeit|aktuell|letzten|vergangenen|laufend|Bayern|Köln|München|in der Praxis|tatsächlich"
    )
    assert len(unanswerable) == 24
    assert not any(
        cues.search(q["question_de"]) or q["question_de"].startswith("Wie viele") for q in unanswerable
    )
    assert (
        "claude-sonnet-5-5" in datasheet
        and "claude-opus-5-5" in datasheet
        and "Author and judge" in datasheet
    )
