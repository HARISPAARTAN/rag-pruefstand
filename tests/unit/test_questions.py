import csv

import pytest

from pruefstand.corpus.gii import parse_law_xml, read_xml_from_zip
from pruefstand.questions.filters import copy_ratio, jaccard, lexical_overlap, numbers_supported, run_filters
from pruefstand.questions.review import export_review, import_review, sample_for_review
from pruefstand.questions.schema import ABSTAIN_ANSWER, check_distribution, validate_authored


@pytest.fixture(scope="module")
def units_by_id():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    xml = read_xml_from_zip(root / "fixtures" / "ci" / "gii" / "messbg.xml.zip")
    units, _ = parse_law_xml(xml, "messbg", "https://www.gesetze-im-internet.de", "2026-10-25")
    return {unit.unit_id: unit for unit in units}


def record(**overrides):
    base = {
        "type": "factoid",
        "question_de": "Muss man für einen Wechsel des Messstellenbetreibers extra bezahlen?",
        "question_en": "Does switching the metering point operator cost an extra fee?",
        "gold_unit_ids": ["messbg:§14:abs3"],
        "gold_passage_de": "Für den Wechsel des Messstellenbetreibers darf kein gesondertes Entgelt erhoben werden.",
        "gold_answer_de": "Nein, für den Wechsel darf kein gesondertes Entgelt erhoben werden.",
        "gold_answer_en": "No, no separate fee may be charged for the switch.",
        "source_unit_ids": ["messbg:§14:abs3"],
    }
    base.update(overrides)
    return base


def test_valid_record_passes(units_by_id):
    assert validate_authored(record(), units_by_id) == []


def test_gold_passage_must_be_verbatim(units_by_id):
    errors = validate_authored(record(gold_passage_de="Der Wechsel ist immer kostenlos."), units_by_id)
    assert any("not found verbatim" in e for e in errors)


def test_unknown_unit_and_missing_question_mark(units_by_id):
    errors = validate_authored(
        record(gold_unit_ids=["messbg:§999:abs1"], question_de="Kein Fragezeichen"), units_by_id
    )
    assert "question_de must end with '?'" in errors
    assert "unknown unit id messbg:§999:abs1" in errors


def test_unanswerable_rules(units_by_id):
    good = record(type="unanswerable", gold_unit_ids=[], gold_passage_de="", gold_answer_de=ABSTAIN_ANSWER)
    assert validate_authored(good, units_by_id) == []
    bad = record(type="unanswerable", gold_unit_ids=[], gold_passage_de="", gold_answer_de="Nein.")
    assert validate_authored(bad, units_by_id) == [
        "unanswerable gold_answer_de must be the abstention sentence"
    ]


def test_crossref_needs_two_units(units_by_id):
    errors = validate_authored(record(type="crossref"), units_by_id)
    assert errors == ["crossref questions need at least two different gold units"]


def test_check_distribution_reports_every_gap():
    errors = check_distribution([record()])
    assert "messbg/factoid: 1 questions, target 8" in errors
    assert "enwg_2005/factoid: 0 questions, target 24" in errors


def test_filter_primitives():
    assert copy_ratio("eins zwei drei vier", "eins zwei drei vier fünf") == 1.0
    assert copy_ratio("eins zwei", "eins zwei") == 0.0
    assert lexical_overlap("Messstellenbetreiber wechseln", "Wechsel des Messstellenbetreibers") == 1.0
    assert numbers_supported("Nach 2 Jahren", "frühestens nach 2 Jahren")
    assert not numbers_supported("Nach 3 Jahren", "frühestens nach 2 Jahren")
    assert jaccard("Messstelle Betreiber Wechsel", "Messstelle Betreiber Wechsel") == 1.0


def test_run_filters_flags_copy_and_duplicates(units_by_id):
    copied = record(
        question_de="Darf für den Wechsel des Messstellenbetreibers kein gesondertes Entgelt erhoben werden?"
    )
    results, errors = run_filters([record(), copied, record()], units_by_id)
    assert results[0]["passed"] is True
    assert any(index == 1 and "copies" in problem for index, problem in errors)
    assert any(index == 2 and "near duplicate of record 1" in problem for index, problem in errors)


def test_review_roundtrip(tmp_path, units_by_id):
    questions = []
    for index, kind in enumerate(["factoid", "factoid", "definition", "unanswerable"], start=1):
        overrides = {"type": kind}
        if kind == "unanswerable":
            overrides.update(gold_unit_ids=[], gold_passage_de="", gold_answer_de=ABSTAIN_ANSWER)
        questions.append({"qid": f"k1v1-{index:04d}", "law": "messbg", **record(**overrides)})
    assert sample_for_review(questions, 8, 1) == sample_for_review(questions, 8, 1)
    sheet = tmp_path / "review.csv"
    assert export_review(questions, units_by_id, sheet, n=8, seed=1) == 4
    with open(sheet, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["verdict"] = "edit"
    rows[0]["edited_question_de"] = "Kostet der Wechsel des Messstellenbetreibers etwas?"
    rows[1]["verdict"] = "accept"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    updated, count, errors = import_review(sheet, questions, "2026-11-01")
    assert errors == [] and count == 2
    assert updated[0]["question_de"] == "Kostet der Wechsel des Messstellenbetreibers etwas?"
    assert updated[0]["review"]["original"]["question_de"].startswith("Muss man")
    assert updated[1]["review"]["status"] == "accept"
    assert "review" not in updated[2]


def test_review_import_rejects_bad_verdict(tmp_path, units_by_id):
    questions = [{"qid": "k1v1-0001", "law": "messbg", **record()}]
    sheet = tmp_path / "review.csv"
    export_review(questions, units_by_id, sheet, n=4, seed=1)
    with open(sheet, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["verdict"] = "vielleicht"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _, count, errors = import_review(sheet, questions, "2026-11-01")
    assert count == 0 and "verdict 'vielleicht'" in errors[0]
