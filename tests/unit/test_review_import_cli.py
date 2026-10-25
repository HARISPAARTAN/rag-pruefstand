"""`questions review-import` records verdicts beside the frozen set and writes edits into the next set."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pruefstand.bench import load_questions
from pruefstand.cli import app
from pruefstand.config import load_config
from pruefstand.io import read_jsonl
from pruefstand.questions.review import REVIEW_COLUMNS, apply_checked_verdicts

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data" / "questions"
UNITS = ROOT / "data" / "corpus" / "k1_units.jsonl.gz"
EDIT = "Welches Gewicht haben erneuerbare Energien in Schutzgüterabwägungen, bevor die Stromerzeugung nahezu klimaneutral ist?"


EDIT_2 = "Darf ein Netzbetreiber einen unterschriebenen Vertrag fordern, bevor er die gesetzlichen Pflichten erfüllt?"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def workspace(tmp_path):
    """A scratch copy of the first ten questions with a config pointing at the committed units."""
    lines = (QUESTIONS / "k1_v1.jsonl").read_text(encoding="utf-8").splitlines()[:10]
    (tmp_path / "data" / "questions" / "review").mkdir(parents=True)
    path = tmp_path / "data" / "questions" / "k1_v1.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    config = tmp_path / "pruefstand.toml"
    config.write_text(
        f'[questions]\npath = "data/questions/k1_v1.jsonl"\nversion = "k1_v1"\n'
        f'[corpus]\nunits_path = "{UNITS.as_posix()}"\n[profile.full]\n',
        encoding="utf-8",
    )
    return tmp_path, config, path


def sheet(directory: Path, verdicts: dict[str, dict]) -> Path:
    path = directory / "sheet.csv"
    questions = {
        q["qid"]: q
        for q in map(json.loads, (directory / "data/questions/k1_v1.jsonl").open(encoding="utf-8"))
    }
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        for qid, fields in verdicts.items():
            writer.writerow(
                {
                    **{c: "" for c in REVIEW_COLUMNS},
                    "qid": qid,
                    "question_de": questions[qid]["question_de"],
                    **fields,
                }
            )
    return path


def run(config, path):
    return CliRunner().invoke(app, ["questions", "review-import", str(path), "--config", str(config)])


def test_accept_and_reject_are_recorded_beside_the_frozen_set(workspace):
    root, config, questions_path = workspace
    before = sha(questions_path)
    result = run(
        config,
        sheet(
            root, {"k1v1-0001": {"verdict": "accept"}, "k1v1-0002": {"verdict": "reject", "note": "falsch"}}
        ),
    )
    assert result.exit_code == 0, result.output
    assert sha(questions_path) == before
    verdicts = read_jsonl(root / "data/questions/review/k1_v1_verdicts.jsonl")
    assert [(v["qid"], v["status"]) for v in verdicts] == [("k1v1-0001", "accept"), ("k1v1-0002", "reject")]
    assert verdicts[1]["note"] == "falsch"
    assert not (root / "data/questions/k1_v2.jsonl").exists()


def test_an_edit_goes_into_k1_v2_and_k1_v1_stays_byte_identical(workspace):
    root, config, questions_path = workspace
    before = sha(questions_path)
    result = run(
        config,
        sheet(
            root,
            {
                "k1v1-0001": {"verdict": "edit", "edited_question_de": EDIT},
                "k1v1-0003": {"verdict": "accept"},
            },
        ),
    )
    assert result.exit_code == 0, result.output
    assert sha(questions_path) == before and "k1_v2" in result.output
    successor = {q["qid"]: q for q in read_jsonl(root / "data/questions/k1_v2.jsonl")}
    assert len(successor) == 10
    assert successor["k1v1-0001"]["question_de"] == EDIT
    assert successor["k1v1-0001"]["review"]["status"] == "edit"
    assert successor["k1v1-0001"]["review"]["original"]["question_de"].startswith("Wie sind erneuerbare")
    assert successor["k1v1-0003"]["review"]["status"] == "accept"
    assert successor["k1v1-0002"]["review"]["status"] == "pending"
    assert {q["provenance"]["created_at"] for q in successor.values()} == {"2026-10-25"}
    assert successor["k1v1-0002"]["question_de"] == read_jsonl(questions_path)[1]["question_de"]
    verdicts = read_jsonl(root / "data/questions/review/k1_v1_verdicts.jsonl")
    assert {v["qid"] for v in verdicts} == {"k1v1-0001", "k1v1-0003"}


def test_a_later_import_adds_to_the_verdicts_and_rebuilds_the_successor_from_all_of_them(workspace):
    root, config, questions_path = workspace
    assert (
        run(config, sheet(root, {"k1v1-0001": {"verdict": "edit", "edited_question_de": EDIT}})).exit_code
        == 0
    )
    assert run(config, sheet(root, {"k1v1-0004": {"verdict": "reject"}})).exit_code == 0
    successor = {q["qid"]: q for q in read_jsonl(root / "data/questions/k1_v2.jsonl")}
    assert (
        successor["k1v1-0001"]["question_de"] == EDIT
        and successor["k1v1-0004"]["review"]["status"] == "reject"
    )
    assert len(read_jsonl(root / "data/questions/review/k1_v1_verdicts.jsonl")) == 2


def test_an_edit_that_fails_validation_writes_nothing(workspace):
    root, config, questions_path = workspace
    result = run(config, sheet(root, {"k1v1-0001": {"verdict": "edit", "edited_question_de": "Keine Frage"}}))
    assert result.exit_code == 1 and "must end with '?'" in result.output
    assert not (root / "data/questions/k1_v2.jsonl").exists()
    assert not (root / "data/questions/review/k1_v1_verdicts.jsonl").exists()


def test_a_bad_sheet_changes_nothing(workspace):
    root, config, questions_path = workspace
    before = sha(questions_path)
    result = run(
        config, sheet(root, {"k1v1-0001": {"verdict": "accept"}, "k1v1-0002": {"verdict": "unsure"}})
    )
    assert result.exit_code == 1 and sha(questions_path) == before
    assert list((root / "data/questions/review").glob("*.jsonl")) == []


def test_checked_subset_reads_accept_and_reject_but_not_edit_verdicts(workspace):
    root, config, _ = workspace
    cfg = load_config(config, "full")
    assert all(q["review"]["status"] == "pending" for q in load_questions(cfg))
    sheet_path = sheet(
        root,
        {
            "k1v1-0001": {"verdict": "accept"},
            "k1v1-0002": {"verdict": "edit", "edited_question_de": EDIT_2},
            "k1v1-0003": {"verdict": "reject"},
        },
    )
    result = run(config, sheet_path)
    assert result.exit_code == 0, result.output
    status = {q["qid"]: q["review"]["status"] for q in load_questions(cfg)}
    assert status["k1v1-0001"] == "accept" and status["k1v1-0003"] == "reject"
    assert status["k1v1-0002"] == "pending"


def test_without_a_verdict_file_the_published_questions_load_unchanged():
    cfg = load_config(ROOT / "pruefstand.toml", "full")
    assert not (QUESTIONS / "review" / "k1_v1_verdicts.jsonl").exists()
    assert load_questions(cfg) == read_jsonl(QUESTIONS / "k1_v1.jsonl")
    assert apply_checked_verdicts([{"qid": "a", "review": {}}], []) == [{"qid": "a", "review": {}}]


def test_the_frozen_set_keeps_its_known_created_at_value():
    """2026-10-26 is a clock assumption of the authoring agent (the work was done on 2026-10-25); it stays in
    k1_v1 because the published run pins the file's SHA-256, and is corrected in k1_v2."""
    for path in (QUESTIONS / "k1_v1.jsonl", ROOT / "fixtures" / "ci" / "questions.jsonl"):
        assert {q["provenance"]["created_at"] for q in read_jsonl(path)} == {"2026-10-26"}
    pinned = json.loads((ROOT / "runs" / "full-2026-10-25" / "manifest.json").read_text(encoding="utf-8"))
    assert pinned["question_set"]["sha256"] == sha(QUESTIONS / "k1_v1.jsonl")
    datasheet = (QUESTIONS / "DATASHEET.md").read_text(encoding="utf-8")
    assert "2026-10-26" in datasheet and "k1_v2" in datasheet
