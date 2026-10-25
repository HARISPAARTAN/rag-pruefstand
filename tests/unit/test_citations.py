import pytest

from pruefstand.citations import (
    Citation,
    citation_precision,
    citation_recall,
    parse_citations,
    parse_report,
    parse_unit_id,
)

E = "enwg_2005"
CASES = [
    ("[EnWG § 14a Abs. 1]", [Citation(E, "§ 14a", "1")]),
    ("[EnWG § 14a]", [Citation(E, "§ 14a", None)]),
    ("[MsbG § 5 Abs. 1; MsbG § 5 Abs. 2]", [Citation("messbg", "§ 5", "1"), Citation("messbg", "§ 5", "2")]),
    ("[EEG 2023 § 21 Abs. 1]", [Citation("eeg_2014", "§ 21", "1")]),
    ("[KWKG 2025 § 7 Abs. 3]", [Citation("kwkg_2016", "§ 7", "3")]),
    ("[EnFG § 12 Absatz 2]", [Citation("enfg", "§ 12", "2")]),
    ("[MessbG § 3 Abs. 1]", [Citation("messbg", "§ 3", "1")]),
    ("nach § 14a Absatz 1 EnWG", [Citation(E, "§ 14a", "1")]),
    ("nach § 14a Abs. 1 Satz 2 EnWG", [Citation(E, "§ 14a", "1")]),
    ("gemäß § 3 Nr. 5 EnWG", [Citation(E, "§ 3", None)]),
    ("gemäß § 3 Nummer 24a EnWG", [Citation(E, "§ 3", None)]),
    ("§§ 20, 21 EnWG", [Citation(E, "§ 20", None), Citation(E, "§ 21", None)]),
    ("§§ 20 und 21 EnWG", [Citation(E, "§ 20", None), Citation(E, "§ 21", None)]),
    ("§ 12 Abs. 1 und 2 EnFG", [Citation("enfg", "§ 12", "1"), Citation("enfg", "§ 12", "2")]),
    ("§ 12 Absätze 1 und 2 EnFG", [Citation("enfg", "§ 12", "1"), Citation("enfg", "§ 12", "2")]),
    ("§ 5 Abs. 1 S. 2 des MsbG", [Citation("messbg", "§ 5", "1")]),
    ("§ 9 Abs. 1 EEG 2023", [Citation("eeg_2014", "§ 9", "1")]),
    ("[EnWG § 14a Abs. 1] und [EnWG § 14a Abs. 1]", [Citation(E, "§ 14a", "1")]),
    ("Siehe [EnWG § 1 Abs. 1] sowie § 2 Abs. 1 EnWG.", [Citation(E, "§ 1", "1"), Citation(E, "§ 2", "1")]),
    ("Keine Fundstelle, nur Text über Absatz 3.", []),
    # lists, ranges and qualifier forms (Task 6 review I7)
    ("[EnWG §§ 20, 21]", [Citation(E, "§ 20", None), Citation(E, "§ 21", None)]),
    ("[EnWG § 12 Abs. 1 und 2]", [Citation(E, "§ 12", "1"), Citation(E, "§ 12", "2")]),
    (
        "[EnWG § 12 Abs. 1 bis 3]",
        [Citation(E, "§ 12", "1"), Citation(E, "§ 12", "2"), Citation(E, "§ 12", "3")],
    ),
    (
        "[EnWG § 12 Abs. 1, 2 und 4]",
        [Citation(E, "§ 12", "1"), Citation(E, "§ 12", "2"), Citation(E, "§ 12", "4")],
    ),
    ("§ 14a Abs. 1 Satz 1 und 2 EnWG", [Citation(E, "§ 14a", "1")]),
    ("§ 21 Absatz 1 Nummer 2 Buchstabe a EnWG", [Citation(E, "§ 21", "1")]),
    ("§ 21 Abs. 1 Nr. 2 und 3 EnWG", [Citation(E, "§ 21", "1")]),
    ("§ 3 Nr. 24a, § 14 EnWG", [Citation(E, "§ 3", None), Citation(E, "§ 14", None)]),
    ("§ 14 und § 15 EnWG", [Citation(E, "§ 14", None), Citation(E, "§ 15", None)]),
    (
        "§ 14 Abs. 1 und Absatz 2 sowie § 15 EnWG",
        [Citation(E, "§ 14", "1"), Citation(E, "§ 14", "2"), Citation(E, "§ 15", None)],
    ),
    ("§§ 20 bis 22 EnWG", [Citation(E, "§ 20", None), Citation(E, "§ 21", None), Citation(E, "§ 22", None)]),
    ("[EnWG § 14a Abs. 1; § 14b]", [Citation(E, "§ 14a", "1"), Citation(E, "§ 14b", None)]),
    ("[§ 14a Abs. 1 EnWG]", [Citation(E, "§ 14a", "1")]),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_citation_forms(text, expected):
    assert parse_citations(text) == expected


def test_parse_unit_id_strips_duplicate_suffix():
    assert parse_unit_id("enwg_2005:§14a:abs1") == (E, "§ 14a", "1")
    assert parse_unit_id("enwg_2005:§14a:abs1_2") == (E, "§ 14a", "1")


def test_precision_and_recall():
    cites = [Citation(E, "§ 14a", "1"), Citation(E, "§ 3", None), Citation(E, "§ 99", "1")]
    context = ["enwg_2005:§14a:abs1", "enwg_2005:§3:abs0"]
    assert citation_precision(cites, context) == pytest.approx(2 / 3)
    assert citation_precision([], context) is None
    assert citation_recall(cites, ["enwg_2005:§14a:abs1", "enwg_2005:§14a:abs2"]) == 0.5
    assert citation_recall(cites, []) is None


def test_unparsed_mentions_are_counted_as_a_diagnostic():
    clean = parse_report("Siehe [EnWG § 1 Abs. 1] sowie § 2 Abs. 1 EnWG.")
    assert (clean.mentions, clean.unparsed) == (2, 0)
    bare = parse_report("Nach § 99 gilt etwas anderes, vgl. [EnWG § 5 Abs. 1].")
    assert bare.citations == [Citation(E, "§ 5", "1")]
    assert (bare.mentions, bare.unparsed) == (2, 1)
    unknown_law = parse_report("[BGB § 433 Abs. 1]")
    assert unknown_law.citations == [] and (unknown_law.mentions, unknown_law.unparsed) == (1, 1)
    assert parse_report("Kein Paragraphenzeichen.") == parse_report("Kein Paragraphenzeichen.")
    assert parse_report("Kein Paragraphenzeichen.").mentions == 0
    assert parse_report("Kein Paragraphenzeichen.").unparsed == 0
    mixed = parse_report("§§ 20, 21 EnWG und § 7 des Bürgerlichen Gesetzbuchs")
    assert len(mixed.citations) == 2 and (mixed.mentions, mixed.unparsed) == (2, 1)
