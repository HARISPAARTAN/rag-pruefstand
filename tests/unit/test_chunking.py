import lzma
from dataclasses import replace

import pytest

from pruefstand.chunking import (
    CLAUSE_BOUNDARY,
    SENTENCE_BOUNDARY,
    c2_header,
    chunk_c1,
    chunk_c2,
    count_tokens,
    load_tokenizer,
    split_unit_text,
)
from pruefstand.corpus.gii import Unit
from pruefstand.errors import PruefstandError


@pytest.fixture(scope="module")
def tokenizer():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return load_tokenizer(str(root / "data" / "tokenizer" / "multilingual-e5-base.tokenizer.json.xz"))


def make_unit(slug: str, norm: str, absatz: str, text: str) -> Unit:
    number = norm.removeprefix("§ ")
    return Unit(
        unit_id=f"{slug}:§{number}:abs{absatz}",
        law_slug=slug,
        law_abbr=slug.upper(),
        law_title="Test",
        norm=norm,
        heading="Überschrift",
        absatz=absatz,
        text=text,
        norm_char_start=0,
        norm_char_end=len(text),
        source_url="",
        snapshot_date="2026-10-25",
        stand="",
    )


def long_text(n_sentences: int) -> str:
    return " ".join(
        f"Der Netzbetreiber muss im Fall {i} die Messstelle nach Absatz 2 prüfen." for i in range(n_sentences)
    )


def test_c1_windows_overlap_exactly(tokenizer):
    units = [make_unit("lawa", f"§ {i}", "1", long_text(12)) for i in range(1, 6)]
    chunks = chunk_c1(units, tokenizer, window=100, overlap=20)
    law_text = "\n".join(u.text for u in units)
    offsets = tokenizer.encode(law_text, add_special_tokens=False).offsets
    assert len(chunks) > 3
    for index, chunk in enumerate(chunks[:-1]):
        begin = index * 80
        assert chunk.n_tokens == 100
        assert chunk.text == law_text[offsets[begin][0] : offsets[begin + 99][1]]
        next_begin = (index + 1) * 80
        assert offsets[next_begin][0] < offsets[begin + 99][1]
    assert chunks[-1].n_tokens <= 100
    assert chunks[-1].text.endswith(law_text[-30:])


def test_c1_covers_map_back_to_units(tokenizer):
    units = [make_unit("lawa", "§ 1", "1", long_text(3)), make_unit("lawa", "§ 1", "2", long_text(3))]
    chunks = chunk_c1(units, tokenizer, window=1000, overlap=10)
    assert len(chunks) == 1
    covers = chunks[0].covers
    assert [c["unit_id"] for c in covers] == ["lawa:§1:abs1", "lawa:§1:abs2"]
    assert covers[0]["start"] == 0 and covers[0]["end"] == len(units[0].text)
    assert covers[1]["end"] == len(units[1].text)


def test_c1_never_mixes_laws(tokenizer):
    units = [make_unit("lawa", "§ 1", "1", long_text(4)), make_unit("lawb", "§ 1", "1", long_text(4))]
    for chunk in chunk_c1(units, tokenizer, window=60, overlap=10):
        assert {c["unit_id"].split(":")[0] for c in chunk.covers} == {chunk.law_slug}


def test_c2_never_crosses_an_absatz_and_keeps_header(tokenizer):
    units = [
        make_unit("lawa", "§ 14a", "1", long_text(40)),
        make_unit("lawa", "§ 14a", "2", "(2) Kurzer Absatz."),
    ]
    chunks = chunk_c2(units, tokenizer, max_tokens=120)
    assert len(chunks) > 2
    for chunk in chunks:
        assert len(chunk.covers) == 1
        assert chunk.n_tokens <= 120
        assert chunk.text.startswith(chunk.header + "\n")
    assert chunks[0].header == "LAWA § 14a Überschrift, Abs. 1"
    assert chunks[-1].chunk_id == "C2:lawa:§14a:abs2:0"


def test_split_unit_text_is_contiguous_and_bounded(tokenizer):
    text = long_text(30) + " " + ", ".join(f"Nummer {i} gilt" for i in range(200))
    pieces = split_unit_text(text, tokenizer, 64)
    assert pieces[0][0] == 0 and pieces[-1][1] == len(text)
    for (_, end), (next_start, _) in zip(pieces, pieces[1:], strict=False):
        assert text[end:next_start].strip() == ""
    assert all(count_tokens(tokenizer, text[s:e]) <= 64 for s, e in pieces)


def test_sentence_boundary_splits_real_sentence_ends():
    text = "Der Betreiber prüft die Anlage. Danach meldet er das Ergebnis; „Zitat“ folgt. § 5 gilt."
    assert SENTENCE_BOUNDARY.split(text) == [
        "Der Betreiber prüft die Anlage.",
        "Danach meldet er das Ergebnis;",
        "„Zitat“ folgt.",
        "§ 5 gilt.",
    ]


def test_sentence_boundary_needs_an_uppercase_or_quote_after_the_period():
    assert SENTENCE_BOUNDARY.split("Es gilt Satz 1. entsprechend 2. und weiter.") == [
        "Es gilt Satz 1. entsprechend 2. und weiter."
    ]
    assert len(SENTENCE_BOUNDARY.split("Es gilt. Danach folgt es.")) == 2


@pytest.mark.parametrize(
    "text",
    [
        "Betreiber von Gasversorgungsnetzen, 11. Netzbetreiber im Sinne des Gesetzes",
        "Die Frist endet am 31. Dezember jedes Jahres",
        "nach Maßgabe von 7. Anlagen, 12. Speicher und 123. Betreiber",
        "Dies gilt z. B. Netzbetreiber und d. h. Anlagenbetreiber",
        "wie u. a. Messstellenbetreiber und i. V. m. Netzbetreibern",
        "nach Teil XIV. Anlagen und nach VI. Teil sowie IV. Abschnitt",
    ],
)
def test_sentence_boundary_ignores_ordinals_letters_and_roman_numerals(text):
    assert SENTENCE_BOUNDARY.split(text) == [text]


@pytest.mark.parametrize(
    "abbreviation",
    ["Abs", "Nr", "Satz", "S", "Art", "ggf", "bzw", "vgl", "Buchst", "etc", "usw", "ABl", "BGBl", "Anl"],
)
def test_sentence_boundary_ignores_abbreviations(abbreviation):
    text = f"Der Betreiber prüft {abbreviation}. Messstellenbetreiber melden es."
    assert SENTENCE_BOUNDARY.split(text) == [text]


def test_sentence_boundary_still_splits_after_a_number_in_a_reference():
    text = "Es gilt nach § 14. Der Betreiber meldet. Nach Absatz 2. Dann folgt es."
    assert SENTENCE_BOUNDARY.split(text) == [
        "Es gilt nach § 14.",
        "Der Betreiber meldet.",
        "Nach Absatz 2.",
        "Dann folgt es.",
    ]


def test_clause_boundary_splits_after_commas_only():
    assert CLAUSE_BOUNDARY.split("eins, zwei drei,  vier") == ["eins,", "zwei drei,", "vier"]


def test_split_unit_text_prefers_sentence_ends(tokenizer):
    sentence = (
        "Der Betreiber meldet es bzw. Messstellenbetreiber prüfen es ggf. Anlagenbetreiber nach Abs. 2 Nr. 3."
    )
    text = " ".join([sentence] * 12)
    limit = count_tokens(tokenizer, sentence) * 2 - 4
    pieces = split_unit_text(text, tokenizer, limit)
    assert len(pieces) == 12
    assert all(text[start:end] == sentence for start, end in pieces)


def test_split_unit_text_falls_back_to_clauses_then_tokens(tokenizer):
    text = ", ".join(f"Nummer {i} gilt für den Netzbetreiber" for i in range(80))
    pieces = split_unit_text(text, tokenizer, 64)
    assert len(pieces) > 3
    assert all(text[start:end].endswith(",") for start, end in pieces[:-1])
    assert all(text[start:end].startswith("Nummer") for start, end in pieces)
    no_commas = " ".join(f"wort{i}" for i in range(300))
    hard = split_unit_text(no_commas, tokenizer, 64)
    assert len(hard) > 3
    assert all(count_tokens(tokenizer, no_commas[s:e]) <= 64 for s, e in hard)


def test_c2_covers_record_where_a_part_starts_inside_the_unit(tokenizer):
    unit = make_unit("lawa", "§ 3", "1", long_text(40))
    chunks = chunk_c2([unit], tokenizer, max_tokens=80)
    assert len(chunks) > 3
    assert chunks[0].covers[0]["start"] == 0
    for chunk in chunks:
        (cover,) = chunk.covers
        assert chunk.text == f"{chunk.header}\n{unit.text[cover['start'] : cover['end']]}"
    assert [c.covers[0]["start"] for c in chunks[1:]] == [c.covers[0]["end"] + 1 for c in chunks[:-1]]
    assert chunks[-1].covers[0]["end"] == len(unit.text)


def test_c1_covers_of_a_non_first_unit_are_relative_to_that_unit(tokenizer):
    units = [make_unit("lawa", f"§ {i}", "1", long_text(3)) for i in range(1, 5)]
    by_id = {unit.unit_id: unit for unit in units}
    chunks = chunk_c1(units, tokenizer, window=60, overlap=10)
    assert len(chunks) > 3
    starts_mid_unit = 0
    for chunk in chunks:
        assert chunk.text == "\n".join(by_id[c["unit_id"]].text[c["start"] : c["end"]] for c in chunk.covers)
        for cover in chunk.covers[1:]:
            assert cover["start"] == 0
        starts_mid_unit += chunk.covers[0]["start"] > 0 and chunk.covers[0]["unit_id"] != units[0].unit_id
    assert starts_mid_unit > 0


def test_c2_header_of_an_unnumbered_unit_has_no_absatz():
    unit = make_unit("lawa", "§ 2", "0", "Im Sinne dieses Gesetzes ist 1. Anlage.")
    assert c2_header(unit) == "LAWA § 2 Überschrift"
    assert c2_header(replace(unit, heading="")) == "LAWA § 2"
    assert c2_header(replace(unit, absatz="2")) == "LAWA § 2 Überschrift, Abs. 2"


def test_load_tokenizer_rejects_a_file_that_differs_from_the_pin(tmp_path):
    path = tmp_path / "tokenizer.json.xz"
    path.write_bytes(lzma.compress(b'{"not": "the pinned tokenizer"}'))
    with pytest.raises(PruefstandError, match="SHA-256"):
        load_tokenizer(str(path))
