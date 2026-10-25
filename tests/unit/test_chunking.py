import pytest

from pruefstand.chunking import chunk_c1, chunk_c2, count_tokens, load_tokenizer, split_unit_text
from pruefstand.corpus.gii import Unit


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
