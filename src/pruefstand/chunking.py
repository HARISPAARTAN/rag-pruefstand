"""Two chunking strategies over the same units.

C1: fixed windows of e5 tokens with overlap over the concatenated text of each law, ignoring structure.
C2: one chunk per Absatz with a header line; long Absätze are split at sentence, then clause, then token
boundaries, and every part keeps the header. Every chunk records the unit character spans it covers.
"""

from __future__ import annotations

import bisect
import lzma
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from pruefstand.config import Config
from pruefstand.corpus.gii import Unit
from pruefstand.errors import PruefstandError
from pruefstand.io import read_jsonl, sha256_bytes, write_jsonl

TOKENIZER_SHA256 = "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"
# A period does not end a sentence after a bare number of one to three digits (ordinal or list marker such as
# "11." or "31. Dezember"), a single letter ("z. B.", "d. h.", "i. V. m.", "u. a."), a Roman numeral of up to
# four letters, or one of these common German abbreviations.
ABBREVIATIONS = (
    "Abs",
    "Nr",
    "Nrn",
    "Satz",
    "S",
    "Art",
    "ggf",
    "bzw",
    "vgl",
    "Buchst",
    "etc",
    "usw",
    "ca",
    "bspw",
    "inkl",
    "evtl",
    "insb",
    "sog",
    "ff",
    "Hs",
    "Alt",
    "Ziff",
    "Anl",
    "Abschn",
    "Mio",
    "Mrd",
    "ABl",
    "BGBl",
)
# Exception to the number guard: after a legal reference word ("Absatz 2.", "§ 14.") a number is a reference
# and the period can end the sentence.
REFERENCE_WORDS = ("Absatz", "Satz", "Nummer", "Nr.", "Abs.", "§", "§§", "Buchstabe", "Anlage", "Teil")
_BARE_NUMBER = "".join(rf"(?<!\b\d{{{n}}}\.)" for n in (1, 2, 3))
_AFTER_REFERENCE = "|".join(
    rf"(?<={re.escape(word)} \d{{{n}}}\.)" for word in REFERENCE_WORDS for n in (1, 2, 3)
)
_ORDINAL_GUARDS = f"(?:{_BARE_NUMBER}|{_AFTER_REFERENCE})"
_LETTER_GUARD = r"(?<!\b[A-Za-zÄÖÜäöüß]\.)"
_ROMAN_GUARDS = "".join(rf"(?<!\b[IVXLCDM]{{{n}}}\.)" for n in (2, 3, 4))
_ABBREVIATION_GUARDS = "".join(rf"(?<!\b{abbr}\.)" for abbr in ABBREVIATIONS)
# A sentence end is [.;:!?] plus whitespace, followed by an uppercase letter, an opening quote or a paragraph sign.
SENTENCE_BOUNDARY = re.compile(
    _ORDINAL_GUARDS
    + _LETTER_GUARD
    + _ROMAN_GUARDS
    + _ABBREVIATION_GUARDS
    + r"(?<=[.;:!?])\s+(?=[A-ZÄÖÜ§\"„“»‚])"
)
CLAUSE_BOUNDARY = re.compile(r"(?<=,)\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    chunking: str
    corpus: str
    tenant: str
    law_slug: str
    text: str
    header: str
    n_tokens: int
    covers: list[dict] = field(default_factory=list)


def fetch_tokenizer(model_id: str, revision: str, out_path: Path) -> None:
    """Download tokenizer.json at a pinned revision, check its SHA-256 and store it xz-compressed."""
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/tokenizer.json"
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise PruefstandError(f"{url}: download failed: {exc}") from exc
    digest = sha256_bytes(data)
    if digest != TOKENIZER_SHA256:
        raise PruefstandError(f"tokenizer.json SHA-256 {digest} differs from pinned {TOKENIZER_SHA256}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(lzma.compress(data, preset=9))


@lru_cache(maxsize=4)
def load_tokenizer(path: str) -> Tokenizer:
    if not Path(path).is_file():
        raise PruefstandError(f"{path} not found; run `pruefstand corpus fetch-tokenizer` first")
    try:
        data = lzma.decompress(Path(path).read_bytes())
    except (OSError, lzma.LZMAError) as exc:
        raise PruefstandError(f"{path}: cannot read tokenizer: {exc}") from exc
    digest = sha256_bytes(data)
    if digest != TOKENIZER_SHA256:
        raise PruefstandError(f"{path}: tokenizer SHA-256 {digest} differs from pinned {TOKENIZER_SHA256}")
    return Tokenizer.from_str(data.decode("utf-8"))


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def _covers_for_span(start: int, end: int, unit_starts: list[int], units: list[Unit]) -> list[dict]:
    covers = []
    index = max(bisect.bisect_right(unit_starts, start) - 1, 0)
    while index < len(units) and unit_starts[index] < end:
        unit_start = unit_starts[index]
        unit_end = unit_start + len(units[index].text)
        lo, hi = max(start, unit_start), min(end, unit_end)
        if hi > lo:
            covers.append({"unit_id": units[index].unit_id, "start": lo - unit_start, "end": hi - unit_start})
        index += 1
    return covers


def chunk_c1(
    units: list[Unit], tokenizer: Tokenizer, window: int, overlap: int, corpus: str = "K1"
) -> list[Chunk]:
    if not 0 <= overlap < window:
        raise PruefstandError("overlap must be >= 0 and smaller than the window")
    stride = window - overlap
    chunks: list[Chunk] = []
    by_law: dict[str, list[Unit]] = {}
    for unit in units:
        by_law.setdefault(unit.law_slug, []).append(unit)
    for slug, law_units in by_law.items():
        starts: list[int] = []
        position = 0
        for unit in law_units:
            starts.append(position)
            position += len(unit.text) + 1
        law_text = "\n".join(unit.text for unit in law_units)
        offsets = tokenizer.encode(law_text, add_special_tokens=False).offsets
        n = len(offsets)
        begin = 0
        index = 0
        while n:
            stop = min(begin + window, n)
            char_start, char_end = offsets[begin][0], offsets[stop - 1][1]
            chunks.append(
                Chunk(
                    chunk_id=f"C1:{slug}:{index:05d}",
                    chunking="C1",
                    corpus=corpus,
                    tenant=corpus,
                    law_slug=slug,
                    text=law_text[char_start:char_end],
                    header="",
                    n_tokens=stop - begin,
                    covers=_covers_for_span(char_start, char_end, starts, law_units),
                )
            )
            index += 1
            if stop == n:
                break
            begin += stride
    return chunks


def _split_spans(text: str, start: int, end: int, pattern: re.Pattern) -> list[tuple[int, int]]:
    spans = []
    position = start
    for match in pattern.finditer(text, start, end):
        if match.start() > position:
            spans.append((position, match.start()))
        position = match.end()
    if position < end:
        spans.append((position, end))
    return spans


def _hard_split(
    text: str, start: int, end: int, tokenizer: Tokenizer, max_tokens: int
) -> list[tuple[int, int]]:
    """Cut a span into pieces of at most max_tokens tokens at token boundaries."""
    offsets = tokenizer.encode(text[start:end], add_special_tokens=False).offsets
    spans = []
    begin = 0
    while begin < len(offsets):
        stop = min(begin + max_tokens, len(offsets))
        while True:
            piece = (start + offsets[begin][0], start + offsets[stop - 1][1])
            if stop - begin == 1 or count_tokens(tokenizer, text[piece[0] : piece[1]]) <= max_tokens:
                break
            stop -= 1
        spans.append(piece)
        begin = stop
    return spans


def _segments(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[tuple[int, int]]:
    segments = []
    for sentence in _split_spans(text, 0, len(text), SENTENCE_BOUNDARY):
        if count_tokens(tokenizer, text[sentence[0] : sentence[1]]) <= max_tokens:
            segments.append(sentence)
            continue
        for clause in _split_spans(text, sentence[0], sentence[1], CLAUSE_BOUNDARY):
            if count_tokens(tokenizer, text[clause[0] : clause[1]]) <= max_tokens:
                segments.append(clause)
            else:
                segments.extend(_hard_split(text, clause[0], clause[1], tokenizer, max_tokens))
    return segments


def split_unit_text(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[tuple[int, int]]:
    """Greedy packing of segments into pieces of at most max_tokens tokens; returns character spans."""
    if count_tokens(tokenizer, text) <= max_tokens:
        return [(0, len(text))]
    pieces: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for segment in _segments(text, tokenizer, max_tokens):
        if current is None:
            current = segment
            continue
        candidate = (current[0], segment[1])
        if count_tokens(tokenizer, text[candidate[0] : candidate[1]]) <= max_tokens:
            current = candidate
        else:
            pieces.append(current)
            current = segment
    if current is not None:
        pieces.append(current)
    return pieces


def c2_header(unit: Unit) -> str:
    title = f"{unit.law_abbr} {unit.norm}"
    if unit.heading:
        title = f"{title} {unit.heading}"
    return title if unit.absatz == "0" else f"{title}, Abs. {unit.absatz}"


def chunk_c2(units: list[Unit], tokenizer: Tokenizer, max_tokens: int, corpus: str = "K1") -> list[Chunk]:
    chunks: list[Chunk] = []
    for unit in units:
        header = c2_header(unit)
        for part, (start, end) in enumerate(split_unit_text(unit.text, tokenizer, max_tokens)):
            body = unit.text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"C2:{unit.unit_id}:{part}",
                    chunking="C2",
                    corpus=corpus,
                    tenant=corpus,
                    law_slug=unit.law_slug,
                    text=f"{header}\n{body}",
                    header=header,
                    n_tokens=count_tokens(tokenizer, body),
                    covers=[{"unit_id": unit.unit_id, "start": start, "end": end}],
                )
            )
    return chunks


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    write_jsonl(path, (asdict(chunk) for chunk in chunks))


def load_chunks(path: str | Path) -> list[Chunk]:
    if not Path(path).is_file():
        raise PruefstandError(f"{path} not found; run `pruefstand chunk` first")
    return [Chunk(**row) for row in read_jsonl(path)]


def chunks_path(cfg: Config, chunking: str) -> Path:
    return cfg.run_dir / f"chunks_{chunking}.jsonl.gz"
