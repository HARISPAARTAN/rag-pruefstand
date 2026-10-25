"""Parse German statute citations from generated answers and score them against units.

Handles the bracket form the answer prompt asks for ("[EnWG § 14a Abs. 1]", several separated by ";") and free
forms such as "§ 14a Absatz 1 Satz 2 EnWG", "§§ 20, 21 EnWG", "§ 12 Abs. 1 und 2 EnFG", "§ 12 Abs. 1 bis 3 EnFG",
"§ 14a Abs. 1 Satz 1 und 2 EnWG", "§ 21 Absatz 1 Nummer 2 Buchstabe a EnWG", "§ 14 und § 15 EnWG" and
"§ 3 Nr. 24a, § 14 EnWG". Lists and ranges of Absätze (and plain-number norm ranges) expand to one citation each;
Satz, Nummer, Buchstabe and similar qualifiers are read and dropped (the unit is the Absatz).

Every `§` sign in the text is a mention; a mention that no parsed citation covers is counted as unparsed, so the
report can show how much of the citation behaviour the parser did not see (a bias of citation precision/recall).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

LAW_ALIASES = {
    "enwg": "enwg_2005",
    "eeg": "eeg_2014",
    "msbg": "messbg",
    "messbg": "messbg",
    "enfg": "enfg",
    "kwkg": "kwkg_2016",
}
LAW_PATTERN = r"(?:EnWG|EEG|MsbG|MessbG|EnFG|KWKG)(?:\s+20\d\d)?"
NUM = r"\d+[a-z]?"
ABS = r"(?:Abs\.|Absatz|Absätze)"
SEP = r"\s*(?:,|und|bis|sowie|oder|-|–)\s*"
NUMLIST = rf"{NUM}(?:{SEP}{NUM})*"
QTOKEN = r"(?:\d+[a-z]?|[a-z](?![a-zäöüß]))"
QUALIFIER = (
    r"(?:\s+(?:Sätze|Satz|S\.|Nummern|Nummer|Nr\.|Buchstabe|Buchst\.|Halbsatz|Alternative|Alt\.)"
    rf"\s*{QTOKEN}(?:{SEP}{QTOKEN})*)*"
)
CHAIN_SEP = r"\s*(?:,|;|und|sowie)\s*"


def _segment(named: bool) -> str:
    norms, absatz = (r"(?P<norms>", r"(?P<abs>") if named else (r"(?:", r"(?:")
    return (
        rf"§§?\s*{norms}{NUMLIST})"
        rf"(?:\s+{ABS}\s*{absatz}{NUMLIST}(?:{SEP}(?:{ABS}\s*)?{NUMLIST})*))?{QUALIFIER}"
    )


SEGMENT_RE = re.compile(_segment(named=True))
INLINE_RE = re.compile(
    rf"{_segment(named=False)}(?:{CHAIN_SEP}{_segment(named=False)})*"
    rf"\s+(?:des\s+|der\s+)?(?P<law>{LAW_PATTERN})\b"
)
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
BRACKET_LAW_RE = re.compile(rf"\s*(?:(?P<law>{LAW_PATTERN})\s+)?(?=§)")
MENTION_RE = re.compile(r"§§?")


@dataclass(frozen=True)
class Citation:
    law_slug: str
    norm: str
    absatz: str | None

    def matches(self, unit_id: str) -> bool:
        slug, norm, absatz = parse_unit_id(unit_id)
        if slug != self.law_slug or norm != self.norm:
            return False
        return self.absatz is None or self.absatz == absatz


@dataclass(frozen=True)
class ParseReport:
    citations: list[Citation]
    mentions: int  # number of "§" / "§§" signs in the text
    unparsed: int  # mentions that no parsed citation covers


def law_slug(abbr: str) -> str:
    return LAW_ALIASES[abbr.split()[0].lower()]


def parse_unit_id(unit_id: str) -> tuple[str, str, str]:
    """'enwg_2005:§14a:abs1' -> ('enwg_2005', '§ 14a', '1')."""
    slug, norm, absatz = unit_id.split(":")
    return slug, f"§ {norm.removeprefix('§')}", absatz.removeprefix("abs").split("_")[0]


def expand_numbers(text: str) -> list[str]:
    """'1 und 2' -> ['1', '2']; '1 bis 3' -> ['1', '2', '3']; ranges only over plain numbers, at most 20 wide."""
    tokens = re.findall(rf"{NUM}|bis|-|–", text)
    numbers: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("bis", "-", "–"):
            index += 1
            continue
        if index + 2 < len(tokens) and tokens[index + 1] in ("bis", "-", "–"):
            end = tokens[index + 2]
            if token.isdigit() and end.isdigit() and 0 <= int(end) - int(token) <= 20:
                numbers.extend(str(n) for n in range(int(token), int(end) + 1))
            else:
                numbers.extend([token, end])
            index += 3
            continue
        numbers.append(token)
        index += 1
    return list(dict.fromkeys(numbers))


def _segment_citations(match: re.Match[str], slug: str) -> list[Citation]:
    norms = expand_numbers(match.group("norms"))
    absaetze: list[str | None] = [None]
    if match.group("abs") and len(norms) == 1:
        absaetze = list(expand_numbers(match.group("abs")))
    return [Citation(slug, f"§ {norm}", absatz) for norm in norms for absatz in absaetze]


def _covered(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def parse_report(text: str) -> ParseReport:
    citations: list[Citation] = []
    spans: list[tuple[int, int]] = []
    masked = text
    for bracket in BRACKET_RE.finditer(text):
        content_start = bracket.start(1)
        last_law: str | None = None
        for part in re.finditer(r"[^;]+", bracket.group(1)):
            head = BRACKET_LAW_RE.match(part.group(0))
            if head is None:
                continue
            if head.group("law"):
                last_law = law_slug(head.group("law"))
            if last_law is None:
                continue
            for segment in SEGMENT_RE.finditer(part.group(0), head.end()):
                citations.extend(_segment_citations(segment, last_law))
                base = content_start + part.start()
                start, end = base + segment.start(), base + segment.end()
                spans.append((start, end))
                masked = masked[:start] + " " * (end - start) + masked[end:]
    for match in INLINE_RE.finditer(masked):
        slug = law_slug(match.group("law"))
        for segment in SEGMENT_RE.finditer(match.group(0)):
            citations.extend(_segment_citations(segment, slug))
        spans.append((match.start(), match.end()))
    mentions = [m.start() for m in MENTION_RE.finditer(text)]
    unparsed = sum(1 for position in mentions if not _covered(position, spans))
    return ParseReport(list(dict.fromkeys(citations)), len(mentions), unparsed)


def parse_citations(text: str) -> list[Citation]:
    return parse_report(text).citations


def citation_precision(citations: Iterable[Citation], context_unit_ids: Iterable[str]) -> float | None:
    """Share of citations that point to a unit present in the retrieved context; None without citations."""
    cited = list(citations)
    if not cited:
        return None
    units = list(context_unit_ids)
    return sum(1 for c in cited if any(c.matches(u) for u in units)) / len(cited)


def citation_recall(citations: Iterable[Citation], gold_unit_ids: Iterable[str]) -> float | None:
    """Share of gold units cited by at least one citation; None for questions without gold units."""
    gold = list(gold_unit_ids)
    if not gold:
        return None
    cited = list(citations)
    return sum(1 for u in gold if any(c.matches(u) for c in cited)) / len(gold)
