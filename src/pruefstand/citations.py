"""Parse German statute citations from generated answers and score them against units.

Handles the bracket form the answer prompt asks for ("[EnWG § 14a Abs. 1]", several separated by ";") and free
forms such as "§ 14a Absatz 1 Satz 2 EnWG", "§§ 20, 21 EnWG", "§ 12 Abs. 1 und 2 EnFG" or "§ 3 Nr. 5 EnWG".
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
SATZ = r"(?:\s+(?:Satz|S\.)\s*\d+)?"
NR = r"(?:\s+(?:Nr\.|Nummer)\s*\d+[a-z]?)?"
LIST_SEP = r"\s*(?:,|und|bis|sowie)\s*"

BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
BRACKET_PART_RE = re.compile(
    rf"^\s*(?P<law>{LAW_PATTERN})\s+§\s*(?P<norm>{NUM})(?:\s+{ABS}\s*(?P<abs>{NUM}))?"
)
INLINE_RE = re.compile(
    rf"§§?\s*(?P<norms>{NUM}(?:{LIST_SEP}{NUM})*)"
    rf"(?:\s+{ABS}\s*(?P<abs>{NUM}(?:{LIST_SEP}{NUM})*))?{SATZ}{NR}{SATZ}"
    rf"\s+(?:des\s+|der\s+)?(?P<law>{LAW_PATTERN})\b"
)


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


def law_slug(abbr: str) -> str:
    return LAW_ALIASES[abbr.split()[0].lower()]


def parse_unit_id(unit_id: str) -> tuple[str, str, str]:
    """'enwg_2005:§14a:abs1' -> ('enwg_2005', '§ 14a', '1')."""
    slug, norm, absatz = unit_id.split(":")
    return slug, f"§ {norm.removeprefix('§')}", absatz.removeprefix("abs").split("_")[0]


def _numbers(text: str) -> list[str]:
    return re.findall(NUM, text)


def parse_citations(text: str) -> list[Citation]:
    citations: list[Citation] = []
    remainder = text
    for match in BRACKET_RE.finditer(text):
        for part in match.group(1).split(";"):
            part_match = BRACKET_PART_RE.match(part)
            if part_match:
                citations.append(
                    Citation(
                        law_slug(part_match.group("law")),
                        f"§ {part_match.group('norm')}",
                        part_match.group("abs"),
                    )
                )
        remainder = remainder.replace(match.group(0), " ")
    for match in INLINE_RE.finditer(remainder):
        slug = law_slug(match.group("law"))
        norms = _numbers(match.group("norms"))
        absaetze = _numbers(match.group("abs")) if match.group("abs") else [None]
        if len(norms) > 1:
            absaetze = [None]
        for norm in norms:
            for absatz in absaetze:
                citations.append(Citation(slug, f"§ {norm}", absatz))
    return list(dict.fromkeys(citations))


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
