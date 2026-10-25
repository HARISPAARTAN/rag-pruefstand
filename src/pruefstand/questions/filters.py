"""Automatic quality filters for authored questions (lexical, so they run in CI without models).

The spec's embedding-based duplicate filter (cosine > 0.92) is replaced by a content-word Jaccard filter here;
an embedding check is a roadmap item.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from bm25s.stopwords import STOPWORDS_GERMAN

from pruefstand.corpus.gii import Unit

WORD_RE = re.compile(r"\w+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
STOPWORDS = frozenset(STOPWORDS_GERMAN)
COPY_MAX = 0.6
SUPPORT_MIN = 0.5
COLLOQUIAL_OVERLAP_MAX = 0.5
DUPLICATE_MAX = 0.8


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [word for word in words(text) if len(word) >= 4 and word not in STOPWORDS]


def _prefixes(text: str) -> set[str]:
    return {word[:5] for word in content_words(text)}


def copy_ratio(question: str, source: str) -> float:
    """Share of the question's word trigrams that appear verbatim in the source text."""
    tokens = words(question)
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if not trigrams:
        return 0.0
    source_tokens = words(source)
    source_trigrams = {tuple(source_tokens[i : i + 3]) for i in range(len(source_tokens) - 2)}
    return round(sum(1 for gram in trigrams if gram in source_trigrams) / len(trigrams), 4)


def lexical_overlap(text: str, source: str) -> float:
    """Share of content words in text whose first five letters occur among the source's content words."""
    mine = content_words(text)
    if not mine:
        return 0.0
    theirs = _prefixes(source)
    return round(sum(1 for word in mine if word[:5] in theirs) / len(mine), 4)


def numbers_supported(answer: str, passage: str) -> bool:
    return set(NUMBER_RE.findall(answer)) <= set(NUMBER_RE.findall(passage))


def jaccard(a: str, b: str) -> float:
    first, second = set(content_words(a)), set(content_words(b))
    if not first or not second:
        return 0.0
    return round(len(first & second) / len(first | second), 4)


def run_filters(
    records: Sequence[Mapping], units_by_id: Mapping[str, Unit]
) -> tuple[list[dict], list[tuple[int, str]]]:
    """Return per-record filter results and (record index, problem) pairs; no pairs means every record passes."""
    results: list[dict] = []
    errors: list[tuple[int, str]] = []
    for index, record in enumerate(records):
        unit_ids = record["gold_unit_ids"] or record["source_unit_ids"]
        source = " ".join(units_by_id[unit_id].text for unit_id in unit_ids)
        result = {
            "copy_ratio": copy_ratio(record["question_de"], source),
            "lexical_overlap": lexical_overlap(record["question_de"], source),
            "support_ratio": None,
            "numbers_supported": None,
            "max_duplicate_jaccard": 0.0,
        }
        problems = []
        if result["copy_ratio"] > COPY_MAX:
            problems.append(f"copies {result['copy_ratio']:.0%} of the source wording (max {COPY_MAX:.0%})")
        if record["type"] == "colloquial" and result["lexical_overlap"] > COLLOQUIAL_OVERLAP_MAX:
            problems.append(f"colloquial question shares {result['lexical_overlap']:.0%} of legal wording")
        if record["type"] != "unanswerable":
            result["support_ratio"] = lexical_overlap(record["gold_answer_de"], record["gold_passage_de"])
            result["numbers_supported"] = numbers_supported(
                record["gold_answer_de"], record["gold_passage_de"]
            )
            if result["support_ratio"] < SUPPORT_MIN:
                problems.append(
                    f"gold answer only {result['support_ratio']:.0%} supported by the gold passage"
                )
            if not result["numbers_supported"]:
                problems.append("gold answer contains a number that is not in the gold passage")
        for earlier in range(index):
            score = jaccard(record["question_de"], records[earlier]["question_de"])
            result["max_duplicate_jaccard"] = max(result["max_duplicate_jaccard"], score)
            if score > DUPLICATE_MAX:
                problems.append(f"near duplicate of record {earlier + 1} (Jaccard {score:.2f})")
        result["passed"] = not problems
        results.append(result)
        errors.extend((index, problem) for problem in problems)
    return results, errors
