"""Answer generation from the top retrieved chunks, with mandatory citations and an exact abstention sentence."""

from __future__ import annotations

from collections.abc import Mapping

from pruefstand.chunking import Chunk
from pruefstand.citations import citation_precision, citation_recall, parse_citations
from pruefstand.corpus.gii import Unit
from pruefstand.llm.base import LLMRequest, LLMResponse, or_none
from pruefstand.prompts import Prompt
from pruefstand.questions.schema import ABSTAIN_ANSWER, normalise_ws


def chunk_labels(chunk: Chunk, units_by_id: Mapping[str, Unit]) -> list[str]:
    """Citation labels of the units a chunk covers, in order, without repeats."""
    return list(dict.fromkeys(units_by_id[cover["unit_id"]].label for cover in chunk.covers))


def build_context(chunks: list[Chunk], units_by_id: Mapping[str, Unit]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        labels = ", ".join(chunk_labels(chunk, units_by_id))
        blocks.append(f"[{number}] Fundstellen: {labels}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_answer_request(
    question: Mapping,
    chunks: list[Chunk],
    units_by_id: Mapping[str, Unit],
    prompt: Prompt,
    role: Mapping,
    tag: str,
) -> LLMRequest:
    user = f"Kontext:\n{build_context(chunks, units_by_id)}\n\nFrage: {question['question_de']}"
    return LLMRequest(
        provider=role["provider"],
        model=role["model"],
        system=prompt.text,
        messages=(("user", user),),
        max_tokens=int(role["max_tokens"]),
        effort=or_none(role.get("effort")),
        thinking=or_none(role.get("thinking")),
        prompt_id=prompt.prompt_id,
        prompt_sha256=prompt.sha256,
        tag=tag,
    )


def is_abstention(text: str) -> bool:
    return normalise_ws(ABSTAIN_ANSWER) in normalise_ws(text)


def answer_record(
    question: Mapping, config_id: str, chunks: list[Chunk], response: LLMResponse, cost: float, generator: str
) -> dict:
    citations = parse_citations(response.text)
    context_units = [cover["unit_id"] for chunk in chunks for cover in chunk.covers]
    return {
        "config_id": config_id,
        "qid": question["qid"],
        "generator": generator,
        "context_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "text": response.text,
        "citations": [f"{c.law_slug}|{c.norm}|{c.absatz or ''}" for c in citations],
        "citation_precision": citation_precision(citations, context_units),
        "citation_recall": citation_recall(citations, question["gold_unit_ids"]),
        "abstained": is_abstention(response.text),
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": response.usage.cache_read_input_tokens,
            "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
        },
        "latency_ms": response.latency_ms,
        "cost_usd": cost,
    }
