"""Synthetic LLM backend for CI fixtures: returns hand-written answers and judgements, clearly marked synthetic.

These texts were written by the implementing agent, not produced by any model. They exist only so CI can exercise
the replay path end to end without an API key; no reported number may come from them.
"""

from __future__ import annotations

import json

from pruefstand.llm.base import LLMRequest, LLMResponse, Usage

JUDGE_FIELDS = ("faithfulness", "correctness", "completeness", "rationale")


class SyntheticClient:
    provider = "synthetic"

    def __init__(self, responses: dict) -> None:
        self.answers = responses["answers"]
        self.judgements = responses["judgements"]

    def complete(self, request: LLMRequest) -> LLMResponse:
        stage, config_id, qid = request.tag.split("|")
        key = f"{config_id}|{qid}"
        if stage == "answer":
            entry = self.answers[key]
            if not entry["text"].strip():
                raise ValueError(f"synthetic answer for {key} is empty")
            return LLMResponse(
                text=entry["text"],
                stop_reason=entry.get("stop_reason", "end_turn"),
                usage=Usage(input_tokens=entry["input_tokens"], output_tokens=entry["output_tokens"]),
                latency_ms=float(entry["latency_ms"]),
                model=request.model,
                provider_meta={"synthetic": True},
            )
        entry = self.judgements[key]
        verdict = {name: entry[name] for name in JUDGE_FIELDS}
        if not str(verdict["rationale"]).strip():
            raise ValueError(f"synthetic judgement for {key} has no rationale")
        return LLMResponse(
            text=json.dumps(verdict, ensure_ascii=False),
            stop_reason="end_turn",
            usage=Usage(input_tokens=entry["input_tokens"], output_tokens=entry["output_tokens"]),
            parsed=verdict,
            model=request.model,
            batch=True,
            provider_meta={"synthetic": True},
        )
