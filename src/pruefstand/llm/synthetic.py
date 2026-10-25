"""Synthetic LLM backend for CI fixtures: returns hand-written answers and judgements, marked synthetic.

One judgement entry per answer answers both judge calls (faithfulness; correctness and completeness).

These texts were written by the implementing agent, not produced by any model. They exist only so CI can exercise
the replay path end to end without an API key; no reported number may come from them.
"""

from __future__ import annotations

import json

from pruefstand.llm.base import LLMRequest, LLMResponse, Usage


class SyntheticClient:
    provider = "synthetic"
    sdk_version = "synthetic"

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
        if stage == "judge_faithfulness":
            fields, input_tokens, output_tokens = (
                ("faithfulness",),
                entry["input_tokens"],
                entry["output_tokens"] // 2,
            )
        elif stage == "judge_correctness":
            # The reference call carries no context: about a quarter of the combined prompt of the old single call.
            fields = ("correctness", "completeness")
            input_tokens = entry["input_tokens"] // 4
            output_tokens = entry["output_tokens"] - entry["output_tokens"] // 2
        else:
            raise ValueError(f"unknown synthetic stage {stage!r}")
        verdict = {name: entry[name] for name in (*fields, "rationale")}
        if not str(verdict["rationale"]).strip():
            raise ValueError(f"synthetic judgement for {key} has no rationale")
        return LLMResponse(
            text=json.dumps(verdict, ensure_ascii=False),
            stop_reason="end_turn",
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            parsed=verdict,
            model=request.model,
            batch=True,
            provider_meta={"synthetic": True},
        )
