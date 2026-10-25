"""Anthropic backend: Messages API for online calls, Message Batches API for the judge.

Request shape follows the claude-api skill (cached 2026-10-06): effort and structured output live in
`output_config`; `temperature`, `top_p` and `top_k` are never sent (HTTP 400 on every model since 2026-10-21);
a refusal (`stop_reason == "refusal"`) is returned as a normal response and scored, never retried.
Server-side refusal fallbacks are not used: the Batches API rejects them and Haiku 5.5 has none.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pruefstand.llm.base import LLMRequest, LLMResponse, Usage


def build_params(request: LLMRequest) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [{"role": role, "content": content} for role, content in request.messages],
    }
    if request.system:
        params["system"] = request.system
    if request.thinking:
        params["thinking"] = {"type": request.thinking}
    output_config: dict[str, Any] = {}
    if request.effort:
        output_config["effort"] = request.effort
    if request.response_schema:
        output_config["format"] = {"type": "json_schema", "schema": request.response_schema}
    if output_config:
        params["output_config"] = output_config
    return params


def to_response(
    message: Any, request: LLMRequest, latency_ms: float | None, batch: bool = False
) -> LLMResponse:
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    stop_reason = message.stop_reason or ""
    parsed = None
    if request.response_schema and stop_reason != "refusal" and text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    usage = message.usage
    meta: dict[str, Any] = {}
    stop_details = getattr(message, "stop_details", None)
    if stop_reason == "refusal" and stop_details is not None:
        meta["refusal_category"] = getattr(stop_details, "category", None)
    return LLMResponse(
        text=text,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        ),
        latency_ms=None if latency_ms is None else round(latency_ms, 1),
        parsed=parsed,
        request_id=getattr(message, "_request_id", None),
        model=getattr(message, "model", request.model) or request.model,
        batch=batch,
        provider_meta=meta,
    )


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, client: Any = None, poll_seconds: int = 60) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.poll_seconds = poll_seconds

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        message = self._client.messages.create(**build_params(request))
        return to_response(message, request, (time.perf_counter() - start) * 1000.0)

    def submit_batch(self, requests: list[LLMRequest]) -> str:
        batch = self._client.messages.batches.create(
            requests=[
                {"custom_id": f"r{index:05d}", "params": build_params(req)}
                for index, req in enumerate(requests)
            ]
        )
        return batch.id

    def collect_batch(self, batch_id: str, requests: list[LLMRequest]) -> list[LLMResponse]:
        while self._client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(self.poll_seconds)
        by_id: dict[str, LLMResponse] = {}
        for item in self._client.messages.batches.results(batch_id):
            request = requests[int(item.custom_id[1:])]
            if item.result.type == "succeeded":
                by_id[item.custom_id] = to_response(item.result.message, request, None, batch=True)
            else:
                by_id[item.custom_id] = LLMResponse(
                    text="",
                    stop_reason=f"batch_{item.result.type}",
                    usage=Usage(),
                    model=request.model,
                    batch=True,
                )
        return [by_id[f"r{index:05d}"] for index in range(len(requests))]

    def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        return self.collect_batch(self.submit_batch(requests), requests)
