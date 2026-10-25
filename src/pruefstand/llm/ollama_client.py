"""Ollama backend over plain HTTP (POST /api/chat, no SDK) for the on-premise generator."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

from pruefstand.llm.base import LLMRequest, LLMResponse, Usage

DEFAULT_HOST = "http://localhost:11434"


def build_payload(request: LLMRequest) -> dict[str, Any]:
    messages = [{"role": "system", "content": request.system}] if request.system else []
    messages.extend({"role": role, "content": content} for role, content in request.messages)
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": request.max_tokens},
    }
    if request.thinking is not None:
        payload["think"] = request.thinking != "disabled"
    if request.response_schema:
        payload["format"] = request.response_schema
    return payload


class OllamaClient:
    provider = "ollama"

    def __init__(self, host: str | None = None, timeout: float = 900.0) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout

    def complete(self, request: LLMRequest) -> LLMResponse:
        body = json.dumps(build_payload(request)).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        start = time.perf_counter()
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = data.get("message", {}).get("content", "")
        parsed = None
        if request.response_schema and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        return LLMResponse(
            text=text,
            stop_reason=data.get("done_reason", "stop"),
            usage=Usage(
                input_tokens=data.get("prompt_eval_count", 0), output_tokens=data.get("eval_count", 0)
            ),
            latency_ms=round(latency_ms, 1),
            parsed=parsed,
            model=data.get("model", request.model),
            provider_meta={"total_duration_ns": data.get("total_duration")},
        )
