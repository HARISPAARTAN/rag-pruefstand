"""Ollama backend over plain HTTP (POST /api/chat, no SDK) for the on-premise generator.

Context size, temperature and seed are sent explicitly (a request without them would run on the server's
defaults: a context window smaller than the prompt, which Ollama silently clips, and sampling at 0.8). The
prompt token count Ollama returns is checked against the context window; a prompt that does not fit fails the
call instead of being measured on a clipped prompt.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from pruefstand.errors import PruefstandError
from pruefstand.llm.base import OLLAMA_DEFAULTS, LLMRequest, LLMResponse, Usage

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(PruefstandError):
    pass


class PromptDoesNotFit(OllamaError):
    pass


def effective_options(request: LLMRequest) -> dict[str, Any]:
    return {**OLLAMA_DEFAULTS, **(request.options or {})}


def build_payload(request: LLMRequest) -> dict[str, Any]:
    messages = [{"role": "system", "content": request.system}] if request.system else []
    messages.extend({"role": role, "content": content} for role, content in request.messages)
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": request.max_tokens, **effective_options(request)},
    }
    if request.thinking is not None:
        payload["think"] = request.thinking != "disabled"
    if request.response_schema:
        payload["format"] = request.response_schema
    return payload


def normalise_host(host: str) -> str:
    """Ollama's own convention allows 'host:port' without a scheme."""
    host = host.rstrip("/")
    return host if "://" in host else f"http://{host}"


def check_prompt_fits(request: LLMRequest, prompt_tokens: int | None) -> None:
    num_ctx = int(effective_options(request)["num_ctx"])
    if prompt_tokens is None:
        raise PromptDoesNotFit(
            "Ollama returned no prompt_eval_count, so the prompt cannot be verified against "
            f"num_ctx={num_ctx}; the call is rejected (fail closed)"
        )
    if prompt_tokens + request.max_tokens > num_ctx:
        raise PromptDoesNotFit(
            f"prompt of {prompt_tokens} tokens plus {request.max_tokens} output tokens does not fit "
            f"num_ctx={num_ctx} ({request.tag or request.prompt_id}); raise num_ctx in [llm.local]"
        )


class OllamaClient:
    provider = "ollama"
    sdk_version = "ollama http api"

    def __init__(self, host: str | None = None, timeout: float = 900.0) -> None:
        self.host = normalise_host(host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST)
        self.timeout = timeout

    def complete(self, request: LLMRequest) -> LLMResponse:
        body = json.dumps(build_payload(request)).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaError(f"Ollama at {self.host} did not answer: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        prompt_tokens = data.get("prompt_eval_count")
        check_prompt_fits(request, prompt_tokens)
        text = data.get("message", {}).get("content", "")
        parsed = None
        if request.response_schema and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        stop_reason = data.get("done_reason", "stop")
        options = effective_options(request)
        return LLMResponse(
            text=text,
            stop_reason="max_tokens" if stop_reason == "length" else stop_reason,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=data.get("eval_count", 0)),
            latency_ms=round(latency_ms, 1),
            parsed=parsed,
            model=data.get("model", request.model),
            provider_meta={
                "total_duration_ns": data.get("total_duration"),
                "prompt_eval_count": prompt_tokens,
                "num_ctx": options["num_ctx"],
                "temperature": options["temperature"],
                "seed": options["seed"],
            },
        )
