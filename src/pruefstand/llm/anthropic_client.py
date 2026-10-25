"""Anthropic backend: Messages API for online calls, Message Batches API for the judge.

Request shape follows the claude-api skill (cached 2026-10-06): effort and structured output live in
`output_config`; `temperature`, `top_p` and `top_k` are never sent (HTTP 400 on every model since 2026-10-21);
a refusal (`stop_reason == "refusal"`) is returned as a normal response and scored, never retried.
Server-side refusal fallbacks are not used: the Batches API rejects them and Haiku 5.5 has none.

Batch flow: the batch id and the request keys are written to a state file on submit, the wait has a maximum and
survives transient poll errors, errored/expired/canceled items are retried once in a second batch, and items that
still fail come back marked `failed` with the error detail (the caller does not store them in the cassette).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Any

from pruefstand.errors import PruefstandError
from pruefstand.io import read_json, sha256_text, write_json
from pruefstand.llm.base import LLMRequest, LLMResponse, Usage

DEFAULT_MAX_WAIT_SECONDS = 26 * 3600  # a Batch is processed within 24 hours
MAX_POLL_ERRORS = 5


class BatchWaitError(PruefstandError):
    """The batch did not end in time or could not be polled; the batch id is in the message and the state file."""


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


def error_detail(result: Any) -> dict[str, Any]:
    """Type, message and request id of a failed Batch item, as far as the SDK object carries them."""
    wrapper = getattr(result, "error", None)
    inner = getattr(wrapper, "error", None) or wrapper
    return {
        "result_type": getattr(result, "type", ""),
        "error_type": getattr(inner, "type", None),
        "message": getattr(inner, "message", None),
        "request_id": getattr(wrapper, "request_id", None),
    }


def failed_response(
    request: LLMRequest, result_type: str, batch_id: str, detail: dict[str, Any]
) -> LLMResponse:
    return LLMResponse(
        text="",
        stop_reason=f"batch_{result_type}",
        usage=Usage(),
        model=request.model,
        batch=True,
        provider_meta={"failed": True, "batch_id": batch_id, "error": detail},
    )


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        client: Any = None,
        poll_seconds: int = 60,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds
        self._sleep = sleep
        self._monotonic = monotonic

    @property
    def sdk_version(self) -> str:
        try:
            return f"anthropic {metadata.version('anthropic')}"
        except metadata.PackageNotFoundError:
            return "anthropic (version unknown)"

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

    def wait_for_batch(self, batch_id: str) -> None:
        """Poll until the batch ended; a maximum wait and repeated poll errors end with BatchWaitError."""
        started = self._monotonic()
        errors = 0
        while True:
            try:
                status = self._client.messages.batches.retrieve(batch_id).processing_status
                errors = 0
            except Exception as exc:
                errors += 1
                status = None
                if errors >= MAX_POLL_ERRORS:
                    raise BatchWaitError(
                        f"batch {batch_id}: {errors} polls in a row failed ({exc}); run the command again to resume"
                    ) from exc
            if status == "ended":
                return
            if self._monotonic() - started > self.max_wait_seconds:
                raise BatchWaitError(
                    f"batch {batch_id} did not end within {self.max_wait_seconds:.0f} s; it keeps running, "
                    "run the same command again to resume and collect it"
                )
            self._sleep(self.poll_seconds)

    def collect_batch(self, batch_id: str, requests: list[LLMRequest]) -> list[LLMResponse]:
        self.wait_for_batch(batch_id)
        by_id: dict[str, LLMResponse] = {}
        for item in self._client.messages.batches.results(batch_id):
            request = requests[int(item.custom_id[1:])]
            if item.result.type == "succeeded":
                response = to_response(item.result.message, request, None, batch=True)
                by_id[item.custom_id] = replace(
                    response,
                    provider_meta={
                        **response.provider_meta,
                        "batch_id": batch_id,
                        "custom_id": item.custom_id,
                    },
                )
            else:
                by_id[item.custom_id] = failed_response(
                    request, item.result.type, batch_id, error_detail(item.result)
                )
        return [
            by_id.get(f"r{index:05d}")
            or failed_response(request, "missing", batch_id, {"message": "no result for this custom_id"})
            for index, request in enumerate(requests)
        ]

    def _run_pass(
        self, requests: list[LLMRequest], keys: list[str], state: dict, index: int, state_path: Path | None
    ) -> list[LLMResponse]:
        """One batch round: resume the recorded batch id when the request keys match, else submit."""
        fingerprint = sha256_text("\n".join(keys))
        passes = state["passes"]
        if index < len(passes) and passes[index]["fingerprint"] == fingerprint:
            batch_id = passes[index]["batch_id"]
        else:
            batch_id = self.submit_batch(requests)
            del passes[index:]
            passes.append(
                {
                    "kind": "initial" if index == 0 else "retry",
                    "batch_id": batch_id,
                    "fingerprint": fingerprint,
                    "n_requests": len(requests),
                    "keys": keys,
                }
            )
            if state_path is not None:
                write_json(state_path, state)
        return self.collect_batch(batch_id, requests)

    def complete_batch(
        self, requests: list[LLMRequest], keys: list[str] | None = None, state_path: Path | None = None
    ) -> list[LLMResponse]:
        """Submit (or resume), poll with a timeout, retry errored/expired/canceled items once.

        Items that still fail come back with provider_meta['failed'] and the error detail; the caller does not
        store them, so a later run submits them again.
        """
        keys = keys or [f"r{index:05d}" for index in range(len(requests))]
        state: dict = {"passes": []}
        if state_path is not None and Path(state_path).exists():
            state = read_json(state_path)
        responses = self._run_pass(requests, keys, state, 0, state_path)
        failed = [i for i, response in enumerate(responses) if response.provider_meta.get("failed")]
        if failed:
            second = self._run_pass(
                [requests[i] for i in failed], [keys[i] for i in failed], state, 1, state_path
            )
            for i, response in zip(failed, second, strict=True):
                if response.provider_meta.get("failed"):
                    response = replace(response, provider_meta={**response.provider_meta, "retried": True})
                responses[i] = response
        return responses
