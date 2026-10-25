"""Record and replay of LLM calls.

The replay key is SHA-256 over the canonical JSON of the request (the prompt version is part of the request).
Modes: record (live call, stored), replay (cassette only, a miss raises ReplayMiss), live (no store).
"""

from __future__ import annotations

import json
from pathlib import Path

from pruefstand.errors import PruefstandError
from pruefstand.io import canonical_json, sha256_text
from pruefstand.llm.base import LLMClient, LLMRequest, LLMResponse


class ReplayMiss(PruefstandError):
    pass


def replay_key(request: LLMRequest) -> str:
    return sha256_text(canonical_json(request.to_dict()))


class Cassette:
    """JSON lines, one recorded call per line, sorted by key so diffs stay readable."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, request: LLMRequest, response: LLMResponse, synthetic: bool = False) -> str:
        key = replay_key(request)
        self.entries[key] = {
            "key": key,
            "synthetic": synthetic,
            "request": request.to_dict(include_tag=True),
            "response": response.to_dict(),
        }
        return key

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(self.entries[key], ensure_ascii=False, sort_keys=True) for key in sorted(self.entries)
        ]
        self.path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


class ReplayClient:
    def __init__(
        self, mode: str, cassette: Cassette, inner: LLMClient | None = None, synthetic: bool = False
    ) -> None:
        if mode not in ("record", "replay", "live"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode != "replay" and inner is None:
            raise ValueError(f"mode {mode!r} needs a provider client")
        self.mode = mode
        self.cassette = cassette
        self.inner = inner
        self.synthetic = synthetic
        self.provider = inner.provider if inner else "replay"

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.mode == "replay":
            entry = self.cassette.get(replay_key(request))
            if entry is None:
                raise ReplayMiss(f"no recording for {request.prompt_id} ({request.tag or 'untagged'})")
            return LLMResponse.from_dict(entry["response"])
        response = self.inner.complete(request)
        if self.mode == "record":
            self.cassette.put(request, response, synthetic=self.synthetic)
        return response

    def complete_many(self, requests: list[LLMRequest], use_batch: bool) -> list[LLMResponse]:
        """Answer several requests; a live Anthropic judge run goes through the Batch API (50% off)."""
        if self.mode == "replay" or not use_batch or not hasattr(self.inner, "complete_batch"):
            return [self.complete(request) for request in requests]
        responses = self.inner.complete_batch(requests)
        if self.mode == "record":
            for request, response in zip(requests, responses, strict=True):
                self.cassette.put(request, response, synthetic=self.synthetic)
        return responses
