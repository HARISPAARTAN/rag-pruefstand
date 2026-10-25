"""Record and replay of LLM calls.

The replay key is SHA-256 over the canonical JSON of the request: provider, model, prompts (with hash), limits,
stage, provider options. The `tag` (config and question id) is not part of the key; the `stage` is, so identical
requests of different stages (answer, repeat_1, ...) are recorded separately.
Modes: record (reads the cassette first and only calls the provider for missing entries; `refresh` forces calls),
replay (cassette only, a miss raises ReplayMiss), live (no store).

Every entry carries the request, the response and a `record` block with timestamp, SDK version, batch id, the
price table date and the cost at record time. Failed Batch items (errored, expired, canceled) are never stored.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pruefstand.errors import PruefstandError
from pruefstand.io import canonical_json, sha256_text
from pruefstand.llm.base import LLMClient, LLMRequest, LLMResponse

SAVE_EVERY = 25


class ReplayMiss(PruefstandError):
    pass


def replay_key(request: LLMRequest) -> str:
    return sha256_text(canonical_json(request.to_dict()))


def is_failed(response: LLMResponse) -> bool:
    """Failed Batch items carry provider_meta['failed']; they are reported but never stored."""
    return bool(response.provider_meta.get("failed"))


class Cassette:
    """JSON lines, one recorded call per line, sorted by key so diffs stay readable.

    With allow_synthetic=False (every profile except ci) synthetic entries are invisible: a live run can never
    be served, or overwrite, hand-written CI fixtures.
    """

    def __init__(self, path: Path, allow_synthetic: bool = True) -> None:
        self.path = Path(path)
        self.allow_synthetic = allow_synthetic
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry

    def get(self, key: str) -> dict | None:
        entry = self.entries.get(key)
        if entry is not None and entry.get("synthetic") and not self.allow_synthetic:
            return None
        return entry

    def is_hidden_synthetic(self, key: str) -> bool:
        entry = self.entries.get(key)
        return bool(entry and entry.get("synthetic") and not self.allow_synthetic)

    def put(
        self, request: LLMRequest, response: LLMResponse, synthetic: bool = False, record: dict | None = None
    ) -> str:
        key = replay_key(request)
        self.entries[key] = {
            "key": key,
            "synthetic": synthetic,
            "request": request.to_dict(include_tag=True),
            "response": response.to_dict(),
            "record": record or {},
        }
        return key

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(self.entries[key], ensure_ascii=False, sort_keys=True) for key in sorted(self.entries)
        ]
        data = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, self.path)


class ReplayClient:
    def __init__(
        self,
        mode: str,
        cassette: Cassette,
        inner: LLMClient | None = None,
        synthetic: bool = False,
        refresh: bool = False,
        cost_fn: Callable[[LLMRequest, LLMResponse], float] | None = None,
        price_as_of: str = "",
        clock: Callable[[], str] | None = None,
    ) -> None:
        if mode not in ("record", "replay", "live"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode != "replay" and inner is None:
            raise ValueError(f"mode {mode!r} needs a provider client")
        self.mode = mode
        self.cassette = cassette
        self.inner = inner
        self.synthetic = synthetic
        self.refresh = refresh
        self.cost_fn = cost_fn
        self.price_as_of = price_as_of
        self.clock = clock or _now_iso
        self.provider = inner.provider if inner else "replay"
        self.reused = 0
        self.called = 0
        self._unsaved = 0

    def _lookup(self, request: LLMRequest) -> dict | None:
        return self.cassette.get(replay_key(request))

    def recorded(self, request: LLMRequest) -> dict | None:
        """The `record` block of the stored entry (timestamp, cost at record time, ...), or None."""
        entry = self._lookup(request)
        return None if entry is None else entry.get("record") or None

    def _miss(self, request: LLMRequest) -> ReplayMiss:
        if self.cassette.is_hidden_synthetic(replay_key(request)):
            return ReplayMiss(
                f"the recording for {request.prompt_id} ({request.tag or 'untagged'}) is synthetic and not "
                "allowed outside the ci profile"
            )
        return ReplayMiss(f"no recording for {request.prompt_id} ({request.tag or 'untagged'})")

    def _record_block(self, request: LLMRequest, response: LLMResponse) -> dict:
        cost = self.cost_fn(request, response) if self.cost_fn else None
        return {
            "recorded_at": self.clock(),
            "provider": request.provider,
            "model": response.model or request.model,
            "sdk_version": getattr(self.inner, "sdk_version", ""),
            "batch_id": response.provider_meta.get("batch_id"),
            "price_as_of": self.price_as_of,
            "cost_usd": cost,
        }

    def _store(self, request: LLMRequest, response: LLMResponse) -> None:
        if is_failed(response):
            return
        self.cassette.put(request, response, self.synthetic, self._record_block(request, response))
        self._unsaved += 1
        if self._unsaved >= SAVE_EVERY:
            self.save()

    def save(self) -> None:
        self.cassette.save()
        self._unsaved = 0

    def misses(self, requests: list[LLMRequest]) -> list[LLMRequest]:
        """Requests that would reach the provider in the current mode (all of them when refreshing)."""
        if self.mode == "replay":
            return []
        if self.mode == "live" or self.refresh:
            return list(requests)
        return [request for request in requests if self._lookup(request) is None]

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.mode != "live":
            entry = self._lookup(request)
            if self.mode == "replay":
                if entry is None:
                    raise self._miss(request)
                return LLMResponse.from_dict(entry["response"])
            if entry is not None and not self.refresh:
                self.reused += 1
                return LLMResponse.from_dict(entry["response"])
        response = self.inner.complete(request)
        self.called += 1
        if self.mode == "record":
            self._store(request, response)
        return response

    def complete_many(
        self, requests: list[LLMRequest], use_batch: bool, state_path: Path | None = None
    ) -> list[LLMResponse]:
        """Answer several requests; a live Anthropic judge run goes through the Batch API (50% off).

        In record mode stored requests are reused and only missing ones are submitted. `state_path` receives
        the batch id so an interrupted wait can be resumed by running the same command again.
        """
        if self.mode == "replay" or not use_batch or not hasattr(self.inner, "complete_batch"):
            return [self.complete(request) for request in requests]
        results: dict[int, LLMResponse] = {}
        pending: list[int] = []
        for index, request in enumerate(requests):
            entry = self._lookup(request) if self.mode == "record" and not self.refresh else None
            if entry is not None:
                results[index] = LLMResponse.from_dict(entry["response"])
                self.reused += 1
            else:
                pending.append(index)
        if pending:
            batch = [requests[index] for index in pending]
            keys = [replay_key(request) for request in batch]
            # If the wait is interrupted, the batch id in state_path lets the next run fetch the results.
            responses = self.inner.complete_batch(batch, keys=keys, state_path=state_path)
            self.called += len(batch)
            for index, request, response in zip(pending, batch, responses, strict=True):
                results[index] = response
                if self.mode == "record":
                    self._store(request, response)
        return [results[index] for index in range(len(requests))]


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
