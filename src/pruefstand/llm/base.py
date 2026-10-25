"""Provider-agnostic request and response types for every LLM call in the bench."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMRequest:
    provider: str
    model: str
    system: str
    messages: tuple[tuple[str, str], ...]
    max_tokens: int
    effort: str | None = None
    thinking: str | None = None
    response_schema: dict | None = None
    prompt_id: str = ""
    prompt_sha256: str = ""
    tag: str = field(default="", compare=False)

    def to_dict(self, include_tag: bool = False) -> dict:
        data = {
            "provider": self.provider,
            "model": self.model,
            "system": self.system,
            "messages": [{"role": role, "content": content} for role, content in self.messages],
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "thinking": self.thinking,
            "response_schema": self.response_schema,
            "prompt_id": self.prompt_id,
            "prompt_sha256": self.prompt_sha256,
        }
        if include_tag:
            data["tag"] = self.tag
        return data


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    stop_reason: str
    usage: Usage
    latency_ms: float | None = None
    parsed: dict | None = None
    request_id: str | None = None
    model: str = ""
    batch: bool = False
    provider_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LLMResponse:
        values = dict(data)
        values["usage"] = Usage(**values.get("usage", {}))
        return cls(**values)


class LLMClient(Protocol):
    provider: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def or_none(value: str | None) -> str | None:
    """Config uses empty strings for 'not set'; requests use None."""
    return value or None
