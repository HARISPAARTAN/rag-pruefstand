"""Provider-agnostic request and response types for every LLM call in the bench."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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
    # Stage name (answer, judge_faithfulness, judge_correctness, repeat_1, ...) is part of the replay key, so
    # identical requests of different stages (e.g. repeats) are recorded separately.
    stage: str = ""
    # Provider-specific options that change the output (Ollama: num_ctx, temperature, seed); part of the key.
    options: dict | None = None
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
            "stage": self.stage,
            "options": self.options,
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
        known = {f.name for f in fields(cls)}
        values = {name: value for name, value in data.items() if name in known}
        values["usage"] = Usage(**values.get("usage", {}))
        return cls(**values)


class LLMClient(Protocol):
    provider: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def or_none(value: str | None) -> str | None:
    """Config uses empty strings for 'not set'; requests use None."""
    return value or None


OLLAMA_DEFAULTS = {"num_ctx": 16384, "temperature": 0, "seed": 20261025}


def role_options(role: dict) -> dict | None:
    """Provider options that change the output and therefore belong in the request (and the replay key).

    Only Ollama has them: a fixed context window, temperature 0 and a fixed seed. The Anthropic API takes no
    sampling parameters on current models, so its requests carry none.
    """
    if role.get("provider") != "ollama":
        return None
    return {name: role.get(name, default) for name, default in OLLAMA_DEFAULTS.items()}
