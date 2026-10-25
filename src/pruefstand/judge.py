"""LLM judge on a JSON rubric (faithfulness, correctness, completeness, rationale)."""

from __future__ import annotations

import json
from collections.abc import Mapping

from pruefstand.llm.base import LLMRequest, LLMResponse, or_none
from pruefstand.prompts import Prompt

RUBRIC_VERSION = "judge_v1"
JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "enum": [0, 1, 2]},
        "correctness": {"type": "integer", "enum": [0, 1, 2]},
        "completeness": {"type": "integer", "enum": [0, 1]},
        "rationale": {"type": "string"},
    },
    "required": ["faithfulness", "correctness", "completeness", "rationale"],
    "additionalProperties": False,
}
ALLOWED = {"faithfulness": (0, 1, 2), "correctness": (0, 1, 2), "completeness": (0, 1)}


def validate_judgement(data: object) -> list[str]:
    if not isinstance(data, dict):
        return ["judgement is not a JSON object"]
    errors = []
    for name, allowed in ALLOWED.items():
        value = data.get(name)
        if isinstance(value, bool) or value not in allowed:
            errors.append(f"{name} must be one of {allowed}, got {value!r}")
    if not isinstance(data.get("rationale"), str) or not data["rationale"].strip():
        errors.append("rationale must be a non-empty string")
    extra = set(data) - set(JUDGE_SCHEMA["properties"])
    if extra:
        errors.append(f"unexpected fields {sorted(extra)}")
    return errors


def build_judge_request(
    question: Mapping, answer_text: str, context: str, prompt: Prompt, role: Mapping, tag: str
) -> LLMRequest:
    user = (
        f"<frage>\n{question['question_de']}\n</frage>\n\n"
        f"<referenzantwort>\n{question['gold_answer_de']}\n</referenzantwort>\n\n"
        f"<kontext>\n{context}\n</kontext>\n\n"
        f"<antwort>\n{answer_text}\n</antwort>"
    )
    return LLMRequest(
        provider=role["provider"],
        model=role["model"],
        system=prompt.text,
        messages=(("user", user),),
        max_tokens=int(role["max_tokens"]),
        effort=or_none(role.get("effort")),
        thinking=or_none(role.get("thinking")),
        response_schema=JUDGE_SCHEMA,
        prompt_id=prompt.prompt_id,
        prompt_sha256=prompt.sha256,
        tag=tag,
    )


def parse_judgement(response: LLMResponse) -> dict:
    """Scores plus a status: ok, refusal (stored and scored as a failure category) or invalid."""
    empty = {"faithfulness": None, "correctness": None, "completeness": None, "rationale": ""}
    if response.stop_reason == "refusal":
        return {**empty, "status": "refusal"}
    data = response.parsed
    if data is None:
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return {**empty, "status": "invalid", "error": "not JSON"}
    errors = validate_judgement(data)
    if errors:
        return {**empty, "status": "invalid", "error": "; ".join(errors)}
    return {
        **{name: data[name] for name in ("faithfulness", "correctness", "completeness", "rationale")},
        "status": "ok",
    }
