"""LLM judge on a JSON rubric, split into two calls so that no call sees more than it needs.

- faithfulness: question, retrieved context and answer; the gold answer is never part of this request, so the
  judge cannot reward an answer that matches the reference but is not in the context.
- correctness and completeness: question, gold answer and answer; no context.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from pruefstand.llm.base import LLMRequest, LLMResponse, or_none, role_options
from pruefstand.prompts import Prompt

RUBRIC_VERSION = "judge_v2"
KINDS = ("faithfulness", "correctness")
STAGES = {"faithfulness": "judge_faithfulness", "correctness": "judge_correctness"}
ALLOWED = {
    "faithfulness": {"faithfulness": (0, 1, 2)},
    "correctness": {"correctness": (0, 1, 2), "completeness": (0, 1)},
}
SCHEMAS: dict[str, dict] = {
    "faithfulness": {
        "type": "object",
        "properties": {
            "faithfulness": {"type": "integer", "enum": [0, 1, 2]},
            "rationale": {"type": "string"},
        },
        "required": ["faithfulness", "rationale"],
        "additionalProperties": False,
    },
    "correctness": {
        "type": "object",
        "properties": {
            "correctness": {"type": "integer", "enum": [0, 1, 2]},
            "completeness": {"type": "integer", "enum": [0, 1]},
            "rationale": {"type": "string"},
        },
        "required": ["correctness", "completeness", "rationale"],
        "additionalProperties": False,
    },
}


def validate_judgement(data: object, kind: str) -> list[str]:
    if not isinstance(data, dict):
        return ["judgement is not a JSON object"]
    errors = []
    for name, allowed in ALLOWED[kind].items():
        value = data.get(name)
        if isinstance(value, bool) or value not in allowed:
            errors.append(f"{name} must be one of {allowed}, got {value!r}")
    if not isinstance(data.get("rationale"), str) or not data["rationale"].strip():
        errors.append("rationale must be a non-empty string")
    extra = set(data) - set(SCHEMAS[kind]["properties"])
    if extra:
        errors.append(f"unexpected fields {sorted(extra)}")
    return errors


def _request(
    kind: str, user: str, prompt: Prompt, role: Mapping, tag: str, stage: str | None = None
) -> LLMRequest:
    return LLMRequest(
        provider=role["provider"],
        model=role["model"],
        system=prompt.text,
        messages=(("user", user),),
        max_tokens=int(role["max_tokens"]),
        effort=or_none(role.get("effort")),
        thinking=or_none(role.get("thinking")),
        response_schema=SCHEMAS[kind],
        prompt_id=prompt.prompt_id,
        prompt_sha256=prompt.sha256,
        stage=stage or STAGES[kind],
        options=role_options(dict(role)),
        tag=tag,
    )


def build_faithfulness_request(
    question: Mapping, answer_text: str, context: str, prompt: Prompt, role: Mapping, tag: str
) -> LLMRequest:
    """Context-only call: the reference answer is deliberately absent."""
    user = (
        f"<frage>\n{question['question_de']}\n</frage>\n\n"
        f"<kontext>\n{context}\n</kontext>\n\n"
        f"<antwort>\n{answer_text}\n</antwort>"
    )
    return _request("faithfulness", user, prompt, role, tag)


def build_correctness_request(
    question: Mapping, answer_text: str, prompt: Prompt, role: Mapping, tag: str
) -> LLMRequest:
    """Reference call: the retrieved context is deliberately absent."""
    user = (
        f"<frage>\n{question['question_de']}\n</frage>\n\n"
        f"<referenzantwort>\n{question['gold_answer_de']}\n</referenzantwort>\n\n"
        f"<antwort>\n{answer_text}\n</antwort>"
    )
    return _request("correctness", user, prompt, role, tag)


def parse_part(response: LLMResponse, kind: str) -> dict:
    """Scores of one call plus a status: ok, refusal, failed (Batch item) or invalid."""
    empty = {**dict.fromkeys(ALLOWED[kind]), "rationale": ""}
    if response.stop_reason == "refusal":
        return {**empty, "status": "refusal"}
    if response.provider_meta.get("failed"):
        detail = response.provider_meta.get("error") or {}
        return {**empty, "status": "failed", "error": f"{response.stop_reason}: {json.dumps(detail)}"}
    data = response.parsed
    if data is None:
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return {**empty, "status": "invalid", "error": "not JSON"}
    errors = validate_judgement(data, kind)
    if errors:
        return {**empty, "status": "invalid", "error": "; ".join(errors)}
    return {**{name: data[name] for name in (*ALLOWED[kind], "rationale")}, "status": "ok"}


def combine_judgement(faithfulness: dict, correctness: dict) -> dict:
    """One record per answer; the overall status is the worst of the two calls."""
    statuses = (faithfulness["status"], correctness["status"])
    overall = "ok"
    for bad in ("failed", "refusal", "invalid"):
        if bad in statuses:
            overall = bad
            break
    record = {
        "faithfulness": faithfulness["faithfulness"],
        "correctness": correctness["correctness"],
        "completeness": correctness["completeness"],
        "rationale_faithfulness": faithfulness["rationale"],
        "rationale": correctness["rationale"],
        "status": overall,
        "status_faithfulness": faithfulness["status"],
        "status_correctness": correctness["status"],
    }
    errors = [part["error"] for part in (faithfulness, correctness) if "error" in part]
    if errors:
        record["error"] = " | ".join(errors)
    return record
