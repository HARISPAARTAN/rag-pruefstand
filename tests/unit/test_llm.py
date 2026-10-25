import io
import json
from types import SimpleNamespace

import pytest

from pruefstand.judge import JUDGE_SCHEMA, parse_judgement, validate_judgement
from pruefstand.llm import ollama_client
from pruefstand.llm.anthropic_client import AnthropicClient, build_params
from pruefstand.llm.base import LLMRequest, LLMResponse, Usage
from pruefstand.llm.costs import BudgetExceeded, BudgetGuard, Price, cost_usd, load_prices, project_stages
from pruefstand.llm.ollama_client import OllamaClient, build_payload
from pruefstand.llm.replay import Cassette, ReplayClient, ReplayMiss, replay_key


def request(**overrides) -> LLMRequest:
    values = {
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "system": "System",
        "messages": (("user", "Frage?"),),
        "max_tokens": 600,
        "effort": "low",
        "thinking": "disabled",
        "prompt_id": "answer_v1_de",
        "prompt_sha256": "abc",
        "tag": "answer|bm25-C1|k1v1-0001",
    }
    values.update(overrides)
    return LLMRequest(**values)


def response(text="Antwort", **overrides) -> LLMResponse:
    values = {
        "text": text,
        "stop_reason": "end_turn",
        "usage": Usage(100, 20),
        "latency_ms": 5.0,
        "model": "claude-haiku-5-5",
    }
    values.update(overrides)
    return LLMResponse(**values)


class FakeInner:
    provider = "anthropic"

    def __init__(self):
        self.calls = 0

    def complete(self, req):
        self.calls += 1
        return response(f"live {self.calls}")


def test_replay_key_ignores_tag_and_tracks_prompt_version():
    assert replay_key(request()) == replay_key(request(tag="other"))
    assert replay_key(request()) != replay_key(request(prompt_sha256="def"))
    assert len(replay_key(request())) == 64


def test_record_then_replay(tmp_path):
    path = tmp_path / "cassette.jsonl"
    inner = FakeInner()
    recorder = ReplayClient("record", Cassette(path), inner)
    assert recorder.complete(request()).text == "live 1"
    recorder.cassette.save()
    replayer = ReplayClient("replay", Cassette(path))
    replayed = replayer.complete(request())
    assert replayed.text == "live 1" and replayed.usage == Usage(100, 20)
    assert inner.calls == 1
    with pytest.raises(ReplayMiss, match="answer_v1_de"):
        replayer.complete(request(messages=(("user", "Andere Frage?"),)))


def test_cassette_file_is_sorted_and_stable(tmp_path):
    path = tmp_path / "cassette.jsonl"
    cassette = Cassette(path)
    cassette.put(request(), response())
    cassette.put(request(messages=(("user", "Zweite?"),)), response("zwei"))
    cassette.save()
    first = path.read_bytes()
    Cassette(path).save()
    assert path.read_bytes() == first
    keys = [json.loads(line)["key"] for line in first.decode("utf-8").splitlines()]
    assert keys == sorted(keys)


def test_replay_mode_needs_no_inner_but_record_does(tmp_path):
    with pytest.raises(ValueError):
        ReplayClient("record", Cassette(tmp_path / "c.jsonl"))
    with pytest.raises(ValueError):
        ReplayClient("bogus", Cassette(tmp_path / "c.jsonl"), FakeInner())


def test_anthropic_params_never_send_sampling_parameters():
    params = build_params(request(response_schema=JUDGE_SCHEMA, thinking="adaptive"))
    assert params["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
    }
    assert params["thinking"] == {"type": "adaptive"}
    assert params["system"] == "System"
    assert not {"temperature", "top_p", "top_k"} & set(params)
    assert "thinking" not in build_params(request(thinking=None))


def _message(text, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=50, output_tokens=10, cache_read_input_tokens=5, cache_creation_input_tokens=0
        ),
        model="claude-sonnet-5-5",
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
    )


class FakeBatches:
    def __init__(self):
        self.submitted = None

    def create(self, requests):
        self.submitted = requests
        return SimpleNamespace(id="batch_1")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended")

    def results(self, batch_id):
        payload = '{"faithfulness": 2, "correctness": 2, "completeness": 1, "rationale": "ok"}'
        return [
            SimpleNamespace(custom_id="r00001", result=SimpleNamespace(type="errored")),
            SimpleNamespace(
                custom_id="r00000", result=SimpleNamespace(type="succeeded", message=_message(payload))
            ),
        ]


def test_anthropic_batch_maps_results_back_in_order():
    batches = FakeBatches()
    fake = SimpleNamespace(messages=SimpleNamespace(batches=batches, create=lambda **kw: _message("x")))
    client = AnthropicClient(client=fake, poll_seconds=0)
    reqs = [
        request(response_schema=JUDGE_SCHEMA),
        request(messages=(("user", "B?"),), response_schema=JUDGE_SCHEMA),
    ]
    results = client.complete_batch(reqs)
    assert [item["custom_id"] for item in batches.submitted] == ["r00000", "r00001"]
    assert results[0].parsed["faithfulness"] == 2 and results[0].batch is True
    assert results[0].usage.cache_read_input_tokens == 5
    assert results[1].stop_reason == "batch_errored"


def test_anthropic_refusal_is_returned_not_raised():
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: _message("", "refusal")))
    result = AnthropicClient(client=fake).complete(request())
    assert result.stop_reason == "refusal" and result.provider_meta["refusal_category"] == "cyber"


def test_ollama_payload_and_response(monkeypatch):
    payload = build_payload(request(provider="ollama", model="qwen3:4b"))
    assert payload["messages"][0] == {"role": "system", "content": "System"}
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"] == {"num_predict": 600}
    seen = {}

    def fake_urlopen(http_request, timeout):
        seen["url"] = http_request.full_url
        body = {
            "model": "qwen3:4b",
            "message": {"content": "Antwort"},
            "done_reason": "stop",
            "prompt_eval_count": 40,
            "eval_count": 7,
        }
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)
    result = OllamaClient(host="http://ollama.test:11434").complete(
        request(provider="ollama", model="qwen3:4b")
    )
    assert seen["url"] == "http://ollama.test:11434/api/chat"
    assert result.text == "Antwort" and result.usage == Usage(40, 7)


def test_costs_with_cache_and_batch(root):
    price = Price(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=0,
    )
    assert cost_usd(usage, price) == 3.2
    assert cost_usd(usage, price, batch=True) == 1.6
    table = load_prices(root / "costs.toml")
    assert table.price("claude-haiku-5-5").input == 0.10
    assert table.eur_per_usd == 0.8934
    with pytest.raises(KeyError):
        table.price("unknown-model")
    roles = {"judge": {"model": "claude-sonnet-5-5"}}
    stages = [
        {
            "name": "judge",
            "role": "judge",
            "calls": 1000,
            "input_tokens": 4000,
            "output_tokens": 500,
            "batch": True,
        }
    ]
    assert project_stages(stages, roles, table)[0].usd == 6.5


def test_budget_guard_stops_before_and_during_a_run():
    guard = BudgetGuard(1.0)
    guard.check_projection(0.9)
    with pytest.raises(BudgetExceeded):
        guard.check_projection(1.1)
    guard.add(0.6)
    with pytest.raises(BudgetExceeded):
        guard.add(0.5)


def test_judge_schema_validation_and_refusal():
    assert (
        validate_judgement({"faithfulness": 2, "correctness": 1, "completeness": 0, "rationale": "x"}) == []
    )
    assert validate_judgement({"faithfulness": 3, "correctness": 1, "completeness": 0, "rationale": "x"})
    assert validate_judgement({"faithfulness": True, "correctness": 1, "completeness": 0, "rationale": "x"})
    assert validate_judgement(
        {"faithfulness": 2, "correctness": 1, "completeness": 0, "rationale": "x", "extra": 1}
    )
    assert parse_judgement(response("", stop_reason="refusal"))["status"] == "refusal"
    assert parse_judgement(response("kein json"))["status"] == "invalid"
    ok = parse_judgement(
        response('{"faithfulness": 1, "correctness": 2, "completeness": 1, "rationale": "r"}')
    )
    assert ok == {"faithfulness": 1, "correctness": 2, "completeness": 1, "rationale": "r", "status": "ok"}


def test_committed_cassette_is_entirely_synthetic_and_replays(root):
    path = root / "fixtures" / "llm" / "cassette.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 32
    assert all(entry["synthetic"] is True for entry in entries)
    assert all(entry["response"]["provider_meta"]["synthetic"] is True for entry in entries)
    fixture = json.loads((root / "fixtures" / "llm" / "synthetic_responses.json").read_text(encoding="utf-8"))
    assert fixture["note"].startswith("SYNTHETIC")
    assert all(item["rationale"].startswith("Synthetic:") for item in fixture["judgements"].values())


def test_cost_estimate_exits_nonzero_above_the_cap(root, tmp_path):
    from typer.testing import CliRunner

    from pruefstand.cli import app

    text = (
        (root / "pruefstand.toml")
        .read_text(encoding="utf-8")
        .replace("max_usd_per_run = 30.0", "max_usd_per_run = 5.0")
    )
    config = tmp_path / "pruefstand.toml"
    config.write_text(text, encoding="utf-8")
    (tmp_path / "costs.toml").write_text((root / "costs.toml").read_text(encoding="utf-8"), encoding="utf-8")
    result = CliRunner().invoke(app, ["cost", "estimate", "--config", str(config)])
    assert result.exit_code == 1
    assert "projection exceeds the cap" in result.output
    ok = CliRunner().invoke(app, ["cost", "estimate", "--config", str(root / "pruefstand.toml")])
    assert ok.exit_code == 0 and "USD 17.10" in ok.output
