import io
import json
from types import SimpleNamespace

import pytest

from pruefstand.config import Config
from pruefstand.errors import PruefstandError
from pruefstand.generation import cost_function, save_stage_spend, stage_guard
from pruefstand.judge import (
    ALLOWED,
    SCHEMAS,
    build_correctness_request,
    build_faithfulness_request,
    combine_judgement,
    parse_part,
    validate_judgement,
)
from pruefstand.llm import ollama_client
from pruefstand.llm.anthropic_client import (
    AnthropicClient,
    BatchWaitError,
    build_params,
)
from pruefstand.llm.base import LLMRequest, LLMResponse, Usage, role_options
from pruefstand.llm.costs import (
    BudgetExceeded,
    BudgetGuard,
    Price,
    cost_usd,
    load_prices,
    project_stages,
    worst_case_usd,
)
from pruefstand.llm.ollama_client import OllamaClient, PromptDoesNotFit, build_payload
from pruefstand.llm.replay import Cassette, ReplayClient, ReplayMiss, replay_key
from pruefstand.prompts import Prompt

JUDGE_SCHEMA = SCHEMAS["correctness"]


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
        "stage": "answer",
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
    sdk_version = "fake 1.0"

    def __init__(self):
        self.calls = 0
        self.batches: list[list[LLMRequest]] = []

    def complete(self, req):
        self.calls += 1
        return response(f"live {self.calls}")

    def complete_batch(self, requests, keys=None, state_path=None):
        self.batches.append(list(requests))
        return [response(f"batch {req.messages[0][1]}", batch=True) for req in requests]


# ----------------------------------------------------------------------------------------------- replay key


def test_replay_key_ignores_tag_and_tracks_prompt_version_stage_and_options():
    assert replay_key(request()) == replay_key(request(tag="other"))
    assert replay_key(request()) != replay_key(request(prompt_sha256="def"))
    assert replay_key(request()) != replay_key(request(stage="repeat_1"))
    assert replay_key(request()) != replay_key(request(options={"num_ctx": 8192}))
    assert len(replay_key(request())) == 64


def test_record_then_replay(tmp_path):
    path = tmp_path / "cassette.jsonl"
    inner = FakeInner()
    recorder = ReplayClient("record", Cassette(path), inner)
    assert recorder.complete(request()).text == "live 1"
    recorder.save()
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
    assert not (tmp_path / "cassette.jsonl.tmp").exists()


def test_replay_mode_needs_no_inner_but_record_does(tmp_path):
    with pytest.raises(ValueError):
        ReplayClient("record", Cassette(tmp_path / "c.jsonl"))
    with pytest.raises(ValueError):
        ReplayClient("bogus", Cassette(tmp_path / "c.jsonl"), FakeInner())


def test_response_from_dict_tolerates_unknown_and_missing_fields():
    data = response().to_dict()
    data["added_later"] = 1
    del data["provider_meta"]
    assert LLMResponse.from_dict(data).text == "Antwort"


# ------------------------------------------------------------------------- I3: record reads the cassette first


def test_record_mode_reuses_stored_entries_and_only_calls_for_misses(tmp_path):
    path = tmp_path / "c.jsonl"
    inner = FakeInner()
    first = ReplayClient("record", Cassette(path), inner)
    first.complete(request())
    first.save()
    again = ReplayClient("record", Cassette(path), inner)
    assert again.complete(request()).text == "live 1"
    assert inner.calls == 1 and again.reused == 1 and again.called == 0
    again.complete(request(messages=(("user", "neu?"),)))
    assert inner.calls == 2
    refreshing = ReplayClient("record", Cassette(path), inner, refresh=True)
    assert refreshing.complete(request()).text == "live 3"
    assert refreshing.misses([request()]) == [request()]


def test_stage_makes_identical_requests_separate_entries(tmp_path):
    client = ReplayClient("record", Cassette(tmp_path / "c.jsonl"), FakeInner())
    client.complete(request(stage="repeat_1"))
    client.complete(request(stage="repeat_2"))
    client.complete(request(stage="repeat_1"))
    assert client.called == 2 and client.reused == 1
    assert len(client.cassette.entries) == 2


def test_complete_many_submits_only_missing_requests_and_never_stores_failures(tmp_path):
    inner = FakeInner()
    path = tmp_path / "c.jsonl"
    client = ReplayClient("record", Cassette(path), inner)
    stored = request(messages=(("user", "A"),))
    client.complete(stored)
    batch = [stored, request(messages=(("user", "B"),)), request(messages=(("user", "C"),))]
    results = client.complete_many(batch, use_batch=True)
    assert [len(b) for b in inner.batches] == [2]
    assert results[0].text == "live 1" and results[1].text == "batch B" and results[2].batch is True

    class FailingInner(FakeInner):
        def complete_batch(self, requests, keys=None, state_path=None):
            return [response("", provider_meta={"failed": True, "error": {"message": "Overloaded"}})]

    failing = ReplayClient("record", Cassette(tmp_path / "f.jsonl"), FailingInner())
    out = failing.complete_many([request()], use_batch=True)
    assert out[0].provider_meta["failed"] is True
    assert failing.cassette.entries == {}


# ---------------------------------------------------------------- I4: audit fields, recorded versus recomputed


def test_cassette_entry_carries_audit_fields_and_cost_at_record_time(tmp_path, root):
    table = load_prices(root / "costs.toml")
    client = ReplayClient(
        "record",
        Cassette(tmp_path / "c.jsonl"),
        FakeInner(),
        cost_fn=cost_function(table),
        price_as_of="2026-10-06",
        clock=lambda: "2026-10-25T12:00:00+00:00",
    )
    req = request(model="claude-haiku-5-5")
    client.complete(req)
    entry = client.cassette.entries[replay_key(req)]
    assert entry["record"] == {
        "recorded_at": "2026-10-25T12:00:00+00:00",
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "sdk_version": "fake 1.0",
        "batch_id": None,
        "price_as_of": "2026-10-06",
        "cost_usd": 0.00002,  # 100 * 0.10 + 20 * 0.50 per million
    }
    assert entry["request"]["max_tokens"] == 600 and entry["response"]["usage"]["output_tokens"] == 20
    assert client.recorded(req)["cost_usd"] == 0.00002


def test_replay_can_show_recorded_and_recomputed_cost_apart(tmp_path, root):
    table = load_prices(root / "costs.toml")
    req = request()
    cassette = Cassette(tmp_path / "c.jsonl")
    recorder = ReplayClient("record", cassette, FakeInner(), cost_fn=cost_function(table))
    recorder.complete(req)
    recorder.save()
    doubled = Price(input=0.2, output=1.0, cache_read=0.0, cache_write=0.0)
    replayer = ReplayClient("replay", Cassette(tmp_path / "c.jsonl"))
    replayed = replayer.complete(req)
    recomputed = cost_usd(replayed.usage, doubled)
    assert replayer.recorded(req)["cost_usd"] == 0.00002
    assert recomputed == 0.00004


# --------------------------------------------------------------- I6: synthetic entries stay in the ci profile


def test_synthetic_entries_are_invisible_outside_the_ci_profile(tmp_path):
    path = tmp_path / "c.jsonl"
    synthetic = ReplayClient("record", Cassette(path), FakeInner(), synthetic=True)
    synthetic.complete(request())
    synthetic.save()
    ci = ReplayClient("replay", Cassette(path, allow_synthetic=True))
    assert ci.complete(request()).text == "live 1"
    live = ReplayClient("replay", Cassette(path, allow_synthetic=False))
    with pytest.raises(ReplayMiss, match="synthetic"):
        live.complete(request())
    inner = FakeInner()
    recorder = ReplayClient("record", Cassette(path, allow_synthetic=False), inner)
    assert recorder.misses([request()]) == [request()]
    assert recorder.complete(request()).text == "live 1" and inner.calls == 1


def test_profiles_use_separate_cassette_files(root):
    from pruefstand.config import load_config

    full = load_config(root / "pruefstand.toml", "full")["llm"]["cassette"]
    ci = load_config(root / "pruefstand.toml", "ci")["llm"]["cassette"]
    assert full != ci and "synthetic" in ci and "synthetic" not in full
    assert not (root / full).exists() or all(
        not json.loads(line)["synthetic"]
        for line in (root / full).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


# ---------------------------------------------------------------------------------------- Anthropic request


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


PAYLOAD = '{"correctness": 2, "completeness": 1, "rationale": "ok"}'


class FakeBatches:
    """Batches endpoint: `outcome(batch_number, custom_id)` decides per item; results come back out of order."""

    def __init__(self, outcome=None, status="ended", fail_polls=0):
        self.created: list[list[dict]] = []
        self.outcome = outcome or (lambda batch, custom_id: "succeeded")
        self.status = status
        self.fail_polls = fail_polls
        self.retrieves = 0

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id=f"batch_{len(self.created)}")

    def retrieve(self, batch_id):
        self.retrieves += 1
        if self.fail_polls > 0:
            self.fail_polls -= 1
            raise ConnectionError("temporary")
        return SimpleNamespace(processing_status=self.status)

    def results(self, batch_id):
        number = int(batch_id.split("_")[1])
        items = []
        for entry in reversed(self.created[number - 1]):
            kind = self.outcome(number, entry["custom_id"])
            if kind == "succeeded":
                result = SimpleNamespace(type="succeeded", message=_message(PAYLOAD))
            elif kind == "errored":
                error = SimpleNamespace(type="overloaded_error", message="Overloaded")
                result = SimpleNamespace(
                    type="errored", error=SimpleNamespace(type="error", error=error, request_id="req_9")
                )
            else:
                result = SimpleNamespace(type=kind)
            items.append(SimpleNamespace(custom_id=entry["custom_id"], result=result))
        return items


def make_client(batches, **kwargs):
    fake = SimpleNamespace(messages=SimpleNamespace(batches=batches, create=lambda **kw: _message("x")))
    kwargs.setdefault("poll_seconds", 0)
    kwargs.setdefault("sleep", lambda seconds: None)
    return AnthropicClient(client=fake, **kwargs)


def batch_requests():
    return [
        request(response_schema=JUDGE_SCHEMA),
        request(messages=(("user", "B?"),), response_schema=JUDGE_SCHEMA),
    ]


def test_anthropic_batch_maps_results_back_in_order():
    batches = FakeBatches()
    results = make_client(batches).complete_batch(batch_requests())
    assert [item["custom_id"] for item in batches.created[0]] == ["r00000", "r00001"]
    assert results[0].parsed["correctness"] == 2 and results[0].batch is True
    assert results[0].usage.cache_read_input_tokens == 5
    assert results[1].parsed["completeness"] == 1 and results[1].provider_meta["batch_id"] == "batch_1"


def test_batch_id_is_persisted_and_errored_items_are_retried_once(tmp_path):
    state_path = tmp_path / "batch_judge.json"
    batches = FakeBatches(
        outcome=lambda batch, cid: "errored" if batch == 1 and cid == "r00001" else "succeeded"
    )
    results = make_client(batches).complete_batch(batch_requests(), keys=["k0", "k1"], state_path=state_path)
    assert len(batches.created) == 2 and len(batches.created[1]) == 1
    assert all(not r.provider_meta.get("failed") for r in results)
    assert results[1].provider_meta["batch_id"] == "batch_2"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert [p["batch_id"] for p in state["passes"]] == ["batch_1", "batch_2"]
    assert state["passes"][0]["keys"] == ["k0", "k1"] and state["passes"][1]["keys"] == ["k1"]


def test_items_that_still_fail_keep_their_error_detail(tmp_path):
    # item 1 fails in the first batch and again as the only item (r00000) of the retry batch
    def outcome(batch, cid):
        return "errored" if (batch, cid) in {(1, "r00001"), (2, "r00000")} else "succeeded"

    batches = FakeBatches(outcome=outcome)
    results = make_client(batches).complete_batch(batch_requests(), keys=["k0", "k1"])
    failed = results[1]
    assert len(batches.created) == 2
    assert failed.stop_reason == "batch_errored" and failed.provider_meta["failed"] is True
    assert failed.provider_meta["retried"] is True
    assert failed.provider_meta["error"]["message"] == "Overloaded"
    assert failed.provider_meta["error"]["error_type"] == "overloaded_error"
    expired = make_client(FakeBatches(outcome=lambda b, cid: "expired")).complete_batch(batch_requests())
    assert expired[0].stop_reason == "batch_expired" and expired[0].provider_meta["retried"] is True


def test_batch_wait_has_a_timeout_and_can_be_resumed(tmp_path):
    state_path = tmp_path / "batch_judge.json"
    clock = iter(range(0, 10_000, 100))
    batches = FakeBatches(status="in_progress")
    client = make_client(batches, max_wait_seconds=250, monotonic=lambda: next(clock))
    with pytest.raises(BatchWaitError, match="batch_1"):
        client.complete_batch(batch_requests(), keys=["k0", "k1"], state_path=state_path)
    assert json.loads(state_path.read_text(encoding="utf-8"))["passes"][0]["batch_id"] == "batch_1"
    batches.status = "ended"
    results = make_client(batches).complete_batch(batch_requests(), keys=["k0", "k1"], state_path=state_path)
    assert len(batches.created) == 1  # resumed, not submitted again
    assert results[0].parsed["correctness"] == 2
    batches.created.clear()
    changed = make_client(batches).complete_batch(batch_requests(), keys=["x0", "x1"], state_path=state_path)
    assert len(batches.created) == 1 and changed[0].parsed is not None  # other keys: new batch


def test_transient_poll_errors_are_survived_but_not_forever():
    batches = FakeBatches(fail_polls=2)
    assert make_client(batches).complete_batch(batch_requests())[0].parsed is not None
    assert batches.retrieves == 3
    with pytest.raises(BatchWaitError, match="polls in a row failed"):
        make_client(FakeBatches(fail_polls=99)).complete_batch(batch_requests())


def test_anthropic_refusal_is_returned_not_raised():
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: _message("", "refusal")))
    result = AnthropicClient(client=fake).complete(request())
    assert result.stop_reason == "refusal" and result.provider_meta["refusal_category"] == "cyber"


# ------------------------------------------------------------------------------------------------ Ollama


def test_ollama_payload_sends_context_temperature_and_seed():
    payload = build_payload(request(provider="ollama", model="qwen3:4b"))
    assert payload["messages"][0] == {"role": "system", "content": "System"}
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"] == {"num_predict": 600, "num_ctx": 16384, "temperature": 0, "seed": 20261025}
    custom = build_payload(request(provider="ollama", options={"num_ctx": 8192, "temperature": 0, "seed": 7}))
    assert custom["options"]["num_ctx"] == 8192 and custom["options"]["seed"] == 7
    assert role_options({"provider": "anthropic"}) is None
    assert role_options({"provider": "ollama", "seed": 3}) == {"num_ctx": 16384, "temperature": 0, "seed": 3}


def ollama_body(prompt_tokens=40, **extra):
    body = {
        "model": "qwen3:4b",
        "message": {"content": "Antwort"},
        "done_reason": "stop",
        "eval_count": 7,
    }
    if prompt_tokens is not None:
        body["prompt_eval_count"] = prompt_tokens
    body.update(extra)
    return body


def test_ollama_response_records_prompt_tokens_and_normalises_stop_reason(monkeypatch):
    seen = {}

    def fake_urlopen(http_request, timeout):
        seen["url"] = http_request.full_url
        return io.BytesIO(json.dumps(ollama_body(done_reason="length")).encode("utf-8"))

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)
    req = request(provider="ollama", model="qwen3:4b", options=role_options({"provider": "ollama"}))
    result = OllamaClient(host="ollama.test:11434").complete(req)
    assert seen["url"] == "http://ollama.test:11434/api/chat"
    assert result.text == "Antwort" and result.usage == Usage(40, 7)
    assert result.stop_reason == "max_tokens"
    assert result.provider_meta["prompt_eval_count"] == 40 and result.provider_meta["num_ctx"] == 16384


@pytest.mark.parametrize("prompt_tokens", [16000, 16384, None])
def test_ollama_fails_closed_when_the_prompt_does_not_fit(monkeypatch, prompt_tokens):
    body = json.dumps(ollama_body(prompt_tokens)).encode("utf-8")
    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", lambda r, timeout: io.BytesIO(body))
    with pytest.raises(PromptDoesNotFit):
        OllamaClient(host="http://x").complete(request(provider="ollama", model="qwen3:4b"))


# ------------------------------------------------------------------------------------------------- costs


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


def test_cache_write_tokens_are_priced():
    price = Price(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
    assert cost_usd(Usage(cache_creation_input_tokens=1_000_000), price) == 2.5
    assert cost_usd(Usage(cache_creation_input_tokens=1_000_000, input_tokens=1_000_000), price) == 4.5


def test_batch_discount_is_subtracted_not_multiplied():
    price = Price(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
    usage = Usage(input_tokens=1_000_000)
    assert cost_usd(usage, price, batch=True, batch_discount=0.3) == 1.4
    assert cost_usd(usage, price, batch=True, batch_discount=0.5) == 1.0
    assert cost_usd(usage, price, batch=False, batch_discount=0.3) == 2.0


def test_budget_guard_stops_before_and_during_a_run():
    guard = BudgetGuard(1.0)
    guard.check_projection(0.9)
    with pytest.raises(BudgetExceeded):
        guard.check_projection(1.1)
    guard.add(0.6)
    with pytest.raises(BudgetExceeded):
        guard.add(0.5)


# ----------------------------------------------------------------------------- I8: hard bound on the judge


def test_worst_case_bound_counts_input_and_the_full_max_tokens():
    price = Price(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
    reqs = [request(system="a" * 250, messages=(("user", "b" * 250),), max_tokens=1000) for _ in range(10)]
    # 500 chars / 2.5 = 200 input tokens per request, 1000 output tokens per request
    assert worst_case_usd(reqs, price) == pytest.approx(10 * (200 * 2.0 + 1000 * 10.0) / 1e6)
    assert worst_case_usd(reqs, price, batch=True) == pytest.approx(10 * (200 * 2.0 + 1000 * 10.0) / 2e6)


def test_budget_is_checked_before_the_first_provider_call(tmp_path):
    from pruefstand.generation import check_budget

    table = SimpleNamespace(price=lambda model: Price(2.0, 10.0, 0.2, 2.5), batch_discount=0.5)
    inner = FakeInner()
    client = ReplayClient("record", Cassette(tmp_path / "c.jsonl"), inner)
    reqs = [request(max_tokens=200_000, messages=(("user", f"{n}"),)) for n in range(40)]
    with pytest.raises(BudgetExceeded):
        check_budget(client, reqs, "claude-sonnet-5-5", table, BudgetGuard(30.0), batch=True)
    assert inner.calls == 0 and inner.batches == []
    assert check_budget(client, reqs[:1], "claude-sonnet-5-5", table, BudgetGuard(30.0), batch=True) > 0
    replayer = ReplayClient("replay", Cassette(tmp_path / "c.jsonl"))
    assert check_budget(replayer, reqs, "claude-sonnet-5-5", table, BudgetGuard(0.0), batch=True) == 0.0


def test_configured_judge_stage_fits_the_cap_even_at_max_tokens(root):
    from pruefstand.config import load_config

    cfg = load_config(root / "pruefstand.toml", "full")
    table = load_prices(root / "costs.toml")
    role = cfg["llm"]["judge"]
    calls = sum(stage["calls"] for stage in cfg["cost"]["stage"] if stage["name"].startswith("judge"))
    output_bound = (
        calls * role["max_tokens"] * table.price(role["model"]).output / 1e6 * (1 - table.batch_discount)
    )
    # output at the hard ceiling must leave room for the input side and the answer generator
    assert output_bound < cfg["budget"]["max_usd_per_run"] * 0.6
    assert role["max_tokens"] <= 1500


def test_stage_budget_is_shared_through_spend_json(tmp_path, monkeypatch):
    monkeypatch.setenv("PRUEFSTAND_LLM_MODE", "record")
    cfg = Config(
        data={
            "budget": {"max_usd_per_run": 30.0},
            "paths": {"runs_dir": "runs", "run_id": "r"},
            "llm": {"mode": "replay"},
        },
        profile="full",
        root=tmp_path,
    )
    generation_guard = stage_guard(cfg, "generate")
    generation_guard.add(1.5)
    save_stage_spend(cfg, "generate", generation_guard, 0.0)
    judge_guard = stage_guard(cfg, "judge")
    assert judge_guard.spent == 1.5
    judge_guard.add(2.0)
    save_stage_spend(cfg, "judge", judge_guard, 1.5)
    assert json.loads((tmp_path / "runs" / "r" / "spend.json").read_text(encoding="utf-8")) == {
        "generate": 1.5,
        "judge": 2.0,
    }
    assert stage_guard(cfg, "generate").spent == 2.0  # a rerun of one stage does not count itself twice
    monkeypatch.setenv("PRUEFSTAND_LLM_MODE", "replay")
    save_stage_spend(cfg, "judge", judge_guard, 0.0)  # replay never rewrites the file
    assert json.loads((tmp_path / "runs" / "r" / "spend.json").read_text(encoding="utf-8"))["judge"] == 2.0


# -------------------------------------------------------------------------------------- I1: split judge

QUESTION = {"question_de": "Wer ist zuständig?", "gold_answer_de": "GOLDSATZ-GEHEIM Der Netzbetreiber."}
ROLE = {
    "provider": "anthropic",
    "model": "claude-sonnet-5-5",
    "max_tokens": 1000,
    "effort": "low",
    "thinking": "adaptive",
}


def test_faithfulness_call_never_sees_the_gold_answer_and_correctness_never_the_context():
    prompt = Prompt("judge_faithfulness_v2", "Regeln\n", "h1")
    faith = build_faithfulness_request(QUESTION, "Antwort X", "KONTEXT-GEHEIM [1] Text", prompt, ROLE, "t")
    correct = build_correctness_request(
        QUESTION, "Antwort X", Prompt("judge_correctness_v2", "R\n", "h2"), ROLE, "t"
    )
    assert "GOLDSATZ-GEHEIM" not in json.dumps(faith.to_dict(), ensure_ascii=False)
    assert "<referenzantwort>" not in faith.messages[0][1]
    assert "KONTEXT-GEHEIM" in faith.messages[0][1] and "Antwort X" in faith.messages[0][1]
    assert "KONTEXT-GEHEIM" not in json.dumps(correct.to_dict(), ensure_ascii=False)
    assert "<kontext>" not in correct.messages[0][1]
    assert "GOLDSATZ-GEHEIM" in correct.messages[0][1]
    assert faith.stage == "judge_faithfulness" and correct.stage == "judge_correctness"
    assert replay_key(faith) != replay_key(correct)
    assert set(faith.response_schema["properties"]) == {"faithfulness", "rationale"}
    assert set(correct.response_schema["properties"]) == {"correctness", "completeness", "rationale"}


def test_judge_prompt_files_keep_the_gold_answer_and_the_context_apart(root):
    faithfulness = (root / "prompts" / "judge_faithfulness_v2.md").read_text(encoding="utf-8")
    correctness = (root / "prompts" / "judge_correctness_v2.md").read_text(encoding="utf-8")
    assert "<referenzantwort>" not in faithfulness.split("Anchor")[0].split("Illustrative")[0]
    assert "<kontext>" not in correctness
    assert "faithfulness" not in correctness.split("Score two criteria")[1].split("correctness (0")[0]


# ------------------------------------------------------------------------------- judge schema, mutants


def test_judge_schema_validation_and_refusal():
    good = {"correctness": 2, "completeness": 1, "rationale": "x"}
    assert validate_judgement(good, "correctness") == []
    assert validate_judgement({"faithfulness": 2, "rationale": "x"}, "faithfulness") == []
    assert validate_judgement({**good, "correctness": 3}, "correctness")
    assert validate_judgement({**good, "correctness": True}, "correctness")
    assert validate_judgement({**good, "extra": 1}, "correctness")
    assert validate_judgement({"faithfulness": 2, "rationale": "x", "completeness": 1}, "faithfulness")
    assert validate_judgement([], "correctness") == ["judgement is not a JSON object"]
    assert parse_part(response("", stop_reason="refusal"), "faithfulness")["status"] == "refusal"
    assert parse_part(response("kein json"), "correctness")["status"] == "invalid"
    ok = parse_part(response('{"correctness": 2, "completeness": 1, "rationale": "r"}'), "correctness")
    assert ok == {"correctness": 2, "completeness": 1, "rationale": "r", "status": "ok"}


def test_completeness_two_and_empty_rationale_are_rejected():
    assert ALLOWED["correctness"]["completeness"] == (0, 1)
    errors = validate_judgement({"correctness": 1, "completeness": 2, "rationale": "x"}, "correctness")
    assert any("completeness" in e for e in errors)
    for rationale in ("", "   ", None, 5):
        errors = validate_judgement(
            {"correctness": 1, "completeness": 1, "rationale": rationale}, "correctness"
        )
        assert any("rationale" in e for e in errors), rationale
    assert validate_judgement({"correctness": 1, "completeness": 1}, "correctness")
    assert validate_judgement({"faithfulness": 1}, "faithfulness")


def test_failed_batch_item_is_reported_as_failed_with_its_error():
    failed = response(
        "",
        stop_reason="batch_errored",
        provider_meta={"failed": True, "error": {"message": "Overloaded"}},
    )
    part = parse_part(failed, "faithfulness")
    assert part["status"] == "failed" and "Overloaded" in part["error"] and "batch_errored" in part["error"]


def test_combined_judgement_takes_the_worst_status():
    ok_f = {"faithfulness": 2, "rationale": "f", "status": "ok"}
    ok_c = {"correctness": 1, "completeness": 1, "rationale": "c", "status": "ok"}
    combined = combine_judgement(ok_f, ok_c)
    assert combined["status"] == "ok" and combined["faithfulness"] == 2 and combined["correctness"] == 1
    assert combined["rationale_faithfulness"] == "f" and combined["rationale"] == "c"
    refused = {"faithfulness": None, "rationale": "", "status": "refusal"}
    assert combine_judgement(refused, ok_c)["status"] == "refusal"
    invalid = {
        "correctness": None,
        "completeness": None,
        "rationale": "",
        "status": "invalid",
        "error": "bad",
    }
    both = combine_judgement(refused, invalid)
    assert both["status"] == "refusal" and both["error"] == "bad"
    failed = {"correctness": None, "completeness": None, "rationale": "", "status": "failed", "error": "e"}
    assert combine_judgement(refused, failed)["status"] == "failed"
    assert combine_judgement(ok_f, failed)["status_correctness"] == "failed"


def test_committed_cassette_is_entirely_synthetic_and_replays(root):
    path = root / "fixtures" / "llm" / "synthetic_cassette.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 48  # 16 answers + 16 x (faithfulness, correctness)
    assert all(entry["synthetic"] is True for entry in entries)
    assert all(entry["response"]["provider_meta"]["synthetic"] is True for entry in entries)
    assert all(entry["record"]["recorded_at"] and entry["record"]["sdk_version"] for entry in entries)
    stages = sorted({entry["request"]["stage"] for entry in entries})
    assert stages == ["answer", "judge_correctness", "judge_faithfulness"]
    fixture = json.loads((root / "fixtures" / "llm" / "synthetic_responses.json").read_text(encoding="utf-8"))
    assert fixture["note"].startswith("SYNTHETIC")
    assert all(item["rationale"].startswith("Synthetic:") for item in fixture["judgements"].values())
    faithfulness_requests = [e["request"] for e in entries if e["request"]["stage"] == "judge_faithfulness"]
    assert all("<referenzantwort>" not in r["messages"][0]["content"] for r in faithfulness_requests)


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
    assert ok.exit_code == 0 and "USD 25.2" in ok.output


def test_pruefstand_errors_are_clean():
    assert issubclass(BatchWaitError, PruefstandError) and issubclass(PromptDoesNotFit, PruefstandError)
