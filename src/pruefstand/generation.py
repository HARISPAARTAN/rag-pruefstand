"""Generation and judging stages plus answer-level metrics with bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable

from pruefstand.answer import answer_record, build_answer_request, build_context
from pruefstand.bench import load_questions
from pruefstand.chunking import load_chunks
from pruefstand.config import Config
from pruefstand.corpus.gii import load_units
from pruefstand.io import read_json, read_jsonl, write_json, write_jsonl
from pruefstand.judge import (
    RUBRIC_VERSION,
    build_correctness_request,
    build_faithfulness_request,
    combine_judgement,
    parse_part,
)
from pruefstand.llm.anthropic_client import DEFAULT_MAX_WAIT_SECONDS, AnthropicClient
from pruefstand.llm.base import LLMRequest, LLMResponse
from pruefstand.llm.costs import BudgetGuard, PriceTable, cost_usd, load_prices, worst_case_usd
from pruefstand.llm.ollama_client import OllamaClient
from pruefstand.llm.replay import Cassette, ReplayClient, is_failed, replay_key
from pruefstand.prompts import load_prompt
from pruefstand.stats import bootstrap_ci, percentile


def source_hits_path(cfg: Config):
    """Hits of the retrieval run this profile reads (its own run, or the published one for the pilot)."""
    return cfg.source_dir / "hits.jsonl.gz"


def answers_path(cfg: Config):
    return cfg.run_dir / "answers.jsonl"


def judgements_path(cfg: Config):
    return cfg.run_dir / "judgements.jsonl"


def spend_path(cfg: Config):
    return cfg.run_dir / "spend.json"


def stage_guard(cfg: Config, stage: str) -> BudgetGuard:
    """One cap for the whole run: what the other stages spent (spend.json) counts against it."""
    guard = BudgetGuard(float(cfg["budget"]["max_usd_per_run"]))
    path = spend_path(cfg)
    if path.exists():
        guard.spent = sum(value for name, value in read_json(path).items() if name != stage)
    return guard


def save_stage_spend(cfg: Config, stage: str, guard: BudgetGuard, before: float) -> None:
    """Persist what this stage spent (live runs only; replay leaves the committed run untouched)."""
    if cfg.llm_mode() == "replay":
        return
    path = spend_path(cfg)
    spend = read_json(path) if path.exists() else {}
    spend[stage] = round(guard.spent - before, 8)
    write_json(path, spend)


def cost_function(table: PriceTable) -> Callable[[LLMRequest, LLMResponse], float]:
    def cost(request: LLMRequest, response: LLMResponse) -> float:
        return cost_usd(
            response.usage,
            table.price(request.model),
            batch=response.batch,
            batch_discount=table.batch_discount,
        )

    return cost


def make_client(cfg: Config, role: str, refresh: bool = False) -> ReplayClient:
    """Client for a role. Synthetic cassette entries are visible in the ci profile only."""
    mode = cfg.llm_mode()
    cassette = Cassette(cfg.path(cfg["llm"]["cassette"]), allow_synthetic=cfg.profile == "ci")
    table = load_prices(cfg.path(cfg["llm"]["costs"]))
    inner = None
    if mode != "replay":
        provider = cfg["llm"][role]["provider"]
        if provider == "anthropic":
            inner = AnthropicClient(
                poll_seconds=int(cfg["llm"]["batch_poll_seconds"]),
                max_wait_seconds=float(cfg["llm"].get("batch_max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS)),
            )
        elif provider == "ollama":
            inner = OllamaClient()
        else:
            raise ValueError(f"unknown provider {provider!r}")
    return ReplayClient(
        mode,
        cassette,
        inner,
        refresh=refresh,
        cost_fn=cost_function(table),
        price_as_of=str(table.meta["as_of"]),
    )


def selected_questions(cfg: Config) -> list[dict]:
    """All questions, or for a capped profile the first N-1 answerable plus the first unanswerable one."""
    questions = sorted(load_questions(cfg), key=lambda q: q["qid"])
    limit = int(cfg["generation"]["max_questions"])
    if limit <= 0:
        return questions
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    chosen = answerable[: limit - 1] + unanswerable[:1]
    return sorted(chosen, key=lambda q: q["qid"])


def chunk_lookup(cfg: Config) -> dict:
    lookup = {}
    for chunking in cfg["retrieval"]["chunkings"]:
        for chunk in load_chunks(cfg.source_dir / f"chunks_{chunking}.jsonl.gz"):
            lookup[chunk.chunk_id] = chunk
    return lookup


def check_budget(
    client: ReplayClient,
    requests: list[LLMRequest],
    model: str,
    table: PriceTable,
    guard: BudgetGuard,
    batch: bool,
) -> float:
    """Hard bound before anything is sent: input bound plus max_tokens of output for every request that would
    reach the provider, against the remaining cap. Replay needs no check."""
    if client.mode == "replay":
        return 0.0
    bound = worst_case_usd(
        client.misses(requests), table.price(model), batch=batch, batch_discount=table.batch_discount
    )
    guard.check_projection(bound)
    return bound


def run_generation(
    cfg: Config, client: ReplayClient, guard: BudgetGuard, table: PriceTable | None = None
) -> list[dict]:
    table = table or load_prices(cfg.path(cfg["llm"]["costs"]))
    role = cfg["llm"]["generator"]
    prompt = load_prompt(cfg.path(role["prompt"]))
    units_by_id = {unit.unit_id: unit for unit in load_units(cfg.path(cfg["corpus"]["units_path"]))}
    hits = {(row["config_id"], row["qid"]): row for row in read_jsonl(source_hits_path(cfg))}
    chunk_by_id = chunk_lookup(cfg)
    n_context = int(cfg["generation"]["context_chunks"])
    jobs = []
    for config_id in cfg["generation"]["variants"]:
        for question in selected_questions(cfg):
            ranked = hits[(config_id, question["qid"])]["ranked"][:n_context]
            chunks = [chunk_by_id[hit["chunk_id"]] for hit in ranked]
            tag = f"answer|{config_id}|{question['qid']}"
            jobs.append(
                (
                    config_id,
                    question,
                    chunks,
                    build_answer_request(question, chunks, units_by_id, prompt, role, tag),
                )
            )
    check_budget(client, [job[3] for job in jobs], role["model"], table, guard, batch=False)
    price = table.price(role["model"])
    records = []
    try:
        for config_id, question, chunks, request in jobs:
            called_before = client.called
            response = client.complete(request)
            billed_now = client.mode != "replay" and client.called > called_before
            cost = cost_usd(response.usage, price, response.batch, table.batch_discount)
            if billed_now:
                guard.add(cost)
            record = answer_record(question, config_id, chunks, response, cost, role["model"])
            record["synthetic"] = bool(response.provider_meta.get("synthetic", False))
            record["replay_key"] = replay_key(request)
            record["recorded_cost_usd"] = (client.recorded(request) or {}).get("cost_usd")
            record["billed_now"] = billed_now
            records.append(record)
    finally:
        if client.mode == "record":
            client.save()
    write_jsonl(answers_path(cfg), records)
    return records


def judge_requests(cfg: Config, answers: list[dict], role: dict) -> list[tuple[dict, LLMRequest, LLMRequest]]:
    """Per answer two requests: faithfulness without the gold answer, correctness without the context."""
    faithfulness_prompt = load_prompt(cfg.path(role["faithfulness_prompt"]))
    correctness_prompt = load_prompt(cfg.path(role["prompt"]))
    units_by_id = {unit.unit_id: unit for unit in load_units(cfg.path(cfg["corpus"]["units_path"]))}
    questions = {question["qid"]: question for question in load_questions(cfg)}
    chunk_by_id = chunk_lookup(cfg)
    pairs = []
    for answer in answers:
        question = questions[answer["qid"]]
        context = build_context([chunk_by_id[cid] for cid in answer["context_chunk_ids"]], units_by_id)
        suffix = f"{answer['config_id']}|{answer['qid']}"
        pairs.append(
            (
                answer,
                build_faithfulness_request(
                    question,
                    answer["text"],
                    context,
                    faithfulness_prompt,
                    role,
                    f"judge_faithfulness|{suffix}",
                ),
                build_correctness_request(
                    question, answer["text"], correctness_prompt, role, f"judge_correctness|{suffix}"
                ),
            )
        )
    return pairs


def run_judging(
    cfg: Config, client: ReplayClient, guard: BudgetGuard, table: PriceTable | None = None
) -> list[dict]:
    table = table or load_prices(cfg.path(cfg["llm"]["costs"]))
    role = cfg["llm"]["judge"]
    pairs = judge_requests(cfg, read_jsonl(answers_path(cfg)), role)
    requests = [request for _, faithfulness, correctness in pairs for request in (faithfulness, correctness)]
    use_batch = bool(role.get("batch", False))
    check_budget(client, requests, role["model"], table, guard, batch=use_batch)
    fresh = {replay_key(request) for request in client.misses(requests)}
    price = table.price(role["model"])
    try:
        responses = client.complete_many(
            requests, use_batch=use_batch, state_path=cfg.run_dir / "batch_judge.json"
        )
    finally:
        if client.mode == "record":
            client.save()
    judgements = []
    for index, (answer, faithfulness_request, correctness_request) in enumerate(pairs):
        calls = {
            "faithfulness": (faithfulness_request, responses[2 * index]),
            "correctness": (correctness_request, responses[2 * index + 1]),
        }
        cost, billed_now, recorded = 0.0, False, []
        for request, response in calls.values():
            call_cost = cost_usd(response.usage, price, response.batch, table.batch_discount)
            cost += call_cost
            if client.mode != "replay" and replay_key(request) in fresh and not is_failed(response):
                guard.add(call_cost)
                billed_now = True
            recorded.append((client.recorded(request) or {}).get("cost_usd"))
        judgements.append(
            {
                "config_id": answer["config_id"],
                "qid": answer["qid"],
                "judge_model": role["model"],
                "rubric_version": RUBRIC_VERSION,
                **combine_judgement(
                    parse_part(calls["faithfulness"][1], "faithfulness"),
                    parse_part(calls["correctness"][1], "correctness"),
                ),
                "cost_usd": round(cost, 8),
                "recorded_cost_usd": None if None in recorded else round(sum(recorded), 8),
                "billed_now": billed_now,
                "synthetic": any(bool(r.provider_meta.get("synthetic", False)) for _, r in calls.values()),
                "replay_key_faithfulness": replay_key(faithfulness_request),
                "replay_key_correctness": replay_key(correctness_request),
            }
        )
    write_jsonl(judgements_path(cfg), judgements)
    return judgements


def compute_answer_metrics(cfg: Config) -> dict:
    n_boot, seed = cfg["stats"]["bootstrap"], cfg["stats"]["seed"]
    table = load_prices(cfg.path(cfg["llm"]["costs"]))
    questions = {question["qid"]: question for question in load_questions(cfg)}
    answers = read_jsonl(answers_path(cfg))
    judged = {(j["config_id"], j["qid"]): j for j in read_jsonl(judgements_path(cfg))}
    variants = {}
    for config_id in cfg["generation"]["variants"]:
        rows = sorted((a for a in answers if a["config_id"] == config_id), key=lambda a: a["qid"])
        verdicts = [judged[(config_id, a["qid"])] for a in rows]
        faithful = [j for j in verdicts if j["status_faithfulness"] == "ok"]
        correct = [j for j in verdicts if j["status_correctness"] == "ok"]

        def ci(values: list[float]) -> dict:
            return bootstrap_ci(values, n_boot, seed)

        n = len(rows)
        mean_cost = sum(a["cost_usd"] for a in rows) / n if n else 0.0
        judge_cost = sum(j["cost_usd"] for j in verdicts) / n if n else 0.0
        recorded = [a["recorded_cost_usd"] for a in rows]
        recorded_mean = None if not n or None in recorded else sum(recorded) / n
        latencies = [a["latency_ms"] for a in rows if a["latency_ms"] is not None]
        variants[config_id] = {
            "n_answers": n,
            "faithfulness": ci([j["faithfulness"] for j in faithful]),
            "correctness": ci([j["correctness"] for j in correct]),
            "correctness_share_2": ci([1.0 if j["correctness"] == 2 else 0.0 for j in correct]),
            "completeness": ci([j["completeness"] for j in correct]),
            "citation_precision": ci(
                [a["citation_precision"] for a in rows if a["citation_precision"] is not None]
            ),
            "citation_recall": ci([a["citation_recall"] for a in rows if a["citation_recall"] is not None]),
            "paragraph_mentions": sum(a["paragraph_mentions"] for a in rows),
            "unparsed_mentions": sum(a["unparsed_mentions"] for a in rows),
            "answers_with_unparsed_mentions": sum(1 for a in rows if a["unparsed_mentions"] > 0),
            "abstention_accuracy": ci(
                [1.0 if a["abstained"] == (not questions[a["qid"]]["answerable"]) else 0.0 for a in rows]
            ),
            "refusals": sum(1 for a in rows if a["stop_reason"] == "refusal"),
            "judge_refusals": sum(1 for j in verdicts if j["status"] == "refusal"),
            "judge_invalid": sum(1 for j in verdicts if j["status"] == "invalid"),
            "judge_failed": sum(1 for j in verdicts if j["status"] == "failed"),
            "mean_input_tokens": round(sum(a["usage"]["input_tokens"] for a in rows) / n, 1) if n else 0.0,
            "mean_output_tokens": round(sum(a["usage"]["output_tokens"] for a in rows) / n, 1) if n else 0.0,
            "usd_per_1000": round(mean_cost * 1000, 4),
            "recorded_usd_per_1000": None if recorded_mean is None else round(recorded_mean * 1000, 4),
            "eur_per_1000": round(mean_cost * 1000 * table.eur_per_usd, 4),
            "judge_usd_per_1000": round(judge_cost * 1000, 4),
            "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        }
    result = {
        "run_id": cfg.run_id,
        "synthetic": any(a.get("synthetic") for a in answers),
        "generator": cfg["llm"]["generator"]["model"],
        "judge": cfg["llm"]["judge"]["model"],
        "rubric_version": RUBRIC_VERSION,
        "prices_as_of": table.meta["as_of"],
        "eur_per_usd": table.eur_per_usd,
        "cost_note": (
            "usd_per_1000 is recomputed from the usage with the current costs.toml; "
            "recorded_usd_per_1000 is the cost stored in the cassette at record time."
        ),
        "variants": variants,
    }
    write_json(cfg.run_dir / "answer_metrics.json", result)
    return result
