"""Generation and judging stages plus answer-level metrics with bootstrap intervals."""

from __future__ import annotations

from pruefstand.answer import answer_record, build_answer_request, build_context
from pruefstand.bench import hits_path, load_questions
from pruefstand.chunking import chunks_path, load_chunks
from pruefstand.config import Config
from pruefstand.corpus.gii import load_units
from pruefstand.io import read_jsonl, write_json, write_jsonl
from pruefstand.judge import RUBRIC_VERSION, build_judge_request, parse_judgement
from pruefstand.llm.anthropic_client import AnthropicClient
from pruefstand.llm.base import Usage
from pruefstand.llm.costs import BudgetGuard, PriceTable, cost_usd, load_prices
from pruefstand.llm.ollama_client import OllamaClient
from pruefstand.llm.replay import Cassette, ReplayClient, replay_key
from pruefstand.prompts import load_prompt
from pruefstand.stats import bootstrap_ci, percentile


def answers_path(cfg: Config):
    return cfg.run_dir / "answers.jsonl"


def judgements_path(cfg: Config):
    return cfg.run_dir / "judgements.jsonl"


def make_client(cfg: Config, role: str) -> ReplayClient:
    mode = cfg.llm_mode()
    cassette = Cassette(cfg.path(cfg["llm"]["cassette"]))
    inner = None
    if mode != "replay":
        provider = cfg["llm"][role]["provider"]
        if provider == "anthropic":
            inner = AnthropicClient(poll_seconds=int(cfg["llm"]["batch_poll_seconds"]))
        elif provider == "ollama":
            inner = OllamaClient()
        else:
            raise ValueError(f"unknown provider {provider!r}")
    return ReplayClient(mode, cassette, inner)


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


def stage_estimate(cfg: Config, role: str) -> tuple[int, int]:
    for stage in cfg.data.get("cost", {}).get("stage", []):
        if stage["role"] == role:
            return int(stage["input_tokens"]), int(stage["output_tokens"])
    return 4000, int(cfg["llm"][role]["max_tokens"])


def _project(cfg: Config, role: str, n_calls: int, table: PriceTable, batch: bool) -> float:
    est_in, est_out = stage_estimate(cfg, role)
    usage = Usage(input_tokens=n_calls * est_in, output_tokens=n_calls * est_out)
    return cost_usd(
        usage, table.price(cfg["llm"][role]["model"]), batch=batch, batch_discount=table.batch_discount
    )


def chunk_lookup(cfg: Config) -> dict:
    lookup = {}
    for chunking in cfg["retrieval"]["chunkings"]:
        for chunk in load_chunks(chunks_path(cfg, chunking)):
            lookup[chunk.chunk_id] = chunk
    return lookup


def run_generation(
    cfg: Config, client: ReplayClient, guard: BudgetGuard, table: PriceTable | None = None
) -> list[dict]:
    table = table or load_prices(cfg.path(cfg["llm"]["costs"]))
    role = cfg["llm"]["generator"]
    prompt = load_prompt(cfg.path(role["prompt"]))
    units_by_id = {unit.unit_id: unit for unit in load_units(cfg.path(cfg["corpus"]["units_path"]))}
    hits = {(row["config_id"], row["qid"]): row for row in read_jsonl(hits_path(cfg))}
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
    if client.mode != "replay":
        guard.check_projection(_project(cfg, "generator", len(jobs), table, batch=False))
    price = table.price(role["model"])
    records = []
    try:
        for config_id, question, chunks, request in jobs:
            response = client.complete(request)
            cost = cost_usd(response.usage, price, batch=response.batch, batch_discount=table.batch_discount)
            if client.mode != "replay":
                guard.add(cost)
            record = answer_record(question, config_id, chunks, response, cost, role["model"])
            record["synthetic"] = bool(response.provider_meta.get("synthetic", False))
            record["replay_key"] = replay_key(request)
            records.append(record)
    finally:
        if client.mode == "record":
            client.cassette.save()
    write_jsonl(answers_path(cfg), records)
    return records


def run_judging(
    cfg: Config, client: ReplayClient, guard: BudgetGuard, table: PriceTable | None = None
) -> list[dict]:
    table = table or load_prices(cfg.path(cfg["llm"]["costs"]))
    role = cfg["llm"]["judge"]
    prompt = load_prompt(cfg.path(role["prompt"]))
    units_by_id = {unit.unit_id: unit for unit in load_units(cfg.path(cfg["corpus"]["units_path"]))}
    questions = {question["qid"]: question for question in load_questions(cfg)}
    chunk_by_id = chunk_lookup(cfg)
    answers = read_jsonl(answers_path(cfg))
    requests = []
    for answer in answers:
        context = build_context([chunk_by_id[cid] for cid in answer["context_chunk_ids"]], units_by_id)
        tag = f"judge|{answer['config_id']}|{answer['qid']}"
        requests.append(
            build_judge_request(questions[answer["qid"]], answer["text"], context, prompt, role, tag)
        )
    use_batch = bool(role.get("batch", False))
    if client.mode != "replay":
        guard.check_projection(_project(cfg, "judge", len(requests), table, batch=use_batch))
    price = table.price(role["model"])
    try:
        responses = client.complete_many(requests, use_batch=use_batch)
    finally:
        if client.mode == "record":
            client.cassette.save()
    judgements = []
    for answer, request, response in zip(answers, requests, responses, strict=True):
        cost = cost_usd(response.usage, price, batch=response.batch, batch_discount=table.batch_discount)
        if client.mode != "replay":
            guard.add(cost)
        judgements.append(
            {
                "config_id": answer["config_id"],
                "qid": answer["qid"],
                "judge_model": role["model"],
                "rubric_version": RUBRIC_VERSION,
                **parse_judgement(response),
                "cost_usd": cost,
                "synthetic": bool(response.provider_meta.get("synthetic", False)),
                "replay_key": replay_key(request),
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
        ok = [judged[(config_id, a["qid"])] for a in rows if judged[(config_id, a["qid"])]["status"] == "ok"]

        def ci(values: list[float]) -> dict:
            return bootstrap_ci(values, n_boot, seed)

        n = len(rows)
        mean_cost = sum(a["cost_usd"] for a in rows) / n if n else 0.0
        judge_cost = sum(judged[(config_id, a["qid"])]["cost_usd"] for a in rows) / n if n else 0.0
        latencies = [a["latency_ms"] for a in rows if a["latency_ms"] is not None]
        variants[config_id] = {
            "n_answers": n,
            "faithfulness": ci([j["faithfulness"] for j in ok]),
            "correctness": ci([j["correctness"] for j in ok]),
            "correctness_share_2": ci([1.0 if j["correctness"] == 2 else 0.0 for j in ok]),
            "completeness": ci([j["completeness"] for j in ok]),
            "citation_precision": ci(
                [a["citation_precision"] for a in rows if a["citation_precision"] is not None]
            ),
            "citation_recall": ci([a["citation_recall"] for a in rows if a["citation_recall"] is not None]),
            "abstention_accuracy": ci(
                [1.0 if a["abstained"] == (not questions[a["qid"]]["answerable"]) else 0.0 for a in rows]
            ),
            "refusals": sum(1 for a in rows if a["stop_reason"] == "refusal"),
            "judge_refusals": sum(1 for a in rows if judged[(config_id, a["qid"])]["status"] == "refusal"),
            "judge_invalid": sum(1 for a in rows if judged[(config_id, a["qid"])]["status"] == "invalid"),
            "mean_input_tokens": round(sum(a["usage"]["input_tokens"] for a in rows) / n, 1) if n else 0.0,
            "mean_output_tokens": round(sum(a["usage"]["output_tokens"] for a in rows) / n, 1) if n else 0.0,
            "usd_per_1000": round(mean_cost * 1000, 4),
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
        "variants": variants,
    }
    write_json(cfg.run_dir / "answer_metrics.json", result)
    return result
