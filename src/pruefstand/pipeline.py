"""Stage functions over files in runs/<run_id>/ and the `all` sequence."""

from __future__ import annotations

from collections.abc import Callable

from pruefstand.aggregate import compute_retrieval_metrics
from pruefstand.bench import run_retrieval
from pruefstand.chunking import chunk_c1, chunk_c2, chunks_path, load_tokenizer, write_chunks
from pruefstand.config import Config
from pruefstand.corpus.gii import build_units, load_units
from pruefstand.latency import run_latency
from pruefstand.manifest import now_iso, write_manifest


def stage_corpus_build(cfg: Config) -> list:
    corpus = cfg["corpus"]
    return build_units(
        corpus["laws"],
        corpus["base_url"],
        cfg.path(corpus["zip_dir"]),
        cfg.path(corpus["snapshot_path"]),
        cfg.path(corpus["units_path"]),
    )


def stage_chunk(cfg: Config) -> dict[str, int]:
    units = load_units(cfg.path(cfg["corpus"]["units_path"]))
    tokenizer = load_tokenizer(str(cfg.path(cfg["tokenizer"]["path"])))
    chunking = cfg["chunking"]
    corpus_id = cfg["corpus"]["id"]
    c1 = chunk_c1(units, tokenizer, chunking["c1_window"], chunking["c1_overlap"], corpus_id)
    c2 = chunk_c2(units, tokenizer, chunking["c2_max_tokens"], corpus_id)
    write_chunks(chunks_path(cfg, "C1"), c1)
    write_chunks(chunks_path(cfg, "C2"), c2)
    return {"C1": len(c1), "C2": len(c2)}


def stage_retrieve(cfg: Config) -> int:
    return run_retrieval(cfg)


def stage_latency(cfg: Config) -> dict:
    return run_latency(cfg)


def stage_metrics(cfg: Config) -> dict:
    return compute_retrieval_metrics(cfg)


def stage_generate(cfg: Config, refresh: bool = False) -> list[dict]:
    from pruefstand.generation import make_client, run_generation, save_stage_spend, stage_guard

    guard = stage_guard(cfg, "generate")
    before = guard.spent
    try:
        return run_generation(cfg, make_client(cfg, "generator", refresh), guard)
    finally:
        save_stage_spend(cfg, "generate", guard, before)


def stage_judge(cfg: Config, refresh: bool = False) -> list[dict]:
    from pruefstand.generation import make_client, run_judging, save_stage_spend, stage_guard

    guard = stage_guard(cfg, "judge")
    before = guard.spent
    try:
        return run_judging(cfg, make_client(cfg, "judge", refresh), guard)
    finally:
        save_stage_spend(cfg, "judge", guard, before)


def stage_answer_metrics(cfg: Config) -> dict:
    from pruefstand.generation import compute_answer_metrics

    return compute_answer_metrics(cfg)


def stage_report(cfg: Config) -> list[str]:
    from pruefstand.report.build import build_report

    return build_report(cfg)


def planned_stages(cfg: Config) -> list[tuple[str, Callable[[Config], object]]]:
    stages: list[tuple[str, Callable[[Config], object]]] = [
        ("corpus build", stage_corpus_build),
        ("chunk", stage_chunk),
        ("retrieve", stage_retrieve),
    ]
    if int(cfg["latency"]["sample"]) > 0:
        stages.append(("latency", stage_latency))
    stages.append(("metrics", stage_metrics))
    if cfg["llm"]["run_in_all"]:
        stages += [
            ("generate", stage_generate),
            ("judge", stage_judge),
            ("answer metrics", stage_answer_metrics),
        ]
    stages.append(("report", stage_report))
    return stages


def run_all(cfg: Config, echo: Callable[[str], None] = print) -> list[str]:
    started = now_iso()
    done = []
    spend = 0.0
    for name, stage in planned_stages(cfg):
        echo(f"[{cfg.profile}] {name} ...")
        result = stage(cfg)
        if name in ("generate", "judge"):
            spend += sum(record["cost_usd"] for record in result if record.get("billed_now"))
        done.append(name)
    # Replayed and reused calls cost nothing now; their cost stays in answers.jsonl and judgements.jsonl.
    write_manifest(cfg, started, done, spend_usd=spend)
    return done
