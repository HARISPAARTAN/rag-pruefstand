"""Stage functions over files in runs/<run_id>/ and the `all` sequence."""

from __future__ import annotations

from collections.abc import Callable

from pruefstand.aggregate import compute_retrieval_metrics
from pruefstand.bench import run_retrieval
from pruefstand.chunking import chunk_c1, chunk_c2, chunks_path, load_tokenizer, write_chunks
from pruefstand.config import Config
from pruefstand.corpus.gii import build_units, load_units
from pruefstand.errors import PruefstandError
from pruefstand.latency import run_latency
from pruefstand.manifest import check_retrieval_provenance, now_iso, write_manifest, write_stage_manifest


def own_run_only(cfg: Config, stage: str) -> None:
    """A profile that reads the retrieval artefacts of another run (pilot) must never write into them."""
    if cfg.reads_foreign_run:
        raise PruefstandError(
            f"stage {stage!r} writes retrieval artefacts, but profile {cfg.profile!r} only reads those of run "
            f"{cfg.source_run_id}; run it with the profile that owns the run"
        )


def stage_corpus_build(cfg: Config) -> list:
    own_run_only(cfg, "corpus build")
    corpus = cfg["corpus"]
    return build_units(
        corpus["laws"],
        corpus["base_url"],
        cfg.path(corpus["zip_dir"]),
        cfg.path(corpus["snapshot_path"]),
        cfg.path(corpus["units_path"]),
    )


def stage_chunk(cfg: Config) -> dict[str, int]:
    own_run_only(cfg, "chunk")
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
    own_run_only(cfg, "retrieve")
    return run_retrieval(cfg)


def stage_latency(cfg: Config) -> dict:
    own_run_only(cfg, "latency")
    return run_latency(cfg)


def stage_metrics(cfg: Config) -> dict:
    own_run_only(cfg, "metrics")
    return compute_retrieval_metrics(cfg)


def _paid_stage(cfg: Config, stage: str, role: str, runner: str, refresh: bool) -> list[dict]:
    """Run generate or judge. Record and live runs first check the retrieval provenance and afterwards leave an
    entry in stage_manifests.json (also when the stage stops half way); replay writes nothing."""
    from pruefstand import generation

    paid = cfg.llm_mode() != "replay"
    if paid:
        check_retrieval_provenance(cfg)
    started = now_iso()
    guard = generation.stage_guard(cfg, stage)
    before = guard.spent
    records = None
    try:
        records = getattr(generation, runner)(cfg, generation.make_client(cfg, role, refresh), guard)
        return records
    finally:
        generation.save_stage_spend(cfg, stage, guard, before)
        if paid:
            write_stage_manifest(cfg, stage, started, records, guard.spent - before)


def stage_generate(cfg: Config, refresh: bool = False) -> list[dict]:
    return _paid_stage(cfg, "generate", "generator", "run_generation", refresh)


def stage_judge(cfg: Config, refresh: bool = False) -> list[dict]:
    return _paid_stage(cfg, "judge", "judge", "run_judging", refresh)


def stage_answer_metrics(cfg: Config) -> dict:
    from pruefstand.generation import compute_answer_metrics

    return compute_answer_metrics(cfg)


def stage_report(cfg: Config) -> list[str]:
    own_run_only(cfg, "report")
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
