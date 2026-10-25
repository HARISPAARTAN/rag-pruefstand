"""Retrieval commands: models fetch, retrieve, latency, metrics, search, all."""

from __future__ import annotations

from typing import Annotated

import typer

from pruefstand.aggregate import summary_lines
from pruefstand.bench import measure_reranker_truncation
from pruefstand.commands.common import DEFAULT_CONFIG_PATH, ConfigOption, ProfileOption, guarded, load
from pruefstand.config import load_config
from pruefstand.corpus.gii import load_units
from pruefstand.pipeline import run_all, stage_latency, stage_metrics, stage_retrieve
from pruefstand.retrieval.bm25 import BM25Index
from pruefstand.verify import MAX_REPORTED, compare_runs

models_app = typer.Typer(
    no_args_is_help=True, help="Download the pinned Hugging Face models (needs the models extra)."
)


@models_app.command("fetch")
def models_fetch(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Download e5-base (1.11 GB), MiniLM (91 MB) and bge-reranker-base (1.11 GB) at pinned revisions."""
    from huggingface_hub import snapshot_download

    cfg = load(profile, config)
    for name, model in cfg["models"].items():
        target = cfg.path(model["dir"])
        snapshot_download(
            model["id"], revision=model["revision"], local_dir=target, allow_patterns=model["files"]
        )
        typer.echo(f"{name}: {model['id']}@{model['revision'][:10]} -> {target}")


def retrieve(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Run all retrieval variants and write hits.jsonl.gz."""
    cfg = load(profile, config)
    with guarded():
        rows = stage_retrieve(cfg)
    typer.echo(f"wrote {rows} hit rows to {cfg.run_dir}")


def latency(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Measure p50/p95 latency per stage on the configured question sample (live models, CPU)."""
    cfg = load(profile, config)
    with guarded():
        result = stage_latency(cfg)
    typer.echo(f"latency measured on {result.get('n_questions', 0)} questions")


def rerank_truncation(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Measure how many reranked (query, chunk) pairs exceed the reranker's max_length (committed hits, no retrieval)."""
    cfg = load(profile, config)
    with guarded():
        report = measure_reranker_truncation(cfg)
    for chunking, per_chunking in report.items():
        row = per_chunking.get("reranker")
        if row:
            typer.echo(
                f"{chunking}: {row['n_truncated']}/{row['n_pairs']} pairs above {row['max_length']} tokens "
                f"({row['truncation_rate'] * 100:.1f}%)"
            )
        else:
            typer.echo(f"{chunking}: not measured")


def metrics(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Compute retrieval metrics with bootstrap intervals and paired comparisons."""
    cfg = load(profile, config)
    with guarded():
        result = stage_metrics(cfg)
    for line in summary_lines(result):
        typer.echo(line)


def search(
    query: Annotated[str, typer.Argument(help="Search text")],
    law: Annotated[str, typer.Option(help="Restrict to one law slug, e.g. messbg")] = "",
    k: Annotated[int, typer.Option(help="Number of units to show")] = 8,
    profile: ProfileOption = "full",
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """BM25 search over units (authoring aid: check that an answer exists or does not)."""
    cfg = load(profile, config)
    units = [u for u in load_units(cfg.path(cfg["corpus"]["units_path"])) if not law or u.law_slug == law]
    index = BM25Index([f"{u.label} {u.heading} {u.text}" for u in units])
    for position, score in index.search(query, k):
        unit = units[position]
        typer.echo(f"{score:7.3f}  {unit.unit_id}  {unit.text[:150]}")


def all_stages(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Run every stage of the profile in order and write the manifest."""
    cfg = load(profile, config)
    with guarded():
        done = run_all(cfg, echo=typer.echo)
    typer.echo(f"done: {', '.join(done)}")


def verify(
    against: Annotated[
        str, typer.Option(help="Run id of the published run; default: the full profile's")
    ] = "",
    profile: ProfileOption = "verify",
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Compare a reproduction run (profile verify) with the published run: hits by content, metrics by value."""
    cfg = load(profile, config)
    published_id = against or load_config(cfg.path(config), "full").run_id
    published = cfg.path(cfg["paths"]["runs_dir"]) / published_id
    with guarded():
        differences = compare_runs(published, cfg.run_dir)
    if differences:
        for line in differences[:MAX_REPORTED]:
            typer.echo(f"  - {line}", err=True)
        more = len(differences) - MAX_REPORTED
        typer.echo(f"{len(differences)} difference(s){f' ({more} not shown)' if more > 0 else ''}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"identical: {cfg.run_id} reproduces {published_id} (hits.jsonl.gz and metrics.json)")


def register(app: typer.Typer) -> None:
    app.add_typer(models_app, name="models")
    app.command("retrieve")(retrieve)
    app.command("latency")(latency)
    app.command("rerank-truncation")(rerank_truncation)
    app.command("metrics")(metrics)
    app.command("search")(search)
    app.command("all")(all_stages)
    app.command("verify")(verify)
