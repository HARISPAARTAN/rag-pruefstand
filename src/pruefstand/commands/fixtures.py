"""CI fixture commands: build the offline MsbG slice with cached embeddings and rerank scores."""

from __future__ import annotations

import shutil

import typer

from pruefstand.commands.common import DEFAULT_CONFIG_PATH, ConfigOption, guarded
from pruefstand.config import Config, deep_merge, load_config
from pruefstand.io import read_json, read_jsonl, write_json, write_jsonl
from pruefstand.pipeline import stage_chunk, stage_corpus_build, stage_retrieve

fixtures_app = typer.Typer(
    no_args_is_help=True, help="Build the committed CI fixtures (needs the models extra)."
)
CI_LAW = "messbg"
CI_QUESTIONS = 30


@fixtures_app.command("build-ci")
def build_ci(config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Copy the MsbG zip and its questions into fixtures/ci and fill fixtures/ci/cache with real model outputs."""
    full = load_config(config, "full")
    ci = load_config(config, "ci")
    questions = [q for q in read_jsonl(full.path(full["questions"]["path"])) if q["law"] == CI_LAW]
    if len(questions) != CI_QUESTIONS:
        typer.echo(f"expected {CI_QUESTIONS} {CI_LAW} questions, found {len(questions)}", err=True)
        raise typer.Exit(code=1)
    write_jsonl(ci.path(ci["questions"]["path"]), questions)
    zip_dir = ci.path(ci["corpus"]["zip_dir"])
    zip_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(full.path(full["corpus"]["zip_dir"]) / f"{CI_LAW}.xml.zip", zip_dir / f"{CI_LAW}.xml.zip")
    snapshot = read_json(full.path(full["corpus"]["snapshot_path"]))
    snapshot["laws"] = {CI_LAW: snapshot["laws"][CI_LAW]}
    write_json(ci.path(ci["corpus"]["snapshot_path"]), snapshot)
    compute = Config(
        data=deep_merge(ci.data, {"retrieval": {"allow_compute": True}}), profile="ci", root=ci.root
    )
    with guarded():
        stage_corpus_build(compute)
        stage_chunk(compute)
        stage_retrieve(compute)
    typer.echo(f"CI fixtures written: {len(questions)} questions, caches in {ci['retrieval']['cache_dir']}")


def register(app: typer.Typer) -> None:
    app.add_typer(fixtures_app, name="fixtures")
