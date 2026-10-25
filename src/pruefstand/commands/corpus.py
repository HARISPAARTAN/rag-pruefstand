"""Corpus commands: fetch, tokenizer, build, chunk, show."""

from __future__ import annotations

from typing import Annotated

import typer

from pruefstand.chunking import fetch_tokenizer
from pruefstand.commands.common import DEFAULT_CONFIG_PATH, ConfigOption, ProfileOption, guarded, load
from pruefstand.corpus.gii import fetch_laws, load_units
from pruefstand.pipeline import stage_chunk, stage_corpus_build

corpus_app = typer.Typer(no_args_is_help=True, help="Fetch and parse the GII law texts.")


@corpus_app.command("fetch")
def fetch(
    update: Annotated[
        bool,
        typer.Option("--update", help="Replace a changed committed snapshot (starts a new corpus version)."),
    ] = False,
    profile: ProfileOption = "full",
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Download xml.zip per law and write snapshot.json; a changed text is refused without --update."""
    cfg = load(profile, config)
    corpus = cfg["corpus"]
    with guarded():
        snapshot = fetch_laws(
            corpus["laws"],
            corpus["base_url"],
            cfg.path(corpus["zip_dir"]),
            cfg.path(corpus["snapshot_path"]),
            update=update,
        )
    for slug, entry in snapshot["laws"].items():
        typer.echo(
            f"{slug}: {entry['bytes']} bytes, sha256 {entry['sha256'][:16]}, build {entry['builddate']}"
        )


@corpus_app.command("fetch-tokenizer")
def fetch_tokenizer_cmd(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Download the pinned e5 tokenizer.json (17 MB) and store it xz-compressed (about 2.4 MB)."""
    cfg = load(profile, config)
    e5 = cfg["models"]["e5"]
    with guarded():
        fetch_tokenizer(e5["id"], e5["revision"], cfg.path(cfg["tokenizer"]["path"]))
    typer.echo(f"wrote {cfg['tokenizer']['path']}")


@corpus_app.command("build")
def build(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Parse the downloaded zips into units (one per Absatz)."""
    cfg = load(profile, config)
    with guarded():
        for stats in stage_corpus_build(cfg):
            typer.echo(
                f"{stats.law_slug}: {stats.n_sections} sections, {stats.n_units} units, "
                f"{stats.n_repealed_skipped} repealed, {stats.n_non_section_skipped} non-section, "
                f"{stats.n_footnotes_removed} footnotes removed"
            )


def chunk(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Write C1 and C2 chunks for the run."""
    cfg = load(profile, config)
    with guarded():
        counts = stage_chunk(cfg)
    typer.echo(f"C1: {counts['C1']} chunks, C2: {counts['C2']} chunks")


def show(
    prefix: Annotated[str, typer.Argument(help="Unit id or prefix, e.g. 'messbg:§5:'")],
    profile: ProfileOption = "full",
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Print the full text of every unit whose id starts with PREFIX."""
    cfg = load(profile, config)
    with guarded():
        units = load_units(cfg.path(cfg["corpus"]["units_path"]))
    matches = [unit for unit in units if unit.unit_id.startswith(prefix)]
    for unit in matches:
        typer.echo(f"{unit.unit_id} | {unit.label} {unit.heading}\n{unit.text}\n")
    if not matches:
        typer.echo("no unit matches", err=True)
        raise typer.Exit(code=1)


def register(app: typer.Typer) -> None:
    app.add_typer(corpus_app, name="corpus")
    app.command("chunk")(chunk)
    app.command("show")(show)
