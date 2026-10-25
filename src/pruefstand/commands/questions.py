"""Question-set commands: check, assemble, review-export, review-import."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from pruefstand.commands.common import DEFAULT_CONFIG_PATH, ConfigOption, ProfileOption, load
from pruefstand.corpus.gii import load_units
from pruefstand.io import read_jsonl, write_jsonl
from pruefstand.questions.assemble import assemble
from pruefstand.questions.review import export_review, import_review

questions_app = typer.Typer(no_args_is_help=True, help="Build, check and review the question set.")
BATCH_DIR = Path("data/questions/batches")
PROTOCOL = Path("docs/question-authoring.md")
REVIEW_CSV = Path("data/questions/review/k1_v1_review.csv")


def _units_by_id(cfg) -> dict:
    return {unit.unit_id: unit for unit in load_units(cfg.path(cfg["corpus"]["units_path"]))}


def _fail(errors: list[str]) -> None:
    for error in errors:
        typer.echo(f"  - {error}", err=True)
    typer.echo(f"{len(errors)} problem(s)", err=True)
    raise typer.Exit(code=1)


@questions_app.command("check")
def check(
    batch: Annotated[
        list[Path] | None, typer.Argument(help="Batch files; default: all in data/questions/batches")
    ] = None,
    targets: Annotated[bool, typer.Option(help="Also check the per-law, per-type targets")] = False,
    profile: ProfileOption = "full",
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Validate authored batch files (fields, grounding, filters) without writing anything."""
    cfg = load(profile, config)
    paths = batch or sorted(cfg.path(BATCH_DIR).glob("*.jsonl"))
    questions, errors = assemble(
        paths, _units_by_id(cfg), "check", "check", cfg.path(PROTOCOL), check_targets=targets
    )
    if errors:
        _fail(errors)
    typer.echo(f"ok: {len(questions)} records in {len(paths)} file(s)")


@questions_app.command("assemble")
def assemble_cmd(
    author_model: Annotated[
        str, typer.Option(help="Exact model id of the authoring agent, e.g. claude-sonnet-5-5")
    ],
    created_at: Annotated[str, typer.Option(help="Build date, ISO format")] = "",
    profile: ProfileOption = "full",
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Merge all batches into data/questions/k1_v1.jsonl with ids and provenance (all checks must pass)."""
    cfg = load(profile, config)
    paths = sorted(cfg.path(BATCH_DIR).glob("*.jsonl"))
    stamp = created_at or date.today().isoformat()
    questions, errors = assemble(paths, _units_by_id(cfg), author_model, stamp, cfg.path(PROTOCOL))
    if errors:
        _fail(errors)
    write_jsonl(cfg.path(cfg["questions"]["path"]), questions)
    typer.echo(f"wrote {len(questions)} questions to {cfg['questions']['path']}")


@questions_app.command("review-export")
def review_export(
    n: Annotated[int, typer.Option(help="Number of questions to review")] = 100,
    profile: ProfileOption = "full",
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Write the stratified review sheet (CSV, UTF-8 with BOM for Excel)."""
    cfg = load(profile, config)
    questions = read_jsonl(cfg.path(cfg["questions"]["path"]))
    count = export_review(questions, _units_by_id(cfg), cfg.path(REVIEW_CSV), n, int(cfg["stats"]["seed"]))
    typer.echo(f"wrote {count} rows to {REVIEW_CSV}")


@questions_app.command("review-import")
def review_import(
    csv_path: Annotated[Path, typer.Argument(help="Filled review sheet")] = REVIEW_CSV,
    profile: ProfileOption = "full",
    config: ConfigOption = DEFAULT_CONFIG_PATH,
) -> None:
    """Apply the owner's verdicts (accept / edit / reject) to the question set."""
    cfg = load(profile, config)
    path = cfg.path(cfg["questions"]["path"])
    questions, updated, errors = import_review(cfg.path(csv_path), read_jsonl(path), date.today().isoformat())
    if errors:
        _fail(errors)
    write_jsonl(path, questions)
    typer.echo(f"updated {updated} questions")


def register(app: typer.Typer) -> None:
    app.add_typer(questions_app, name="questions")
