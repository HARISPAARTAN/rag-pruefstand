"""Command line: `uv run pruefstand --help`."""

from __future__ import annotations

import typer

from pruefstand import __version__
from pruefstand.commands import corpus
from pruefstand.commands.common import DEFAULT_CONFIG_PATH, ConfigOption, ProfileOption, load

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="RAG-Prüfstand: German RAG evaluation bench."
)


@app.callback()
def main() -> None:
    """RAG-Prüfstand: German RAG evaluation bench (keeps the CLI a command group even with one command)."""


@app.command()
def info(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Show version, profile, run id and the configuration hash."""
    cfg = load(profile, config)
    typer.echo(
        f"pruefstand {__version__} | profile {cfg.profile} | run {cfg.run_id} | config {cfg.config_hash()[:12]}"
    )


for module in (corpus,):
    module.register(app)
