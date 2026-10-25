"""Shared CLI options and error handling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from pruefstand.config import DEFAULT_CONFIG, Config, load_config
from pruefstand.errors import PruefstandError

ProfileOption = Annotated[
    str, typer.Option("--profile", "-p", help="Profile from pruefstand.toml: full or ci.")
]
ConfigOption = Annotated[Path, typer.Option("--config", help="Path to pruefstand.toml.")]


def load(profile: str, config: Path) -> Config:
    """Load the profile; a missing file or unknown profile is a clean exit 1, not a traceback."""
    try:
        return load_config(config, profile)
    except (OSError, ValueError) as exc:
        typer.echo(f"Fehler: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@contextmanager
def guarded() -> Iterator[None]:
    """Turn bench errors (replay miss, cache miss, budget stop, failed check) into exit code 1."""
    try:
        yield
    except PruefstandError as exc:
        typer.echo(f"Fehler: {exc}", err=True)
        raise typer.Exit(code=1) from exc


DEFAULT_CONFIG_PATH = Path(DEFAULT_CONFIG)
