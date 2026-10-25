"""Report command: rebuild reports and site from committed run artefacts (no key, no network)."""

from __future__ import annotations

import typer

from pruefstand.commands.common import DEFAULT_CONFIG_PATH, ConfigOption, ProfileOption, guarded, load
from pruefstand.pipeline import stage_report


def report(profile: ProfileOption = "full", config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Write reports/v1 (Markdown DE/EN, JSON, SVG) and site/ from runs/<run_id>/."""
    cfg = load(profile, config)
    with guarded():
        written = stage_report(cfg)
    for path in written:
        typer.echo(f"wrote {path}")


def register(app: typer.Typer) -> None:
    app.command("report")(report)
