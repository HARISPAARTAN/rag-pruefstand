from typer.testing import CliRunner

from pruefstand import __version__
from pruefstand.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "info" in result.output


def test_info_prints_version_and_hash(root, monkeypatch):
    monkeypatch.chdir(root)
    result = runner.invoke(app, ["info", "--profile", "ci"])
    assert result.exit_code == 0, result.output
    assert f"pruefstand {__version__}" in result.output
    assert "profile ci" in result.output
    assert "| config " in result.output


def test_unknown_profile_is_a_clean_error(root, monkeypatch):
    monkeypatch.chdir(root)
    result = runner.invoke(app, ["info", "--profile", "nope"])
    assert result.exit_code == 1
    assert "unknown profile" in result.output
