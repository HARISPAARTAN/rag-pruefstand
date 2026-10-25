"""End-to-end run of the CI profile: offline, no key, no models, compared with committed golden files.

Refresh the golden files only after an intended change, in the same commit as the change:
    PRUEFSTAND_UPDATE_GOLDEN=1 uv run pytest tests/integration
"""

import os
import shutil

import pytest
from typer.testing import CliRunner

from pruefstand.cli import app
from pruefstand.config import load_config
from pruefstand.io import read_json, write_json

PROFILE = "ci"
GOLDEN = {"metrics.json": "ci_metrics.json", "answer_metrics.json": "ci_answer_metrics.json"}


def updating() -> bool:
    return os.environ.get("PRUEFSTAND_UPDATE_GOLDEN") == "1"


@pytest.mark.integration
def test_ci_profile_runs_offline_and_matches_golden(root, monkeypatch, no_network):
    monkeypatch.chdir(root)
    monkeypatch.setenv("PRUEFSTAND_LLM_MODE", "replay")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = load_config(root / "pruefstand.toml", PROFILE)
    shutil.rmtree(cfg.run_dir, ignore_errors=True)
    result = CliRunner().invoke(app, ["all", "--profile", PROFILE])
    assert result.exit_code == 0, result.output
    for produced, golden in GOLDEN.items():
        data = read_json(cfg.run_dir / produced)
        golden_path = root / "tests" / "golden" / golden
        if updating():
            write_json(golden_path, data)
        assert data == read_json(golden_path), f"{produced} differs from {golden_path.name}"
    assert read_json(cfg.run_dir / "manifest.json")["stages"][-1] == "answer metrics"


@pytest.mark.integration
def test_ci_retrieval_is_byte_stable(root, monkeypatch, no_network):
    monkeypatch.chdir(root)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = load_config(root / "pruefstand.toml", PROFILE)
    runner = CliRunner()
    assert runner.invoke(app, ["corpus", "build", "--profile", PROFILE]).exit_code == 0
    assert runner.invoke(app, ["chunk", "--profile", PROFILE]).exit_code == 0
    first = runner.invoke(app, ["retrieve", "--profile", PROFILE])
    assert first.exit_code == 0, first.output
    hits = (cfg.run_dir / "hits.jsonl.gz").read_bytes()
    assert runner.invoke(app, ["retrieve", "--profile", PROFILE]).exit_code == 0
    assert (cfg.run_dir / "hits.jsonl.gz").read_bytes() == hits
