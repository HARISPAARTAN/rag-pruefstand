"""The reproduction path: the verify profile never writes into published files, `verify` compares by content."""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pruefstand import pipeline
from pruefstand.cli import app
from pruefstand.config import load_config
from pruefstand.errors import PruefstandError
from pruefstand.io import read_json, read_jsonl, write_json, write_jsonl
from pruefstand.verify import compare_runs, diff_json

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "runs" / "full-2026-10-25"


@pytest.fixture
def copy(tmp_path):
    for name in ("hits.jsonl.gz", "metrics.json"):
        shutil.copy(PUBLISHED / name, tmp_path / name)
    return tmp_path


def test_a_faithful_reproduction_has_no_differences_even_with_another_run_id(copy):
    metrics = read_json(copy / "metrics.json")
    metrics["run_id"] = "verify-scratch"
    write_json(copy / "metrics.json", metrics)
    write_jsonl(copy / "hits.jsonl.gz", read_jsonl(PUBLISHED / "hits.jsonl.gz"))  # other gzip bytes allowed
    assert compare_runs(PUBLISHED, copy) == []


def test_a_changed_metric_or_hit_is_reported(copy):
    metrics = read_json(copy / "metrics.json")
    metrics["variants"]["bm25-C2"]["metrics"]["recall@5"]["mean"] += 0.001
    write_json(copy / "metrics.json", metrics)
    rows = read_jsonl(copy / "hits.jsonl.gz")
    rows[0]["ranked"] = rows[0]["ranked"][::-1]
    write_jsonl(copy / "hits.jsonl.gz", rows)
    differences = compare_runs(PUBLISHED, copy)
    assert any(d.startswith("hits.jsonl.gz: 1 of") for d in differences)
    assert any("variants.bm25-C2.metrics.recall@5.mean" in d for d in differences)


def test_a_run_is_never_compared_with_itself_or_with_a_missing_run(tmp_path):
    with pytest.raises(PruefstandError, match="itself"):
        compare_runs(PUBLISHED, PUBLISHED)
    with pytest.raises(PruefstandError, match="not found"):
        compare_runs(PUBLISHED, tmp_path)


def test_diff_json_handles_missing_keys_and_lists():
    assert diff_json({"run_id": "a", "x": [1, 2]}, {"run_id": "b", "x": [1, 3], "y": 1}) == [
        "x[1]: published 2 vs reproduction 3",
        "y: only in reproduction",
    ]


def test_verify_profile_has_its_own_folders_and_cannot_rebuild_the_corpus():
    full = load_config(ROOT / "pruefstand.toml", "full")
    verify = load_config(ROOT / "pruefstand.toml", "verify")
    assert verify.run_dir == ROOT / "runs" / "verify-scratch" != full.run_dir
    for key in ("reports_dir", "site_dir"):
        assert str(verify.path(verify["paths"][key])).startswith(str(verify.run_dir))
    assert verify["corpus"] == full["corpus"]
    assert verify.uses_committed_units and not full.uses_committed_units
    assert "corpus build" not in [name for name, _ in pipeline.planned_stages(verify)]
    assert "corpus build" in [name for name, _ in pipeline.planned_stages(full)]
    with pytest.raises(PruefstandError, match="never rebuilds"):
        pipeline.stage_corpus_build(verify)
    assert verify["latency"]["sample"] == 0


def test_verify_command_refuses_to_compare_the_published_run_with_itself():
    result = CliRunner().invoke(app, ["verify", "--profile", "full"])
    assert result.exit_code == 1 and "itself" in result.output
