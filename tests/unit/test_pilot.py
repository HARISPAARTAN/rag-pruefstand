"""The pilot profile: a few questions, the full run's models and cassette, the published run's retrieval artefacts."""

import pytest

from pruefstand import pipeline
from pruefstand.config import load_config
from pruefstand.errors import PruefstandError
from pruefstand.generation import selected_questions, source_hits_path
from pruefstand.llm.replay import Cassette


@pytest.fixture
def full(root):
    return load_config(root / "pruefstand.toml", "full")


@pytest.fixture
def pilot(root):
    return load_config(root / "pruefstand.toml", "pilot")


def test_pilot_resolves_to_the_full_cassette_and_the_full_llm_settings(full, pilot):
    assert pilot["llm"]["cassette"] == full["llm"]["cassette"] == "cassettes/full-2026-10-25.jsonl"
    assert pilot.path(pilot["llm"]["cassette"]) == full.path(full["llm"]["cassette"])
    for key in ("generator", "judge", "second_judge"):
        assert pilot["llm"][key] == full["llm"][key]
    assert pilot["budget"] == full["budget"] and pilot["llm"]["costs"] == full["llm"]["costs"]
    assert pilot["questions"] == full["questions"]
    assert pilot.retrieval_config_hash() == full.retrieval_config_hash()


def test_pilot_cassette_is_the_real_one_where_synthetic_entries_stay_invisible(pilot, tmp_path):
    from pruefstand.generation import make_client

    client = make_client(pilot, "generator")
    assert client.cassette.path == pilot.path("cassettes/full-2026-10-25.jsonl")
    assert Cassette(tmp_path / "x.jsonl", allow_synthetic=False).entries == {}


def test_pilot_is_small_and_reads_the_published_run(full, pilot):
    assert pilot["generation"]["max_questions"] == 3
    assert pilot["generation"]["variants"] == ["hyb+rr-C2"]
    assert pilot.run_dir == full.path("runs/pilot-2026-10-25") != full.run_dir
    assert pilot.source_dir == full.run_dir and pilot.reads_foreign_run and not full.reads_foreign_run
    assert source_hits_path(pilot) == full.run_dir / "hits.jsonl.gz"
    chosen = selected_questions(pilot)
    assert len(chosen) == 3 and sum(1 for q in chosen if not q["answerable"]) == 1


@pytest.mark.parametrize(
    "stage",
    [
        pipeline.stage_corpus_build,
        pipeline.stage_chunk,
        pipeline.stage_retrieve,
        pipeline.stage_latency,
        pipeline.stage_metrics,
        pipeline.stage_report,
    ],
)
def test_pilot_cannot_write_retrieval_artefacts(pilot, stage):
    with pytest.raises(PruefstandError, match="only reads those of run full-2026-10-25"):
        stage(pilot)


def test_judge_without_answers_is_a_clean_error(pilot):
    from typer.testing import CliRunner

    from pruefstand.cli import app

    result = CliRunner().invoke(app, ["judge", "--profile", "pilot"])
    assert result.exit_code == 1 and "run `pruefstand generate` first" in result.output
    assert not pilot.run_dir.exists()
