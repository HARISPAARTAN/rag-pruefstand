"""Provenance of paid stages: the published run's configuration, the retrieval hash check and stage manifests."""

import dataclasses

import pytest

from pruefstand import pipeline
from pruefstand.config import load_config
from pruefstand.errors import PruefstandError
from pruefstand.io import read_json, write_json
from pruefstand.manifest import (
    STAGE_MANIFESTS,
    check_retrieval_provenance,
    published_retrieval_hash,
    write_stage_manifest,
)

RUN = "full-2026-10-25"
RUNTIME_HASH = "625b4f08e540"  # configuration hash of the published run (git 2f73d04)
RETRIEVAL_HASH = "67af0ba7a702"  # hash of its retrieval-relevant settings, unchanged since


def with_data(cfg, **sections):
    data = {**cfg.data, **sections}
    return dataclasses.replace(cfg, data=data)


@pytest.fixture
def full(root):
    return load_config(root / "pruefstand.toml", "full")


@pytest.fixture
def runtime(root):
    return load_config(root / "runs" / RUN / "config.toml", "full")


def test_committed_runtime_config_is_the_one_the_manifest_names(root, runtime):
    manifest = read_json(root / "runs" / RUN / "manifest.json")
    assert runtime.config_hash() == manifest["config_sha256"]
    assert runtime.config_hash().startswith(RUNTIME_HASH)
    assert runtime.retrieval_config_hash().startswith(RETRIEVAL_HASH)


def test_only_llm_cost_and_question_metadata_changed_since_the_run(full, runtime):
    assert full.config_hash() != runtime.config_hash()
    changed = {key for key in full.data if full.data[key] != runtime.data.get(key)}
    assert changed == {"cost", "llm", "questions"}
    assert {key for key in full["questions"] if full["questions"][key] != runtime["questions"].get(key)} == {
        "authored",
        "author_model",
    }
    assert full.retrieval_config_hash() == runtime.retrieval_config_hash()


def test_retrieval_hash_ignores_llm_cost_and_paths_but_not_retrieval_settings(full):
    llm = {**full["llm"], "generator": {**full["llm"]["generator"], "max_tokens": 1}}
    assert with_data(full, llm=llm).retrieval_config_hash() == full.retrieval_config_hash()
    assert with_data(full, paths={**full["paths"], "run_id": "x"}).retrieval_config_hash() == (
        full.retrieval_config_hash()
    )
    chunking = {**full["chunking"], "c2_max_tokens": 300}
    assert with_data(full, chunking=chunking).retrieval_config_hash() != full.retrieval_config_hash()
    questions = {**full["questions"], "version": "k1_v2"}
    assert with_data(full, questions=questions).retrieval_config_hash() != full.retrieval_config_hash()


def test_check_accepts_the_current_configuration_and_the_pilot(root, full):
    assert check_retrieval_provenance(full).startswith(RETRIEVAL_HASH)
    pilot = load_config(root / "pruefstand.toml", "pilot")
    assert check_retrieval_provenance(pilot).startswith(RETRIEVAL_HASH)


def test_check_refuses_changed_retrieval_settings_and_names_the_section(full):
    changed = with_data(full, retrieval={**full["retrieval"], "rerank_depth": 10})
    with pytest.raises(PruefstandError, match=r"differs from the published run.*changed sections: retrieval"):
        check_retrieval_provenance(changed)


def make_run(tmp_path, cfg, manifest):
    run_dir = tmp_path / "runs" / "published"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "manifest.json", manifest)
    paths = {**cfg["paths"], "runs_dir": str(tmp_path / "runs"), "run_id": "published"}
    return with_data(cfg, paths=paths), run_dir


def test_manifest_with_its_own_retrieval_hash_needs_no_runtime_config(tmp_path, full):
    cfg, _ = make_run(
        tmp_path,
        full,
        {"profile": "full", "config_sha256": "x", "retrieval_config_sha256": full.retrieval_config_hash()},
    )
    assert published_retrieval_hash(cfg) == (full.retrieval_config_hash(), None)
    assert check_retrieval_provenance(cfg) == full.retrieval_config_hash()


def test_old_manifest_without_runtime_config_cannot_be_checked(tmp_path, full):
    cfg, _ = make_run(tmp_path, full, {"profile": "full", "config_sha256": "x"})
    with pytest.raises(PruefstandError, match="cannot be established"):
        check_retrieval_provenance(cfg)


def test_runtime_config_must_match_the_manifest_hash(tmp_path, root, full):
    cfg, run_dir = make_run(tmp_path, full, {"profile": "full", "config_sha256": "0" * 64})
    (run_dir / "config.toml").write_bytes((root / "pruefstand.toml").read_bytes())
    with pytest.raises(PruefstandError, match="does not belong to"):
        check_retrieval_provenance(cfg)


def test_missing_manifest_refuses_a_paid_stage(tmp_path, full):
    cfg = with_data(full, paths={**full["paths"], "runs_dir": str(tmp_path), "run_id": "none"})
    with pytest.raises(PruefstandError, match="not found"):
        check_retrieval_provenance(cfg)


def test_stage_manifest_records_settings_models_cassette_and_times(tmp_path, full):
    cfg = with_data(full, paths={**full["paths"], "runs_dir": str(tmp_path), "run_id": "r"})
    records = [{"billed_now": True}, {"billed_now": False}]
    entry = write_stage_manifest(cfg, "generate", "2026-10-26T08:00:00+00:00", records, 0.123456789)
    write_stage_manifest(cfg, "judge", "2026-10-26T09:00:00+00:00", None, 0.0)
    saved = read_json(tmp_path / "r" / STAGE_MANIFESTS)
    assert saved["generate"] == [entry] and len(saved["judge"]) == 1
    assert entry["model"] == "claude-haiku-5-5" and saved["judge"][0]["model"] == "claude-sonnet-5-5"
    assert entry["cassette"] == "cassettes/full-2026-10-25.jsonl"
    assert entry["retrieval_config_sha256"] == full.retrieval_config_hash()
    assert entry["config_sha256"] == cfg.config_hash() and len(entry["stage_config_sha256"]) == 64
    assert entry["n_records"] == 2 and entry["n_billed_now"] == 1 and entry["spend_usd"] == 0.123457
    assert entry["started_at"] == "2026-10-26T08:00:00+00:00" and entry["finished_at"]
    assert entry["prompts"] and entry["completed"] is True and saved["judge"][0]["completed"] is False
    assert entry["stage_config_sha256"] != saved["judge"][0]["stage_config_sha256"]


@pytest.fixture
def stub_generation(monkeypatch):
    from pruefstand import generation

    calls = []
    monkeypatch.setattr(generation, "make_client", lambda cfg, role, refresh=False: calls.append(role))
    monkeypatch.setattr(generation, "run_generation", lambda cfg, client, guard: [{"billed_now": True}])
    monkeypatch.setattr(generation, "run_judging", lambda cfg, client, guard: [{"billed_now": False}])
    return calls


def test_record_mode_checks_the_hash_and_writes_a_stage_manifest(
    tmp_path, monkeypatch, full, stub_generation
):
    cfg, run_dir = make_run(
        tmp_path,
        full,
        {"profile": "full", "config_sha256": "x", "retrieval_config_sha256": full.retrieval_config_hash()},
    )
    monkeypatch.setenv("PRUEFSTAND_LLM_MODE", "record")
    assert pipeline.stage_generate(cfg) == [{"billed_now": True}]
    assert pipeline.stage_judge(cfg) == [{"billed_now": False}]
    saved = read_json(run_dir / STAGE_MANIFESTS)
    assert saved["generate"][0]["mode"] == "record" and saved["judge"][0]["n_records"] == 1
    assert read_json(run_dir / "manifest.json")["config_sha256"] == "x"  # the retrieval manifest is untouched


def test_record_mode_refuses_when_the_retrieval_settings_changed(
    tmp_path, monkeypatch, full, stub_generation
):
    cfg, run_dir = make_run(
        tmp_path, full, {"profile": "full", "config_sha256": "x", "retrieval_config_sha256": "0" * 64}
    )
    monkeypatch.setenv("PRUEFSTAND_LLM_MODE", "record")
    with pytest.raises(PruefstandError, match="differs from the published run"):
        pipeline.stage_judge(cfg)
    assert stub_generation == [] and not (run_dir / STAGE_MANIFESTS).exists()


def test_replay_neither_checks_nor_writes(tmp_path, monkeypatch, full, stub_generation):
    cfg = with_data(full, paths={**full["paths"], "runs_dir": str(tmp_path), "run_id": "none"})
    monkeypatch.setenv("PRUEFSTAND_LLM_MODE", "replay")
    assert pipeline.stage_generate(cfg) == [{"billed_now": True}]
    assert not (tmp_path / "none").exists()


def test_a_failing_stage_still_leaves_an_incomplete_entry(tmp_path, monkeypatch, full, stub_generation):
    from pruefstand import generation

    def boom(cfg, client, guard):
        raise PruefstandError("budget stop")

    monkeypatch.setattr(generation, "run_generation", boom)
    cfg, run_dir = make_run(
        tmp_path,
        full,
        {"profile": "full", "config_sha256": "x", "retrieval_config_sha256": full.retrieval_config_hash()},
    )
    monkeypatch.setenv("PRUEFSTAND_LLM_MODE", "record")
    with pytest.raises(PruefstandError, match="budget stop"):
        pipeline.stage_generate(cfg)
    entry = read_json(run_dir / STAGE_MANIFESTS)["generate"][0]
    assert entry["completed"] is False and entry["n_records"] is None
