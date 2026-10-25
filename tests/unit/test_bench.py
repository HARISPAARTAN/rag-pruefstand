"""Offline end-to-end test of the retrieval stages with a deterministic fake model hub (synthetic vectors)."""

import copy
import hashlib
import re
from dataclasses import asdict

import numpy as np
import pytest
from typer.testing import CliRunner

from pruefstand.aggregate import compute_retrieval_metrics
from pruefstand.bench import ModelHub, hits_path, run_retrieval
from pruefstand.cli import app
from pruefstand.config import Config, load_config
from pruefstand.corpus.gii import load_units
from pruefstand.io import read_json, read_jsonl, write_jsonl
from pruefstand.latency import run_latency
from pruefstand.pipeline import stage_chunk

DIM = 64


def hashed_vector(text: str) -> np.ndarray:
    """Synthetic bag-of-words embedding: hashed tokens, L2-normalised."""
    vector = np.zeros(DIM, dtype=np.float32)
    for token in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class FakeEncoder:
    def encode(self, texts):
        return np.stack([hashed_vector(text) for text in texts])


class FakeHub(ModelHub):
    """Same interface as ModelHub, no torch: synthetic encoders and a word-overlap 'cross-encoder'."""

    def encoder(self, name):
        return FakeEncoder()

    def encode_fn(self, name):
        return FakeEncoder().encode

    def score(self, pairs):
        out = []
        for query, passage in pairs:
            q, p = set(re.findall(r"\w+", query.lower())), set(re.findall(r"\w+", passage.lower()))
            out.append(len(q & p) / (len(q) or 1))
        return out


@pytest.fixture
def cfg(root, tmp_path) -> Config:
    base = load_config(root / "pruefstand.toml", "ci")
    data = copy.deepcopy(base.data)
    data["paths"]["runs_dir"] = str(tmp_path / "runs")
    data["corpus"]["units_path"] = str(tmp_path / "units.jsonl.gz")
    data["retrieval"]["cache_dir"] = str(tmp_path / "cache")
    data["retrieval"]["allow_compute"] = True
    data["questions"]["path"] = str(tmp_path / "questions.jsonl")
    data["latency"]["sample"] = 3
    config = Config(data=data, profile="ci", root=root)
    # The committed corpus restricted to one law stands in for a freshly built CI corpus.
    messbg = [u for u in load_units(root / "data" / "corpus" / "k1_units.jsonl.gz") if u.law_slug == "messbg"]
    write_jsonl(data["corpus"]["units_path"], (asdict(u) for u in messbg))
    stage_chunk(config)
    units = [u for u in messbg if len(u.text) > 120]
    questions = [
        {
            "qid": f"q{i:02d}",
            "type": "factual" if i % 2 else "multi_hop",
            "answerable": True,
            "question_de": " ".join(unit.text.split()[:12]),
            "gold_unit_ids": [unit.unit_id],
            "review": {"status": "accept"},
        }
        for i, unit in enumerate(units[:6])
    ]
    questions.append(
        {
            "qid": "q99",
            "type": "unanswerable",
            "answerable": False,
            "question_de": "Wie hoch ist die Hundesteuer?",
            "gold_unit_ids": [],
        }
    )
    write_jsonl(data["questions"]["path"], questions)
    return config


def test_retrieval_metrics_latency_end_to_end(cfg):
    hub = FakeHub(cfg)
    n_rows = run_retrieval(cfg, hub)
    assert n_rows == 7 * 12
    rows = read_jsonl(hits_path(cfg))
    assert {row["config_id"] for row in rows} >= {"bm25-C1", "hyb+rr-C2", "mini-C2"}
    assert all(len(row["ranked"]) <= 10 for row in rows)

    result = compute_retrieval_metrics(cfg)
    assert result["n_questions"] == 7 and result["n_answerable"] == 6 and result["n_checked"] == 6
    assert result["corpus"]["n_chunks_C1"] > 0 and result["corpus"]["n_chunks_C2"] > 0
    best = result["variants"]["bm25-C2"]["metrics"]["recall@5"]
    assert best["n"] == 6 and best["mean"] > 0.5
    assert len(result["comparisons"]) == 16
    assert result["headline"]["best"]["config_id"] == "hyb+rr-C2"
    assert (cfg.run_dir / "metrics.json").exists()

    latency = run_latency(cfg, hub)
    assert latency["n_questions"] == 3
    assert set(latency["configs"]["hyb+rr-C2"]) == {"embed", "search", "rerank", "total"}
    assert read_json(cfg.run_dir / "latency.json")["rerank_depth"] == 20


def test_retrieval_is_deterministic_and_cached(cfg):
    run_retrieval(cfg, FakeHub(cfg))
    first = hits_path(cfg).read_bytes()
    cfg.data["retrieval"]["allow_compute"] = False
    run_retrieval(cfg, FakeHub(cfg))
    assert hits_path(cfg).read_bytes() == first


def test_cached_run_without_cache_is_a_clean_cli_error(root, monkeypatch):
    monkeypatch.chdir(root)
    result = CliRunner().invoke(app, ["retrieve", "--profile", "ci"])
    assert result.exit_code == 1
    assert "Fehler" in result.output
    assert "Traceback" not in result.output
