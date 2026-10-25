"""Offline end-to-end test of the retrieval stages with a deterministic fake model hub (synthetic vectors)."""

import copy
import hashlib
import re
from dataclasses import asdict

import numpy as np
import pytest
from typer.testing import CliRunner

from pruefstand.aggregate import compute_retrieval_metrics, per_question_metrics
from pruefstand.bench import ModelHub, hits_path, open_caches, retrieve_chunking, run_retrieval
from pruefstand.chunking import chunks_path, load_chunks
from pruefstand.cli import app
from pruefstand.config import Config, load_config
from pruefstand.corpus.gii import load_units
from pruefstand.errors import PruefstandError
from pruefstand.io import read_json, read_jsonl, write_jsonl
from pruefstand.latency import WARMUP_QUERIES, run_latency
from pruefstand.pipeline import stage_chunk
from pruefstand.retrieval.bm25 import BM25Index
from pruefstand.retrieval.dense import DenseIndex
from pruefstand.retrieval.fusion import rrf
from pruefstand.stats import bootstrap_ci

DIMS = {"e5": 64, "mini": 32}  # different sizes: a MiniLM query searched in the e5 index fails loudly


def hashed_vector(text: str, dim: int = 64) -> np.ndarray:
    """Synthetic bag-of-words embedding: hashed tokens, L2-normalised."""
    vector = np.zeros(dim, dtype=np.float32)
    for token in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % dim] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class FakeEncoder:
    def __init__(self, dim: int, log: list[str]) -> None:
        self.dim = dim
        self.log = log

    def encode(self, texts):
        self.log.extend(texts)
        return np.stack([hashed_vector(text, self.dim) for text in texts])


class FakeHub(ModelHub):
    """Same interface as ModelHub, no torch: synthetic encoders and a word-overlap 'cross-encoder'.

    Records every string it encodes and every pair it scores, so tests can assert what reached the models.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoded: dict[str, list[str]] = {"e5": [], "mini": []}
        self.scored: list[list[tuple[str, str]]] = []

    def encoder(self, name):
        return FakeEncoder(DIMS[name], self.encoded[name])

    def encode_fn(self, name):
        return self.encoder(name).encode

    def score(self, pairs):
        self.scored.append(list(pairs))
        out = []
        for query, passage in pairs:
            q, p = set(re.findall(r"\w+", query.lower())), set(re.findall(r"\w+", passage.lower()))
            out.append(len(q & p) / (len(q) or 1))
        return out

    def token_counts(self, name, texts):
        return [len(re.findall(r"\w+", text)) for text in texts]  # one 'token' per word


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
    assert len(result["comparisons"]) == 13
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


def test_missing_question_set_is_a_clean_cli_error(root, tmp_path):
    """Own config with a question path that never exists, so the result does not depend on repository fixtures."""
    config = tmp_path / "pruefstand.toml"
    text = (root / "pruefstand.toml").read_text(encoding="utf-8")
    extra = '\n[profile.noquestions]\nquestions.path = "does-not-exist.jsonl"\n'
    config.write_text(text + extra, encoding="utf-8")
    result = CliRunner().invoke(app, ["retrieve", "--profile", "noquestions", "--config", str(config)])
    assert result.exit_code == 1
    assert "Fehler" in result.output and "does-not-exist.jsonl" in result.output
    assert "Traceback" not in result.output


def test_cache_miss_without_compute_is_a_pruefstand_error(cfg):
    cfg.data["retrieval"]["allow_compute"] = False
    with pytest.raises(PruefstandError):
        run_retrieval(cfg, FakeHub(cfg))


def chunks_and_questions(cfg, chunking):
    units = load_units(cfg.path(cfg["corpus"]["units_path"]))
    lengths = {unit.unit_id: len(unit.text) for unit in units}
    chunks = load_chunks(chunks_path(cfg, chunking))
    return chunks, read_jsonl(cfg.path(cfg["questions"]["path"])), lengths


def test_models_receive_the_configured_prefixes(cfg):
    hub = FakeHub(cfg)
    run_retrieval(cfg, hub)
    models = cfg["models"]
    questions = [q["question_de"] for q in read_jsonl(cfg.path(cfg["questions"]["path"]))]
    chunk_texts = {c.text for chunking in ("C1", "C2") for c in load_chunks(chunks_path(cfg, chunking))}
    assert models["e5"]["query_prefix"] == "query: " and models["e5"]["passage_prefix"] == "passage: "
    assert set(hub.encoded["e5"]) == {"query: " + q for q in questions} | {
        "passage: " + t for t in chunk_texts
    }
    assert set(hub.encoded["mini"]) == set(questions) | chunk_texts  # no prefix at all for the control
    assert not any(text.startswith(("query: ", "passage: ")) for text in hub.encoded["mini"])


def test_rrf_k_from_config_reaches_the_fusion(cfg):
    tops = {}
    for rrf_k in (60, 7):
        cfg.data["retrieval"]["rrf_k"] = rrf_k
        run_retrieval(cfg, FakeHub(cfg))
        rows = read_jsonl(hits_path(cfg))
        by_id = {(r["config_id"], r["qid"]): r for r in rows}
        agreed = 0
        for (config_id, qid), row in by_id.items():
            if config_id != "hyb-C2":
                continue
            bm25, e5 = by_id[("bm25-C2", qid)], by_id[("e5-C2", qid)]
            if bm25["ranked"][0]["chunk_id"] == e5["ranked"][0]["chunk_id"] == row["ranked"][0]["chunk_id"]:
                # first in both lists: score = 2 / (k + 1)
                assert row["ranked"][0]["score"] == round(2 / (rrf_k + 1), 9)
                agreed += 1
        assert agreed >= 3
        tops[rrf_k] = [r["ranked"][0]["score"] for r in rows if r["config_id"] == "hyb-C2"]
    assert tops[60] != tops[7]


def test_rerank_pool_is_the_top_candidates_of_both_bases(cfg):
    hub = FakeHub(cfg)
    chunks, questions, lengths = chunks_and_questions(cfg, "C2")
    rows = retrieve_chunking(cfg, "C2", chunks, questions, lengths, open_caches(cfg), hub)
    retrieval = cfg["retrieval"]
    depth, rerank_depth = retrieval["candidate_depth"], retrieval["rerank_depth"]
    assert rerank_depth == 20
    texts = [chunk.text for chunk in chunks]
    scored: dict[str, set[str]] = {}
    for call in hub.scored:
        for query, passage in call:
            scored.setdefault(query, set()).add(passage)
    bm25 = BM25Index(texts)
    dense = DenseIndex(np.stack([hashed_vector("passage: " + text) for text in texts]))
    for question in questions:
        query = question["question_de"]
        lexical = bm25.search(query, depth)
        semantic = dense.search(hashed_vector("query: " + query), depth)
        fused = rrf([[d for d, _ in lexical], [d for d, _ in semantic]], k=retrieval["rrf_k"], depth=depth)
        pool = {texts[d] for d, _ in semantic[:rerank_depth]} | {texts[d] for d, _ in fused[:rerank_depth]}
        assert scored[query] == pool
        assert len(pool) >= rerank_depth
    # reranked lists are reorderings of that pool, best score first
    text_of = {chunk.chunk_id: chunk.text for chunk in chunks}
    query_of = {q["qid"]: q["question_de"] for q in questions}
    for row in rows:
        if row["config_id"] in ("e5+rr-C2", "hyb+rr-C2"):
            query = query_of[row["qid"]]
            assert all(text_of[hit["chunk_id"]] in scored[query] for hit in row["ranked"])
            values = [hit["score"] for hit in row["ranked"]]
            assert values == sorted(values, reverse=True)


def test_hits_carry_ideal_gains_and_token_counts(cfg):
    run_retrieval(cfg, FakeHub(cfg))
    rows = read_jsonl(hits_path(cfg))
    for row in rows:
        if row["config_id"].endswith("C2") and row["answerable"]:
            assert row["ideal_gains"] == [1]
        if not row["answerable"]:
            assert row["ideal_gains"] == []
        assert all(hit["n_tokens"] > 0 for hit in row["ranked"])
    questions = {q["qid"]: q for q in read_jsonl(cfg.path(cfg["questions"]["path"]))}
    for per_question in per_question_metrics(rows, questions).values():
        assert all(0.0 <= values["ndcg@10"] <= 1.0 for values in per_question.values())


def test_metrics_record_notes_tokens_and_truncation(cfg):
    cfg.data["models"]["mini"]["max_seq_length"] = 40  # 'tokens' are words in the fake hub
    run_retrieval(cfg, FakeHub(cfg))
    result = compute_retrieval_metrics(cfg)
    note = result["metric_notes"]["text"]
    assert "cross-chunking comparison metrics" in note and "within one chunking" in note
    assert result["metric_notes"]["within_chunking_metrics"] == ["ndcg@10"]
    for chunking in ("C1", "C2"):
        chunks = load_chunks(chunks_path(cfg, chunking))
        counts = [len(re.findall(r"\w+", chunk.text)) for chunk in chunks]
        mini = result["truncation"][chunking]["mini"]
        assert mini["max_seq_length"] == 40 and mini["n_chunks"] == len(chunks)
        assert mini["n_truncated"] == sum(c > 40 for c in counts)
        assert mini["truncation_rate"] == round(sum(c > 40 for c in counts) / len(counts), 4)
        assert result["truncation"][chunking]["e5"]["max_seq_length"] == 512
    rows = read_jsonl(hits_path(cfg))
    for config_id, variant in result["variants"].items():
        tokens = variant["mean_retrieved_tokens_at_k"]
        assert list(tokens) == ["1", "3", "5", "10"]
        assert tokens["1"] <= tokens["3"] <= tokens["5"] <= tokens["10"]
        top1 = [r["ranked"][0]["n_tokens"] for r in rows if r["config_id"] == config_id and r["answerable"]]
        assert tokens["1"] == round(sum(top1) / len(top1), 2)
    for comparison in result["comparisons"]:
        cross = comparison["a"].rsplit("-", 1)[1] != comparison["b"].rsplit("-", 1)[1]
        assert comparison["scope"] == ("cross_chunking" if cross else "within_chunking")
        assert comparison["valid"] is True
        assert not (cross and comparison["metric"] == "ndcg@10")
    assert sum(c["scope"] == "cross_chunking" for c in result["comparisons"]) == 3


def test_metrics_use_the_configured_bootstrap_size_and_seed(cfg):
    run_retrieval(cfg, FakeHub(cfg))
    questions = {q["qid"]: q for q in read_jsonl(cfg.path(cfg["questions"]["path"]))}
    table = per_question_metrics(read_jsonl(hits_path(cfg)), questions)
    qids = sorted(qid for qid, q in questions.items() if q["answerable"])
    for seed, n_boot in ((20261025, 2000), (5, 300)):
        cfg.data["stats"]["seed"], cfg.data["stats"]["bootstrap"] = seed, n_boot
        result = compute_retrieval_metrics(cfg)
        values = [table["e5-C2"][q]["ndcg@10"] for q in qids]
        assert result["variants"]["e5-C2"]["metrics"]["ndcg@10"] == bootstrap_ci(values, n_boot, seed)
        assert result["seed"] == seed and result["n_bootstrap"] == n_boot


def test_unknown_gold_unit_is_rejected(cfg):
    questions = read_jsonl(cfg.path(cfg["questions"]["path"]))
    questions[0]["gold_unit_ids"] = ["messbg:\u00a7999:abs0"]
    write_jsonl(cfg.path(cfg["questions"]["path"]), questions)
    with pytest.raises(PruefstandError, match="not in the corpus"):
        run_retrieval(cfg, FakeHub(cfg))


def test_latency_warms_up_the_full_path_untimed(cfg):
    run_retrieval(cfg, FakeHub(cfg))
    fresh = FakeHub(cfg)
    result = run_latency(cfg, fresh)
    n_chunkings = len(cfg["retrieval"]["chunkings"])
    # two rerank calls (e5 pool, hybrid pool) per query, for the warm-up queries and the measured ones
    assert len(fresh.scored) == n_chunkings * 2 * (WARMUP_QUERIES + result["n_questions"])
    assert all(len(call) == cfg["retrieval"]["rerank_depth"] for call in fresh.scored)


@pytest.mark.parametrize(
    ("model", "key", "value"),
    [
        ("mini", "max_seq_length", 256),
        ("e5", "max_seq_length", 256),
        ("reranker", "max_length", 128),
        ("e5", "revision", "other"),
        ("mini", "revision", "other"),
        ("reranker", "revision", "other"),
    ],
)
def test_changed_model_settings_invalidate_the_cached_scores(cfg, model, key, value):
    run_retrieval(cfg, FakeHub(cfg))
    cfg.data["retrieval"]["allow_compute"] = False
    run_retrieval(cfg, FakeHub(cfg))  # unchanged settings: served from the cache
    cfg.data["models"][model][key] = value
    with pytest.raises(PruefstandError):
        run_retrieval(cfg, FakeHub(cfg))
