"""Online latency per stage on a fixed question sample, measured live with caches bypassed (CPU only)."""

from __future__ import annotations

import platform
import time

from pruefstand.bench import ModelHub, build_indexes, load_questions, open_caches
from pruefstand.chunking import chunks_path, load_chunks
from pruefstand.config import Config
from pruefstand.io import write_json
from pruefstand.retrieval.fusion import rrf
from pruefstand.stats import percentile

STAGES = ("embed", "search", "rerank", "total")
WARMUP_QUERIES = 3  # full-path queries run untimed per chunking before the measured ones


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def summarise(samples: dict[str, list[float]]) -> dict[str, dict[str, float | None]]:
    return {
        stage: {"p50": percentile(values, 50), "p95": percentile(values, 95)}
        for stage, values in samples.items()
    }


def _measure(
    hub: ModelHub, cfg: Config, indexes: dict, texts: list[str], query: str
) -> dict[str, tuple[float, float, float]]:
    """One query through every stage, timed live (caches bypassed); returns (embed, search, rerank) in ms."""
    retrieval, models = cfg["retrieval"], cfg["models"]
    depth, top, rerank_depth = retrieval["candidate_depth"], retrieval["top_k"], retrieval["rerank_depth"]
    start = time.perf_counter()
    bm25 = indexes["bm25"].search(query, depth)
    t_bm25 = _ms(start)
    start = time.perf_counter()
    e5_vector = hub.encoder("e5").encode([models["e5"]["query_prefix"] + query])[0]
    t_e5_embed = _ms(start)
    start = time.perf_counter()
    e5 = indexes["e5"].search(e5_vector, depth)
    t_e5_search = _ms(start)
    start = time.perf_counter()
    mini_vector = hub.encoder("mini").encode([models["mini"]["query_prefix"] + query])[0]
    t_mini_embed = _ms(start)
    start = time.perf_counter()
    indexes["mini"].search(mini_vector, top)
    t_mini_search = _ms(start)
    start = time.perf_counter()
    hyb = rrf([[d for d, _ in bm25], [d for d, _ in e5]], k=retrieval["rrf_k"], depth=depth)
    t_fuse = _ms(start)
    start = time.perf_counter()
    hub.score([(query, texts[d]) for d, _ in e5[:rerank_depth]])
    t_rr_e5 = _ms(start)
    start = time.perf_counter()
    hub.score([(query, texts[d]) for d, _ in hyb[:rerank_depth]])
    t_rr_hyb = _ms(start)
    rows = {
        "bm25": (0.0, t_bm25, 0.0),
        "e5": (t_e5_embed, t_e5_search, 0.0),
        "mini": (t_mini_embed, t_mini_search, 0.0),
        "hyb": (t_e5_embed, t_bm25 + t_e5_search + t_fuse, 0.0),
        "e5+rr": (t_e5_embed, t_e5_search, t_rr_e5),
        "hyb+rr": (t_e5_embed, t_bm25 + t_e5_search + t_fuse, t_rr_hyb),
    }
    return rows


def run_latency(cfg: Config, hub: ModelHub | None = None) -> dict:
    sample = int(cfg["latency"]["sample"])
    if sample <= 0:
        return {}
    hub = hub or ModelHub(cfg)
    retrieval = cfg["retrieval"]
    rerank_depth = retrieval["rerank_depth"]
    questions = sorted((q for q in load_questions(cfg) if q["answerable"]), key=lambda q: q["qid"])[:sample]
    warmup = [questions[i % len(questions)] for i in range(WARMUP_QUERIES)]
    caches = open_caches(cfg)
    configs: dict[str, dict] = {}
    for chunking in retrieval["chunkings"]:
        indexes = build_indexes(cfg, load_chunks(chunks_path(cfg, chunking)), caches, hub)
        texts = indexes["texts"]
        samples = {
            name: {stage: [] for stage in STAGES} for name in ("bm25", "e5", "mini", "hyb", "e5+rr", "hyb+rr")
        }
        for query in (q["question_de"] for q in warmup):
            _measure(
                hub, cfg, indexes, texts, query
            )  # untimed: model load, first batch shapes, search code paths
        for question in questions:
            rows = _measure(hub, cfg, indexes, texts, question["question_de"])
            for name, (embed, search, rerank) in rows.items():
                bucket = samples[name]
                bucket["embed"].append(embed)
                bucket["search"].append(search)
                bucket["rerank"].append(rerank)
                bucket["total"].append(embed + search + rerank)
        for name, bucket in samples.items():
            configs[f"{name}-{chunking}"] = summarise(bucket)
    result = {
        "n_questions": len(questions),
        "rerank_depth": rerank_depth,
        "machine": {
            "cpu": platform.processor(),
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "configs": configs,
    }
    write_json(cfg.run_dir / "latency.json", result)
    return result
