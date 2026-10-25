"""Retrieval bench: six retrievers x two chunkings, top-k hits with span relevance per question."""

from __future__ import annotations

from pathlib import Path

from pruefstand.chunking import Chunk, chunks_path, load_chunks
from pruefstand.config import Config
from pruefstand.corpus.gii import load_units
from pruefstand.errors import PruefstandError
from pruefstand.io import read_jsonl, write_jsonl
from pruefstand.metrics import gold_units_covered
from pruefstand.retrieval.bm25 import BM25Index
from pruefstand.retrieval.cache import EmbeddingCache, RerankCache
from pruefstand.retrieval.dense import DenseIndex
from pruefstand.retrieval.fusion import rrf
from pruefstand.retrieval.rerank import rerank_order


def hits_path(cfg: Config) -> Path:
    return cfg.run_dir / "hits.jsonl.gz"


def config_ids(cfg: Config) -> list[str]:
    retrieval = cfg["retrieval"]
    return [
        f"{retriever}-{chunking}"
        for chunking in retrieval["chunkings"]
        for retriever in retrieval["retrievers"]
    ]


def load_questions(cfg: Config) -> list[dict]:
    path = cfg.path(cfg["questions"]["path"])
    if not path.is_file():
        raise PruefstandError(f"{path} not found; the question set must exist before retrieval can run")
    questions = read_jsonl(path)
    if not questions:
        raise PruefstandError(f"no questions in {cfg['questions']['path']}")
    return questions


def _sentence_encoder(cfg: Config, model: dict):
    from pruefstand.retrieval.dense import SentenceEncoder

    return SentenceEncoder(cfg.path(model["dir"]), model["max_seq_length"], cfg["retrieval"]["batch_size"])


def _cross_encoder(cfg: Config, model: dict):
    from pruefstand.retrieval.rerank import CrossEncoderScorer

    return CrossEncoderScorer(cfg.path(model["dir"]), model["max_length"], cfg["retrieval"]["batch_size"])


def _load_model(factory, model_dir: Path):
    """Build a model wrapper; a missing download or missing models extra becomes a clean PruefstandError."""
    try:
        return factory()
    except FileNotFoundError as exc:
        raise PruefstandError(
            f"{model_dir} not found; run `uv run --extra models pruefstand models fetch`"
        ) from exc
    except ImportError as exc:
        raise PruefstandError(
            "computing embeddings needs the models extra: run with `uv run --extra models ...`"
        ) from exc


class ModelHub:
    """Creates encoders and the cross-encoder on first use, so cached runs never import torch."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._encoders: dict = {}
        self._scorer = None

    def encoder(self, name: str):
        if name not in self._encoders:
            model = self.cfg["models"][name]
            self._encoders[name] = _load_model(
                lambda: _sentence_encoder(self.cfg, model), self.cfg.path(model["dir"])
            )
        return self._encoders[name]

    def encode_fn(self, name: str):
        return lambda texts: self.encoder(name).encode(texts)

    def scorer(self):
        if self._scorer is None:
            model = self.cfg["models"]["reranker"]
            self._scorer = _load_model(lambda: _cross_encoder(self.cfg, model), self.cfg.path(model["dir"]))
        return self._scorer

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self.scorer().score(pairs)


def open_caches(cfg: Config) -> dict:
    retrieval = cfg["retrieval"]
    cache_dir = cfg.path(retrieval["cache_dir"])
    allow = bool(retrieval["allow_compute"])
    models = cfg["models"]
    return {
        "e5": EmbeddingCache(cache_dir, "e5", models["e5"]["id"], models["e5"]["revision"], allow),
        "mini": EmbeddingCache(cache_dir, "mini", models["mini"]["id"], models["mini"]["revision"], allow),
        "reranker": RerankCache(
            cache_dir, "reranker", models["reranker"]["id"], models["reranker"]["revision"], allow
        ),
    }


def build_indexes(cfg: Config, chunks: list[Chunk], caches: dict, hub: ModelHub) -> dict:
    texts = [chunk.text for chunk in chunks]
    tenants = [chunk.tenant for chunk in chunks]
    models = cfg["models"]
    e5_vectors = caches["e5"].get_many(
        [models["e5"]["passage_prefix"] + text for text in texts], hub.encode_fn("e5")
    )
    mini_vectors = caches["mini"].get_many(
        [models["mini"]["passage_prefix"] + text for text in texts], hub.encode_fn("mini")
    )
    return {
        "texts": texts,
        "bm25": BM25Index(texts, tenants),
        "e5": DenseIndex(e5_vectors, tenants),
        "mini": DenseIndex(mini_vectors, tenants),
    }


def retrieve_chunking(
    cfg: Config,
    chunking: str,
    chunks: list[Chunk],
    questions: list[dict],
    unit_lengths: dict[str, int],
    caches: dict,
    hub: ModelHub,
) -> list[dict]:
    retrieval = cfg["retrieval"]
    models = cfg["models"]
    depth, top, rerank_depth = retrieval["candidate_depth"], retrieval["top_k"], retrieval["rerank_depth"]
    indexes = build_indexes(cfg, chunks, caches, hub)
    texts = indexes["texts"]
    query_texts = [question["question_de"] for question in questions]
    e5_queries = caches["e5"].get_many(
        [models["e5"]["query_prefix"] + q for q in query_texts], hub.encode_fn("e5")
    )
    mini_queries = caches["mini"].get_many(
        [models["mini"]["query_prefix"] + q for q in query_texts], hub.encode_fn("mini")
    )

    candidates: list[dict[str, list[tuple[int, float]]]] = []
    for i, query in enumerate(query_texts):
        bm25 = indexes["bm25"].search(query, depth)
        e5 = indexes["e5"].search(e5_queries[i], depth)
        mini = indexes["mini"].search(mini_queries[i], top)
        hyb = rrf([[doc for doc, _ in bm25], [doc for doc, _ in e5]], k=retrieval["rrf_k"], depth=depth)
        candidates.append({"bm25": bm25, "e5": e5, "mini": mini, "hyb": hyb})

    pairs: list[tuple[str, str]] = []
    for i, query in enumerate(query_texts):
        for base in ("e5", "hyb"):
            pairs.extend((query, texts[doc]) for doc, _ in candidates[i][base][:rerank_depth])
    scores = caches["reranker"].get_many(pairs, hub.score)
    position = 0
    for i in range(len(query_texts)):
        for base in ("e5", "hyb"):
            docs = [doc for doc, _ in candidates[i][base][:rerank_depth]]
            candidates[i][f"{base}+rr"] = rerank_order(docs, scores[position : position + len(docs)])
            position += len(docs)

    rows = []
    for i, question in enumerate(questions):
        gold = question["gold_unit_ids"]
        for retriever in retrieval["retrievers"]:
            ranked = candidates[i][retriever][:top]
            rows.append(
                {
                    "config_id": f"{retriever}-{chunking}",
                    "qid": question["qid"],
                    "type": question["type"],
                    "answerable": question["answerable"],
                    "ranked": [
                        {
                            "rank": rank,
                            "chunk_id": chunks[doc].chunk_id,
                            "score": score,
                            "gold_covered": gold_units_covered(chunks[doc].covers, gold, unit_lengths),
                        }
                        for rank, (doc, score) in enumerate(ranked, start=1)
                    ],
                }
            )
    return rows


def run_retrieval(cfg: Config, hub: ModelHub | None = None) -> int:
    """Run all variants, write hits.jsonl.gz, prune and save the caches; returns the number of hit rows."""
    hub = hub or ModelHub(cfg)
    questions = load_questions(cfg)
    units = load_units(cfg.path(cfg["corpus"]["units_path"]))
    unit_lengths = {unit.unit_id: len(unit.text) for unit in units}
    caches = open_caches(cfg)
    rows: list[dict] = []
    for chunking in cfg["retrieval"]["chunkings"]:
        chunks = load_chunks(chunks_path(cfg, chunking))
        rows.extend(retrieve_chunking(cfg, chunking, chunks, questions, unit_lengths, caches, hub))
    for cache in caches.values():
        cache.prune_unused()
        cache.save()
    return write_jsonl(hits_path(cfg), rows)
