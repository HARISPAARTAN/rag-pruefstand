"""Retrieval bench: six retrievers x two chunkings, top-k hits with span relevance per question."""

from __future__ import annotations

from pathlib import Path

from pruefstand.chunking import Chunk, chunks_path, load_chunks
from pruefstand.config import Config
from pruefstand.corpus.gii import load_units
from pruefstand.errors import PruefstandError
from pruefstand.io import read_json, read_jsonl, write_json, write_jsonl
from pruefstand.metrics import (
    LONG_UNIT_CHARS,
    MIN_OVERLAP_CHARS,
    MIN_OVERLAP_SHARE,
    RELEVANCE_RULE,
    gold_units_covered,
    ideal_gains,
    passage_spans,
    validate_gold,
)
from pruefstand.retrieval.bm25 import BM25Index
from pruefstand.retrieval.cache import EmbeddingCache, RerankCache
from pruefstand.retrieval.dense import NORMALIZE, DenseIndex
from pruefstand.retrieval.fusion import rrf
from pruefstand.retrieval.rerank import rerank_order


def hits_path(cfg: Config) -> Path:
    return cfg.run_dir / "hits.jsonl.gz"


def meta_path(cfg: Config) -> Path:
    return cfg.run_dir / "retrieval_meta.json"


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

    def token_counts(self, name: str, texts: list) -> list[int] | None:
        """Token counts incl. special tokens from the tokenizer.json next to the model, as the model sees them.

        An item is a text or a (query, passage) pair (cross-encoder input). Returns None when the model is not
        downloaded (offline CI); needs no torch.
        """
        path = self.cfg.path(self.cfg["models"][name]["dir"]) / "tokenizer.json"
        if not path.is_file():
            return None
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_truncation()  # some tokenizer.json files carry a 128-token limit; we want the raw length
        tokenizer.no_padding()
        return [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]


def open_caches(cfg: Config) -> dict:
    retrieval = cfg["retrieval"]
    cache_dir = cfg.path(retrieval["cache_dir"])
    allow = bool(retrieval["allow_compute"])
    models = cfg["models"]

    def embeddings(name: str) -> EmbeddingCache:
        model = models[name]
        return EmbeddingCache(
            cache_dir,
            name,
            model["id"],
            model["revision"],
            allow,
            max_seq_length=model["max_seq_length"],
            normalize=NORMALIZE,
        )

    reranker = models["reranker"]
    return {
        "e5": embeddings("e5"),
        "mini": embeddings("mini"),
        "reranker": RerankCache(
            cache_dir,
            "reranker",
            reranker["id"],
            reranker["revision"],
            allow,
            max_length=reranker["max_length"],
        ),
    }


def truncation_report(cfg: Config, chunks: list[Chunk], hub: ModelHub) -> dict:
    """Per embedding model: how many chunk texts (with passage prefix) exceed max_seq_length and how much is seen."""
    report: dict = {}
    for name in ("e5", "mini"):
        model = cfg["models"][name]
        if not cfg["retrieval"].get("measure_truncation", True):
            report[name] = None  # offline CI: the golden file must not depend on local model files
            continue
        counts = hub.token_counts(name, [model["passage_prefix"] + chunk.text for chunk in chunks])
        if counts is None:
            report[name] = None
            continue
        limit = model["max_seq_length"]
        n = len(counts)
        report[name] = {
            "max_seq_length": limit,
            "n_chunks": n,
            "n_truncated": sum(count > limit for count in counts),
            "truncation_rate": round(sum(count > limit for count in counts) / n, 4) if n else 0.0,
            "mean_tokens": round(sum(counts) / n, 2) if n else 0.0,
            "mean_visible_share": round(sum(min(c, limit) / c for c in counts if c) / n, 4) if n else 0.0,
        }
    return report


RERANKED_RETRIEVERS = ("e5+rr", "hyb+rr")
RERANK_SCOPE = "(query, chunk) pairs of the reranked top-k of e5+rr and hyb+rr, answerable questions"


def reranker_truncation_report(
    cfg: Config, chunks: list[Chunk], rows: list[dict], questions: list[dict], hub: ModelHub
) -> dict | None:
    """How many (query, chunk) pairs the cross-encoder scores exceed its max_length and lose their tail.

    Counts the pairs behind the final top-k of the reranked retrievers, from the hit rows of one chunking.
    """
    if not cfg["retrieval"].get("measure_truncation", True):
        return None
    reranked = [retriever for retriever in RERANKED_RETRIEVERS if retriever in cfg["retrieval"]["retrievers"]]
    if not reranked:
        return None
    text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}
    query_by_qid = {question["qid"]: question["question_de"] for question in questions}
    pairs = [
        (query_by_qid[row["qid"]], text_by_id[hit["chunk_id"]])
        for row in rows
        if row["answerable"] and row["config_id"].rsplit("-", 1)[0] in reranked
        for hit in row["ranked"]
    ]
    counts = hub.token_counts("reranker", pairs) if pairs else None
    if counts is None:
        return None
    limit = cfg["models"]["reranker"]["max_length"]
    n = len(counts)
    truncated = sum(count > limit for count in counts)
    return {
        "max_length": limit,
        "scope": RERANK_SCOPE.replace("top-k", f"top-{cfg['retrieval']['top_k']}"),
        "n_pairs": n,
        "n_truncated": truncated,
        "truncation_rate": round(truncated / n, 4),
        "mean_tokens": round(sum(counts) / n, 2),
    }


def measure_reranker_truncation(cfg: Config, hub: ModelHub | None = None) -> dict:
    """Add the reranker truncation to retrieval_meta.json from the committed hit rows (no retrieval run)."""
    hub = hub or ModelHub(cfg)
    if not hits_path(cfg).is_file() or not meta_path(cfg).is_file():
        raise PruefstandError(
            f"{hits_path(cfg)} or {meta_path(cfg)} not found; run `pruefstand retrieve` first"
        )
    questions = load_questions(cfg)
    hits = read_jsonl(hits_path(cfg))
    meta = read_json(meta_path(cfg))
    for chunking in cfg["retrieval"]["chunkings"]:
        chunks = load_chunks(chunks_path(cfg, chunking))
        rows = [row for row in hits if row["config_id"].endswith(f"-{chunking}")]
        meta["truncation"].setdefault(chunking, {})["reranker"] = reranker_truncation_report(
            cfg, chunks, rows, questions, hub
        )
    write_json(meta_path(cfg), meta)
    return meta["truncation"]


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
    unit_texts: dict[str, str],
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

    chunks_by_unit: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        for cover in chunk.covers:
            chunks_by_unit.setdefault(cover["unit_id"], []).append(index)

    unit_lengths = {unit_id: len(text) for unit_id, text in unit_texts.items()}
    rows = []
    for i, question in enumerate(questions):
        gold = question["gold_unit_ids"]
        validate_gold(gold, unit_lengths)
        if question["answerable"] and not gold:
            raise PruefstandError(f"{question['qid']}: an answerable question needs at least one gold unit")
        # Long gold units are graded on the gold passage, not on the whole unit (see pruefstand.metrics).
        spans = passage_spans(gold, question.get("gold_passage_de", ""), unit_texts, question["qid"])
        # Ideal gains of this chunking for nDCG: greedy cover of the gold units by the chunks that overlap them.
        overlapping = sorted({index for unit_id in gold for index in chunks_by_unit.get(unit_id, [])})
        ideal = ideal_gains(
            gold_units_covered(chunks[index].covers, gold, unit_lengths, spans) for index in overlapping
        )
        for retriever in retrieval["retrievers"]:
            ranked = candidates[i][retriever][:top]
            rows.append(
                {
                    "config_id": f"{retriever}-{chunking}",
                    "qid": question["qid"],
                    "type": question["type"],
                    "answerable": question["answerable"],
                    "ideal_gains": ideal,
                    "ranked": [
                        {
                            "rank": rank,
                            "chunk_id": chunks[doc].chunk_id,
                            "score": score,
                            "n_tokens": chunks[doc].n_tokens,
                            "gold_covered": gold_units_covered(chunks[doc].covers, gold, unit_lengths, spans),
                        }
                        for rank, (doc, score) in enumerate(ranked, start=1)
                    ],
                }
            )
    return rows


def relevance_meta(questions: list[dict], unit_texts: dict[str, str]) -> dict:
    """The relevance rule and how many questions and gold units it grades on the passage, for the results metadata."""
    passage_based_units = 0
    passage_based_questions = 0
    for question in questions:
        spans = passage_spans(question["gold_unit_ids"], question.get("gold_passage_de", ""), unit_texts)
        passage_based_units += len(spans)
        passage_based_questions += bool(spans)
    return {
        "rule": RELEVANCE_RULE,
        "long_unit_chars": LONG_UNIT_CHARS,
        "min_overlap_chars": MIN_OVERLAP_CHARS,
        "min_overlap_share": MIN_OVERLAP_SHARE,
        "n_passage_based_questions": passage_based_questions,
        "n_passage_based_gold_units": passage_based_units,
    }


def run_retrieval(cfg: Config, hub: ModelHub | None = None) -> int:
    """Run all variants, write hits.jsonl.gz, prune and save the caches; returns the number of hit rows."""
    hub = hub or ModelHub(cfg)
    questions = load_questions(cfg)
    units = load_units(cfg.path(cfg["corpus"]["units_path"]))
    unit_texts = {unit.unit_id: unit.text for unit in units}
    caches = open_caches(cfg)
    rows: list[dict] = []
    truncation: dict = {}
    for chunking in cfg["retrieval"]["chunkings"]:
        chunks = load_chunks(chunks_path(cfg, chunking))
        chunk_rows = retrieve_chunking(cfg, chunking, chunks, questions, unit_texts, caches, hub)
        rows.extend(chunk_rows)
        truncation[chunking] = truncation_report(cfg, chunks, hub)
        truncation[chunking]["reranker"] = reranker_truncation_report(cfg, chunks, chunk_rows, questions, hub)
    write_json(meta_path(cfg), {"truncation": truncation, "relevance": relevance_meta(questions, unit_texts)})
    for cache in caches.values():
        cache.prune_unused()
        cache.save()
    return write_jsonl(hits_path(cfg), rows)
