"""Run-level retrieval metrics with bootstrap intervals, per question type, and pre-declared paired comparisons."""

from __future__ import annotations

from collections import Counter

from pruefstand.bench import config_ids, hits_path, load_questions
from pruefstand.chunking import chunks_path, load_chunks
from pruefstand.config import Config
from pruefstand.corpus.gii import load_units
from pruefstand.errors import PruefstandError
from pruefstand.io import read_jsonl, write_json
from pruefstand.metrics import METRIC_NAMES, question_metrics
from pruefstand.stats import bootstrap_ci, paired_bootstrap

TYPE_METRICS = ("recall@5", "ndcg@10")
COMPARISON_METRICS = ("recall@5", "ndcg@10")
CHECKED_STATUSES = ("accept", "edit")


def per_question_metrics(
    hits: list[dict], questions: dict[str, dict]
) -> dict[str, dict[str, dict[str, float]]]:
    """config_id -> qid -> metric -> value, answerable questions only."""
    table: dict[str, dict[str, dict[str, float]]] = {}
    for row in hits:
        if not row["answerable"]:
            continue
        n_gold = len(questions[row["qid"]]["gold_unit_ids"])
        covered = [hit["gold_covered"] for hit in row["ranked"]]
        table.setdefault(row["config_id"], {})[row["qid"]] = question_metrics(covered, n_gold)
    return table


def corpus_stats(cfg: Config) -> dict:
    units = load_units(cfg.path(cfg["corpus"]["units_path"]))
    stats = {"n_units": len(units), "units_per_law": dict(sorted(Counter(u.law_slug for u in units).items()))}
    for chunking in cfg["retrieval"]["chunkings"]:
        chunks = load_chunks(chunks_path(cfg, chunking))
        tokens = [chunk.n_tokens for chunk in chunks]
        stats[f"n_chunks_{chunking}"] = len(chunks)
        stats[f"mean_tokens_{chunking}"] = round(sum(tokens) / len(tokens), 2) if tokens else 0.0
    return stats


def compute_retrieval_metrics(cfg: Config) -> dict:
    stats_cfg = cfg["stats"]
    n_boot, seed = stats_cfg["bootstrap"], stats_cfg["seed"]
    question_list = load_questions(cfg)
    questions = {question["qid"]: question for question in question_list}
    if not hits_path(cfg).is_file():
        raise PruefstandError(f"{hits_path(cfg)} not found; run `pruefstand retrieve` first")
    table = per_question_metrics(read_jsonl(hits_path(cfg)), questions)
    qids = sorted(qid for qid, question in questions.items() if question["answerable"])
    checked = [qid for qid in qids if questions[qid].get("review", {}).get("status") in CHECKED_STATUSES]
    types = sorted({questions[qid]["type"] for qid in qids})

    variants = {}
    for config_id in config_ids(cfg):
        values = table[config_id]
        variants[config_id] = {
            "metrics": {
                name: bootstrap_ci([values[q][name] for q in qids], n_boot, seed) for name in METRIC_NAMES
            },
            "by_type": {
                kind: {
                    name: bootstrap_ci(
                        [values[q][name] for q in qids if questions[q]["type"] == kind], n_boot, seed
                    )
                    for name in TYPE_METRICS
                }
                for kind in types
            },
            "checked_subset": {
                name: bootstrap_ci([values[q][name] for q in checked], n_boot, seed) for name in TYPE_METRICS
            },
        }

    comparisons = []
    for pair in cfg.data.get("comparison", []):
        a, b = pair["a"], pair["b"]
        for name in COMPARISON_METRICS:
            result = paired_bootstrap(
                [table[a][q][name] for q in qids], [table[b][q][name] for q in qids], n_boot, seed
            )
            comparisons.append({"a": a, "b": b, "metric": name, **result})

    headline_cfg = cfg["headline"]

    def find(a: str, b: str) -> dict:
        return next(c for c in comparisons if c["a"] == a and c["b"] == b and c["metric"] == "recall@5")

    result = {
        "run_id": cfg.run_id,
        "question_set": cfg["questions"]["version"],
        "n_questions": len(question_list),
        "n_answerable": len(qids),
        "n_checked": len(checked),
        "n_bootstrap": n_boot,
        "seed": seed,
        "question_types": dict(sorted(Counter(q["type"] for q in question_list).items())),
        "corpus": corpus_stats(cfg),
        "variants": variants,
        "comparisons": comparisons,
        "headline": {
            "best": {
                "config_id": headline_cfg["best"],
                **variants[headline_cfg["best"]]["metrics"]["recall@5"],
            },
            "dense_vs_english": find(*headline_cfg["dense_vs_english"]),
            "stack_vs_lexical": find(*headline_cfg["stack_vs_lexical"]),
        },
    }
    write_json(cfg.run_dir / "metrics.json", result)
    return result
