"""Run-level retrieval metrics with bootstrap intervals, per question type, and pre-declared paired comparisons."""

from __future__ import annotations

from collections import Counter

from pruefstand.bench import config_ids, hits_path, load_questions, meta_path
from pruefstand.chunking import chunks_path, load_chunks
from pruefstand.config import Config
from pruefstand.corpus.gii import load_units
from pruefstand.errors import PruefstandError
from pruefstand.io import read_json, read_jsonl, write_json
from pruefstand.metrics import (
    CROSS_CHUNKING_METRICS,
    K_VALUES,
    METRIC_NAMES,
    WITHIN_CHUNKING_METRICS,
    question_metrics,
)
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
        table.setdefault(row["config_id"], {})[row["qid"]] = question_metrics(
            covered, n_gold, row["ideal_gains"]
        )
    return table


def mean_tokens_at_k(hits: list[dict]) -> dict[str, dict[str, float]]:
    """config_id -> k -> mean number of retrieved chunk tokens in the top k (answerable questions).

    Recall@k at equal k does not mean an equal context budget when chunks differ in length; these means make a
    token-budget comparison possible later.
    """
    totals: dict[str, dict[int, list[int]]] = {}
    for row in hits:
        if not row["answerable"]:
            continue
        per_k = totals.setdefault(row["config_id"], {k: [] for k in K_VALUES})
        for k in K_VALUES:
            per_k[k].append(sum(hit["n_tokens"] for hit in row["ranked"][:k]))
    return {
        config_id: {str(k): round(sum(values) / len(values), 2) for k, values in per_k.items()}
        for config_id, per_k in totals.items()
    }


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
    hits = read_jsonl(hits_path(cfg))
    table = per_question_metrics(hits, questions)
    tokens_at_k = mean_tokens_at_k(hits)
    qids = sorted(qid for qid, question in questions.items() if question["answerable"])
    checked = [qid for qid in qids if questions[qid].get("review", {}).get("status") in CHECKED_STATUSES]
    types = sorted({questions[qid]["type"] for qid in qids})

    variants = {}
    for config_id in config_ids(cfg):
        values = table[config_id]
        variants[config_id] = {
            "mean_retrieved_tokens_at_k": tokens_at_k[config_id],
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
        cross = a.rsplit("-", 1)[1] != b.rsplit("-", 1)[1]
        for name in COMPARISON_METRICS:
            if cross and name in WITHIN_CHUNKING_METRICS:
                continue  # nDCG is only defined within one chunking: no interval is computed, so none can be quoted
            result = paired_bootstrap(
                [table[a][q][name] for q in qids], [table[b][q][name] for q in qids], n_boot, seed
            )
            comparisons.append(
                {
                    "a": a,
                    "b": b,
                    "metric": name,
                    "scope": "cross_chunking" if cross else "within_chunking",
                    "valid": True,
                    **result,
                }
            )

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
        "metric_notes": {
            "cross_chunking_metrics": list(CROSS_CHUNKING_METRICS),
            "within_chunking_metrics": list(WITHIN_CHUNKING_METRICS),
            "text": (
                "recall@k and MRR@10 are the cross-chunking comparison metrics (C1 against C2). nDCG@10 compares "
                "retrievers within one chunking: gain = share of gold units newly covered, ideal = greedy cover of "
                "the same chunking; cross-chunking pairs are therefore compared on recall@5 only and no cross-chunking "
                "nDCG interval exists. mean_retrieved_tokens_at_k gives the context size behind each k."
            ),
        },
        "truncation": read_json(meta_path(cfg))["truncation"] if meta_path(cfg).is_file() else None,
        "relevance": read_json(meta_path(cfg)).get("relevance") if meta_path(cfg).is_file() else None,
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


def _truncation_note(result: dict, config_id: str) -> str:
    """Suffix for the MiniLM rows: how much of the chunk text the model never sees."""
    retriever, chunking = config_id.rsplit("-", 1)
    if retriever != "mini" or not result.get("truncation"):
        return ""
    mini = (result["truncation"].get(chunking) or {}).get("mini")
    if not mini:
        return ""
    return (
        f"  (MiniLM truncated {mini['n_truncated']}/{mini['n_chunks']} {chunking} chunks"
        f" = {mini['truncation_rate'] * 100:.1f}%)"
    )


def summary_lines(result: dict) -> list[str]:
    """Console summary: recall@5 per variant, the dense-versus-English comparison and the truncation table.

    The English-only control reads only part of many chunks, so the truncation table is printed directly after
    the comparison it qualifies, and the MiniLM rows carry their own truncation note.
    """
    lines = []
    for config_id, variant in result["variants"].items():
        r5 = variant["metrics"]["recall@5"]
        lines.append(
            f"{config_id:12s} recall@5 {r5['mean']:.3f} [{r5['lo']:.3f}, {r5['hi']:.3f}]"
            f"{_truncation_note(result, config_id)}"
        )
    versus = result["headline"]["dense_vs_english"]
    lines.append("")
    lines.append(
        f"dense vs English ({versus['a']} vs {versus['b']}): recall@5 diff {versus['diff']:+.3f} "
        f"[{versus['lo']:+.3f}, {versus['hi']:+.3f}] -> {versus['verdict']}"
    )
    truncation = result.get("truncation")
    if not truncation or not any(
        row for per_model in truncation.values() for row in (per_model or {}).values()
    ):
        lines.append("truncation: not measured (offline profile without model files)")
        return lines
    lines.append("truncation (chunks whose text exceeds the model's max_seq_length):")
    lines.append(
        f"  {'model':5s} {'chunking':8s} {'max_seq':>7s} {'truncated':>12s} {'rate':>7s} {'visible':>8s}"
    )
    for model in ("e5", "mini"):
        for chunking, per_model in truncation.items():
            row = (per_model or {}).get(model)
            if not row:
                continue
            lines.append(
                f"  {model:5s} {chunking:8s} {row['max_seq_length']:7d} "
                f"{row['n_truncated']:>5d}/{row['n_chunks']:<6d} {row['truncation_rate'] * 100:6.1f}% "
                f"{row['mean_visible_share'] * 100:7.1f}%"
            )
    return lines
