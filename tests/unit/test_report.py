import re

from pruefstand.config import load_config
from pruefstand.io import read_json
from pruefstand.report.build import (
    render_html,
    render_markdown,
    report_meta,
    update_readme_block,
)
from pruefstand.report.charts import RETRIEVERS, retrieval_matrix_svg
from pruefstand.report.text import fmt, fmt_ci

CHUNKINGS = ["C1", "C2"]
CONFIG_IDS = [f"{r}-{c}" for c in CHUNKINGS for r in RETRIEVERS]


def ci(mean, n=27):
    return {"mean": mean, "lo": round(mean - 0.1, 6), "hi": round(mean + 0.05, 6), "n": n}


def fake_metrics() -> dict:
    variant = {
        "metrics": {
            name: ci(0.5) for name in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")
        },
        "by_type": {"factoid": {"recall@5": ci(0.6, 8), "ndcg@10": ci(0.5, 8)}},
        "checked_subset": {"recall@5": ci(0.5, 0), "ndcg@10": ci(0.5, 0)},
    }
    diff = {
        "a": "hyb+rr-C2",
        "b": "bm25-C2",
        "metric": "recall@5",
        "diff": 0.12,
        "lo": 0.04,
        "hi": 0.2,
        "n": 27,
        "verdict": "better",
    }
    return {
        "run_id": "test",
        "question_set": "k1_v1",
        "n_questions": 30,
        "n_answerable": 27,
        "n_checked": 0,
        "n_bootstrap": 2000,
        "seed": 20261025,
        "question_types": {"factoid": 27, "unanswerable": 3},
        "corpus": {"n_units": 220, "units_per_law": {"messbg": 220}},
        "variants": {cid: variant for cid in CONFIG_IDS},
        "comparisons": [diff],
        "headline": {
            "best": {"config_id": "hyb+rr-C2", **ci(0.5)},
            "dense_vs_english": {**diff, "a": "e5-C2", "b": "mini-C2", "verdict": "no_measurable_difference"},
            "stack_vs_lexical": diff,
        },
    }


META = {
    "snapshot_date": "2026-10-25",
    "chunkings": CHUNKINGS,
    "config_ids": CONFIG_IDS,
    "generation_variants": ["bm25-C1"],
}


def test_number_formatting_per_language():
    assert fmt(0.5, "de") == "0,500"
    assert fmt(0.5, "en") == "0.500"
    assert fmt(0.12, "de", signed=True) == "+0,120"
    assert fmt(None, "de") == "–"
    assert fmt_ci(ci(0.5), "en") == "0.500 [0.400, 0.550]"


def test_pending_generation_never_shows_numbers():
    text = render_markdown(fake_metrics(), None, None, META, "de")
    generation = text.split("## Generierung und Bewertung", 1)[1].split("## ", 1)[0]
    assert "Ausstehend: Live-Lauf" in generation
    assert "|" not in generation
    assert "| hyb+rr | 0,500 [0,400, 0,550] | 0,500 [0,400, 0,550] |" in text
    assert "besser" in text


def test_english_report_and_synthetic_banner():
    answers = {
        "synthetic": True,
        "variants": {
            "bm25-C1": {
                "faithfulness": ci(2.0, 8),
                "correctness": ci(1.0, 8),
                "citation_precision": ci(0.5, 8),
                "abstention_accuracy": ci(1.0, 8),
                "usd_per_1000": 0.41,
                "eur_per_1000": 0.3663,
                "latency_ms": {"p50": 1200.0, "p95": 1300.0},
            }
        },
    }
    text = render_markdown(fake_metrics(), None, answers, META, "en")
    assert "Synthetic fixtures – not a measurement." in text
    assert "| bm25-C1 | 2.000 [1.900, 2.050] |" in text


def test_html_is_dark_and_escapes_text():
    page = render_html(
        fake_metrics(),
        None,
        None,
        META,
        {"retrieval_matrix": "<svg></svg>", "recall_at_k": "<svg></svg>"},
        "de",
    )
    assert page.startswith("<!doctype html>")
    assert "--bg:#0b0f14" in page and "--accent:#f2a93b" in page
    assert '<html lang="de">' in page and 'href="en.html"' in page
    assert "Ausstehend: Live-Lauf" in page


def test_chart_svg_is_deterministic():
    variants = fake_metrics()["variants"]
    first = retrieval_matrix_svg(variants, CHUNKINGS, "Titel", "dark")
    assert first == retrieval_matrix_svg(variants, CHUNKINGS, "Titel", "dark")
    assert first.startswith("<svg") and "#0b0f14" in first


def test_readme_block_replacement(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("Kopf\n<!-- results:start -->\nalt\n<!-- results:end -->\nFuß\n", encoding="utf-8")
    assert update_readme_block(readme, "- neu")
    assert (
        readme.read_text(encoding="utf-8")
        == "Kopf\n<!-- results:start -->\n- neu\n<!-- results:end -->\nFuß\n"
    )
    assert not update_readme_block(tmp_path / "missing.md", "x")


# ------------------------------------------------------------------------------------------ published run


def published(root):
    cfg = load_config(root / "pruefstand.toml", "full")
    metrics = read_json(cfg.run_dir / "metrics.json")
    latency = read_json(cfg.run_dir / "latency.json")
    return metrics, latency, report_meta(cfg)


def test_published_report_german_wording(root):
    metrics, latency, meta = published(root)
    text = render_markdown(metrics, latency, None, meta, "de")
    assert "Vorab festgelegte Hauptkonfiguration hyb+rr-C2: Recall@5 0,706" in text
    assert "216 modellgenerierte beantwortbare Fragen, 0 menschlich geprüft" in text
    assert "factoid 60, colloquial 48, definition 48, crossref 36, number 24" in text
    assert "24 nicht beantwortbare Fragen" in text
    assert "Geschrieben am 2026-10-25 von claude-sonnet-5-5" in text
    assert "Hybrid-Fusion bringt gegenüber reiner Dense-Suche nichts Messbares" in text
    assert "Der Gewinn kommt vom Reranking" in text
    assert "Das ist kein reiner Spracheffekt" in text
    assert "Autor der Fragen und Primär-Judge des Live-Laufs sind dasselbe Modell" in text
    assert "Windows-11" in text and "AMD64" in text and "keine GPU" in text
    assert "MiniLM-Kürzung" in text and "92,3 %" in text and "Reranker-Kürzung" in text and "15,1 %" in text
    generation = text.split("## Generierung und Bewertung", 1)[1].split("## ", 1)[0]
    assert "Ausstehend: Live-Lauf" in generation and "|" not in generation
    assert "e5+rr-C2" in text  # the variant with the highest point estimate is named, not hidden
    assert not re.search(r"\bbest\b|\bbeste[nrsm]?\b", text, re.IGNORECASE)


def test_published_report_english_wording(root):
    metrics, latency, meta = published(root)
    text = render_markdown(metrics, latency, None, meta, "en")
    assert "Pre-declared headline configuration hyb+rr-C2: recall@5 0.706" in text
    assert "216 model-generated answerable questions, 0 human-checked" in text
    assert "Written on 2026-10-25 by claude-sonnet-5-5" in text
    assert "Hybrid fusion adds nothing measurable over dense retrieval alone" in text
    assert "The gain comes from reranking" in text
    assert "This is not a pure language effect" in text
    assert "MiniLM truncation" in text and "92.3%" in text and "Reranker truncation" in text
    assert "Pending: live run" in text
    assert not re.search(r"\bbest\b", text, re.IGNORECASE)


def test_html_pages_carry_the_same_statements(root):
    metrics, latency, meta = published(root)
    svgs = {"retrieval_matrix": "<svg></svg>", "recall_at_k": "<svg></svg>", "latency": "<svg></svg>"}
    page = render_html(metrics, latency, None, meta, svgs, "de")
    assert "Vorab festgelegte Hauptkonfiguration hyb+rr-C2" in page
    assert "216 modellgenerierte beantwortbare Fragen, 0 menschlich geprüft" in page
    assert "Ausstehend: Live-Lauf" in page and "Hybrid-Fusion bringt" in page
    english = render_html(metrics, latency, None, meta, svgs, "en")
    assert '<html lang="en">' in english and 'href="index.html"' in english
    assert "Pending: live run" in english
