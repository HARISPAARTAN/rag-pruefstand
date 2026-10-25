"""German and English report strings. German is the primary language."""

from __future__ import annotations

TEXT = {
    "de": {
        "title": "RAG-Prüfstand – Ergebnisse",
        "run": "Lauf",
        "questions_note": (
            "Fragensatz {version}: {answerable} modellgenerierte beantwortbare Fragen, {checked} menschlich geprüft "
            "({spread}); dazu {unanswerable} nicht beantwortbare Fragen."
        ),
        "questions_authored": (
            " Geschrieben am {authored} von {author}; die Prüfung von 100 Fragen durch den Owner steht aus."
        ),
        "corpus_note": (
            "Korpus: {units} Einheiten (Absätze) aus {laws} Bundesgesetz(en). Quelle: gesetze-im-internet.de, "
            "Bundesministerium der Justiz; nichtamtliche konsolidierte Fassung, Stand {date}."
        ),
        "summary": "Kurzfassung",
        "headline_best": (
            "Vorab festgelegte Hauptkonfiguration {config}: Recall@5 {mean} (95-%-KI {lo} bis {hi}, n = {n})."
        ),
        "headline_diff": "{a} gegenüber {b} (Recall@5): {diff} (95-%-KI {lo} bis {hi}) – {verdict}.",
        "headline_dense": (
            "{a} gegenüber {b} (Recall@5): {diff} (95-%-KI {lo} bis {hi}) – {verdict}. "
            "Das ist kein reiner Spracheffekt: die Modelle unterscheiden sich auch in Größe, Training und der "
            "Kürzung langer Chunks (siehe Grenzen)."
        ),
        "finding_hybrid_none": (
            "Hybrid-Fusion bringt gegenüber reiner Dense-Suche nichts Messbares: {a} gegenüber {b} (Recall@5): "
            "{diff} (95-%-KI {lo} bis {hi}) – {verdict}."
        ),
        "finding_rerank": "Der Gewinn kommt vom Reranking: {pairs} (Recall@5, 95-%-KI).",
        "finding_rerank_pair": "{a} gegenüber {b} {diff} [{lo}, {hi}]",
        "finding_highest": (
            "Einordnung: Den höchsten Punktschätzer für Recall@5 hat {config} ({mean}, 95-%-KI {lo} bis {hi}). "
            "Ein Vergleich mit der Hauptkonfiguration wurde vorab nicht festgelegt; die Hauptkonfiguration wurde "
            "vor dem Lauf bestimmt, nicht nach dem Ergebnis gewählt."
        ),
        "matrix": "Retrieval-Matrix (Recall@5, 95-%-KI)",
        "all_metrics": "Alle Retrieval-Kennzahlen",
        "comparisons": "Vergleiche (gepaarter Bootstrap, vorab festgelegt)",
        "by_type": "Recall@5 nach Fragetyp",
        "latency": "Latenz pro Stufe (ms, p50 / p95)",
        "latency_missing": "In diesem Profil nicht gemessen.",
        "latency_note": "Gemessen auf {n} Fragen, CPU: {cpu}.",
        "generation": "Generierung und Bewertung",
        "pending_live": (
            "**Ausstehend: Live-Lauf.** Antworten, Judge-Bewertungen, Kosten pro 1.000 Anfragen und API-Latenz "
            "werden erst mit dem Live-Lauf (v0.2.0) gemessen. Hier stehen bewusst keine Zahlen."
        ),
        "synthetic_banner": (
            "**Synthetische Fixtures – keine Messung.** Diese Werte stammen aus handgeschriebenen CI-Fixtures "
            "und prüfen nur die Pipeline."
        ),
        "calibration": "Kalibrierung des Judges",
        "calibration_pending": (
            "**Ausstehend: Live-Lauf und menschliche Labels.** Gewichtetes Cohens Kappa und Rohübereinstimmung "
            "werden auf 80 vom Owner gelabelten Antworten berechnet."
        ),
        "method": "Methode und Grenzen",
        "method_points": [
            "Relevanz über Quellspannen: ein Chunk zählt, wenn er mindestens 50 % oder 200 Zeichen eines Gold-Absatzes abdeckt.",
            "Intervalle: Perzentil-Bootstrap über Fragen ({boot} Ziehungen, Seed {seed}); Unterschiede innerhalb des Intervalls heißen „kein messbarer Unterschied“.",
            "Die Fragen sind modellgeneriert und noch von keinem Menschen geprüft; die Prüfung von 100 Fragen durch den Owner steht aus.",
            "Autor der Fragen und Primär-Judge des Live-Laufs sind dasselbe Modell (claude-sonnet-5-5); Selbstpräferenz ist möglich. Ein zweiter Judge (claude-opus-5-5) und die menschlich geprüfte Teilmenge begrenzen den Effekt, beseitigen ihn nicht. Die Retrieval-Kennzahlen brauchen keinen Judge.",
            "Die Hauptkonfiguration hyb+rr-C2 wurde vor dem Lauf festgelegt; die Fusion (Reciprocal Rank Fusion, k = 60) hat kein auf den Testfragen getuntes Gewicht.",
            "Der Abstand zwischen multilingual-e5-base und all-MiniLM-L6-v2 ist kein reiner Spracheffekt: die Modelle unterscheiden sich auch in Größe, Trainingsdaten und Präfix-Konvention, und MiniLM kürzt lange Chunks. MiniLM ist eine Kontrolle, keine Messung des Sprachvorteils.",
            "Der Reranker ist bge-reranker-base (vor allem auf Chinesisch und Englisch trainiert); ein mehrsprachiger Reranker ist Roadmap.",
            "Alle Modelle laufen auf der CPU (keine GPU); Latenzen gelten nur für die genannte Maschine.",
        ],
        "platform_line": "Plattform: {os}, CPU {cpu}, Python {python}; keine GPU.",
        "mini_truncation": (
            "MiniLM-Kürzung: all-MiniLM-L6-v2 liest höchstens {max_len} Token; bei C1 verlieren {c1_rate} % der "
            "Chunks ihr Ende (sichtbarer Anteil im Mittel {c1_vis} %), bei C2 {c2_rate} % ({c2_vis} %). "
            "multilingual-e5-base kürzt {e5_c1} % (C1) bzw. {e5_c2} % (C2) der Chunks."
        ),
        "reranker_truncation": (
            "Reranker-Kürzung: bge-reranker-base liest höchstens {max_len} Token je (Frage, Chunk)-Paar; bei C1 "
            "verlieren {c1_rate} % der neu bewerteten Paare ihr Ende, bei C2 {c2_rate} %. Die Wirkung auf die "
            "Ergebnisse wurde nicht gemessen."
        ),
        "variant": "Variante",
        "retriever": "Retriever",
        "metric": "Kennzahl",
        "diff": "Differenz",
        "ci": "95-%-KI",
        "verdict": "Urteil",
        "embed": "Embedding",
        "search": "Suche",
        "rerank": "Rerank",
        "total": "Gesamt",
        "verdicts": {
            "better": "besser",
            "worse": "schlechter",
            "no_measurable_difference": "kein messbarer Unterschied",
            "no_data": "keine Daten",
        },
        "gen_cols": [
            "Variante",
            "Treue",
            "Korrektheit",
            "Zitat-Präzision",
            "Enthaltung",
            "USD/1.000",
            "EUR/1.000",
            "p50 ms",
        ],
        "chart_matrix_title": "Recall@5 je Retriever und Chunking",
        "chart_k_title": "Recall@k, Chunking C2",
        "chart_latency_title": "Latenz gesamt (ms), Chunking C2",
        "repo": "Repository",
        "other_lang": "English version",
    },
    "en": {
        "title": "RAG-Prüfstand – results",
        "run": "Run",
        "questions_note": (
            "Question set {version}: {answerable} model-generated answerable questions, {checked} human-checked "
            "({spread}); plus {unanswerable} unanswerable questions."
        ),
        "questions_authored": (
            " Written on {authored} by {author}; the owner's check of 100 questions is pending."
        ),
        "corpus_note": (
            "Corpus: {units} units (paragraphs) from {laws} federal law(s). Source: gesetze-im-internet.de, Federal Ministry "
            "of Justice; non-official consolidated version as of {date}."
        ),
        "summary": "Summary",
        "headline_best": (
            "Pre-declared headline configuration {config}: recall@5 {mean} (95% CI {lo} to {hi}, n = {n})."
        ),
        "headline_diff": "{a} vs {b} (recall@5): {diff} (95% CI {lo} to {hi}) – {verdict}.",
        "headline_dense": (
            "{a} vs {b} (recall@5): {diff} (95% CI {lo} to {hi}) – {verdict}. "
            "This is not a pure language effect: the models also differ in size, training and in how much of long "
            "chunks they read (see limitations)."
        ),
        "finding_hybrid_none": (
            "Hybrid fusion adds nothing measurable over dense retrieval alone: {a} vs {b} (recall@5): "
            "{diff} (95% CI {lo} to {hi}) – {verdict}."
        ),
        "finding_rerank": "The gain comes from reranking: {pairs} (recall@5, 95% CI).",
        "finding_rerank_pair": "{a} vs {b} {diff} [{lo}, {hi}]",
        "finding_highest": (
            "For context: the highest point estimate for recall@5 belongs to {config} ({mean}, 95% CI {lo} to {hi}). "
            "No comparison with the headline configuration was declared in advance; the headline configuration was "
            "fixed before the run, not chosen after seeing results."
        ),
        "matrix": "Retrieval matrix (recall@5, 95% CI)",
        "all_metrics": "All retrieval metrics",
        "comparisons": "Comparisons (paired bootstrap, fixed before the run)",
        "by_type": "Recall@5 by question type",
        "latency": "Latency per stage (ms, p50 / p95)",
        "latency_missing": "Not measured in this profile.",
        "latency_note": "Measured on {n} questions, CPU: {cpu}.",
        "generation": "Generation and judging",
        "pending_live": (
            "**Pending: live run.** Answers, judge scores, cost per 1,000 queries and API latency are measured "
            "only in the live run (v0.2.0). No numbers are shown here on purpose."
        ),
        "synthetic_banner": (
            "**Synthetic fixtures – not a measurement.** These values come from hand-written CI fixtures and only "
            "test the pipeline."
        ),
        "calibration": "Judge calibration",
        "calibration_pending": (
            "**Pending: live run and human labels.** Weighted Cohen's kappa and raw agreement are computed on 80 "
            "answers labelled by the owner."
        ),
        "method": "Method and limitations",
        "method_points": [
            "Relevance by source spans: a chunk counts when it covers at least 50% or 200 characters of a gold Absatz.",
            "Intervals: percentile bootstrap over questions ({boot} resamples, seed {seed}); differences inside the interval are called “no measurable difference”.",
            "The questions are model-generated and not yet checked by any human; the owner's check of 100 questions is pending.",
            "The author of the questions and the primary judge of the live run are the same model (claude-sonnet-5-5); self-preference is possible. A second judge (claude-opus-5-5) and the human-checked subset bound the effect but do not remove it. The retrieval metrics need no judge.",
            "The headline configuration hyb+rr-C2 was fixed before the run; the fusion (reciprocal rank fusion, k = 60) has no weight tuned on the test questions.",
            "The gap between multilingual-e5-base and all-MiniLM-L6-v2 is not a pure language effect: the models also differ in size, training data and prefix convention, and MiniLM truncates long chunks. MiniLM is a control, not a measurement of the language advantage.",
            "The reranker is bge-reranker-base (trained mainly on Chinese and English); a multilingual reranker is on the roadmap.",
            "All models run on the CPU (no GPU); latencies hold only for the machine named.",
        ],
        "platform_line": "Platform: {os}, CPU {cpu}, Python {python}; no GPU.",
        "mini_truncation": (
            "MiniLM truncation: all-MiniLM-L6-v2 reads at most {max_len} tokens; with C1, {c1_rate}% of the chunks "
            "lose their tail (visible share {c1_vis}% on average), with C2 {c2_rate}% ({c2_vis}%). "
            "multilingual-e5-base truncates {e5_c1}% (C1) and {e5_c2}% (C2) of the chunks."
        ),
        "reranker_truncation": (
            "Reranker truncation: bge-reranker-base reads at most {max_len} tokens per (question, chunk) pair; with "
            "C1, {c1_rate}% of the reranked pairs lose their tail, with C2 {c2_rate}%. The effect on the results "
            "was not measured."
        ),
        "variant": "Variant",
        "retriever": "Retriever",
        "metric": "Metric",
        "diff": "Difference",
        "ci": "95% CI",
        "verdict": "Verdict",
        "embed": "Embedding",
        "search": "Search",
        "rerank": "Rerank",
        "total": "Total",
        "verdicts": {
            "better": "better",
            "worse": "worse",
            "no_measurable_difference": "no measurable difference",
            "no_data": "no data",
        },
        "gen_cols": [
            "Variant",
            "Faithfulness",
            "Correctness",
            "Citation precision",
            "Abstention",
            "USD/1,000",
            "EUR/1,000",
            "p50 ms",
        ],
        "chart_matrix_title": "Recall@5 per retriever and chunking",
        "chart_k_title": "Recall@k, chunking C2",
        "chart_latency_title": "Total latency (ms), chunking C2",
        "repo": "Repository",
        "other_lang": "Deutsche Fassung",
    },
}


def fmt(value: float | None, lang: str, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "–"
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "de" else text


def fmt_ci(entry: dict, lang: str) -> str:
    if entry.get("mean") is None:
        return "–"
    return f"{fmt(entry['mean'], lang)} [{fmt(entry['lo'], lang)}, {fmt(entry['hi'], lang)}]"
