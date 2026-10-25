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
        "finding_rerank": "Über der Dense-Suche bringt erst das Reranking einen messbaren Gewinn: {pairs} (Recall@5, 95-%-KI).",
        "finding_rerank_pair": "{a} gegenüber {b} {diff} [{lo}, {hi}]",
        "finding_rerank_close": (
            "Die untere Grenze des Vergleichs {a} gegenüber {b} ({lo}) liegt nahe null; dieser Befund ist knapp."
        ),
        "finding_dense_share": (
            "Der Abstand zu BM25 ({total}, {stack} gegenüber {lexical}) geht nach den Punktschätzern überwiegend auf das "
            "Dense-Modell zurück: {dense} {dense_mean} gegenüber {lexical} {lexical_mean} ({dense_gap}). Dieser "
            "Vergleich wurde vorab nicht festgelegt und hat hier kein Intervall."
        ),
        "finding_highest": (
            "Einordnung: Den höchsten Punktschätzer für Recall@5 hat {config} ({mean}, 95-%-KI {lo} bis {hi}). "
            "Ein Vergleich mit der Hauptkonfiguration wurde vorab nicht festgelegt; die Hauptkonfiguration wurde "
            "vor dem Lauf bestimmt, nicht nach dem Ergebnis gewählt."
        ),
        "matrix": "Retrieval-Matrix (Recall@5, 95-%-KI)",
        "all_metrics": "Alle Retrieval-Kennzahlen",
        "all_metrics_note": (
            "nDCG@10 ist nur innerhalb eines Chunkings vergleichbar: C1 und C2 sind auf je eigene Ideale normiert. "
            "C1 und C2 werden mit Recall@k und MRR@10 verglichen."
        ),
        "comparisons": "Vergleiche (gepaarter Bootstrap, vorab festgelegt)",
        "context": "Kontextmenge: mittlere Token der Top-k-Chunks",
        "context_note": (
            "Bei gleichem k enthält der Kontext mit C1 etwa das Doppelte bis Vierfache der Token wie mit C2 (e5-Tokenizer, "
            "C2 mit Kopfzeile). Die Vergleiche C1 gegen C2 sind deshalb Vergleiche bei gleichem k, nicht bei gleicher Kontextmenge."
        ),
        "by_type": "Recall@5 nach Fragetyp",
        "latency": "Latenz pro Stufe (ms, p50 / p95)",
        "latency_missing": "In diesem Profil nicht gemessen.",
        "latency_note": (
            "Gemessen auf {n} Fragen: den ersten {n} beantwortbaren nach Fragen-ID ({laws}), also kein Querschnitt "
            "der Gesetze. CPU: {cpu}."
        ),
        "latency_machine": " Maschine: {detail}. Eine Wiederholung auf einer anderen CPU kann niedrige Dezimalstellen verschieben.",
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
            "Relevanz über Quellspannen, auf zwei Ebenen. Gold-Absätze bis {long_chars} Zeichen (Einheitsebene): ein Chunk zählt, wenn er mindestens {min_chars} Zeichen oder {share} % des Absatzes abdeckt, je nachdem, was weniger ist. Längere Gold-Absätze (vor allem Definitionsnormen, Passagenebene): ein Chunk zählt nur, wenn er einen Auszug der Gold-Passage um mindestens {min_chars} Zeichen oder {share} % des Auszugs überlappt; ein anderer Eintrag derselben langen Liste zählt nicht. {passage} von {answerable} beantwortbaren Fragen werden auf der Passage bewertet.",
            "nDCG@10 ist nur innerhalb eines Chunkings vergleichbar: der Gewinn eines Rangs ist der Anteil der Gold-Einheiten, den der Chunk erstmals abdeckt; das Ideal ist die gierige Minimalüberdeckung der Gold-Einheiten durch die Chunks desselben Chunkings, die Kennzahl ist auf höchstens 1 begrenzt. Zwischen C1 und C2 werden Recall@k und MRR@10 verglichen.",
            "Intervalle: Perzentil-Bootstrap über Fragen ({boot} Ziehungen, Seed {seed}); enthält das Intervall einer Differenz die Null, heißt es „kein messbarer Unterschied“.",
            "Die Tabelle der Vergleiche hat {n_comparisons} Zeilen aus {n_pairs} vorab festgelegten Paaren und ist nicht für mehrfaches Testen korrigiert. Die Fragen werden als unabhängig neu gezogen, obwohl einige Fragen Gold-Absätze teilen; die Intervalle können dadurch zu eng sein. Ein Intervall wie [1,000, 1,000] (alle Fragen eines Typs getroffen, wenige Fragen) ist keine Gewissheit.",
            "Die Fragen sind modellgeneriert und noch von keinem Menschen geprüft; die Prüfung von 100 Fragen durch den Owner steht aus.",
            "Autor der Fragen und Primär-Judge des Live-Laufs sind dasselbe Modell (claude-sonnet-5-5); Selbstpräferenz ist möglich. Ein zweiter Judge (claude-opus-5-5) und die menschlich geprüfte Teilmenge begrenzen den Effekt, beseitigen ihn nicht. Die Retrieval-Kennzahlen brauchen keinen Judge.",
            "Die Hauptkonfiguration hyb+rr-C2 wurde vor dem Lauf festgelegt; die Fusion (Reciprocal Rank Fusion, k = 60) hat kein auf den Testfragen getuntes Gewicht.",
            "Der Abstand zwischen multilingual-e5-base und all-MiniLM-L6-v2 ist kein reiner Spracheffekt: die Modelle unterscheiden sich auch in Größe, Trainingsdaten und Präfix-Konvention, und MiniLM kürzt lange Chunks. MiniLM ist eine Kontrolle, keine Messung des Sprachvorteils.",
            "Der Reranker ist bge-reranker-base (vor allem auf Chinesisch und Englisch trainiert); ein mehrsprachiger Reranker ist Roadmap.",
            "Alle Modelle laufen auf der CPU (keine GPU); Latenzen gelten nur für die genannte Maschine.",
        ],
        "platform_line": "Plattform: {os}, CPU {cpu}, Python {python}; keine GPU.",
        "provenance_line": (
            "Konfigurations-Hash zur Laufzeit {run_hash} (git {git}). Seither geänderte Abschnitte von pruefstand.toml: "
            "{sections}. Hash der Retrieval-Einstellungen: {retrieval_hash}, {same}."
        ),
        "provenance_same": "unverändert; die Treffer gehören zu dieser Konfiguration",
        "provenance_diff": "ABWEICHEND; die veröffentlichten Treffer gehören nicht zu dieser Konfiguration",
        "provenance_none": "keine",
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
        "finding_rerank": "On top of dense retrieval, reranking is what adds a measurable gain: {pairs} (recall@5, 95% CI).",
        "finding_rerank_pair": "{a} vs {b} {diff} [{lo}, {hi}]",
        "finding_rerank_close": (
            "The lower bound of the comparison {a} vs {b} ({lo}) is close to zero; this finding is borderline."
        ),
        "finding_dense_share": (
            "By the point estimates, most of the gap to BM25 ({total}, {stack} vs {lexical}) comes from the dense "
            "model: {dense} {dense_mean} vs {lexical} {lexical_mean} ({dense_gap}). That comparison was not declared "
            "in advance and has no interval here."
        ),
        "finding_highest": (
            "For context: the highest point estimate for recall@5 belongs to {config} ({mean}, 95% CI {lo} to {hi}). "
            "No comparison with the headline configuration was declared in advance; the headline configuration was "
            "fixed before the run, not chosen after seeing results."
        ),
        "matrix": "Retrieval matrix (recall@5, 95% CI)",
        "all_metrics": "All retrieval metrics",
        "all_metrics_note": (
            "nDCG@10 is comparable only within one chunking: C1 and C2 are normalised against their own ideals. "
            "C1 and C2 are compared with recall@k and MRR@10."
        ),
        "comparisons": "Comparisons (paired bootstrap, fixed before the run)",
        "context": "Context size: mean tokens of the top-k chunks",
        "context_note": (
            "At equal k the context holds about two to four times as many tokens with C1 as with C2 (e5 tokenizer, "
            "C2 with header line). The C1-versus-C2 comparisons are therefore comparisons at equal k, not at equal context size."
        ),
        "by_type": "Recall@5 by question type",
        "latency": "Latency per stage (ms, p50 / p95)",
        "latency_missing": "Not measured in this profile.",
        "latency_note": (
            "Measured on {n} questions: the first {n} answerable ones by question id ({laws}), so not a cross-section "
            "of the laws. CPU: {cpu}."
        ),
        "latency_machine": " Machine: {detail}. A re-run on another CPU may move low decimals.",
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
            "Relevance by source spans, on two levels. Gold Absätze up to {long_chars} characters (unit level): a chunk counts when it covers at least {min_chars} characters or {share}% of the Absatz, whichever is smaller. Longer gold Absätze (mostly definition norms, passage level): a chunk counts only when it overlaps an excerpt of the gold passage by at least {min_chars} characters or {share}% of the excerpt; another entry of the same long list does not count. {passage} of {answerable} answerable questions are graded on the passage.",
            "nDCG@10 is comparable only within one chunking: the gain of a rank is the share of the gold units the chunk covers for the first time; the ideal is the greedy minimal cover of the gold units by the chunks of the same chunking, and the metric is capped at 1. C1 and C2 are compared with recall@k and MRR@10.",
            "Intervals: percentile bootstrap over questions ({boot} resamples, seed {seed}); when the interval of a difference contains zero it is called “no measurable difference”.",
            "The comparison table has {n_comparisons} rows from {n_pairs} pairs fixed in advance and is not corrected for multiple testing. Questions are resampled as independent although some share gold Absätze, so the intervals may be too narrow. An interval such as [1.000, 1.000] (every question of a type hit, few questions) is not certainty.",
            "The questions are model-generated and not yet checked by any human; the owner's check of 100 questions is pending.",
            "The author of the questions and the primary judge of the live run are the same model (claude-sonnet-5-5); self-preference is possible. A second judge (claude-opus-5-5) and the human-checked subset bound the effect but do not remove it. The retrieval metrics need no judge.",
            "The headline configuration hyb+rr-C2 was fixed before the run; the fusion (reciprocal rank fusion, k = 60) has no weight tuned on the test questions.",
            "The gap between multilingual-e5-base and all-MiniLM-L6-v2 is not a pure language effect: the models also differ in size, training data and prefix convention, and MiniLM truncates long chunks. MiniLM is a control, not a measurement of the language advantage.",
            "The reranker is bge-reranker-base (trained mainly on Chinese and English); a multilingual reranker is on the roadmap.",
            "All models run on the CPU (no GPU); latencies hold only for the machine named.",
        ],
        "platform_line": "Platform: {os}, CPU {cpu}, Python {python}; no GPU.",
        "provenance_line": (
            "Configuration hash at run time {run_hash} (git {git}). Sections of pruefstand.toml changed since: "
            "{sections}. Hash of the retrieval settings: {retrieval_hash}, {same}."
        ),
        "provenance_same": "unchanged; the hits belong to this configuration",
        "provenance_diff": "DIFFERENT; the published hits do not belong to this configuration",
        "provenance_none": "none",
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
