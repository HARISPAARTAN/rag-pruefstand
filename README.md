# RAG-Prüfstand

[![CI](https://github.com/HARISPAARTAN/rag-pruefstand/actions/workflows/ci.yml/badge.svg)](https://github.com/HARISPAARTAN/rag-pruefstand/actions/workflows/ci.yml)

Ein reproduzierbarer Prüfstand für Retrieval-Augmented Generation (RAG) auf deutschem Recht: fünf Bundesgesetze aus
dem Energierecht, ein versionierter Fragensatz mit Herkunftsangabe (240 Fragen, davon 216 beantwortbar, modellgeneriert,
0 menschlich geprüft), zwölf Retrieval-Varianten auf CPU und Konfidenzintervalle für jede Zahl. English version:
[README.en.md](README.en.md). Bericht als Seite: https://harispaartan.github.io/rag-pruefstand/

## Problem

Viele RAG-Demos zeigen Antworten, aber keine Messung. Dieser Prüfstand misst, wie gut verschiedene Retrieval-Varianten
die richtige Fundstelle finden, mit festem Fragensatz, festem Korpus und Intervallen statt Einzelwerten. Deutsche
Rechtstexte sind ein harter Fall: lange Komposita, Querverweise wie „nach § 14a Absatz 1" und Definitionen in langen
Listen.

## Ergebnis in drei Zahlen

<!-- results:start -->
- Vorab festgelegte Hauptkonfiguration hyb+rr-C2: Recall@5 0,706 (95-%-KI 0,649 bis 0,762, n = 216).
- e5-C2 gegenüber mini-C2 (Recall@5): +0,408 (95-%-KI +0,341 bis +0,474) – besser. Das ist kein reiner Spracheffekt: die Modelle unterscheiden sich auch in Größe, Training und der Kürzung langer Chunks (siehe Grenzen).
- hyb+rr-C2 gegenüber bm25-C2 (Recall@5): +0,171 (95-%-KI +0,113 bis +0,231) – besser.
<!-- results:end -->

Vollständiger Bericht: [reports/v1/results.md](reports/v1/results.md). Stand v0.1.0: Retrieval ist gemessen, auf einer
Windows-CPU. Antworten, Judge-Bewertungen und Kosten pro 1.000 Anfragen sind **ausstehend** (Live-Lauf, v0.2.0); der
Bericht zeigt dafür bewusst keine Zahlen.

## Entscheidungen

| Thema | Entscheidung | Warum |
|---|---|---|
| Korpus | EnWG, EEG 2023, MsbG, EnFG, KWKG 2025 von gesetze-im-internet.de | gemeinfrei (§ 5 UrhG), schwer für Retrieval, einfrierbar |
| Relevanz | Überlappung mit Gold-Absätzen | fairer Vergleich beider Chunking-Strategien ([ADR-002](docs/adr/002-span-relevance.md)) |
| Suche | bm25s mit deutschem Stemming; exakter Kosinus in NumPy | keine Datenbank nötig ([ADR-005](docs/adr/005-no-vector-database.md)) |
| Modelle | multilingual-e5-base, englisches MiniLM als Kontrolle, bge-reranker-base | laufen auf CPU, Revisionen gepinnt |
| Fusion | Reciprocal Rank Fusion, k = 60 | kein Gewicht, das auf den Testfragen getunt wird; im Lauf bringt sie gegenüber reiner Dense-Suche nichts Messbares, der Gewinn kommt vom Reranking |
| Fragen | 216 beantwortbare und 24 nicht beantwortbare Fragen, am 2026-10-25 vom implementierenden Modell geschrieben, automatisch gefiltert, 0 menschlich geprüft | das Datenblatt nennt das offen ([ADR-006](docs/adr/006-question-set.md)) |
| Judge | eigene Rubrik in zwei Aufrufen (Treue ohne Referenzantwort, Korrektheit ohne Kontext), JSON-Ausgabe, Batch-API | prüfbar statt Framework-Standardprompts ([ADR-003](docs/adr/003-own-judge.md)) |
| LLM-Aufrufe | eine Schnittstelle, Aufzeichnen und Abspielen, getrennte Kassetten je Profil | CI ohne Schlüssel ([ADR-004](docs/adr/004-provider-and-replay.md)) |

## Schnellstart

Offline, ohne Schlüssel und ohne Modelle (dasselbe wie in CI):

```bash
uv sync --locked
uv run pruefstand all --profile ci
uv run pruefstand report
```

`report` baut Bericht und Seite aus den eingecheckten Läufen neu; `uv run pruefstand metrics` auf dem veröffentlichten
Lauf braucht zusätzlich die lokalen Chunk-Dateien (`uv run pruefstand chunk`).

Vollständiger Retrieval-Lauf auf CPU (lädt etwa 2,3 GB Modelle nach `.cache/models`):

```bash
uv sync --locked --extra models
uv run pruefstand corpus fetch
uv run pruefstand models fetch
uv run pruefstand all
```

Live-Lauf mit API (kostet Geld, Prognose mit `uv run pruefstand cost estimate`, etwa 25 USD, Obergrenze 30 USD): Schlüssel
in `ANTHROPIC_API_KEY` setzen, dann `PRUEFSTAND_LLM_MODE=record` und `uv run pruefstand generate` und
`uv run pruefstand judge`. Der Lauf liest zuerst die Kassette `cassettes/full-2026-10-25.jsonl` und bezahlt nur fehlende
Aufrufe; die Batch-ID des Judges steht in `runs/full-2026-10-25/batch_judge.json`, derselbe Befehl nimmt einen
unterbrochenen Lauf wieder auf.

## Methode

Einheit ist der Absatz. C1 schneidet feste Fenster (350 Token, 50 Überlappung), C2 einen Chunk pro Absatz mit Kopfzeile.
Gemessen werden Recall@1/3/5/10, MRR@10 und nDCG@10 mit 95-%-Bootstrap-Intervallen; acht Vergleiche sind vorab
festgelegt und werden gepaart getestet. Die Hauptkonfiguration hyb+rr-C2 wurde vor dem Lauf festgelegt; sie wurde nicht
nach dem Ergebnis ausgewählt. Details: [docs/method.md](docs/method.md), Rubrik: [docs/rubric.md](docs/rubric.md).

## Daten und Lizenzen

Code: MIT. Gesetzestexte: gesetze-im-internet.de, Bundesministerium der Justiz; nichtamtliche konsolidierte Fassung,
Stand 2026-10-25. Fragensatz: CC BY 4.0, modellgeneriert am 2026-10-25; die menschliche Prüfung von 100 Fragen steht aus
([Datenblatt](data/questions/DATASHEET.md)). Alle Quellen: [DATA_LICENSES.md](DATA_LICENSES.md).

## Kosten

v0.1.0 hat keinen bezahlten API-Aufruf gemacht. Prognose für einen Live-Lauf und Preistabelle:
[docs/cost-model.md](docs/cost-model.md).

## Grenzen

- Die Fragen sind modellgeneriert (216 beantwortbare, 0 menschlich geprüft) und noch nicht von einem Menschen geprüft.
- Autor der Fragen und Primär-Judge des Live-Laufs sind dasselbe Modell (claude-sonnet-5-5); das ist offengelegt und
  wird durch einen zweiten Judge und die geprüfte Teilmenge nur begrenzt.
- Der Abstand zwischen e5 und MiniLM ist kein reiner Spracheffekt: Größe, Training und die Kürzung langer Chunks durch
  MiniLM wirken mit.
- Der Reranker ist vor allem auf Chinesisch und Englisch trainiert und kürzt lange Paare auf 384 Token; ein
  mehrsprachiger Reranker ist Roadmap.
- Alle Messungen laufen auf einer Windows-CPU ohne GPU; Latenzen gelten nur für diese Maschine.
- Ergebnisse gelten für deutsches Gesetzesdeutsch, nicht für deutsche Texte allgemein.

## Roadmap

- v0.2.0: Live-Lauf (Antworten, Judge, Kosten, Latenz der API), Prüfung von 100 Fragen, Judge-Kalibrierung mit
  gewichtetem Kappa, zweiter Judge, Wiederholungen, lokaler Generator über Ollama.
- Danach: Fehleranalyse mit fester Taxonomie, DSK-Zuordnung, Korpus K2 (Bürgerservice München) nach Lizenzklärung,
  mehrsprachiger Reranker, Veröffentlichung des Fragensatzes auf Hugging Face.
