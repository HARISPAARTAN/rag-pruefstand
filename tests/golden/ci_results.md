# RAG-Prüfstand – Ergebnisse (Lauf ci)

> Fragensatz k1_v1: 27 modellgenerierte beantwortbare Fragen, 0 menschlich geprüft (factoid 8, colloquial 6, definition 6, crossref 4, number 3); dazu 3 nicht beantwortbare Fragen. Geschrieben am 2026-10-25 von claude-sonnet-5-5; die Prüfung von 100 Fragen durch den Owner steht aus.
> Korpus: 220 Einheiten (Absätze) aus 1 Bundesgesetz(en). Quelle: gesetze-im-internet.de, Bundesministerium der Justiz; nichtamtliche konsolidierte Fassung, Stand 2026-10-25.

## Kurzfassung

- Vorab festgelegte Hauptkonfiguration hyb+rr-C2: Recall@5 0,778 (95-%-KI 0,630 bis 0,926, n = 27).
- e5-C2 gegenüber mini-C2 (Recall@5): +0,463 (95-%-KI +0,278 bis +0,648) – besser. Das ist kein reiner Spracheffekt: die Modelle unterscheiden sich auch in Größe, Training und der Kürzung langer Chunks (siehe Grenzen).
- hyb+rr-C2 gegenüber bm25-C2 (Recall@5): +0,278 (95-%-KI +0,111 bis +0,463) – besser.
- Hybrid-Fusion bringt gegenüber reiner Dense-Suche nichts Messbares: hyb-C2 gegenüber e5-C2 (Recall@5): -0,074 (95-%-KI -0,222 bis +0,074) – kein messbarer Unterschied.
- Der Abstand zu BM25 (+0,278, hyb+rr-C2 gegenüber bm25-C2) geht nach den Punktschätzern überwiegend auf das Dense-Modell zurück: e5-C2 0,704 gegenüber bm25-C2 0,500 (+0,204). Dieser Vergleich wurde vorab nicht festgelegt und hat hier kein Intervall.
- Einordnung: Den höchsten Punktschätzer für Recall@5 hat e5+rr-C1 (0,870, 95-%-KI 0,741 bis 0,981). Ein Vergleich mit der Hauptkonfiguration wurde vorab nicht festgelegt; die Hauptkonfiguration wurde vor dem Lauf bestimmt, nicht nach dem Ergebnis gewählt.

## Retrieval-Matrix (Recall@5, 95-%-KI)

| Retriever | C1 | C2 |
|---|---|---|
| bm25 | 0,537 [0,352, 0,722] | 0,500 [0,296, 0,685] |
| e5 | 0,833 [0,685, 0,963] | 0,704 [0,519, 0,852] |
| mini | 0,407 [0,241, 0,574] | 0,241 [0,093, 0,407] |
| hyb | 0,667 [0,481, 0,815] | 0,630 [0,444, 0,796] |
| e5+rr | 0,870 [0,741, 0,981] | 0,852 [0,704, 0,963] |
| hyb+rr | 0,833 [0,685, 0,963] | 0,778 [0,630, 0,926] |

![Recall@5 je Retriever und Chunking](charts/retrieval_matrix.svg)

![Recall@k, Chunking C2](charts/recall_at_k.svg)

## Alle Retrieval-Kennzahlen

| Variante | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| bm25-C1 | 0,370 [0,204, 0,537] | 0,537 [0,352, 0,722] | 0,537 [0,352, 0,722] | 0,556 [0,370, 0,722] | 0,486 [0,304, 0,652] | 0,499 [0,320, 0,670] |
| e5-C1 | 0,407 [0,222, 0,593] | 0,778 [0,630, 0,926] | 0,833 [0,685, 0,963] | 0,963 [0,889, 1,000] | 0,626 [0,489, 0,767] | 0,702 [0,589, 0,814] |
| mini-C1 | 0,111 [0,019, 0,241] | 0,296 [0,148, 0,463] | 0,407 [0,241, 0,574] | 0,574 [0,370, 0,759] | 0,279 [0,154, 0,419] | 0,335 [0,204, 0,470] |
| hyb-C1 | 0,389 [0,204, 0,556] | 0,574 [0,389, 0,741] | 0,667 [0,481, 0,815] | 0,796 [0,648, 0,944] | 0,561 [0,398, 0,712] | 0,603 [0,458, 0,742] |
| e5+rr-C1 | 0,648 [0,463, 0,815] | 0,833 [0,685, 0,963] | 0,870 [0,741, 0,981] | 0,889 [0,741, 1,000] | 0,762 [0,623, 0,889] | 0,799 [0,669, 0,918] |
| hyb+rr-C1 | 0,574 [0,389, 0,759] | 0,796 [0,630, 0,926] | 0,833 [0,685, 0,963] | 0,852 [0,704, 0,963] | 0,701 [0,552, 0,843] | 0,742 [0,600, 0,873] |
| bm25-C2 | 0,389 [0,222, 0,556] | 0,481 [0,296, 0,667] | 0,500 [0,296, 0,685] | 0,556 [0,370, 0,741] | 0,479 [0,296, 0,651] | 0,477 [0,307, 0,642] |
| e5-C2 | 0,519 [0,333, 0,704] | 0,667 [0,481, 0,833] | 0,704 [0,519, 0,852] | 0,796 [0,648, 0,926] | 0,644 [0,468, 0,796] | 0,668 [0,516, 0,803] |
| mini-C2 | 0,148 [0,037, 0,296] | 0,241 [0,093, 0,407] | 0,241 [0,093, 0,407] | 0,370 [0,204, 0,537] | 0,219 [0,095, 0,365] | 0,253 [0,123, 0,400] |
| hyb-C2 | 0,426 [0,241, 0,611] | 0,500 [0,296, 0,685] | 0,630 [0,444, 0,796] | 0,778 [0,593, 0,926] | 0,530 [0,357, 0,691] | 0,588 [0,433, 0,734] |
| e5+rr-C2 | 0,630 [0,444, 0,796] | 0,796 [0,648, 0,926] | 0,852 [0,704, 0,963] | 0,852 [0,704, 0,963] | 0,753 [0,605, 0,889] | 0,772 [0,634, 0,902] |
| hyb+rr-C2 | 0,593 [0,407, 0,760] | 0,722 [0,556, 0,870] | 0,778 [0,630, 0,926] | 0,815 [0,667, 0,963] | 0,707 [0,556, 0,856] | 0,726 [0,573, 0,867] |

nDCG@10 ist nur innerhalb eines Chunkings vergleichbar: C1 und C2 sind auf je eigene Ideale normiert. C1 und C2 werden mit Recall@k und MRR@10 verglichen.

### Kontextmenge: mittlere Token der Top-k-Chunks

| Variante | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| bm25-C1 | 350 | 1051 | 1737 | 3422 |
| e5-C1 | 350 | 1050 | 1751 | 3502 |
| mini-C1 | 350 | 1050 | 1750 | 3501 |
| hyb-C1 | 350 | 1051 | 1751 | 3501 |
| e5+rr-C1 | 350 | 1050 | 1751 | 3502 |
| hyb+rr-C1 | 350 | 1051 | 1751 | 3501 |
| bm25-C2 | 162 | 512 | 859 | 1744 |
| e5-C2 | 147 | 444 | 756 | 1572 |
| mini-C2 | 85 | 291 | 513 | 1115 |
| hyb-C2 | 153 | 506 | 823 | 1716 |
| e5+rr-C2 | 176 | 483 | 827 | 1716 |
| hyb+rr-C2 | 168 | 507 | 848 | 1816 |

Bei gleichem k enthält der Kontext mit C1 etwa das Doppelte bis Vierfache der Token wie mit C2 (e5-Tokenizer, C2 mit Kopfzeile). Die Vergleiche C1 gegen C2 sind deshalb Vergleiche bei gleichem k, nicht bei gleicher Kontextmenge.

## Vergleiche (gepaarter Bootstrap, vorab festgelegt)

| A | B | Kennzahl | Differenz | 95-%-KI | Urteil |
|---|---|---|---|---|---|
| hyb+rr-C2 | bm25-C2 | recall@5 | +0,278 | [+0,111, +0,463] | besser |
| hyb+rr-C2 | bm25-C2 | ndcg@10 | +0,248 | [+0,110, +0,410] | besser |
| e5-C2 | mini-C2 | recall@5 | +0,463 | [+0,278, +0,648] | besser |
| e5-C2 | mini-C2 | ndcg@10 | +0,415 | [+0,263, +0,565] | besser |
| hyb-C2 | e5-C2 | recall@5 | -0,074 | [-0,222, +0,074] | kein messbarer Unterschied |
| hyb-C2 | e5-C2 | ndcg@10 | -0,079 | [-0,196, +0,020] | kein messbarer Unterschied |
| e5+rr-C2 | e5-C2 | recall@5 | +0,148 | [-0,019, +0,315] | kein messbarer Unterschied |
| e5+rr-C2 | e5-C2 | ndcg@10 | +0,104 | [-0,043, +0,255] | kein messbarer Unterschied |
| hyb+rr-C2 | hyb-C2 | recall@5 | +0,148 | [+0,000, +0,315] | kein messbarer Unterschied |
| hyb+rr-C2 | hyb-C2 | ndcg@10 | +0,137 | [+0,033, +0,270] | besser |
| bm25-C2 | bm25-C1 | recall@5 | -0,037 | [-0,111, +0,000] | kein messbarer Unterschied |
| e5-C2 | e5-C1 | recall@5 | -0,130 | [-0,315, +0,037] | kein messbarer Unterschied |
| hyb+rr-C2 | hyb+rr-C1 | recall@5 | -0,056 | [-0,241, +0,130] | kein messbarer Unterschied |

## Recall@5 nach Fragetyp

| Variante | colloquial | crossref | definition | factoid | number |
|---|---|---|---|---|---|
| bm25-C1 | 0,167 [0,000, 0,500] | 0,625 [0,250, 1,000] | 0,000 [0,000, 0,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| e5-C1 | 0,500 [0,167, 0,833] | 0,875 [0,625, 1,000] | 0,833 [0,500, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| mini-C1 | 0,167 [0,000, 0,500] | 0,500 [0,125, 0,875] | 0,333 [0,000, 0,667] | 0,500 [0,125, 0,875] | 0,667 [0,000, 1,000] |
| hyb-C1 | 0,500 [0,167, 0,833] | 0,750 [0,500, 1,000] | 0,167 [0,000, 0,500] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| e5+rr-C1 | 0,500 [0,167, 0,833] | 0,875 [0,625, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| hyb+rr-C1 | 0,500 [0,167, 0,833] | 0,875 [0,625, 1,000] | 0,833 [0,500, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| bm25-C2 | 0,000 [0,000, 0,000] | 0,625 [0,250, 1,000] | 0,000 [0,000, 0,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| e5-C2 | 0,500 [0,167, 0,833] | 0,750 [0,500, 1,000] | 0,333 [0,000, 0,667] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| mini-C2 | 0,167 [0,000, 0,500] | 0,125 [0,000, 0,375] | 0,000 [0,000, 0,000] | 0,500 [0,125, 0,875] | 0,333 [0,000, 1,000] |
| hyb-C2 | 0,167 [0,000, 0,500] | 0,750 [0,500, 1,000] | 0,333 [0,000, 0,667] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| e5+rr-C2 | 0,333 [0,000, 0,667] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| hyb+rr-C2 | 0,167 [0,000, 0,500] | 1,000 [1,000, 1,000] | 0,833 [0,500, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |

## Latenz pro Stufe (ms, p50 / p95)

In diesem Profil nicht gemessen.

## Generierung und Bewertung

**Synthetische Fixtures – keine Messung.** Diese Werte stammen aus handgeschriebenen CI-Fixtures und prüfen nur die Pipeline.

| Variante | Treue | Korrektheit | Zitat-Präzision | Enthaltung | USD/1.000 | EUR/1.000 | p50 ms |
|---|---|---|---|---|---|---|---|
| bm25-C1 | 1,375 [0,750, 1,875] | 1,125 [0,625, 1,625] | 1,000 [1,000, 1,000] | 0,750 [0,375, 1,000] | 0,3894 | 0,3479 | 1072 |
| hyb+rr-C2 | 1,875 [1,625, 2,000] | 1,750 [1,250, 2,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 0,3338 | 0,2982 | 942 |

## Kalibrierung des Judges

**Ausstehend: Live-Lauf und menschliche Labels.** Gewichtetes Cohens Kappa und Rohübereinstimmung werden auf 80 vom Owner gelabelten Antworten berechnet.

## Methode und Grenzen

- Relevanz über Quellspannen, auf zwei Ebenen. Gold-Absätze bis 2.000 Zeichen (Einheitsebene): ein Chunk zählt, wenn er mindestens 200 Zeichen oder 50 % des Absatzes abdeckt, je nachdem, was weniger ist. Längere Gold-Absätze (vor allem Definitionsnormen, Passagenebene): ein Chunk zählt nur, wenn er einen Auszug der Gold-Passage um mindestens 200 Zeichen oder 50 % des Auszugs überlappt; ein anderer Eintrag derselben langen Liste zählt nicht. 8 von 27 beantwortbaren Fragen werden auf der Passage bewertet.
- nDCG@10 ist nur innerhalb eines Chunkings vergleichbar: der Gewinn eines Rangs ist der Anteil der Gold-Einheiten, den der Chunk erstmals abdeckt; das Ideal ist die gierige Minimalüberdeckung der Gold-Einheiten durch die Chunks desselben Chunkings, die Kennzahl ist auf höchstens 1 begrenzt. Zwischen C1 und C2 werden Recall@k und MRR@10 verglichen.
- Intervalle: Perzentil-Bootstrap über Fragen (2000 Ziehungen, Seed 20261025); enthält das Intervall einer Differenz die Null, heißt es „kein messbarer Unterschied“.
- Die Tabelle der Vergleiche hat 13 Zeilen aus 8 vorab festgelegten Paaren und ist nicht für mehrfaches Testen korrigiert. Die Fragen werden als unabhängig neu gezogen, obwohl einige Fragen Gold-Absätze teilen; die Intervalle können dadurch zu eng sein. Ein Intervall wie [1,000, 1,000] (alle Fragen eines Typs getroffen, wenige Fragen) ist keine Gewissheit.
- Die Fragen sind modellgeneriert und noch von keinem Menschen geprüft; die Prüfung von 100 Fragen durch den Owner steht aus.
- Autor der Fragen und Primär-Judge des Live-Laufs sind dasselbe Modell (claude-sonnet-5-5); Selbstpräferenz ist möglich. Ein zweiter Judge (claude-opus-5-5) und die menschlich geprüfte Teilmenge begrenzen den Effekt, beseitigen ihn nicht. Die Retrieval-Kennzahlen brauchen keinen Judge.
- Die Hauptkonfiguration hyb+rr-C2 wurde vor dem Lauf festgelegt; die Fusion (Reciprocal Rank Fusion, k = 60) hat kein auf den Testfragen getuntes Gewicht.
- Der Abstand zwischen multilingual-e5-base und all-MiniLM-L6-v2 ist kein reiner Spracheffekt: die Modelle unterscheiden sich auch in Größe, Trainingsdaten und Präfix-Konvention, und MiniLM kürzt lange Chunks. MiniLM ist eine Kontrolle, keine Messung des Sprachvorteils.
- Der Reranker ist bge-reranker-base (vor allem auf Chinesisch und Englisch trainiert); ein mehrsprachiger Reranker ist Roadmap.
- Alle Modelle laufen auf der CPU (keine GPU); Latenzen gelten nur für die genannte Maschine.

Repository: https://github.com/HARISPAARTAN/rag-pruefstand
