# RAG-Prüfstand – Ergebnisse (Lauf full-2026-10-25)

> Fragensatz k1_v1: 216 modellgenerierte beantwortbare Fragen, 0 menschlich geprüft (factoid 60, colloquial 48, definition 48, crossref 36, number 24); dazu 24 nicht beantwortbare Fragen. Geschrieben am 2026-10-25 von claude-sonnet-5-5; die Prüfung von 100 Fragen durch den Owner steht aus.
> Korpus: 2523 Einheiten (Absätze) aus 5 Bundesgesetz(en). Quelle: gesetze-im-internet.de, Bundesministerium der Justiz; nichtamtliche konsolidierte Fassung, Stand 2026-10-25.

## Kurzfassung

- Vorab festgelegte Hauptkonfiguration hyb+rr-C2: Recall@5 0,706 (95-%-KI 0,649 bis 0,762, n = 216).
- e5-C2 gegenüber mini-C2 (Recall@5): +0,408 (95-%-KI +0,341 bis +0,474) – besser. Das ist kein reiner Spracheffekt: die Modelle unterscheiden sich auch in Größe, Training und der Kürzung langer Chunks (siehe Grenzen).
- hyb+rr-C2 gegenüber bm25-C2 (Recall@5): +0,171 (95-%-KI +0,113 bis +0,231) – besser.
- Hybrid-Fusion bringt gegenüber reiner Dense-Suche nichts Messbares: hyb-C2 gegenüber e5-C2 (Recall@5): +0,005 (95-%-KI -0,044 bis +0,053) – kein messbarer Unterschied.
- Der Gewinn kommt vom Reranking: e5+rr-C2 gegenüber e5-C2 +0,076 [+0,021, +0,137]; hyb+rr-C2 gegenüber hyb-C2 +0,059 [+0,006, +0,110] (Recall@5, 95-%-KI).
- Einordnung: Den höchsten Punktschätzer für Recall@5 hat e5+rr-C2 (0,718, 95-%-KI 0,661 bis 0,774). Ein Vergleich mit der Hauptkonfiguration wurde vorab nicht festgelegt; die Hauptkonfiguration wurde vor dem Lauf bestimmt, nicht nach dem Ergebnis gewählt.

## Retrieval-Matrix (Recall@5, 95-%-KI)

| Retriever | C1 | C2 |
|---|---|---|
| bm25 | 0,560 [0,497, 0,622] | 0,535 [0,473, 0,599] |
| e5 | 0,651 [0,589, 0,714] | 0,642 [0,583, 0,702] |
| mini | 0,289 [0,233, 0,347] | 0,234 [0,183, 0,292] |
| hyb | 0,665 [0,604, 0,725] | 0,647 [0,586, 0,707] |
| e5+rr | 0,682 [0,620, 0,738] | 0,718 [0,661, 0,774] |
| hyb+rr | 0,698 [0,635, 0,752] | 0,706 [0,649, 0,762] |

![Recall@5 je Retriever und Chunking](charts/retrieval_matrix.svg)

![Recall@k, Chunking C2](charts/recall_at_k.svg)

## Alle Retrieval-Kennzahlen

| Variante | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| bm25-C1 | 0,316 [0,258, 0,374] | 0,475 [0,413, 0,539] | 0,560 [0,497, 0,622] | 0,653 [0,594, 0,712] | 0,454 [0,398, 0,512] | 0,488 [0,434, 0,541] |
| e5-C1 | 0,344 [0,284, 0,405] | 0,563 [0,504, 0,627] | 0,651 [0,589, 0,714] | 0,716 [0,658, 0,775] | 0,510 [0,456, 0,568] | 0,546 [0,494, 0,598] |
| mini-C1 | 0,083 [0,050, 0,117] | 0,210 [0,161, 0,263] | 0,289 [0,233, 0,347] | 0,363 [0,299, 0,424] | 0,191 [0,149, 0,235] | 0,220 [0,178, 0,262] |
| hyb-C1 | 0,410 [0,351, 0,472] | 0,592 [0,528, 0,654] | 0,665 [0,604, 0,725] | 0,745 [0,688, 0,803] | 0,563 [0,506, 0,622] | 0,594 [0,539, 0,648] |
| e5+rr-C1 | 0,429 [0,367, 0,489] | 0,618 [0,556, 0,679] | 0,682 [0,620, 0,738] | 0,752 [0,695, 0,808] | 0,579 [0,522, 0,634] | 0,607 [0,553, 0,658] |
| hyb+rr-C1 | 0,428 [0,367, 0,490] | 0,607 [0,539, 0,664] | 0,698 [0,635, 0,752] | 0,771 [0,715, 0,824] | 0,584 [0,528, 0,639] | 0,616 [0,563, 0,666] |
| bm25-C2 | 0,353 [0,291, 0,414] | 0,465 [0,403, 0,529] | 0,535 [0,473, 0,599] | 0,600 [0,538, 0,662] | 0,475 [0,414, 0,535] | 0,485 [0,428, 0,541] |
| e5-C2 | 0,441 [0,376, 0,502] | 0,573 [0,510, 0,633] | 0,642 [0,583, 0,702] | 0,732 [0,675, 0,787] | 0,574 [0,517, 0,631] | 0,593 [0,540, 0,644] |
| mini-C2 | 0,116 [0,076, 0,160] | 0,204 [0,153, 0,259] | 0,234 [0,183, 0,292] | 0,301 [0,243, 0,361] | 0,185 [0,141, 0,232] | 0,206 [0,162, 0,253] |
| hyb-C2 | 0,427 [0,364, 0,486] | 0,562 [0,501, 0,625] | 0,647 [0,586, 0,707] | 0,752 [0,698, 0,806] | 0,565 [0,507, 0,621] | 0,595 [0,541, 0,645] |
| e5+rr-C2 | 0,535 [0,472, 0,600] | 0,677 [0,617, 0,734] | 0,718 [0,661, 0,774] | 0,769 [0,715, 0,821] | 0,667 [0,609, 0,721] | 0,673 [0,620, 0,722] |
| hyb+rr-C2 | 0,522 [0,458, 0,585] | 0,678 [0,618, 0,735] | 0,706 [0,649, 0,762] | 0,777 [0,724, 0,828] | 0,658 [0,602, 0,715] | 0,668 [0,617, 0,719] |

## Vergleiche (gepaarter Bootstrap, vorab festgelegt)

| A | B | Kennzahl | Differenz | 95-%-KI | Urteil |
|---|---|---|---|---|---|
| hyb+rr-C2 | bm25-C2 | recall@5 | +0,171 | [+0,113, +0,231] | besser |
| hyb+rr-C2 | bm25-C2 | ndcg@10 | +0,184 | [+0,140, +0,230] | besser |
| e5-C2 | mini-C2 | recall@5 | +0,408 | [+0,341, +0,474] | besser |
| e5-C2 | mini-C2 | ndcg@10 | +0,387 | [+0,337, +0,436] | besser |
| hyb-C2 | e5-C2 | recall@5 | +0,005 | [-0,044, +0,053] | kein messbarer Unterschied |
| hyb-C2 | e5-C2 | ndcg@10 | +0,002 | [-0,034, +0,039] | kein messbarer Unterschied |
| e5+rr-C2 | e5-C2 | recall@5 | +0,076 | [+0,021, +0,137] | besser |
| e5+rr-C2 | e5-C2 | ndcg@10 | +0,080 | [+0,039, +0,122] | besser |
| hyb+rr-C2 | hyb-C2 | recall@5 | +0,059 | [+0,006, +0,110] | besser |
| hyb+rr-C2 | hyb-C2 | ndcg@10 | +0,074 | [+0,038, +0,110] | besser |
| bm25-C2 | bm25-C1 | recall@5 | -0,025 | [-0,072, +0,023] | kein messbarer Unterschied |
| e5-C2 | e5-C1 | recall@5 | -0,009 | [-0,067, +0,046] | kein messbarer Unterschied |
| hyb+rr-C2 | hyb+rr-C1 | recall@5 | +0,008 | [-0,045, +0,067] | kein messbarer Unterschied |

## Recall@5 nach Fragetyp

| Variante | colloquial | crossref | definition | factoid | number |
|---|---|---|---|---|---|
| bm25-C1 | 0,167 [0,062, 0,271] | 0,667 [0,542, 0,778] | 0,375 [0,229, 0,500] | 0,800 [0,700, 0,900] | 0,958 [0,875, 1,000] |
| e5-C1 | 0,438 [0,312, 0,583] | 0,685 [0,565, 0,792] | 0,521 [0,375, 0,667] | 0,783 [0,667, 0,883] | 0,958 [0,875, 1,000] |
| mini-C1 | 0,083 [0,021, 0,167] | 0,287 [0,181, 0,398] | 0,250 [0,125, 0,375] | 0,350 [0,233, 0,467] | 0,625 [0,458, 0,792] |
| hyb-C1 | 0,396 [0,250, 0,521] | 0,769 [0,657, 0,866] | 0,438 [0,312, 0,583] | 0,867 [0,767, 0,950] | 1,000 [1,000, 1,000] |
| e5+rr-C1 | 0,500 [0,375, 0,646] | 0,704 [0,583, 0,815] | 0,583 [0,438, 0,708] | 0,767 [0,650, 0,867] | 1,000 [1,000, 1,000] |
| hyb+rr-C1 | 0,438 [0,292, 0,583] | 0,741 [0,625, 0,843] | 0,604 [0,458, 0,750] | 0,833 [0,733, 0,917] | 1,000 [1,000, 1,000] |
| bm25-C2 | 0,167 [0,062, 0,271] | 0,542 [0,440, 0,639] | 0,229 [0,104, 0,354] | 0,883 [0,800, 0,967] | 1,000 [1,000, 1,000] |
| e5-C2 | 0,542 [0,396, 0,667] | 0,519 [0,417, 0,625] | 0,354 [0,229, 0,500] | 0,900 [0,817, 0,967] | 0,958 [0,875, 1,000] |
| mini-C2 | 0,104 [0,021, 0,188] | 0,208 [0,111, 0,319] | 0,021 [0,000, 0,062] | 0,350 [0,233, 0,467] | 0,667 [0,458, 0,833] |
| hyb-C2 | 0,375 [0,250, 0,500] | 0,690 [0,569, 0,792] | 0,333 [0,208, 0,458] | 0,950 [0,883, 1,000] | 1,000 [1,000, 1,000] |
| e5+rr-C2 | 0,500 [0,354, 0,646] | 0,699 [0,593, 0,792] | 0,521 [0,375, 0,667] | 0,950 [0,883, 1,000] | 1,000 [1,000, 1,000] |
| hyb+rr-C2 | 0,458 [0,312, 0,604] | 0,708 [0,606, 0,801] | 0,500 [0,354, 0,646] | 0,950 [0,883, 1,000] | 1,000 [1,000, 1,000] |

## Latenz pro Stufe (ms, p50 / p95)

Gemessen auf 50 Fragen, CPU: AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD.

| Variante | Embedding | Suche | Rerank | Gesamt |
|---|---|---|---|---|
| bm25-C1 | 0,0 / 0,0 | 0,3 / 0,3 | 0,0 / 0,0 | 0,3 / 0,3 |
| e5-C1 | 23,0 / 27,1 | 0,5 / 0,6 | 0,0 / 0,0 | 23,5 / 27,6 |
| mini-C1 | 6,0 / 7,6 | 0,4 / 0,5 | 0,0 / 0,0 | 6,4 / 8,1 |
| hyb-C1 | 23,0 / 27,1 | 0,8 / 1,0 | 0,0 / 0,0 | 23,8 / 28,0 |
| e5+rr-C1 | 23,0 / 27,1 | 0,5 / 0,6 | 1841,9 / 1888,0 | 1867,7 / 1913,4 |
| hyb+rr-C1 | 23,0 / 27,1 | 0,8 / 1,0 | 1821,8 / 1891,2 | 1845,3 / 1916,7 |
| bm25-C2 | 0,0 / 0,0 | 0,3 / 0,4 | 0,0 / 0,0 | 0,3 / 0,4 |
| e5-C2 | 22,5 / 26,3 | 0,7 / 0,8 | 0,0 / 0,0 | 23,2 / 27,1 |
| mini-C2 | 5,5 / 6,5 | 0,5 / 0,6 | 0,0 / 0,0 | 6,0 / 7,0 |
| hyb-C2 | 22,5 / 26,3 | 1,1 / 1,2 | 0,0 / 0,0 | 23,6 / 27,5 |
| e5+rr-C2 | 22,5 / 26,3 | 0,7 / 0,8 | 1597,5 / 1681,9 | 1629,1 / 1705,9 |
| hyb+rr-C2 | 22,5 / 26,3 | 1,1 / 1,2 | 1591,2 / 1671,1 | 1617,6 / 1695,5 |

![Latenz gesamt (ms), Chunking C2](charts/latency.svg)

## Generierung und Bewertung

**Ausstehend: Live-Lauf.** Antworten, Judge-Bewertungen, Kosten pro 1.000 Anfragen und API-Latenz werden erst mit dem Live-Lauf (v0.2.0) gemessen. Hier stehen bewusst keine Zahlen.

## Kalibrierung des Judges

**Ausstehend: Live-Lauf und menschliche Labels.** Gewichtetes Cohens Kappa und Rohübereinstimmung werden auf 80 vom Owner gelabelten Antworten berechnet.

## Methode und Grenzen

- Relevanz über Quellspannen: ein Chunk zählt, wenn er mindestens 50 % oder 200 Zeichen eines Gold-Absatzes abdeckt.
- Intervalle: Perzentil-Bootstrap über Fragen (2000 Ziehungen, Seed 20261025); Unterschiede innerhalb des Intervalls heißen „kein messbarer Unterschied“.
- Die Fragen sind modellgeneriert und noch von keinem Menschen geprüft; die Prüfung von 100 Fragen durch den Owner steht aus.
- Autor der Fragen und Primär-Judge des Live-Laufs sind dasselbe Modell (claude-sonnet-5-5); Selbstpräferenz ist möglich. Ein zweiter Judge (claude-opus-5-5) und die menschlich geprüfte Teilmenge begrenzen den Effekt, beseitigen ihn nicht. Die Retrieval-Kennzahlen brauchen keinen Judge.
- Die Hauptkonfiguration hyb+rr-C2 wurde vor dem Lauf festgelegt; die Fusion (Reciprocal Rank Fusion, k = 60) hat kein auf den Testfragen getuntes Gewicht.
- Der Abstand zwischen multilingual-e5-base und all-MiniLM-L6-v2 ist kein reiner Spracheffekt: die Modelle unterscheiden sich auch in Größe, Trainingsdaten und Präfix-Konvention, und MiniLM kürzt lange Chunks. MiniLM ist eine Kontrolle, keine Messung des Sprachvorteils.
- Der Reranker ist bge-reranker-base (vor allem auf Chinesisch und Englisch trainiert); ein mehrsprachiger Reranker ist Roadmap.
- Alle Modelle laufen auf der CPU (keine GPU); Latenzen gelten nur für die genannte Maschine.
- Plattform: Windows-11-10.0.26200-SP0, CPU AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, Python 3.13.11; keine GPU.
- MiniLM-Kürzung: all-MiniLM-L6-v2 liest höchstens 512 Token; bei C1 verlieren 92,3 % der Chunks ihr Ende (sichtbarer Anteil im Mittel 92,1 %), bei C2 9,4 % (99,2 %). multilingual-e5-base kürzt 0,0 % (C1) bzw. 0,0 % (C2) der Chunks.
- Reranker-Kürzung: bge-reranker-base liest höchstens 384 Token je (Frage, Chunk)-Paar; bei C1 verlieren 15,1 % der neu bewerteten Paare ihr Ende, bei C2 4,3 %. Die Wirkung auf die Ergebnisse wurde nicht gemessen.

Repository: https://github.com/HARISPAARTAN/rag-pruefstand
