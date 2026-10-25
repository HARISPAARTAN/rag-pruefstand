# Question authoring protocol k1_v1

This file is the instruction the authoring agent followed to write the 240 questions of `k1_v1`. Its SHA-256 is stored
in every question's `provenance.prompt_sha256`, so any change to it after `questions assemble` breaks the link between
the data and the method. Do not edit it after the set is assembled; write `k1_v2` with a new protocol instead.

## Who writes the questions

The questions are written by the implementing agent (a Claude model) at build time, while it reads the law texts with
the CLI. No API call is made for this; the agent's exact model id is recorded in `provenance.model`. The questions are
model-generated, not human-written. A human check of 100 questions by the owner is planned and recorded per question in
`review.status`; until then every record says `pending`.

## Tools

- `uv run pruefstand show "<unit id prefix>"` prints the full text of units, for example `show "messbg:§5:"`.
- `uv run pruefstand search "<text>" --law <slug>` lists the best BM25 matches; use it to find units and to check that
  an "unanswerable" question really has no answer in any of the five laws (search without `--law` as well).
- `uv run pruefstand questions check <batch file>` validates a batch: fields, verbatim gold passage, filters.

Law slugs: `enwg_2005` (EnWG), `eeg_2014` (EEG 2023), `messbg` (MsbG), `enfg` (EnFG), `kwkg_2016` (KWKG 2025).
Unit ids look like `enwg_2005:§14a:abs1`; unnumbered paragraphs use `abs0`, for example `messbg:§2:abs0`.

## Record format

One JSON object per line, UTF-8, these eight fields and no others:

| Field | Content |
|---|---|
| `type` | `factoid`, `definition`, `colloquial`, `crossref`, `number` or `unanswerable` |
| `question_de` | the question in German, ending with `?` |
| `question_en` | a faithful English translation (reviewing aid only) |
| `gold_unit_ids` | ids of the units that contain the answer; empty for `unanswerable` |
| `gold_passage_de` | verbatim excerpt(s) of the gold units that support the answer, several joined by ` … `; empty for `unanswerable` |
| `gold_answer_de` | a short German answer (one or two sentences) in your own words |
| `gold_answer_en` | a faithful English translation of the answer |
| `source_unit_ids` | the units you read to write the question (for `unanswerable`: the neighbouring topic) |

Example (factoid, MsbG):

```json
{"type": "factoid", "question_de": "Muss man für einen Wechsel des Messstellenbetreibers extra bezahlen?", "question_en": "Does switching the metering point operator cost an extra fee?", "gold_unit_ids": ["messbg:§14:abs3"], "gold_passage_de": "Für den Wechsel des Messstellenbetreibers darf kein gesondertes Entgelt erhoben werden.", "gold_answer_de": "Nein, für den Wechsel darf kein gesondertes Entgelt erhoben werden.", "gold_answer_en": "No, no separate fee may be charged for the switch.", "source_unit_ids": ["messbg:§14:abs3"]}
```

Example (unanswerable):

```json
{"type": "unanswerable", "question_de": "Wie hoch ist das Bußgeld für einen manipulierten Stromzähler in Bayern?", "question_en": "How high is the fine for a tampered electricity meter in Bavaria?", "gold_unit_ids": [], "gold_passage_de": "", "gold_answer_de": "Die Frage lässt sich mit den vorliegenden Quellen nicht beantworten.", "gold_answer_en": "The question cannot be answered from the available sources.", "source_unit_ids": ["messbg:§3:abs2"]}
```

## Question types

- `factoid`: one fact stated in one Absatz (who must do what, under which condition). Use your own wording.
- `definition`: what a legal term means, answered by a definition norm (for example § 3 EnWG, § 3 EEG 2023, § 2 MsbG,
  § 2 EnFG, § 2 KWKG). Ask about the term, not about its number in the list.
- `colloquial`: the question a citizen or a small business would ask, in everyday German, avoiding the legal terms of the
  gold unit. Example: "Darf mein Netzbetreiber meine Wallbox drosseln?" for a rule on controllable consumption devices.
  The filter rejects a colloquial question when more than 50% of its content words share a five-letter prefix with the
  gold unit text.
- `crossref`: the answer needs at least two different units, for example a rule plus the definition it relies on, or a
  norm that refers to another norm ("nach § 14a Absatz 1"). List every needed unit in `gold_unit_ids` and give one
  excerpt per unit in `gold_passage_de`.
- `number`: a deadline, amount, percentage, threshold or count. The number in the answer must appear in the gold passage
  in the same notation (the filter compares digit strings).
- `unanswerable`: a plausible question from the same domain that none of the five laws answers (regional rules, prices,
  case law, other laws, future plans). Search the whole corpus before you keep it. `gold_unit_ids` is `[]`,
  `gold_passage_de` is `""` and `gold_answer_de` is exactly
  `Die Frage lässt sich mit den vorliegenden Quellen nicht beantworten.`

## Rules for every answerable question

1. Gold passage: copy the excerpt from the output of `show`; it must appear verbatim in a gold unit (whitespace may
   differ). Keep it as short as possible while it still contains every fact of the answer.
2. Gold answer: short, factual, in your own words, fully supported by the passage. At least half of its content words
   must share a five-letter prefix with the passage (support filter), and every number must appear in the passage.
3. No copying: at most 60% of the question's word trigrams may appear in the source text (copy filter). Rephrase.
4. No duplicates: two questions with a content-word Jaccard similarity above 0.8 are rejected.
5. Spread: cover the whole law, not only its first sections; at most three questions per §; prefer norms that a
   practitioner would actually look up (obligations, deadlines, definitions, rights of customers and operators).
6. Neutral content: no personal data, no real persons or companies, no requests for legal advice on a specific case,
   no facts that are not in the text (prices, dates of later amendments, court decisions).
7. The English fields translate the German ones faithfully; they are never used for retrieval or scoring.

## Targets

Exact counts per law and type (240 questions); `questions check --targets` enforces them.

| Law | factoid | definition | colloquial | crossref | number | unanswerable | Total |
|---|---|---|---|---|---|---|---|
| enwg_2005 | 24 | 19 | 19 | 15 | 10 | 9 | 96 |
| eeg_2014 | 15 | 12 | 12 | 9 | 6 | 6 | 60 |
| messbg | 8 | 6 | 6 | 4 | 3 | 3 | 30 |
| enfg | 7 | 6 | 5 | 4 | 2 | 3 | 27 |
| kwkg_2016 | 6 | 5 | 6 | 4 | 3 | 3 | 27 |
| Total | 60 | 48 | 48 | 36 | 24 | 24 | 240 |

## Batches

Write the questions in batch files `data/questions/batches/<law slug>_<nn>.jsonl` (for example `enwg_2005_01.jsonl`),
at most 24 records per file, all records of a file from the same law. Run `questions check` on each file before you
start the next one and fix every reported problem in place. When all files pass, run `questions check --targets`, then
`questions assemble`, which numbers the questions `k1v1-0001` to `k1v1-0240` in file order and adds provenance.
