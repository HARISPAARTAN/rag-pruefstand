# Datasheet: RAG-Prüfstand question set k1_v1

## Summary

240 German questions about five federal energy laws (EnWG, EEG 2023, MsbG, EnFG, KWKG 2025), each with a type, the ids
of the gold units (Absätze) that answer it, a verbatim gold passage, a short gold answer, English renderings for review
and full provenance. File: `data/questions/k1_v1.jsonl` (one JSON object per line). Licence: CC BY 4.0 (see
`LICENSE.md` in this folder).

## Status of the human check

**Pending.** The questions were model-generated on 2026-10-26 with Claude Code by the authoring agent `claude-sonnet-5-5` (a Claude model
acting as the implementing agent of this repository) while it read the law texts. No question has been checked by a
human yet. A stratified sample of 100 questions is in `review/k1_v1_review.csv`: all 24 unanswerable questions plus 76
answerable ones, drawn by type in proportion to the type sizes (21 factoid, 17 definition, 17 colloquial, 13 crossref,
8 number). The owner's verdicts (accept, edit, reject) will be imported with `pruefstand questions review-import` and
recorded per question in `review`. Until then, treat every question and gold answer as unverified. Results are reported
on all 240 questions and, once the check is done, separately on the checked subset.

## Author and judge are the same model

All 240 questions, gold answers and gold passages were written by `claude-sonnet-5-5`. The configuration
(`pruefstand.toml`, `[llm.judge]`) names the same model as the primary answer judge of the planned live run. Self-preference
and shared blind spots between author and judge are therefore possible: the judge may rate answers more favourably that
resemble the gold answers it would have written itself. Two measures bound this effect. The second judge in the live
run, `claude-opus-5-5`, is a different model. The human-checked subset (the 100-question sample above) is reported
separately once the owner's verdicts are in. Neither measure removes the overlap; judge scores must be read with it in
mind. Retrieval metrics do not involve a judge.

## Motivation

The set measures retrieval and answer quality of RAG pipelines on German legal text: compound words, cross-references
("nach § 14a Absatz 1"), definitions in long lists, and citizen wording that differs from legal wording.

## Composition

| Type | Share | Meaning |
|---|---|---|
| factoid | 25% (60) | one fact from one Absatz |
| definition | 20% (48) | meaning of a legal term from a definition norm |
| colloquial | 20% (48) | everyday wording that avoids the legal terms |
| crossref | 15% (36) | answer needs passages from at least two different norms |
| number | 10% (24) | deadline, amount, percentage or threshold |
| unanswerable | 10% (24) | plausible, but no answer in the five laws |

Per law: EnWG 96, EEG 2023 60, MsbG 30, EnFG 27, KWKG 2025 27. The 30 MsbG questions are meant to serve as the offline CI
subset (`fixtures/ci/questions.jsonl`, added together with the CI fixtures).

### Crossref questions

The 36 crossref questions need passages from at least two different norms (section level, for example EnWG § 12a and
§ 12b); the schema check rejects two Absätze of the same section. Six of them reach across laws (0054, 0083, 0128, 0152,
0154, 0207; for example EnWG § 17 Abs. 2a with EEG § 8 and KWKG § 3 on connection priority). The typical shape is a rule
that refers to a definition or to another norm, an exception that sits in a different section, or a deadline in one
norm and its consequence in another. These are two-passage questions, not a measurement of "reference following" in
a strict sense: most can be answered by a retriever that finds both passages independently, and a few (0107, 0221, 0236)
are one list or one process spread over two norms. Conclusions about cross-reference handling should be drawn
carefully from this type.

### Unanswerable questions

The 24 unanswerable questions carry the fixed answer "Die Frage lässt sich mit den vorliegenden Quellen nicht
beantworten." and no gold unit. They contain no recency or locality cue ("derzeit", "im letzten Jahr", a named city or
state) and none starts with "Wie viele"; their length and phrasing follow the answerable questions. Fifteen are hard
negatives, that is plausible questions about the five laws that the laws do not answer (a rate the law delegates to an
ordinance, a number that is only published elsewhere, a fee or compensation the law does not regulate): 0019, 0020,
0059, 0060, 0074, 0087, 0110, 0111, 0158, 0159, 0182, 0209, 0210, 0225, 0239. The other nine ask for case law (0039,
0086, 0134), for rules of another law (0040, 0135, 0181, 0183, 0240) or for a legislative plan (0197). Each was checked
with key-term searches (and, for most, BM25 search) over all five laws; the check is not exhaustive and an owner review may find a
partial answer. The set contains no counterfactual questions. Abstention rates measured on these 24 questions are
estimates for model-written negatives, not for real user questions.

### Definition units and the relevance rule

The 48 definition questions point at five definition norms that the corpus keeps as one unit each: EnWG § 3 (32,331
characters), EEG § 3 (16,954), KWKG § 2 (12,090), MsbG § 2 (7,812) and EnFG § 2 (6,105). Counting every question that
uses one of them as a gold unit, EnWG § 3 serves 21 questions, EEG § 3 15, MsbG § 2 7, EnFG § 2 6 and KWKG § 2 5. A
retrieved chunk that merely comes from such a list must not count as a hit for a different definition, so the bench uses
passage-based relevance for long units:

- Gold unit of at most 2,000 characters: a chunk is relevant if it covers at least min(200 characters, 50% of the unit).
- Gold unit longer than 2,000 characters (11 units; 62 questions use at least one: 48 definition, 10 crossref, 3
  colloquial and 1 number question): every excerpt of `gold_passage_de` is located in the
  unit text, and a chunk is relevant only if it overlaps one excerpt by at least min(200 characters, 50% of the excerpt).

The rule is implemented in `pruefstand.metrics` and `pruefstand.bench`, applied to all metrics (recall, MRR, nDCG and
the ideal gains), and recorded in `retrieval_meta.json` of every run and in the metrics file. Before this rule the
review measured a 8 to 13 point inflation of recall@5 on definition questions with BM25. Definition questions remain
harder than the rest of the set because the passage sits in a 5,000 to 32,000 character list.

### Naming the law

57 of the 240 questions name a law in the question text (37 of the 48 definition questions, because terms such as
"Letztverbraucher", "Unternehmen" or "Netzentnahme" are defined in several laws). The other 183 do not; they rely on
topic words that are, in the large majority, specific to one law. The author checked the definition questions for terms
defined in more than one law; questions outside this check can still touch a topic that a second law mentions.

## Collection process

- Source texts: XML downloads from gesetze-im-internet.de, fetched on 2026-10-25 (hashes in
  `data/corpus/snapshot.json`), parsed into one unit per Absatz.
- Method: the authoring agent followed `docs/question-authoring.md` (its SHA-256 is in each record's provenance) and
  wrote the questions in batches per law in `data/questions/batches/`. No API call and no separate generator model.
- Automatic checks, all passed by every record (results stored in `provenance.filters`): gold passage verbatim in the
  gold units, every gold unit holding at least one excerpt of at least 20 characters; crossref gold units from at
  least two different sections; word-trigram copy ratio of the question at most 0.6; colloquial questions share at most
  50% of content words with the legal wording; gold answer at least 50% supported by the passage and every number in it
  present in the passage; content-word Jaccard to every earlier question at most 0.8.
- Revision: the set was assembled once (commit 2055b6b) and revised before any published run after an independent
  review (commit ea2d39c for the relevance rule, 3a37031 for the questions). The revision rewrote 32 of the 36 crossref
  questions (0107, 0194, 0221 and 0236 stay), 23 of the 24 unanswerable questions (0210 stays), fixed the five
  questions the review found unsound or ambiguous (0066, 0152, 0162, 0172, 0232) and the minor wording issues it listed
  (0082, 0091, 0150, 0163, 0180), and added the law name to 19 questions. The protocol file was not changed, so its hash
  is the same in all records. The checks on sections and excerpts were added to the tooling after the protocol was
  frozen; they are described here, not in the protocol.

## Known limitations

- There is no separate difficulty rating; the question type is the stratification variable, and the per-law and per-type
  counts are fixed by the authoring protocol.
- Model-written questions can mirror the wording of the source more than real user questions do; the colloquial type
  and the copy filter reduce this but do not remove it.
- The duplicate filter is lexical (Jaccard), not semantic; near-paraphrases with different words can remain. Several
  units are gold for two questions each.
- Gold answers are short summaries; a correct but more detailed answer can be judged differently.
- The legal texts are a non-official consolidation dated 2026-10-25; later amendments can make answers outdated.
- Author and judge are the same model (see above); the unanswerable and crossref types are small (24 and 36
  questions) and written by one author.

## Uses

Intended: benchmarking retrieval and RAG answer quality on German legal text. Not intended: legal advice or any decision
about a real case.

## Distribution and maintenance

Distributed in this repository under CC BY 4.0. Corrections go into a new version (`k1_v2`); `k1_v1` stays frozen from
the first published run so published numbers remain reproducible.
