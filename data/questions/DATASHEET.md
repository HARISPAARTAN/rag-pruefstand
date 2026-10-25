# Datasheet: RAG-Prüfstand question set k1_v1

## Summary

240 German questions about five federal energy laws (EnWG, EEG 2023, MsbG, EnFG, KWKG 2025), each with a type, the ids
of the gold units (Absätze) that answer it, a verbatim gold passage, a short gold answer, English renderings for review
and full provenance. File: `data/questions/k1_v1.jsonl` (one JSON object per line). Licence: CC BY 4.0 (see
`LICENSE.md` in this folder).

## Status of the human check

**Pending.** The questions were model-generated on 2026-10-26 with Claude Code by the authoring agent `claude-sonnet-5-5` (a Claude model
acting as the implementing agent of this repository) while it read the law texts. No question has been checked by a
human yet. A stratified sample of 100 questions is in `review/k1_v1_review.csv`; the owner's verdicts (accept, edit,
reject) will be imported with `pruefstand questions review-import` and recorded per question in `review`. Until then,
treat every question and gold answer as unverified. Results are reported on all 240 questions and, once the check is
done, separately on the checked subset.

## Motivation

The set measures retrieval and answer quality of RAG pipelines on German legal text: compound words, cross-references
("nach § 14a Absatz 1"), definitions in long lists, and citizen wording that differs from legal wording.

## Composition

| Type | Share | Meaning |
|---|---|---|
| factoid | 25% (60) | one fact from one Absatz |
| definition | 20% (48) | meaning of a legal term from a definition norm |
| colloquial | 20% (48) | everyday wording that avoids the legal terms |
| crossref | 15% (36) | answer needs at least two units |
| number | 10% (24) | deadline, amount, percentage or threshold |
| unanswerable | 10% (24) | plausible, but no answer in the five laws |

Per law: EnWG 96, EEG 2023 60, MsbG 30, EnFG 27, KWKG 2025 27. The 30 MsbG questions are meant to serve as the offline CI
subset (`fixtures/ci/questions.jsonl`, added together with the CI fixtures).

## Collection process

- Source texts: XML downloads from gesetze-im-internet.de, fetched on 2026-10-25 (hashes in
  `data/corpus/snapshot.json`), parsed into one unit per Absatz.
- Method: the authoring agent followed `docs/question-authoring.md` (its SHA-256 is in each record's provenance) and
  wrote the questions in batches per law in `data/questions/batches/`. No API call and no separate generator model.
- Automatic checks, all passed by every record (results stored in `provenance.filters`): gold passage verbatim in the
  gold units; word-trigram copy ratio of the question at most 0.6; colloquial questions share at most 50% of content
  words with the legal wording; gold answer at least 50% supported by the passage and every number in it present in
  the passage; content-word Jaccard to every earlier question at most 0.8.

## Known limitations

- There is no separate difficulty rating; the question type is the stratification variable, and the per-law and per-type
  counts are fixed by the authoring protocol.
- Model-written questions can mirror the wording of the source more than real user questions do; the colloquial type
  and the copy filter reduce this but do not remove it.
- The duplicate filter is lexical (Jaccard), not semantic; near-paraphrases with different words can remain.
- Gold answers are short summaries; a correct but more detailed answer can be judged differently.
- The legal texts are a non-official consolidation dated 2026-10-25; later amendments can make answers outdated.

## Uses

Intended: benchmarking retrieval and RAG answer quality on German legal text. Not intended: legal advice or any decision
about a real case.

## Distribution and maintenance

Distributed in this repository under CC BY 4.0. Corrections go into a new version (`k1_v2`); `k1_v1` stays frozen so
published numbers remain reproducible.
