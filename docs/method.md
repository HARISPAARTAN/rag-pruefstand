# Method

## Corpus and units

Five federal energy laws from gesetze-im-internet.de (EnWG, EEG 2023, MsbG, EnFG, KWKG 2025), downloaded once and frozen
by SHA-256 in `data/corpus/snapshot.json`. The parser keeps one unit per Absatz with law, §, heading, Absatz number,
text, character offsets within the norm and a deep link; it drops footnotes, amendment notes, the table of contents and
the annexes, and counts what it removed. Repealed norms ("(weggefallen)") are skipped.

## Question set

240 German questions (216 answerable in five types, 24 unanswerable), written by the implementing model
(`claude-sonnet-5-5`) on 2026-10-25 while it read the units, filtered automatically, 0 human-checked; see
`data/questions/DATASHEET.md` and ADR-006. Retrieval metrics use the 216 answerable questions; the by-type spread is
factoid 60, colloquial 48, definition 48, crossref 36, number 24.

## Chunking

- C1: fixed windows of 350 multilingual-e5 tokens with 50 tokens overlap over the concatenated text of each law,
  ignoring structure.
- C2: one chunk per Absatz with a header line such as "EnWG § 14a Steuerbare Verbrauchseinrichtungen, Abs. 1"; an Absatz
  longer than 350 tokens is split at sentence boundaries (then clauses, then tokens), each part keeping the header.

Every chunk records which unit spans it covers (`covers`), so relevance can be judged the same way for both strategies.

## Relevance rule

A retrieved chunk covers a gold unit when it overlaps at least min(50% of the unit's length, 200 characters) of that
unit. Metrics are counted per gold unit, not per chunk, so C1 and C2 are compared fairly.

## Retrieval variants

Six retrievers × two chunkings = twelve variants: `bm25` (bm25s, German Snowball stemming, German stop words), `e5`
(multilingual-e5-base, exact cosine), `mini` (English-only all-MiniLM-L6-v2 as a control), `hyb` (reciprocal rank fusion,
k = 60, of bm25 and e5 top-50), `e5+rr` and `hyb+rr` (top-20 reranked by bge-reranker-base, max length 384). Everything
runs on CPU (the published run: a Windows 11 machine without GPU); embeddings and reranker scores are cached by model
revision and text hash.

The control model reads at most 512 tokens and cuts long chunks (the report states the share per chunking), the reranker
reads at most 384 tokens per pair (share also stated). The e5-versus-MiniLM gap therefore mixes language coverage with
model size, training data and truncation; it is not a pure language effect.

## Metrics and statistics

Per question: recall@1/3/5/10 (share of gold units covered in the top k), MRR@10 (first covering rank) and nDCG@10 (gain 1
for each newly covered gold unit, capped at 1). Unanswerable questions are excluded from retrieval metrics. Intervals are
95% percentile bootstrap intervals over questions (2,000 resamples, seed 20261025). The eight comparisons in
`pruefstand.toml` were fixed before the run; each uses a paired bootstrap of the per-question difference, and a difference
whose interval contains zero is reported as "no measurable difference". The configuration `hyb+rr-C2` is the pre-declared
headline configuration, not the variant with the highest point estimate (that is `e5+rr-C2`; no comparison between the two
was declared). Latency is measured per stage (embedding, search, rerank) on a 50-question sample with warm models; p50 and
p95 hold for the machine named in the report.

## Generation and judging (live run)

Four configurations fixed in advance (bm25-C1, e5-C2, hyb-C2, hyb+rr-C2) send their top-5 chunks to the generator with
`prompts/answer_v1_de.md`. The judge (rubric `judge_v2`, `docs/rubric.md`) scores every answer in two separate calls:
faithfulness from question, retrieved context and answer (the gold answer is never part of this request), correctness and
completeness from question, gold answer and answer (no context). Citation precision and recall come from a parser that
reads brackets, lists, ranges and chains and counts every `§` it cannot attribute as an unparsed mention. All calls go
through one provider interface with record and replay; CI replays a small hand-written synthetic cassette and never calls
an API. Author and primary judge are the same model, which is disclosed in the datasheet; a second judge and the
human-checked subset bound the effect. v0.1.0 contains no measured generation or judge numbers; the report says so
explicitly.

## Reproducibility

`manifest.json` in each run directory records the git SHA, configuration hash, prompt hashes, corpus and question-set
hashes, model revisions, package versions, machine and spend. `uv run pruefstand report` rebuilds the report and the site
from the committed artefacts without network or key; `uv run pruefstand all --profile ci` reproduces the committed CI
metrics exactly. Cassettes are per profile: the CI profile reads `fixtures/llm/synthetic_cassette.jsonl`, the full profile
`cassettes/full-2026-10-25.jsonl`, and synthetic entries are invisible outside the CI profile.
