# Method

## Corpus and units

Five federal energy laws from gesetze-im-internet.de (EnWG, EEG 2023, MsbG, EnFG, KWKG 2025), downloaded once and frozen
by SHA-256 in `data/corpus/snapshot.json`. The parser keeps one unit per Absatz with law, §, heading, Absatz number,
text, character offsets within the norm and a deep link; it drops footnotes, amendment notes, the table of contents and
the annexes, and counts what it removed. Only sections that are repealed as a whole ("§ 5 (weggefallen)") are skipped;
48 bare Absätze of the form "(3) (weggefallen)" inside otherwise valid sections remain as short units (a known
imperfection of k1, to be fixed with a corpus revision).

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

The rule depends on the length of the gold unit (`metrics.json`, key `relevance`, holds the rule text and the counts):

- **Unit level, gold units of at most 2,000 characters.** A retrieved chunk is relevant to a gold unit when it covers at
  least min(200 characters, 50% of the unit's length) of that unit.
- **Passage level, gold units longer than 2,000 characters** (the definition norms; EnWG § 3 has 32,331 characters). The
  excerpts of the question's gold passage are located in the unit text, and a chunk is relevant only when it overlaps one
  excerpt by at least min(200 characters, 50% of the excerpt). A chunk of the same long unit that misses the passage does
  not count, so a retriever that returns a different definition from the same list gets no credit. This applies to 62
  of the 216 answerable questions (all 48 definition questions, 10 crossref, 3 colloquial and 1 number question).

Metrics are counted per gold unit, not per chunk, so C1 and C2 are compared on the same labels.

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

Per question: recall@1/3/5/10 (share of gold units covered in the top k), MRR@10 (reciprocal rank of the first covering
chunk) and nDCG@10. recall@k and MRR@10 are the metrics that compare C1 with C2. nDCG@10 compares retrievers **within one
chunking only**: the gain of a rank is the share of the question's gold units that the chunk covers for the first time, and
the ideal DCG is the greedy minimal cover of the gold units by the chunks of the same chunking (repeatedly take the chunk
covering the most still uncovered gold units, place its gain at ranks 1, 2, ...). A chunk that covers two gold units earns
two gain units and is measured against an ideal that also needs such a chunk; the result is guarded to at most 1
(`min(1, DCG / ideal)`), which matters only in contrived cases where the greedy cover is not optimal. A value of 1.0 means
"as good as this chunking allows", so no cross-chunking nDCG interval exists. Unanswerable questions are excluded from
retrieval metrics. Intervals are 95% percentile bootstrap intervals over questions (2,000 resamples, seed 20261025). The
eight comparisons in
`pruefstand.toml` were fixed before the run; each uses a paired bootstrap of the per-question difference, and a difference
whose interval contains zero is reported as "no measurable difference". The configuration `hyb+rr-C2` is the pre-declared
headline configuration, not the variant with the highest point estimate (that is `e5+rr-C2`; no comparison between the two
was declared). The comparison table has 13 rows (eight pairs; nDCG@10 only for the five pairs within one chunking) and is
not corrected for multiple testing; the lower bound of `hyb+rr-C2` against `hyb-C2` (+0.006) is borderline. Questions are
resampled as independent although some share gold units, so the intervals may be too narrow, and an interval such as
[1.000, 1.000] in the by-type table (all 24 questions of a type hit) is not certainty. Which part of the +0.171 over BM25
comes from the dense model and which from reranking is read from point estimates: e5-C2 against bm25-C2 is +0.107 (not a
pre-declared comparison, no interval), reranking adds +0.076 on the e5 path and +0.059 on the hybrid path; hybrid fusion
adds nothing measurable over dense retrieval. At equal k the context of C1 holds about two to four times the tokens of C2
(the report shows the means), so C1-versus-C2 comparisons are comparisons at equal k, not at equal context size.
Latency is measured per stage (embedding, search, rerank) on the first 50 answerable questions by question id (all of them
EEG questions) with warm models; p50 and p95 hold for the machine named in the report (AMD Ryzen 9 9900X, 12 cores, 61.6 GB
RAM, Windows 11), and a re-run on another CPU may move low decimals.

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
metrics exactly. The published retrieval run is reproduced with `uv run --extra models pruefstand all --profile verify`
followed by `uv run pruefstand verify` (see the README): the verify profile starts from the committed units, writes only
to `runs/verify-scratch/`, and `verify` compares hits by content and `metrics.json` by value.

**Configuration hash of the published run.** The manifest of `runs/full-2026-10-25` records the configuration hash
`625b4f08e540` (git `2f73d04`). `pruefstand info` at v0.1.1 prints `b083c0ea9088`. The difference comes from three
top-level sections of `pruefstand.toml` that changed after the run: `cost` (projection table), `llm` (the LLM layer of
the live run: judge settings, cassette path, batch limits) and `questions` (the added keys `authored` and
`author_model`). None of them influences retrieval. The hash of the retrieval-relevant settings (corpus, tokenizer,
chunking, retrieval, models, stats, latency, comparisons, headline and `questions.path`/`questions.version`) is
`67af0ba7a702` in both versions. The configuration the run used is committed as `runs/full-2026-10-25/config.toml`
(taken from git `2f73d04`); a unit test checks that its hash equals the manifest and that its retrieval hash equals the
current one.

**Paid stages.** `generate` and `judge` in record or live mode first compare the current retrieval hash with the
published run and stop on a difference, and afterwards append an entry to `stage_manifests.json` in the run folder:
configuration hash, retrieval hash, hash of the stage settings, model id, prompt hashes, cassette file and hash,
question-set hash, SDK version, spend and timestamps. `manifest.json` itself is not rewritten by these stages. New
manifests carry `retrieval_config_sha256`.

**Question set.** `k1_v1.jsonl` is frozen: its SHA-256 is pinned by the manifest. `questions review-import` writes the
owner's verdicts to `data/questions/review/k1_v1_verdicts.jsonl` and puts every edit, after validating it again, into a
new set `k1_v2.jsonl`. Cassettes are per profile: the CI profile reads `fixtures/llm/synthetic_cassette.jsonl`, the full profile
`cassettes/full-2026-10-25.jsonl`, and synthetic entries are invisible outside the CI profile.
