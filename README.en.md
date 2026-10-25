# RAG-Prüfstand

[![CI](https://github.com/HARISPAARTAN/rag-pruefstand/actions/workflows/ci.yml/badge.svg)](https://github.com/HARISPAARTAN/rag-pruefstand/actions/workflows/ci.yml)

A reproducible bench for retrieval-augmented generation (RAG) on German law: five federal energy laws, a versioned question
set with provenance (240 questions, 216 of them answerable, model-generated, 0 human-checked), twelve retrieval variants
on CPU, and confidence intervals for every retrieval metric. Deutsche Fassung: [README.md](README.md). Report as a page:
https://harispaartan.github.io/rag-pruefstand/en.html

## Problem

Many RAG demos show answers but no measurement. This bench measures how well retrieval variants find the right source,
with a fixed question set, a frozen corpus and intervals instead of single numbers. German legal text is a hard case:
long compound words, cross-references such as "nach § 14a Absatz 1" and definitions in long lists.

## Result in three numbers

<!-- results:start -->
- Pre-declared headline configuration hyb+rr-C2: recall@5 0.706 (95% CI 0.649 to 0.762, n = 216).
- e5-C2 vs mini-C2 (recall@5): +0.408 (95% CI +0.341 to +0.474) – better. This is not a pure language effect: the models also differ in size, training and in how much of long chunks they read (see limitations).
- hyb+rr-C2 vs bm25-C2 (recall@5): +0.171 (95% CI +0.113 to +0.231) – better.
<!-- results:end -->

Full report: [reports/v1/results.en.md](reports/v1/results.en.md). Status v0.1.1: retrieval is measured, on a Windows
CPU. Answers, judge scores and cost per 1,000 queries are **pending** (live run, v0.2.0); the report deliberately shows
no numbers for them.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Corpus | EnWG, EEG 2023, MsbG, EnFG, KWKG 2025 from gesetze-im-internet.de | public domain (§ 5 UrhG), hard for retrieval, freezable |
| Relevance | overlap with the gold paragraph, with the gold passage for gold paragraphs over 2,000 characters (62 of 216 questions) | fair comparison of both chunking strategies ([ADR-002](docs/adr/002-span-relevance.md)) |
| Search | bm25s with German stemming; exact cosine in NumPy | no database needed ([ADR-005](docs/adr/005-no-vector-database.md)) |
| Models | multilingual-e5-base, English MiniLM as a control, bge-reranker-base | run on CPU, revisions pinned |
| Fusion | reciprocal rank fusion, k = 60 | no weight tuned on the test questions; in the run it adds nothing measurable over dense retrieval alone. By the point estimates most of the gap to BM25 (+0.171) comes from the dense model (+0.107); reranking adds +0.076 on top (e5 path) or +0.059 (hybrid path, lower bound +0.006, borderline) |
| Questions | 216 answerable and 24 unanswerable questions, written on 2026-10-25 by the implementing model, filtered automatically, 0 human-checked | the datasheet says so openly ([ADR-006](docs/adr/006-question-set.md)) |
| Judge | own rubric in two calls (faithfulness without the reference answer, correctness without the context), JSON output, Batch API | inspectable instead of framework default prompts ([ADR-003](docs/adr/003-own-judge.md)) |
| LLM calls | one interface, record and replay, separate cassettes per profile | CI without a key ([ADR-004](docs/adr/004-provider-and-replay.md)) |

## Quick start

Offline, without a key and without models (the same as CI):

```bash
uv sync --locked
uv run pruefstand all --profile ci
uv run pruefstand report
```

`report` rebuilds the report and the page from the committed runs; `uv run pruefstand metrics` on the published run also
needs the local chunk files (`uv run pruefstand chunk`).

Reproducing the retrieval run (downloads about 2.3 GB of models to `.cache/models`). The run starts from the committed
corpus (`data/corpus/k1_units.jsonl.gz`) and writes only to `runs/verify-scratch/` (git-ignored); the corpus snapshot, the
units, the published run `runs/full-2026-10-25`, `reports/` and `site/` stay untouched:

```bash
uv sync --locked --extra models
uv run --extra models pruefstand models fetch
uv run --extra models pruefstand all --profile verify
uv run pruefstand verify
```

`verify` compares hits (by content) and `metrics.json` (by value, run id ignored) with the published run and exits with
code 1 and a list when anything differs. Without the cache `.cache/scores`, `all` recomputes the embeddings (about 22
minutes on a 12-core CPU); on another CPU low decimals may differ. The law texts cannot be fetched and compared again:
gesetze-im-internet.de serves only the current consolidation. That is why the units file is part of the repository, and
`uv run pruefstand corpus fetch` refuses to overwrite the snapshot when a law has changed (only `--update` replaces it, as
a new corpus version with a new run). `uv run pruefstand all` with the `full` profile, in contrast, creates a new run in
`runs/full-2026-10-25` and is not a reproduction.

Live run with the API (costs money; projection with `uv run pruefstand cost estimate`, about USD 25, cap USD 30): set
`ANTHROPIC_API_KEY` and choose mode `record`, in Bash `export PRUEFSTAND_LLM_MODE=record`, in PowerShell
`$env:PRUEFSTAND_LLM_MODE = "record"`. **Run the pilot first**, then the full judge:

```bash
uv run pruefstand generate --profile pilot
uv run pruefstand judge --profile pilot
```

The pilot takes three questions (the first two answerable ones by question id, both from the EEG, and the first
unanswerable one) and the variant hyb+rr-C2, with the same models, prompts and judge settings as the full run. It reads
hits and chunks of the published run, writes answers, judgements and spend to `runs/pilot-2026-10-25/` and shares the
cassette `cassettes/full-2026-10-25.jsonl` with the full run: what the pilot recorded is not paid again by the full run.
Then check `judge_invalid`, `judge_failed` and `judge_refusals` in `runs/pilot-2026-10-25/answer_metrics.json`, answers with
`stop_reason` `max_tokens` in `answers.jsonl` (the judge's limit of 1,000 tokens includes thinking), the Batch round trip
in `batch_judge.json` and the cost per call. Start the full run only when that is clean; changing the judge limit
afterwards changes the requests and pays the pilot calls again. The three pilot questions are a smoke test, not a
reportable subset.

```bash
uv run pruefstand generate
uv run pruefstand judge
```

The full run reads the cassette first and pays only for missing calls; the judge's Batch id is kept in
`runs/full-2026-10-25/batch_judge.json`, and the same command resumes an interrupted run. Before every paid call the
command compares the hash of the retrieval settings with the published run (`runs/full-2026-10-25/config.toml` is its
configuration, hash 625b4f08e540) and stops on a difference. `generate` and `judge` write one entry per run to
`stage_manifests.json` in the run folder (configuration hash, models, cassette, times). The cassette, `spend.json`,
`batch_judge.json` and `stage_manifests.json` are the audit trail and are committed after the live run.

## Method

The unit is the paragraph (Absatz). C1 cuts fixed windows (350 tokens, 50 overlap), C2 one chunk per paragraph with a
header line. Metrics are recall@1/3/5/10, MRR@10 and nDCG@10 with 95% bootstrap intervals; eight comparisons are fixed in
advance and tested pairwise. The headline configuration hyb+rr-C2 was fixed before the run; it was not selected by its
result. Details: [docs/method.md](docs/method.md), rubric: [docs/rubric.md](docs/rubric.md).

## Data and licences

Code: MIT. Law texts: gesetze-im-internet.de, Federal Ministry of Justice; non-official consolidated version as of
2026-10-25. Question set: CC BY 4.0, model-generated on 2026-10-25; the human check of 100 questions is pending
([datasheet](data/questions/DATASHEET.md)). `questions review-import` writes the verdicts to
`data/questions/review/` and edits into a new set `k1_v2`; `k1_v1` stays unchanged. All sources: [DATA_LICENSES.md](DATA_LICENSES.md).

## Cost

v0.1.0 and v0.1.1 made no paid API call. Projection for a live run and price table: [docs/cost-model.md](docs/cost-model.md).

## Limitations

- The questions are model-generated (216 answerable, 0 human-checked) and not yet checked by a human.
- The author of the questions and the primary judge of the live run are the same model (claude-sonnet-5-5); this is
  disclosed and only bounded by a second judge and the checked subset.
- The gap between e5 and MiniLM is not a pure language effect: size, training and MiniLM's truncation of long chunks
  contribute.
- The reranker is trained mainly on Chinese and English and truncates long pairs at 384 tokens; a multilingual reranker is
  on the roadmap.
- All measurements ran on a Windows CPU without a GPU; latencies hold only for that machine.
- Results hold for German legal language, not for German text in general.

## Roadmap

- v0.2.0: live run (answers, judge, cost, API latency), check of 100 questions, judge calibration with weighted kappa,
  second judge, repeats, local generator through Ollama.
- Later: failure analysis with a fixed taxonomy, DSK mapping, corpus K2 (Munich citizen services) once its licence is
  clear, a multilingual reranker, publishing the question set on Hugging Face.
