# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog; versions follow semantic
versioning.

## [0.1.1] - 2026-10-25

Patch release: text corrections and the safeguards needed before the first paid call. No committed run artefact, no
published retrieval number and no question record changed; `hits.jsonl.gz`, `metrics.json` and `k1_v1.jsonl` are
byte-identical to 0.1.0 (`uv run pruefstand metrics` and `uv run pruefstand verify` reproduce them).

### Fixed

- The report (German and English), the site, `docs/method.md` and ADR-002 now describe the relevance rule as it runs: unit
  level for gold units of up to 2,000 characters, passage level above (62 of 216 answerable questions are graded on the
  gold passage), and nDCG@10 as it is computed (share of newly covered gold units as gain, greedy minimal cover of the same
  chunking as ideal, capped at 1, comparable only within one chunking). `docs/method.md` no longer says that all
  "(weggefallen)" norms are skipped: only fully repealed sections are, 48 bare "(n) (weggefallen)" Absätze remain units.
- Attribution of the +0.171 recall@5 of hyb+rr-C2 over bm25-C2 (report summary, both READMEs, this changelog): by the point
  estimates about +0.107 comes from the dense model (e5-C2 against bm25-C2, not a pre-declared comparison, no interval),
  reranking adds +0.076 [+0.021, +0.137] on the e5 path and +0.059 [+0.006, +0.110] on the hybrid path (borderline), and
  hybrid fusion adds nothing over dense retrieval. The sentence "the gain comes from reranking" is gone.
- Smaller report corrections: "confidence intervals for every retrieval metric" (not every number), "pre-declared" instead
  of "pre-registered", the interval rule ("contains zero"), a note on 13 comparisons without multiplicity correction and on
  questions that share gold units, the first-50-EEG latency sample, the CPU and RAM of the measuring machine, the mean
  tokens of the top-k chunks per variant (C1 holds two to four times the tokens of C2 at equal k), and a stale comment in
  `pruefstand.toml`.
- Reproduction no longer overwrites published files. The README path uses the new `verify` profile (committed units in,
  `runs/verify-scratch/` out) and `pruefstand verify`; `corpus fetch` refuses to replace a changed snapshot without
  `--update` and keeps `snapshot.json` untouched when nothing changed.
- `questions review-import` no longer rewrites `k1_v1.jsonl`: verdicts go to `data/questions/review/k1_v1_verdicts.jsonl`,
  edits are validated again and written to `k1_v2.jsonl`, and the original wording is always taken from the frozen set.
- The datasheet and `DATA_LICENSES.md` state that every k1_v1 record carries `provenance.created_at = 2026-10-26` although
  the questions were written on 2026-10-25 (a clock assumption of the authoring agent). The value stays in `k1_v1` because
  the published run pins the file's SHA-256; `k1_v2` corrects it. A test pins the known value.
- `judge` without answers is a clean error instead of a traceback.

### Added

- `generate` and `judge` write one entry per record or live run to `stage_manifests.json` in the run folder
  (configuration hash, retrieval hash, stage settings hash, model, prompt hashes, cassette file and hash, question-set
  hash, SDK version, spend, timestamps) and do not touch `manifest.json`. New manifests carry `retrieval_config_sha256`.
- Before every record or live stage the retrieval-relevant settings hash is compared with the published run, and the stage
  stops on a difference. The v0.1.0 manifest only holds the hash of the whole configuration, so the configuration of the
  run (git `2f73d04`) is committed as `runs/full-2026-10-25/config.toml`.
- `[profile.pilot]`: three questions, the full run's models, judge settings and cassette file, hits and chunks of the
  published run, its own run folder; stages that write retrieval artefacts are refused. The README tells the owner to run
  it before the full judge.
- `[profile.verify]` and `pruefstand verify` (see above); PowerShell syntax next to Bash for the record mode in the README.
- The Pages workflow fails when the committed report, site or README results blocks differ from a fresh `pruefstand report`.

### Notes

- Configuration hash. The manifest of the published run records `625b4f08e540` (git `2f73d04`); `pruefstand info` prints
  `b083c0ea9088`. Only the `cost`, `llm` and `questions` sections changed (`questions` only gained `authored` and
  `author_model`); the hash of the retrieval-relevant settings is `67af0ba7a702` in both. The published numbers are not
  affected. The report now states this.
- Not yet done, scheduled before the first paid call: key preflight and `anthropic.APIError` mapping, `PRUEFSTAND_LLM_MODE`
  validation as a clean error, refusing `live` mode on profiles with a cassette, and the budget edge for a resumed Batch.

## [0.1.0] - 2026-10-25

### Added

- Corpus K1: EnWG, EEG 2023, MsbG, EnFG and KWKG 2025 from gesetze-im-internet.de, frozen by SHA-256, parsed into one unit
  per Absatz with footnotes and amendment notes removed.
- Chunking C1 (350-token windows, 50 overlap) and C2 (one chunk per Absatz with header, split at sentence boundaries).
- Retrieval bench: bm25, multilingual-e5-base, English MiniLM control, RRF hybrid, each with and without
  bge-reranker-base, on both chunkings (twelve variants), on a Windows CPU with cached embeddings and scores.
- Metrics recall@1/3/5/10, MRR@10, nDCG@10 with 95% bootstrap intervals, eight pre-declared paired comparisons (fixed in the root commit), p50/p95
  latency per stage, and the truncation rates of MiniLM and the reranker. hyb+rr-C2 is the pre-declared headline
  configuration; in the run, hybrid fusion adds nothing measurable over dense retrieval. (The first release also said the gain over
  BM25 comes from reranking; that was inaccurate, see 0.1.1.)
- Question set k1_v1: 240 German questions (216 answerable, 24 unanswerable) with provenance, gold passages and answers
  (CC BY 4.0), model-generated on 2026-10-25 by the implementing agent and filtered automatically, 0 human-checked;
  datasheet and review sheet for the pending human check of 100 questions.
- LLM layer: one provider interface with Anthropic (Messages and Message Batches API) and Ollama backends, record and
  replay, price table, cost projection and a budget guard with a hard upper bound; answer and judge prompts; judge rubric
  judge_v2 in two calls (faithfulness never sees the gold answer, correctness never sees the context); citation parser for
  lists, ranges and chains with an unparsed-mention diagnostic.
- Live-run safeguards: record mode reads the cassette first and pays only for missing calls, the stage is part of the
  replay key, every cassette entry stores timestamp, SDK version, Batch id and the cost at record time, Batch ids are kept
  in the run directory and resumed, errored or expired items are retried once, Ollama requests carry num_ctx,
  temperature 0 and a fixed seed and fail when the prompt does not fit, and the CI profile keeps its synthetic cassette in
  a separate file that other profiles refuse to read.
- Report in German and English (Markdown, JSON, SVG) and a static dark site, rebuilt from committed run artefacts.
- Offline CI profile on the MsbG slice with golden files, a hand-written synthetic LLM cassette and GitHub Actions; a
  GitHub Pages workflow for the site.

### Not yet measured

- Generation, judge scores, cost per 1,000 queries and API latency: pending the live run (v0.2.0). The report shows an
  explicit pending state and no numbers for them.
