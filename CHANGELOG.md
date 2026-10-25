# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog; versions follow semantic
versioning.

## [0.1.0] - 2026-10-25

### Added

- Corpus K1: EnWG, EEG 2023, MsbG, EnFG and KWKG 2025 from gesetze-im-internet.de, frozen by SHA-256, parsed into one unit
  per Absatz with footnotes and amendment notes removed.
- Chunking C1 (350-token windows, 50 overlap) and C2 (one chunk per Absatz with header, split at sentence boundaries).
- Retrieval bench: bm25, multilingual-e5-base, English MiniLM control, RRF hybrid, each with and without
  bge-reranker-base, on both chunkings (twelve variants), on a Windows CPU with cached embeddings and scores.
- Metrics recall@1/3/5/10, MRR@10, nDCG@10 with 95% bootstrap intervals, eight pre-registered paired comparisons, p50/p95
  latency per stage, and the truncation rates of MiniLM and the reranker. hyb+rr-C2 is the pre-declared headline
  configuration; in the run, hybrid fusion adds nothing measurable over dense retrieval and the gain comes from reranking.
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
