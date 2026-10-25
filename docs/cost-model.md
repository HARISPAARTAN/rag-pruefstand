# Cost model

Prices are data in `costs.toml` (USD per million tokens, with source URL and date); `uv run pruefstand cost estimate`
prints the projection from the `[[cost.stage]]` table in `pruefstand.toml` without making any call.

## Price table (as of 2026-10-06)

Source: https://platform.claude.com/docs/en/about-claude/pricing, taken from the claude-api skill's model table cached on
2026-10-06, not fetched live. Recheck before the live run.

| Model | Input | Output | Cache read | Cache write (5 min) |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.10 | 0.50 | 0.01 | 0.125 |
| claude-sonnet-5-5 | 2.00 | 10.00 | 0.20 | 2.50 |
| claude-opus-5-5 | 4.00 | 20.00 | 0.20 | 5.00 |
| qwen3:4b (Ollama, local) | 0 | 0 | 0 | 0 |

Haiku 5.5 rates apply to prompts up to 100K tokens (longer prompts cost 0.50 / 2.50); every prompt in this bench is far
below that. The Message Batches API costs 50% of these prices. EUR figures use the ECB reference rate of 2026-10-23
(1 EUR = 1.1193 USD, so 0.8934 EUR per USD).

## Projection of one full live run

Projection of the planned v0.2.0 run, not a measurement. The judge now makes two calls per answer (faithfulness without
the gold answer, correctness without the context; see `docs/rubric.md`), so it appears twice.

| Stage | Model | Calls | Tokens in / out per call | Mode | USD |
|---|---|---|---|---|---|
| Answers, 4 configurations | claude-haiku-5-5 | 960 | 4,000 / 400 | online | 0.58 |
| Prompt ablation | claude-haiku-5-5 | 240 | 4,000 / 400 | online | 0.14 |
| Repeats (50 × 3 × 2) | claude-haiku-5-5 | 300 | 4,000 / 400 | online | 0.18 |
| Local generator | qwen3:4b | 100 | 4,000 / 400 | local | 0.00 |
| Judge: faithfulness (context only) | claude-sonnet-5-5 | 1,600 | 4,500 / 700 | batch | 12.80 |
| Judge: correctness (with gold answer) | claude-sonnet-5-5 | 1,600 | 1,500 / 800 | batch | 8.80 |
| Second judge: faithfulness | claude-opus-5-5 | 100 | 4,500 / 700 | batch | 1.60 |
| Second judge: correctness | claude-opus-5-5 | 100 | 1,500 / 800 | batch | 1.10 |
| Total | | | | | 25.20 (EUR 22.51) |

Token counts per call are estimates (±50%); thinking tokens are billed as output. The measured context at k = 5 is about
850 to 1,750 e5 tokens plus the prompt, so the input figures are on the conservative side; judge output with thinking is
the unknown that matters (every 100 extra output tokens per judge call adds about USD 1.6 over 3,200 calls). The question
set costs nothing here: it was written by the implementing agent, not by API calls.

## Guards

- `[budget] max_usd_per_run = 30`. Before anything is sent, a stage computes a hard upper bound for the requests that would
  reach the provider (not those already in the cassette): an input bound at 2.5 characters per token plus `max_tokens`
  of output per request, times the Batch discount where it applies. If the bound plus what the run already spent passes
  the cap, the stage stops with `BudgetExceeded` (exit code 1) before the first call or submit. Output cannot exceed
  `max_tokens` (thinking tokens count inside it), so the bound holds unless the input bound is wrong.
- The judge stage is bounded this way as a whole: with `max_tokens = 1000` the worst case for the 960 headline answers is
  USD 16.2 (computed from the real requests with answers of 2,400 characters), for all 1,600 planned judged answers
  about USD 27.0, below the cap; at the old 4,000 it would have been USD 45 for 960 answers. A verdict that needs more
  than 1,000 tokens is recorded as `invalid`, never silently retried.
- One cap per run: each live stage writes what it spent to `runs/<id>/spend.json`, and the other stages count it.
- Measured spend is computed from the usage the API returns (input, output, cache read and cache write tokens), stored per
  answer and judgement (`cost_usd`), per stage in `spend.json` and in the `stage_manifests.json` entry of each live
  `generate` or `judge` run (`spend_usd`, only for calls billed in that run; `manifest.json` records `spend_usd` for
  runs made with `all`).
- Every cassette entry stores the cost at record time, the price table date, a timestamp, the SDK version and the Batch
  id. A replay reports `cost_usd` (recomputed with the current `costs.toml`) and `recorded_cost_usd` (as stored) apart,
  so a later price edit cannot silently rewrite history.
- The Anthropic Console spend limit set by the owner is the hard stop outside this code.

## Resuming after a failure

Record mode reads the cassette first and only pays for requests that are not stored, so a crashed run costs nothing
twice (`--refresh` forces new calls). The judge Batch id and the request keys are written to `runs/<id>/batch_judge.json`
before the wait starts; running the same command again picks the batch up instead of submitting it again. The wait has a
maximum (26 hours); errored, expired or canceled items are retried once in a second batch and, if they still fail, are
reported with their error and left out of the cassette.

## Actual spend

v0.1.0 made no paid API call: USD 0. The first actual spend will be recorded here after the live run (v0.2.0), next to
the projection, so the estimate can be checked.

## Pilot before the full judge

`--profile pilot` runs three questions (the first two answerable ones by question id, both EEG, and the first unanswerable
one) with the variant hyb+rr-C2: 3 answers and 6 judge calls with the models and settings of the full run, sharing the
full run's cassette file. At the projected token counts that is a few US cents. The pilot answers three open questions the
tests cannot: how many judge verdicts end in `max_tokens` (the 1,000-token limit includes adaptive thinking), whether the
Batch API returns the shapes the code expects, and what a call really costs. Its spend is kept in
`runs/pilot-2026-10-25/spend.json`, separate from the cap of the full run.
