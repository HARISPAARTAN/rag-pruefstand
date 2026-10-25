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

| Stage | Model | Calls | Tokens in / out per call | Mode | USD |
|---|---|---|---|---|---|
| Answers, 4 configurations | claude-haiku-5-5 | 960 | 4,000 / 400 | online | 0.58 |
| Prompt ablation | claude-haiku-5-5 | 240 | 4,000 / 400 | online | 0.14 |
| Repeats (50 × 3 × 2) | claude-haiku-5-5 | 300 | 4,000 / 400 | online | 0.18 |
| Local generator | qwen3:4b | 100 | 4,000 / 400 | local | 0.00 |
| Judge | claude-sonnet-5-5 | 1,600 | 5,000 / 800 | batch | 14.40 |
| Second judge | claude-opus-5-5 | 100 | 5,000 / 800 | batch | 1.80 |
| Total | | | | | 17.10 (EUR 15.28) |

Token counts per call are estimates (±50%); thinking tokens are billed as output. The question set costs nothing here:
it was written by the implementing agent, not by API calls.

## Guards

- `[budget] max_usd_per_run = 30`: before the first call of a stage the projected cost is checked against the cap, and
  the run stops as soon as measured spend passes it (`BudgetExceeded`, exit code 1).
- Spend is computed from the usage the API returns (input, output, cache read and cache write tokens) and stored per
  answer and judgement (`cost_usd`) and per run (`manifest.json`, `spend_usd`).
- The Anthropic Console spend limit set by the owner is the hard stop outside this code.

## Actual spend

v0.1.0 made no paid API call: USD 0. The first actual spend will be recorded here after the live run (v0.2.0), next to
the projection, so the estimate can be checked.
