# Judge rubric judge_v2

Two judge prompts, one per call; this page explains them. Judge model `claude-sonnet-5-5`, adaptive thinking, effort low,
structured JSON output validated against the schemas in `pruefstand.judge`, sent together through the Message Batches
API (50% price). The rubric is written in English; questions, answers and context stay in German.

Why two calls: in the first version (`judge_v1`) one request carried the question, the gold answer, the retrieved context
and the answer. A model answer that matches the gold answer through the judge's own knowledge but is not in the retrieved
context could then be scored faithful, which is exactly the failure faithfulness is meant to catch. In `judge_v2` each
call sees only what its criterion needs:

| Call | Prompt | The judge sees | It never sees | Scores |
|---|---|---|---|---|
| faithfulness | `prompts/judge_faithfulness_v2.md` | question, retrieved context, answer | the gold answer | faithfulness 0, 1, 2 and a rationale |
| correctness | `prompts/judge_correctness_v2.md` | question, gold answer, answer | the retrieved context | correctness 0, 1, 2, completeness 0, 1 and a rationale |

Score meanings:

- faithfulness 2: every claim is in the context; 1: the main claim is supported, a minor one is not; 0: the main claim is
  unsupported or contradicts the context. A claim that is true in law but absent from the context counts as unsupported.
  The exact abstention sentence makes no claim and scores 2.
- correctness 2: same essential content as the gold answer, numbers and parties match; 1: core right, a detail wrong or
  missing; 0: wrong, contradicting, or an abstention although the gold answer has content. For unanswerable questions an
  abstention scores 2 and any other answer 0.
- completeness 1: all asked-for parts answered (or a correct abstention); 0: a part is missing.

Anchor examples (German), question "Ab wann darf der Anschlussnutzer nach dem Einbau eines intelligenten Messsystems den
Messstellenbetreiber wechseln?", gold "Frühestens zwei Jahre nach der Ausstattung der Messstelle.". The faithfulness
anchors in the prompt carry their own context line, so each example is self-contained.

| Answer | faithfulness | correctness | completeness |
|---|---|---|---|
| "Frühestens nach Ablauf von zwei Jahren ab Ausstattung der Messstelle [MsbG § 5 Abs. 1]." | 2 | 2 | 1 |
| "Nach einem Jahr [MsbG § 5 Abs. 1]." | 0 | 0 | 1 |
| "Der Wechsel ist jederzeit möglich, frühestens aber nach zwei Jahren [MsbG § 5 Abs. 1]." | 1 | 1 | 1 |

Refusals (`stop_reason == "refusal"`), invalid JSON and failed Batch items are stored with status `refusal`, `invalid` or
`failed` (with the error detail) and counted in the report; refusals and invalid output are never retried silently, errored
or expired Batch items are retried once in a second batch and, if they still fail, are not stored in the cassette so the
next run submits them again. The per-answer status is the worst of the two calls; faithfulness and correctness are
aggregated over the answers whose own call is `ok`.

Output limits: `max_tokens` is 1,000 per judge call and includes thinking tokens; a verdict that hits the limit is
`invalid`. The limit is also the budget bound, see `docs/cost-model.md`.

Calibration (live run, v0.2.0): 80 answers stratified by configuration and score are labelled by the owner without seeing
the judge's scores; agreement is reported as quadratic weighted Cohen's kappa per criterion and raw agreement. If kappa is
below 0.6 the judge numbers are labelled indicative in the headline. A second judge (`claude-opus-5-5`, batch) scores 100
items for inter-judge agreement.

Deterministic checks need no LLM: citation precision (share of cited norms that are in the retrieved context), citation
recall (share of gold units cited) and abstention accuracy (the exact sentence on unanswerable questions, no abstention
on answerable ones). The citation parser reads brackets, lists ("Abs. 1 und 2", "Satz 1 und 2"), ranges ("Abs. 1 bis 3",
"§§ 20 bis 22") and chains ("§ 14 und § 15 EnWG"); every `§` it cannot attribute to a law is counted as an unparsed
mention and reported next to the citation scores, because unparsed citations bias precision and recall.
