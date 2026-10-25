# Judge rubric judge_v1

The judge prompt is `prompts/judge_v1.md`; this page explains it. Judge model `claude-sonnet-5-5`, adaptive thinking,
effort low, structured JSON output validated against `pruefstand.judge.JUDGE_SCHEMA`, sent through the Message Batches
API (50% price). The rubric is written in English; questions, answers and context stay in German.

| Criterion | Scores | Question the judge answers |
|---|---|---|
| faithfulness | 0, 1, 2 | Is every claim in the answer supported by the retrieved context? |
| correctness | 0, 1, 2 | Does the answer agree with the gold answer? |
| completeness | 0, 1 | Does the answer cover every part of the question? |
| rationale | text | One sentence naming the decisive point. |

Score meanings:

- faithfulness 2: every claim is in the context; 1: the main claim is supported, a minor one is not; 0: the main claim is
  unsupported or contradicts the context. The exact abstention sentence makes no claim and scores 2.
- correctness 2: same essential content as the gold answer, numbers and parties match; 1: core right, a detail wrong or
  missing; 0: wrong, contradicting, or an abstention although the gold answer has content. For unanswerable questions an
  abstention scores 2 and any other answer 0.
- completeness 1: all asked-for parts answered (or a correct abstention); 0: a part is missing.

Anchor examples (German), question "Ab wann darf der Anschlussnutzer nach dem Einbau eines intelligenten Messsystems den
Messstellenbetreiber wechseln?", gold "Frühestens zwei Jahre nach der Ausstattung der Messstelle.":

| Answer | faithfulness | correctness | completeness |
|---|---|---|---|
| "Frühestens nach Ablauf von zwei Jahren ab Ausstattung der Messstelle [MsbG § 5 Abs. 1]." | 2 | 2 | 1 |
| "Nach einem Jahr [MsbG § 5 Abs. 1]." | 0 | 0 | 1 |
| "Der Wechsel ist jederzeit möglich, frühestens aber nach zwei Jahren [MsbG § 5 Abs. 1]." | 1 | 1 | 1 |

Refusals (`stop_reason == "refusal"`) and invalid JSON are stored with status `refusal` or `invalid`, counted in the
report and never retried silently.

Calibration (live run, v0.2.0): 80 answers stratified by configuration and score are labelled by the owner without seeing
the judge's scores; agreement is reported as quadratic weighted Cohen's kappa per criterion and raw agreement. If kappa is
below 0.6 the judge numbers are labelled indicative in the headline. A second judge (`claude-opus-5-5`, batch) scores 100
items for inter-judge agreement.

Deterministic checks need no LLM: citation precision (share of cited norms that are in the retrieved context), citation
recall (share of gold units cited) and abstention accuracy (the exact sentence on unanswerable questions, no abstention
on answerable ones).
