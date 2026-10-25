# ADR-003: Own judge rubric instead of RAGAS or DeepEval as the engine

Status: accepted, 2026-10-25 (amended the same day: two calls)

Context: evaluation frameworks ship English default prompts that change between versions and are hard to inspect.

Decision: a short rubric (`prompts/judge_faithfulness_v2.md` and `prompts/judge_correctness_v2.md`, explained in
`docs/rubric.md`) with structured JSON output, versioned by SHA-256, plus deterministic checks (citation precision and
recall, abstention accuracy) that need no LLM. The judge works in two calls per answer so that no call sees more than it
needs: faithfulness is judged from the question, the retrieved context and the answer, and never sees the gold answer;
correctness and completeness are judged from the question, the gold answer and the answer, without the context. A single
call with everything in view would let a model answer that matches the gold answer through the judge's own knowledge pass
as faithful.

Consequences: every judge decision can be traced to a prompt hash and a recorded response; the judge costs two Batch calls
per answer (see `docs/cost-model.md`). Author and primary judge are the same model (disclosed in the datasheet), and the
judge must be calibrated against human labels (live run); until then judge numbers are not reported.
