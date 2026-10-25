# ADR-006: Question set written by the implementing agent, checked by filters, reviewed by a human

Status: accepted, 2026-10-25

Context: no public German question set exists for these laws, and v0.1.0 has no API budget for a generator model.

Decision: the implementing agent (a Claude model, `claude-sonnet-5-5`) writes the 240 questions of `k1_v1` on 2026-10-25
(216 answerable, 24 unanswerable) while reading the units, following `docs/question-authoring.md`, in batches per law.
Automatic filters check verbatim gold passages, copying, colloquial wording, answer support, numbers and lexical
duplicates. The owner reviews a stratified sample of 100 questions with an English aid; results are reported on all
questions and on the checked subset. The spec's embedding-based duplicate filter is replaced by a content-word Jaccard
filter so the check runs in CI without models.

Consequences: the set is model-generated and labelled as such in the datasheet and in every report (216 answerable
questions, 0 human-checked); until the review is imported every record says `pending`. The author model is also the
primary answer judge of the live run, which is disclosed. A gap between the full set and the checked subset is itself a
finding about synthetic questions.
