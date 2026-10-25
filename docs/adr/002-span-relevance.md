# ADR-002: Relevance by overlap with gold spans

Status: accepted, 2026-10-25; amended in v0.1.1 to describe the rule as it runs (passage level for long units)

Context: questions point to gold units (Absätze), but the two chunking strategies cut the text differently. Labelling gold
chunks per strategy would double the work and favour the strategy that matches the labels.

Decision: every chunk stores the unit spans it covers. The rule depends on the length of the gold unit. For gold units of
at most 2,000 characters a chunk counts when it overlaps at least min(200 characters, 50% of the unit). For longer gold
units (the definition norms, up to 32,331 characters) the unit is too coarse a label: a chunk counts only when it overlaps
an excerpt of the question's gold passage by at least min(200 characters, 50% of the excerpt), so that another entry of the
same list earns nothing. 62 of the 216 answerable questions are graded on the passage. Metrics are counted per gold unit.
nDCG@10 uses the share of newly covered gold units as gain and the greedy minimal cover of the same chunking as ideal
(capped at 1), so it compares retrievers within one chunking; C1 against C2 is compared with recall@k and MRR@10.

Consequences: C1 and C2 are compared on the same labels. A C1 window that covers several gold units of a cross-reference
question can satisfy them at once, which is correct behaviour, not a loophole. The unit-level rule alone would have
counted every chunk of a 32,000-character definition list as relevant, which is why the passage level exists.
