# ADR-002: Relevance by overlap with gold spans

Status: accepted, 2026-10-25

Context: questions point to gold units (Absätze), but the two chunking strategies cut the text differently. Labelling gold
chunks per strategy would double the work and favour the strategy that matches the labels.

Decision: every chunk stores the unit spans it covers; a chunk counts for a gold unit when it overlaps at least
min(50% of the unit, 200 characters). Metrics are counted per gold unit.

Consequences: C1 and C2 are compared on the same labels. A C1 window that covers several gold units of a cross-reference
question can satisfy them at once, which is correct behaviour, not a loophole.
