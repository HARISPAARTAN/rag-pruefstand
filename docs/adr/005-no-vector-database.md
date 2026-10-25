# ADR-005: No vector database

Status: accepted, 2026-10-25

Context: the corpus has a few thousand chunks per strategy.

Decision: exact cosine similarity on a NumPy matrix (float64, scores rounded to six decimals, ties broken by index) and
bm25s in memory. Embeddings and reranker scores are cached on disk, keyed by model revision and text hash. Run artefacts
are JSON and gzip-compressed JSON lines instead of Parquet: small, diffable and readable without extra libraries.

Consequences: no service, no container, exact and deterministic results. A vector database becomes worth it only at a
corpus size this bench does not have.
