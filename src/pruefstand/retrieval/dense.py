"""Dense retrieval: exact cosine search over a NumPy matrix of normalised embeddings.

sentence-transformers and torch are imported only when an embedding has to be computed, so CI (which installs
no `models` extra) runs on cached vectors alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pruefstand.retrieval.ranking import tenant_mask, top_k


class DenseIndex:
    def __init__(self, vectors: np.ndarray, tenants: list[str] | None = None) -> None:
        self.vectors = vectors.astype(np.float64)
        self.tenants = tenants or ["K1"] * len(vectors)

    def search(self, query_vector: np.ndarray, k: int, tenant: str | None = None) -> list[tuple[int, float]]:
        scores = self.vectors @ query_vector.astype(np.float64)
        return top_k(scores, k, allowed=tenant_mask(self.tenants, tenant))


class SentenceEncoder:
    """Thin wrapper around a locally downloaded sentence-transformers model (CPU only)."""

    def __init__(self, model_dir: Path, max_seq_length: int, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} missing; run `uv run pruefstand models fetch` first")
        self.model = SentenceTransformer(str(model_dir), device="cpu")
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(vectors, dtype=np.float32)
