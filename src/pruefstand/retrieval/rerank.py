"""Cross-encoder reranking of the top candidates (bge-reranker-base on CPU)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def rerank_order(candidates: Sequence[int], scores: Sequence[float]) -> list[tuple[int, float]]:
    """Sort candidates by rerank score, best first; ties keep the original candidate order."""
    ranked = sorted(
        (
            (position, doc, round(float(score), 6))
            for position, (doc, score) in enumerate(zip(candidates, scores, strict=True))
        ),
        key=lambda item: (-item[2], item[0]),
    )
    return [(doc, score) for _, doc, score in ranked]


class CrossEncoderScorer:
    def __init__(self, model_dir: Path, max_length: int, batch_size: int) -> None:
        from sentence_transformers import CrossEncoder

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} missing; run `uv run pruefstand models fetch` first")
        self.model = CrossEncoder(str(model_dir), device="cpu", max_length=max_length)
        self.batch_size = batch_size

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Raw logits (identity activation), not sigmoid probabilities.

        The sigmoid squeezes unrelated pairs towards 0 so tightly that rounding to 6 decimals produced ties; the
        logit is monotonic in the probability, so the ranking is the same, and keeps its resolution.
        """
        from torch import nn

        values = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) > 256,
            activation_fn=nn.Identity(),
        )
        return [float(value) for value in values]
