"""Content-addressed caches for embeddings and rerank scores.

Keys are SHA-256 over model id, revision and the exact input text, so a cached value can never be reused for a
different model or text. CI reads committed caches with allow_compute = false; a miss then raises CacheMiss.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from pruefstand.errors import PruefstandError
from pruefstand.io import sha256_text


class CacheMiss(PruefstandError):
    pass


def embedding_key(model_id: str, revision: str, text: str) -> str:
    return sha256_text(f"{model_id}@{revision}\n{text}")


def rerank_key(model_id: str, revision: str, query: str, passage: str) -> str:
    return sha256_text(f"{model_id}@{revision}\n{query}\n\x1f\n{passage}")


class EmbeddingCache:
    def __init__(self, cache_dir: Path, name: str, model_id: str, revision: str, allow_compute: bool) -> None:
        self.path = Path(cache_dir) / f"emb-{name}.npz"
        self.model_id = model_id
        self.revision = revision
        self.allow_compute = allow_compute
        self._rows: dict[str, np.ndarray] = {}
        self._used: set[str] = set()
        self._dirty = False
        if self.path.exists():
            with np.load(self.path) as data:
                for key, vector in zip(data["keys"].tolist(), data["vectors"], strict=True):
                    self._rows[key] = vector.astype(np.float32)

    def __len__(self) -> int:
        return len(self._rows)

    def keys(self) -> set[str]:
        return set(self._rows)

    def get_many(self, texts: Sequence[str], encode: Callable[[list[str]], np.ndarray]) -> np.ndarray:
        keys = [embedding_key(self.model_id, self.revision, text) for text in texts]
        missing = sorted(
            {key: text for key, text in zip(keys, texts, strict=True) if key not in self._rows}.items()
        )
        if missing:
            if not self.allow_compute:
                raise CacheMiss(
                    f"{len(missing)} embeddings missing from {self.path} and computing is disabled"
                )
            vectors = encode([text for _, text in missing])
            for (key, _), vector in zip(missing, vectors, strict=True):
                self._rows[key] = np.asarray(vector, dtype=np.float32)
            self._dirty = True
        self._used.update(keys)
        return np.stack([self._rows[key] for key in keys]) if keys else np.zeros((0, 0), dtype=np.float32)

    def prune_unused(self) -> int:
        """Drop every entry not requested since loading (deletion propagation); returns the number removed."""
        stale = [key for key in self._rows if key not in self._used]
        for key in stale:
            del self._rows[key]
        self._dirty = self._dirty or bool(stale)
        return len(stale)

    def save(self) -> None:
        """Write only when entries were added or pruned, so cached runs leave committed fixtures untouched."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._rows)
        vectors = np.stack([self._rows[key] for key in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(self.path, keys=np.array(keys, dtype="<U64"), vectors=vectors)


class RerankCache:
    def __init__(self, cache_dir: Path, name: str, model_id: str, revision: str, allow_compute: bool) -> None:
        self.path = Path(cache_dir) / f"rerank-{name}.json"
        self.model_id = model_id
        self.revision = revision
        self.allow_compute = allow_compute
        self._scores: dict[str, float] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        )
        self._used: set[str] = set()
        self._dirty = False

    def __len__(self) -> int:
        return len(self._scores)

    def get_many(
        self, pairs: Sequence[tuple[str, str]], score: Callable[[list[tuple[str, str]]], Sequence[float]]
    ) -> list[float]:
        keys = [rerank_key(self.model_id, self.revision, query, passage) for query, passage in pairs]
        missing = sorted(
            {key: pair for key, pair in zip(keys, pairs, strict=True) if key not in self._scores}.items()
        )
        if missing:
            if not self.allow_compute:
                raise CacheMiss(
                    f"{len(missing)} rerank scores missing from {self.path} and computing is disabled"
                )
            values = score([pair for _, pair in missing])
            for (key, _), value in zip(missing, values, strict=True):
                self._scores[key] = round(float(value), 6)
            self._dirty = True
        self._used.update(keys)
        return [self._scores[key] for key in keys]

    def prune_unused(self) -> int:
        stale = [key for key in self._scores if key not in self._used]
        for key in stale:
            del self._scores[key]
        self._dirty = self._dirty or bool(stale)
        return len(stale)

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._scores, sort_keys=True, indent=0) + "\n"
        self.path.write_bytes(text.encode("utf-8"))
