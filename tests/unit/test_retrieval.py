import numpy as np
import pytest

from pruefstand.retrieval.bm25 import BM25Index, analyze
from pruefstand.retrieval.cache import CacheMiss, EmbeddingCache, RerankCache, embedding_key
from pruefstand.retrieval.dense import DenseIndex
from pruefstand.retrieval.fusion import rrf
from pruefstand.retrieval.ranking import top_k
from pruefstand.retrieval.rerank import rerank_order

TEXTS = [
    "Der Messstellenbetreiber muss die Messstelle betreiben.",
    "Der Netzbetreiber schließt die Anlage an das Netz an.",
    "Für den Wechsel des Messstellenbetreibers darf kein Entgelt erhoben werden.",
]


def test_analyze_stems_and_drops_stopwords():
    tokens = analyze("Die Messstellenbetreiber und der Netzbetreiber")
    assert "die" not in tokens and "und" not in tokens
    assert tokens == ["messstellenbetreib", "netzbetreib"]


def test_bm25_ranks_matching_text_first():
    index = BM25Index(TEXTS)
    hits = index.search("Entgelt für den Wechsel", k=3)
    assert hits[0][0] == 2
    assert all(score > 0 for _, score in hits)
    assert index.search("der die das", k=3) == []


def test_top_k_breaks_ties_by_index_and_rounds():
    scores = np.array([0.5, 0.7000001, 0.7, 0.1])
    assert top_k(scores, 3) == [(1, 0.7), (2, 0.7), (0, 0.5)]
    assert top_k(scores, 2, positive_only=True) == [(1, 0.7), (2, 0.7)]


def test_rrf_known_example():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    expected_1 = 1 / 61 + 1 / 62
    expected_3 = 1 / 63 + 1 / 61
    assert [doc for doc, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(expected_1)
    assert fused[1][1] == pytest.approx(expected_3)
    assert rrf([[1, 2, 3]], k=60, depth=2) == [(1, round(1 / 61, 9)), (2, round(1 / 62, 9))]


def test_rerank_order_sorts_by_score_and_keeps_ties_stable():
    assert rerank_order([7, 8, 9], [0.1, 0.9, 0.1]) == [(8, 0.9), (7, 0.1), (9, 0.1)]


def test_dense_index_cosine_and_tenant_filter():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32)
    index = DenseIndex(vectors, tenants=["K1", "K2", "K1"])
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert [doc for doc, _ in index.search(query, 3)] == [0, 2, 1]
    for tenant in ("K1", "K2"):
        hits = index.search(query, 3, tenant=tenant)
        assert hits and all(index.tenants[doc] == tenant for doc, _ in hits)


def test_bm25_tenant_filter_never_returns_other_tenant():
    index = BM25Index(TEXTS, tenants=["K1", "K2", "K2"])
    hits = index.search("Messstellenbetreiber Netzbetreiber Entgelt", k=3, tenant="K1")
    assert [doc for doc, _ in hits] == [0]


def test_embedding_cache_miss_raises_when_compute_disabled(tmp_path):
    cache = EmbeddingCache(tmp_path, "e5", "model", "rev", allow_compute=False)
    with pytest.raises(CacheMiss):
        cache.get_many(["passage: a"], lambda texts: np.zeros((len(texts), 2)))


def test_embedding_cache_roundtrip_and_prune(tmp_path):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    cache = EmbeddingCache(tmp_path, "e5", "model", "rev", allow_compute=True)
    first = cache.get_many(["aa", "bbb"], encode)
    cache.save()
    assert first.shape == (2, 2) and sorted(calls[0]) == ["aa", "bbb"]
    assert first[0].tolist() == [2.0, 1.0]
    reloaded = EmbeddingCache(tmp_path, "e5", "model", "rev", allow_compute=False)
    assert np.array_equal(reloaded.get_many(["bbb"], encode), np.array([[3.0, 1.0]], dtype=np.float32))
    assert reloaded.prune_unused() == 1
    reloaded.save()
    assert EmbeddingCache(tmp_path, "e5", "model", "rev", False).keys() == {
        embedding_key("model", "rev", "bbb")
    }


def test_cache_key_depends_on_model_revision():
    assert embedding_key("m", "r1", "text") != embedding_key("m", "r2", "text")


def test_rerank_cache_scores_and_miss(tmp_path):
    cache = RerankCache(tmp_path, "rr", "model", "rev", allow_compute=True)
    assert cache.get_many([("q", "p1"), ("q", "p2")], lambda pairs: [0.25, 0.75]) == [0.25, 0.75]
    cache.save()
    frozen = RerankCache(tmp_path, "rr", "model", "rev", allow_compute=False)
    assert frozen.get_many([("q", "p2")], lambda pairs: []) == [0.75]
    with pytest.raises(CacheMiss):
        frozen.get_many([("q", "p3")], lambda pairs: [0.0])
