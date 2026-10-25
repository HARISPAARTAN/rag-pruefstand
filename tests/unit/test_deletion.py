"""Deletion propagation: dropping a law and rebuilding leaves no chunk and no cached embedding of it."""

import numpy as np

from pruefstand.chunking import chunk_c1, chunk_c2, load_tokenizer
from pruefstand.corpus.gii import parse_law_xml
from pruefstand.retrieval.cache import EmbeddingCache, embedding_key

BASE = "https://www.gesetze-im-internet.de"


def fake_encode(texts):
    return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


def test_deleted_law_leaves_no_chunk_or_embedding(root, tmp_path):
    xml = (root / "tests" / "fixtures" / "gii_sample.xml").read_bytes()
    kept, _ = parse_law_xml(xml, "testg", BASE, "2026-10-25")
    dropped, _ = parse_law_xml(xml.replace(b"TestG", b"OtherG"), "otherg", BASE, "2026-10-25")
    tokenizer = load_tokenizer(str(root / "data" / "tokenizer" / "multilingual-e5-base.tokenizer.json.xz"))

    before = chunk_c2(kept + dropped, tokenizer, 350) + chunk_c1(kept + dropped, tokenizer, 350, 50)
    cache = EmbeddingCache(tmp_path, "e5", "model", "rev", allow_compute=True, max_seq_length=512)
    cache.get_many([chunk.text for chunk in before], fake_encode)
    cache.save()
    assert any(chunk.law_slug == "otherg" for chunk in before)

    after = chunk_c2(kept, tokenizer, 350) + chunk_c1(kept, tokenizer, 350, 50)
    rebuilt = EmbeddingCache(tmp_path, "e5", "model", "rev", allow_compute=False, max_seq_length=512)
    rebuilt.get_many([chunk.text for chunk in after], fake_encode)
    assert rebuilt.prune_unused() > 0
    rebuilt.save()

    assert {chunk.law_slug for chunk in after} == {"testg"}
    assert all(cover["unit_id"].startswith("testg:") for chunk in after for cover in chunk.covers)
    reloaded = EmbeddingCache(tmp_path, "e5", "model", "rev", allow_compute=False, max_seq_length=512)
    assert reloaded.keys() == {embedding_key("model", "rev", chunk.text, 512) for chunk in after}
