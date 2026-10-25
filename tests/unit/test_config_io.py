import gzip

import pytest

from pruefstand.config import deep_merge, load_config
from pruefstand.io import canonical_json, read_jsonl, sha256_text, write_jsonl


def test_deep_merge_keeps_unrelated_keys():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = deep_merge(base, {"a": {"y": 5}, "c": 4})
    assert merged == {"a": {"x": 1, "y": 5}, "b": 3, "c": 4}
    assert base["a"]["y"] == 2


def test_ci_profile_overrides_paths(root):
    cfg = load_config(root / "pruefstand.toml", "ci")
    assert cfg.run_id == "ci"
    assert cfg["corpus"]["laws"] == ["messbg"]
    assert cfg["retrieval"]["allow_compute"] is False
    assert cfg["retrieval"]["top_k"] == 10
    assert cfg.run_dir == root / "runs" / "ci"


def test_unknown_profile_raises(root):
    with pytest.raises(ValueError, match="unknown profile"):
        load_config(root / "pruefstand.toml", "nope")


def test_llm_mode_env_override(root, monkeypatch):
    cfg = load_config(root / "pruefstand.toml", "full")
    monkeypatch.setenv("PRUEFSTAND_LLM_MODE", "record")
    assert cfg.llm_mode() == "record"
    monkeypatch.setenv("PRUEFSTAND_LLM_MODE", "bogus")
    with pytest.raises(ValueError):
        cfg.llm_mode()


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": "ä"}) == canonical_json({"a": "ä", "b": 1}) == '{"a":"ä","b":1}'
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_jsonl_gz_roundtrip_is_byte_stable(tmp_path):
    rows = [{"id": 1, "text": "Absatz"}, {"id": 2, "text": "§ 14a"}]
    first, second = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    write_jsonl(first, rows)
    write_jsonl(second, rows)
    assert first.read_bytes() == second.read_bytes()
    assert read_jsonl(first) == rows
    assert gzip.decompress(first.read_bytes()).decode("utf-8").count("\n") == 2
