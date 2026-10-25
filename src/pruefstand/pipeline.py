"""Stage functions over files in runs/<run_id>/."""

from __future__ import annotations

from pruefstand.chunking import chunk_c1, chunk_c2, chunks_path, load_tokenizer, write_chunks
from pruefstand.config import Config
from pruefstand.corpus.gii import build_units, load_units


def stage_corpus_build(cfg: Config) -> list:
    corpus = cfg["corpus"]
    return build_units(
        corpus["laws"],
        corpus["base_url"],
        cfg.path(corpus["zip_dir"]),
        cfg.path(corpus["snapshot_path"]),
        cfg.path(corpus["units_path"]),
    )


def stage_chunk(cfg: Config) -> dict[str, int]:
    units = load_units(cfg.path(cfg["corpus"]["units_path"]))
    tokenizer = load_tokenizer(str(cfg.path(cfg["tokenizer"]["path"])))
    chunking = cfg["chunking"]
    corpus_id = cfg["corpus"]["id"]
    c1 = chunk_c1(units, tokenizer, chunking["c1_window"], chunking["c1_overlap"], corpus_id)
    c2 = chunk_c2(units, tokenizer, chunking["c2_max_tokens"], corpus_id)
    write_chunks(chunks_path(cfg, "C1"), c1)
    write_chunks(chunks_path(cfg, "C2"), c2)
    return {"C1": len(c1), "C2": len(c2)}
